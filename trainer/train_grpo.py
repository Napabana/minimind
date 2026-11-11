import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import math
import re
import gc
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AutoTokenizer
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model, LMForRewardModel
from trainer.rollout_engine import create_rollout_engine

warnings.filterwarnings('ignore')


def rep_penalty(text, n=3, cap=0.5):
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0


def calculate_rewards(prompts, responses, reward_model):
    rewards = torch.zeros(len(responses), device=args.device)

    with torch.no_grad():
        reward_model_scores = []
        batch_size = len(prompts)

        for i in range(batch_size):
            for j in range(args.num_generations):
                response_idx = i * args.num_generations + j
                response = responses[response_idx]
                prompt = prompts[i]
                #两个捕获组 
                #第一个是三选一 抓到角色名字
                #第二个是(.*?) 非贪婪匹配，到第一个 <\|im_end\|> 就停。
                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                #re.DOTALL：让 . 也能匹配换行符 \n
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]
                answer = response
                
                #1.长度奖励20~800 规则评分
                rewards[response_idx] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
                #2.分割思考和答案，思考长度合理20~300奖励
                #3.结束标记有恰好只有依次加分 格式检查
                if '</think>' in response:
                    thinking_content, answer_content = response.split('</think>', 1)
                    rewards[response_idx] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
                    rewards[response_idx] += 0.25 if response.count('</think>') == 1 else -0.25
                    answer = answer_content.strip()
                    #4.正式答案计算重复度（比如连续 n-gram 重复率），重复越多扣越多 ——抑制 RL 微调常见的“复读机”退化。
                rewards[response_idx] -= rep_penalty(answer)
                
                #rewards评估回答质量
                score = reward_model.get_score(messages, answer)
                reward_model_scores.append(score)

        # 总 reward = 长度(±0.5) + 思考长度(±1.0) + 格式(±0.25) − 重复惩罚 + reward_model 分数 
        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        #rewards:[B x G]
        #数学题可以使用准确答案校验作为 Reward；代码任务可以使用测试结果或验证器；而 MiniMind 当前通用 GRPO 脚本采用的是上述混合奖励。
        rewards += reward_model_scores

    return rewards


def grpo_train_epoch(epoch, loader, iters, rollout_engine, ref_model, reward_model, start_step=0, wandb=None, use_sglang=False):
    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch['prompt']  # list[str], length B
        prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False,
                                  padding_side="left", add_special_tokens=False).to(args.device)
        if args.max_seq_len:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]

        rollout_result = rollout_engine.rollout(
            prompt_ids=prompt_inputs["input_ids"],
            attention_mask=prompt_inputs["attention_mask"],
            num_generations=args.num_generations,
            max_new_tokens=args.max_gen_len,
            temperature=0.8,
        )
        outputs = rollout_result.output_ids
        completion_ids = rollout_result.completion_ids
        completions = rollout_result.completions
        #`detach()` 表示这些旧概率不参与梯度计算。
        #Rollout 时的策略概率
        old_per_token_logps = rollout_result.per_token_logps.to(args.device).detach()
        #有prompt_lens个提示词，提示词长度
        prompt_lens = rollout_result.prompt_lens.to(args.device)
        full_mask = (outputs != tokenizer.pad_token_id).long()
        logp_pos = prompt_lens.unsqueeze(1) - 1 + torch.arange(completion_ids.size(1), device=args.device).unsqueeze(0)
        full_mask.scatter_(1, logp_pos + 1, rollout_result.completion_mask.to(args.device, dtype=full_mask.dtype))

        rewards = calculate_rewards(prompts, completions, reward_model).to(args.device)  # [B*num_gen]

        model_unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
        #对生成结果重新计算概率：
        with autocast_ctx:
            res = model_unwrapped(outputs, attention_mask=full_mask)
            aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
            #logits是[B,L,V]batch、序列长度、词表大小
            #B 第几个句子
            #L 句子里第几个位置：对应每个token
            #V 候选答案的打分表
            #logits[b,t,:] 位置 t 上，“如果下一个词是词表里的第 0 个词，打 x 分；是第 1 个词，打 y 分；……”    
            # logits[b, t] 是一个长度为 V 的向量：             
            #    [词0的分数, 词1的分数, ..., 词6399的分数]   

            #logits丢掉要预测的最后一个token：在词表维做log_softmax → 每个位置得到“预测每个 token 的对数概率”。
            
            #和output的label对齐 label扔掉第一个，ubsqueeze(-1):[B,L-1,1]
            # #gather：沿词表维度，用token_id当下标去查，模型给真实发生的token打了多少logp: 
            #我们知道下一个位置的真实的id，假设位置t实际下一个词id为302，就是要从6400个词中取出302，logits[b,t,302]
            
            #再去掉最后一维，这里就是每个token位置的logp(token)
            #gather:用logp_pos 预先生成好的位置下标，只挑出生成的answer部分对应的logp，丢掉提示词部分和无用的padding
            # dim=1 沿序列维挑，logp_pos[b, k] 给出“样本 b 的 answer 第 k 个词在原序列的位置   p”，就取 logp[b, p]。也是“每行下标不同，一次取完”。    
            #当前 Policy 概率，参与梯度计算
            per_token_logps = F.log_softmax(res.logits[:, :-1, :], dim=-1).gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)

        with torch.no_grad():
            #Reference 概率，用于 KL
            ref_per_token_logps = F.log_softmax(ref_model(outputs, attention_mask=full_mask).logits[:, :-1, :], dim=-1).gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)

        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            for i in range(len(prompts)):
                Logger(f"[DEBUG] step={step}, sample[{i}]")
                Logger('-'*100)
                Logger(f"{'=' * 30} [DEBUG] sample[{i}] CONTEXT_BEGIN {'=' * 30}")
                Logger(prompts[i])
                Logger(f"{'=' * 31} [DEBUG] sample[{i}] CONTEXT_END {'=' * 31}")
                for j in range(args.num_generations):
                    idx = i * args.num_generations + j
                    Logger(f"{'=' * 28} [DEBUG] gen[{j}] RESPONSE_BEGIN {'=' * 28}")
                    Logger(completions[idx])
                    Logger(f"{'=' * 29} [DEBUG] gen[{j}] RESPONSE_END {'=' * 29}")
                    Logger(f"[DEBUG] gen[{j}] reward={rewards[idx].item():.4f}")
                Logger('='*100)

        #按组重排： 把[Bxnum_gen]拆成[B, num_gen] 
        grouped_rewards = rewards.view(-1, args.num_generations)  # [B, num_gen]
        #组内均值：mean(dim=1)沿行内求平均 → 每个 prompt 得到一个组平均分 ，假设B=2 [μ1, μ2]。  
        #repeat_interleave 是逐元素重复，为了让每个回答都能减去它所属组的平均 Reward
        #[B]->[Bxnum_gen]
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)  # [B*num_gen]
        #组内标准差：std(dim=1, unbiased=False)沿行内求标准差 → 每个 prompt 得到一个组标准差  [σ1, σ2]。
        std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)  # [B*num_gen]
        #advantages = (rewards - mean_r) / (std_r + 1e-4)
        advantages = (rewards - mean_r) / (std_r + 1e-4)  # [B*num_gen]

        completion_pad_mask = rollout_result.completion_mask.to(args.device).bool()
        is_eos = (completion_ids == tokenizer.eos_token_id) & completion_pad_mask  # [B*num_gen, R]
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1) - 1, dtype=torch.long, device=args.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        completion_mask = ((torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)) & completion_pad_mask).int()  # [B*num_gen, R]

        kl_div = ref_per_token_logps - per_token_logps
        #Di,t:per_token_kl
        per_token_kl = torch.exp(kl_div) - kl_div - 1  # [B*num_gen, R]
        # ρi,t:ratio
        ratio = torch.exp(per_token_logps - old_per_token_logps)  # [B*num_gen, R]
        if args.loss_type == "cispo":
            #Clipped Importance Sampling Policy Optimization
            #只设置了上界 ，且裁剪后不沿ratio本身传播梯度
            clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
            per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - args.beta * per_token_kl)
        else:
            #clip函数
            clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
            #advantage就是A 优势
            per_token_loss1 = ratio * advantages.unsqueeze(1)
            per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
            #对应公式 min(ρA,clip(ρ,1-ε,1+ε)*A)-βD
            #[B, K]
            per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)

# `completion_mask` 很重要。
# 它屏蔽 Padding 和 EOS 后面的无效生成位置，同时保留有效的生成 Token（包括 EOS）。
# 先对每条回答的有效 Token Loss 求平均，再对 Batch 中所有回答求平均。
        policy_loss = (
            (per_token_loss * completion_mask).sum(dim=1) 
            / completion_mask.sum(dim=1).clamp(min=1)
            #clamp(min=1)：防除零——万一某条回答一个有效 token 都没有，分母兜底为 1，不出 NaN。 
        ).mean()

        loss = (policy_loss + aux_loss) / args.accumulation_steps  # scalar
        loss.backward()

        if step % args.accumulation_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step % args.log_interval == 0 or step == iters:
            policy_loss_val = loss.item() * args.accumulation_steps
            current_aux_loss = aux_loss.item()
            avg_reward_val = rewards.mean().item()
            avg_len_val = completion_mask.sum(dim=1).float().mean().item()
            kl_ref_val = ((ref_per_token_logps - per_token_logps) * completion_mask).sum().item() / max(completion_mask.sum().item(), 1)
            advantages_mean_val = advantages.mean().item()
            advantages_std_val = advantages.std().item()
            current_lr = optimizer.param_groups[0]['lr']

            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                   f'Reward: {avg_reward_val:.4f}, KL_ref: {kl_ref_val:.4f}, '
                   f'Adv Std: {advantages_std_val:.4f}, Adv Mean: {advantages_mean_val:.4f}, '
                   f'Actor Loss: {policy_loss_val:.4f}, Avg Response Len: {avg_len_val:.2f}, Learning Rate: {current_lr:.8f}')

            if wandb and is_main_process():
                wandb.log({
                    "reward": avg_reward_val,
                    "kl_ref": kl_ref_val,
                    "advantages_std": advantages_std_val,
                    "advantages_mean": advantages_mean_val,
                    "policy_loss": policy_loss_val,
                    "avg_response_len": avg_len_val,
                    "learning_rate": current_lr
                })

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scheduler=scheduler)
            model.train()
            del state_dict

        if step % args.save_interval == 0 or step == iters: rollout_engine.update_policy(model)

        del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps
        del completions, rewards, grouped_rewards, mean_r, std_r, advantages, completion_mask, completion_pad_mask, prompt_lens, logp_pos

    if step > start_step and step % args.accumulation_steps != 0:
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind GRPO (Group Relative Policy Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='grpo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_seq_len', default=768, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1024, help="生成的最大长度")
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif.jsonl", help="RLAIF数据路径")
    parser.add_argument("--num_generations", type=int, default=6, help="每个prompt生成的样本数")
    parser.add_argument("--beta", type=float, default=0.1, help="KL惩罚系数")
    parser.add_argument("--loss_type", type=str, default="cispo", choices=["grpo", "cispo"], help="loss类型")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GRPO的PPO clip epsilon")
    parser.add_argument("--epsilon_high", type=float, default=5.0, help="epsilon上界")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-GRPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--debug_mode", action="store_true", help="是否打印训练调试采样")
    parser.add_argument("--debug_interval", type=int, default=20, help="debug模式下每隔多少step打印一次采样")
    parser.add_argument("--thinking_ratio", type=float, default=0.9, help="按概率开启thinking（0.0~1.0）")
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"], help="rollout引擎类型")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8998", help="SGLang服务器URL")
    parser.add_argument("--sglang_model_path", type=str, default="../model", help="SGLang tokenizer路径")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_grpo", help="SGLang共享存储路径")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               max_seq_len=args.max_seq_len + args.max_gen_len, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-GRPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 初始化模型和数据 ==========
    base_weight = args.from_weight
    # Policy模型
    model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    # Reference模型
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    #计算KLpenalty KL惩罚
    ref_model = ref_model.eval().requires_grad_(False)
    # Reward模型 这个奖励模型是什么模型，单独的小模型吗？ 默认配置为 `internlm2-1_8b-reward` 对应的模型路径
    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
    # Rollout引擎（可插拔替换，只负责 policy 推理） 使用policy模型进行采样
    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine,
        policy_model=model,
        tokenizer=tokenizer,
        device=args.device,
        #把混合精度上下文传给内部函数。让被包裹的代码块在前向计算时以低精度（如 fp16/bf16）运行——省显存、加速，并保证与训练时精度一致。
        autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url,
        sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path,
    )
    # 数据和优化器
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len, thinking_ratio=args.thinking_ratio)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    total_optimizer_steps = math.ceil(iters / args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scheduler.load_state_dict(ckp_data['scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
        rollout_engine.update_policy(model)
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    rollout_engine.update_policy(model)
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            grpo_train_epoch(epoch, loader, len(loader) + skip, rollout_engine, ref_model, reward_model, start_step, wandb, use_sglang = (args.rollout_engine == "sglang"))
        else:
            grpo_train_epoch(epoch, loader, len(loader), rollout_engine, ref_model, reward_model, 0, wandb, use_sglang = (args.rollout_engine == "sglang"))
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()