import math, torch, torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Config
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size #模型内部宽度
        self.num_hidden_layers = num_hidden_layers #隐藏层数量
        self.use_moe = use_moe #默认不用
        self.dropout = kwargs.get("dropout", 0.0)
        #此表默认大小6400
        self.vocab_size = kwargs.get("vocab_size", 6400)
        #BOS（序列开始）token 的 ID。
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        #EOS（序列结束）token 的 ID。
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        #使用 Flash Attention，以更省显存、更高效地计算注意力
        self.flash_attn = kwargs.get("flash_attn", True)
        #Query 注意力头数量。多个头可以从不同角度处理 token 间的关系。
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        #Key、Value 的头数。少于 Query 头数时属于 GQA，可降低 KV Cache 和计算开销。
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        #每个注意力头的维度；默认是 768 // 8 = 96
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        #前馈网络使用的激活函数。
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        #前馈网络的中间层大小。 默认配置计算后是 2432，并对齐到 64 的倍数。π 并不是 Transformer 理论规定的常数，也没有圆周率的几何含义。它只是作者为 FFN 中间层选取的一个扩宽比例：
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        #配置支持的最大序列位置数量，即上下文长度上限。
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        #RMSNorm 中防止除零、提高数值稳定性的小量。
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        #RoPE 旋转位置编码的基频参数，会影响模型表示长距离位置的方式。
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        #输入 Embedding 与输出词表映射共享权重，以减少参数量。
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        #推理时是否启用 RoPE 长上下文缩放；默认关闭。
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,#控制不同频率从“不缩放”到“缩放”的过渡范围。
            "factor": 16,#位置频率的缩放倍数。
            "original_max_position_embeddings": 2048,#缩放算法把 2048 当作原始上下文长度。
            "attention_factor": 1.0,#对 RoPE 产生的正余弦值进行幅度调整；1.0 表示不额外放大或缩小。
            "type": "yarn" #type="yarn"：使用 YaRN 长上下文缩放方法。
        } if self.inference_rope_scaling else None
        ### MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4)#4个专家 FFN。
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)#每个 token 只选择 1 个专家处理。
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)#每个专家内部 FFN 的中间层宽度。
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)#是否重新归一化被选中专家的路由权重。
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)#MoE 负载均衡辅助损失的权重，用来避免 token 过度集中到少数专家

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Model
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class RMSNorm(torch.nn.Module):#输入输出shape相同
    #一个 token 的全部特征 → 一起计算尺度 → 一起缩放
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    #输入[batch_size, sequence_length, hidden_size]
    def norm(self, x):#x / sqrt(mean(x²) + eps)  mean(-1) 在最后一维求平方平均值。
        #先求出均值，最后一维个数变成1，再广播到原来的维度大小
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)#将当前张量转换成与另一个张量相同的数据类型。

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    # dim：每个注意力头的维度，默认配置下是 96。
    # end：提前计算多少个位置，默认 32768。
    # rope_base：控制不同维度旋转频率的基数，默认 1e6。
    # rope_scaling：可选的长上下文缩放配置；默认是 None。

    #(torch.arange(0, dim, 2)[: (dim // 2)] 保证只保留偶数个【当dim是奇数时】
    #freq_i=1/rope_base^(2_i/dim)
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None: # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)
    t = torch.arange(end, device=freqs.device)
    #每一行对应一个 token 位置。
    # 每一列对应一个旋转频率。
    # 每个元素表示“某个位置在某个频率下的旋转角度”。
    freqs = torch.outer(t, freqs).float()#[i, j] = a[i] × b[j] -->  [32768] outer [48] → [32768, 48]

    #cos：对每个元素计算cos值 cat：拼接最后一维：让前后两半使用相同的旋转系数。
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin

#q：Query，shape 通常为 [batch, seq_len, heads, head_dim]
#cos.shape = [seq_len, head_dim]
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    #先把最后一位切半，后半部分取负，接上前半部分：相当于让配对维度做一次 90° 旋转
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    #旋转结果 = 原向量 × cos(角度)+ 旋转90°后的向量 × sin(角度)
    #unsqueeze(1) 给 cos 和 sin 增加“注意力头”这一维，使同一位置的所有注意力头都能使用对应的位置旋转参数。
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed
# GQA（Grouped Query Attention）中复制 Key/Value 头，让它们的头数与 Query 对齐
#x:[batch_size, seq_len, num_key_value_heads, head_dim]
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    #n_rep 表示每个 KV 头需要重复多少次：和query对齐
    if n_rep == 1: return x
    #增加重复维度，把kvhead都扩展n_rep份，最后合并
    #expand 主要创建重复视图，reshape 再把“KV 头”和“重复次数”合并成 8 个头。
    #expand：把长度为 1 的维度逻辑上扩展，不通常复制数据。
    #reshape：改变张量的形状，不改变数据内容。
    return (x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim))
#GQA
class Attention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        #query头数量，默认8
        self.n_local_heads = config.num_attention_heads
        #kv头数量，默认4
        self.n_local_kv_heads = self.num_key_value_heads
        #每个 KV 头需要共享给几个 Query 头  2
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        #每一个注意力头的宽度 96
        self.head_dim = config.head_dim
        #表示当前注意力是否使用因果注意力（Causal Attention）。是否禁止 token 关注未来 token。
        self.is_causal = True
        #输入768 输出768
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        #把多个注意力头合并后的结果投影回 hidden_size
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        #对每个 Query、Key头的 96 维向量做 RMSNorm
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        #对注意力权重做 dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        #对注意力模块最终输出做 dropout
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        #记录当前是否真的能够使用 Flash Attention
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        #拆分最后一维
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings#取出 RoPE 的 cos 和 sin
        #Value 不参与 RoPE，因为 Q 和 K 的点积负责决定“关注哪个位置”，Value 负责提供被取出的内容。
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        #在生成新token时，拼接过去和当前的维度，在dim=1的维度拼接
        # 自回归生成每次只增加一个 token。过去 token 的 K、V 不会改变，因此可以直接复用
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        #use_cache=True：返回合并后的 K、V，供下一次生成使用
        past_kv = (xk, xv) if use_cache else None
        # xq：[B, 当前序列长度, 8, 96]
        # xk：[B, KV序列长度, 4, 96]
        # xv：[B, KV序列长度, 4, 96]
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))
        #处理后，交换维度：注意力计算希望把 head 单独拿出来并行计算：这样就可以处理后面两维的xq @ xk.transpose(-2, -1)
        #让每个注意力头内部形成标准的矩阵计算：[Query长度, head_dim] @ [head_dim, Key长度]。
        # xq：[B, 8, 当前序列长度, 96]
        # xk：[B, 8, KV序列长度, 96]
        # xv：[B, 8, KV序列长度, 96]

        #环境启用Flash Attention & seq_len > 1 & 适合直接使用内置因果遮罩 &  没有 padding mask，或者 mask 全是 1
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            #一次计算完，最后得到输出
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
        else:
            #Q K^T / sqrt(d_k) 计算注意力分数  [B, 8, Tq, Tk] :   / sqrt(d_k)把分数控制在更稳定的范围。
            # 对每个 batch
            #     对每个注意力头
            #         每个 Query token
            #         分别与所有 Key token 计算一个相关性分数
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            #添加因果遮罩：禁止看未来
            # #  -seq_len: 只对当前新增的 Key 区域施加当前序列内部的因果遮罩；过去缓存的 Key 默认全部可见。
            if self.is_causal: scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
            #注意力掩码：屏蔽无效位置 1：有效 token 0：padding 或无效 token
            #attention_mask.shape
            # [B, seq_len]
            #         ↓ unsqueeze(1)
            # [B, 1, seq_len]
            #         ↓ unsqueeze(2)
            # [B, 1, 1, seq_len]
            if attention_mask is not None: scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
        # [B, 8, T, 96]->[B, T, 8, 96]->[B, T, 8*96]
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv
#FFN  SwiGLU
#把 gate_proj 理解成一道可学习的门：它决定 up_proj(x) 中哪些信息应该保留、增强或抑制。down_proj 再把扩宽后的表示压回 768 维，以便残差连接和下一层继续处理。
class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        # gate_proj：768  → 2432
        # up_proj：  768  → 2432
        # down_proj：2432 → 768
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        #self.act_fn 是 SiLU；整个 FeedForward 结构是 SiLU 门控的 SwiGLU 风格 FFN。
        #ACT2FN 是 Transformers 提供的激活函数映射表 ACT2FN["silu"] == SiLU 激活函数 
        #实际执行的是F.silu(...)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

# MoE-FFN
class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        #输出该token和4个专家的匹配程度
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        #包含 4 个独立的 FeedForward 专家。
        self.experts = nn.ModuleList([FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])
        #这里没用到，用的是每个专家自己FeedForward中的
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        #合并batch和seqlen [B, T, 768] → [B×T, 768] 可以直接表示token的数量了
        x_flat = x.view(-1, hidden_dim)
        #每个token获得4个专家的分数，softmax转分数为概率
        scores = F.softmax(self.gate(x_flat), dim=-1)
        #只选择1个，得到编号和权重
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)
        #是否重新归一化路由权重 其实默认topk为1的话，那专家权重基本会为1
        if self.config.norm_topk_prob: topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        #创建一个全零的结果容器
        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            #mask：二维：哪个 token 的哪个 top-k 专家槽位是当前专家
            mask = (topk_idx == i) #取出最匹配当前专家的token的id
            if mask.any():
                #取出True的token id
                #一维：哪些 token 需要交给当前专家
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = topk_weight[mask].view(-1, 1)
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            #为了让计算图记录专家参数与训练有关：防止某个专家参数未被使用
            elif self.training:
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
        #避免所有 token 都挤到同一个专家，让 4 个专家的任务量尽量均衡。
        if self.training and self.config.router_aux_loss_coef > 0:
            #load记录每个专家实际处理的token的概率
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            #scores.mean(0) 所有 token 平均认为每个专家应该获得多少概率。
            #负载均衡辅助损失：实际负载 × 平均路由概率 然后相加 
            #self.config.router_aux_loss_coef 控制辅助损失的影响强度，避免这个损失压过语言模型本身的训练损失。
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()
            #[B×T, 768]↓ view[B, T, 768]
        return y.view(batch_size, seq_len, hidden_dim)

class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        #让不同 token 交换信息
        self.self_attn = Attention(config)
        #进入 Attention 前做 RMSNorm
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        #进入 FFN/MoE 前做 RMSNorm
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        #对每个 token 的特征进行加工
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        #残差
        residual = hidden_states
        #pre-norm 先norm再进入子模块
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        #第一次残差输出= 原始输入 + Attention(RMSNorm(原始输入))
        hidden_states += residual
        #第二次残差=第一次残差输出+MLP(RMSNorm(第一次残差输出))
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value

class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        #词元嵌入层：6400 个 token ID → 每个 token 变成 768 维向量
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        Dropout
        self.dropout = nn.Dropout(config.dropout)
        #默认8个MiniMindBlock
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        # 最终 RMSNorm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        #初始化 RoPE 表
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        #register_buffer 表示： 运行张量，不是训练的参数，会一起移动到GPU
        #不会保存到模型参数文件
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        #一个兼容性判断
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        #当前输入从哪个位置开始计算 RoPE。
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        if self.freqs_cos[0, 0] == 0:#正常应该为1，如果为0说明与计算的PoPE buffer没有成功保留，需要重算，并移动到和hidden_state相同的设备
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
        #根据当前 token 的真实位置切片：区分没有缓存和有缓存两种情况
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present) #收集当前层缓存，供下次生成使用。
        hidden_states = self.norm(hidden_states)
        #只收集使用 MoE 的层
        #hidden_states.new_zeros(1).squeeze()创建一个与 hidden_states 同设备、同数据类型的标量 0，作为 sum 的初始值。
        # 这样即使模型没有任何 MoE 层，列表为空，也能得到模型相同设备、相同类型的 PyTorch Tensor。
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss
#最外层语言模型类
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        # 把隐藏状态转换成词表中每个 token 的预测分数。
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings: self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        #logits_to_keep 控制需要为哪些位置计算 logits。
        #Logits（逻辑值）是神经网络最后一层输出的原始分数，尚未经过归一化处理，用于表示模型对各类别的倾向性。
        #默认为0，保留全部位置，如果是1的话，只保留最后一个token
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        #需要用当前和之前预测下一个token
        if labels is not None: 
            #x：logits 删除最后一个位置    y：labels 删除第一个位置：
            #删除第一个标签，使两者错位一格后正确对齐。
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            #表示 label 为 -100 的位置不计算损失，通常用于忽略 padding 或不需要监督的部分。
            #数据处理时通常把不需要训练的位置改成：-100，-100 是一个特殊占位值，表示这个位置没有有效监督目标，不应该拿它和模型预测做比较。
            #因为合法tokenID 是0，1，，，，6399
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
    
    # https://github.com/jingyaogong/minimind/discussions/611
    @torch.inference_mode() #表示是推理过程
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        # 如果 kwargs 中传了 input_ids，优先使用它。
        # 否则使用 inputs。
        # 同时将 input_ids 从 kwargs 中删除，避免后面重复传递。
        # repeat(num_return_sequences, 1)：让同一个输入生成多份结果。第0维重复num_return_sequences，第二维重复1
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)
        #布尔标记： False：这条序列还在生成
        # True：这条序列已经生成 EOS
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        #流式输入
        if streamer: streamer.put(input_ids.cpu())
        for _ in range(max_new_tokens):
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            # 只取“尚未进入 KV Cache”的 token。
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
            # 每生成一个新 token，就在 mask 最后追加一个 1。
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
            # temperature=1：分数不变。
            # temperature<1：拉大分数差异，概率更集中，输出更稳定。
            # temperature>1：缩小分数差异，概率更平坦，输出更随机。
            logits = outputs.logits[:, -1, :] / temperature
            #重复惩罚，正数除法，负数乘法
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[i]); score = logits[i, seen]; logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
            #把比第topk小的都变为负无穷
            if top_k > 0: 
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            #Top-p（核采样 / Nucleus Sampling），作用是只保留累计概率达到 top_p 的最小候选集合，其余 token 不参与采样。
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                # 这里的 mask=True 表示“要删除”，不是“要保留”
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            #do_sample=True 转成概率随机选；false“直接选最大
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            #已完成序列强制生成 EOS
            if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
            #把新 token 加到序列末尾
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer: streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)#更新
                if finished.all(): break#示 batch 中所有序列都结束
        if streamer: streamer.end()
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids