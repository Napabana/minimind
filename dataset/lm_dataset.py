from torch.utils.data import Dataset
import torch
import json
import os
import random
from datasets import load_dataset, Features, Sequence, Value
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def pre_processing_chat(conversations, add_system_ratio=0.2):
    # tool use 数据完整保留不做处理
    if any(conv.get('tools') for conv in conversations): return conversations

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    # 概率性添加system
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations

def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    # 以80%概率移除空思考标签
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content

class PretrainDataset(Dataset):
    #路径，分词器，每条样本最终固定长度
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        #self.samples 中每条数据至少需要有： {"text": "一段预训练文本"}
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        tokens = self.tokenizer(str(sample['text']), add_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids
        #手动添加BOS 和 EOS
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        #补齐pad到max_length
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        #转换成整数张量
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100#错位在模型中完成
        return input_ids, labels



class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        features = Features({'conversations': [{'role': Value('string'), 'content': Value('string'), 'reasoning_content': Value('string'), 'tools': Value('string'), 'tool_calls': Value('string')}]})
        self.samples = load_dataset('json', data_files=jsonl_path, split='train', features=features)
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def generate_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = pre_processing_chat(sample['conversations'])
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)
        # # === 调试打印 ===
        # print(f"\n--- Sample {index} ---")
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([input_ids[i+1]])!r:16s} label={y}")
        # # ================
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        self.samples = load_dataset('json', data_files=file_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        chosen = sample['chosen']  # 是一个 list，里面包含若干 {role, content}
        rejected = sample['rejected']  # 同上
        # tokenize=False：先得到格式化的文本，不立即转换 Token ID。
        # add_generation_prompt=False：因为数据中已经包含完整的 Assistant 回答，不需要额外添加待生成的 Assistant 前缀。
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        chosen_prompt = post_processing_chat(chosen_prompt)

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = post_processing_chat(rejected_prompt)
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        chosen_input_ids = chosen_encoding['input_ids']
        # 只计算 Assistant 生成部分的概率。忽略提问只看问答
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)
        # 使用自回归语言模型的 Next Token Prediction。
        #x_chosen 获取除了最后一个元素之外的所有元素。
        #y_chosen 获取除了第一个元素之外的所有元素。
        # mask_chosen 掩码跟着后移
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    # 找到 Assistant 消息的起始位置。
    # 找到 Assistant 消息结束位置。
    # 将 Assistant 正文和结束标记对应的位置设为 1。
    def generate_loss_mask(self, input_ids):
        # 初始化 Loss Mask，长度与 input_ids 相同。
        # 默认全部为 0，表示暂时不参与 Loss 计算。
        loss_mask = [0] * len(input_ids)
        # i：当前扫描到的 Token 位置。
        i = 0

        # 从头到尾扫描整个 Token 序列。
        while i < len(input_ids):
            # 判断当前位置开始的 Token 序列，
            # 是否与 Assistant 消息起始标记 bos_id 完全一致。
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:

                # 找到 Assistant 消息起始标记。
                # start 指向 Assistant 正文的第一个 Token。
                # 不包含 bos_id 本身。
                start = i + len(self.bos_id)

                # end 从正文起始位置开始，
                # 用于向后搜索 Assistant 消息的结束标记。
                end = start
                
                # 在当前 Assistant 消息内部向后扫描。
                while end < len(input_ids):

                    # 判断当前位置开始的 Token 序列，
                    # 是否与 Assistant 消息结束标记 eos_id 完全一致。
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        # 找到结束标记，停止搜索。
                        break

                    # 尚未找到结束标记，继续向后移动一个 Token。
                    end += 1

                # 将 Assistant 正文和结束标记对应的位置设为 1。
                # end 是结束标记的起始位置，因此需要加上 eos_id 长度。
                # min 用于限制标记范围不超过 max_length。
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1

                # 当前 Assistant 消息处理完毕。
                # 跳过结束标记，继续扫描后面的消息。
                # 如果没有找到结束标记，则直接结束整个扫描。
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)

            else:
                # 当前 Token 不是 Assistant 消息的起始标记，
                # 继续向后移动一个 Token。
                i += 1

        # 返回生成的 Loss Mask。
        return loss_mask

#产出干净的，待模型作答的问题文本
class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024, thinking_ratio=0.5):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.thinking_ratio = thinking_ratio  # 按概率开启 thinking
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        #清洗/格式化  原始对话数据，变成标准的 chat 格式（如 [{"role": "user", "content": ...}, {"role": "assistant",   "content": ...}]）。           
        conversations = pre_processing_chat(conversations)
        use_thinking = random.random() < self.thinking_ratio
        return self.tokenizer.apply_chat_template(
            #读取的是已有对话的前缀
            #关键一步。原始对话最后一条是 assistant 的回答，切掉它，只保留"问题前缀"。因为 GRPO 不需要现成答案——答案要模型自己生成再打分。
            conversations[:-1],
            #GRPO训练是 提示词(字符串)+模型生成的answer(字符串) 拼接后再tokenize: 所以这里保留了字符串是可拼接可复制的
            tokenize=False,
            open_thinking=use_thinking,
            #add_generation_prompt=True 会在末尾补上assistant 的开头标记，这样拼出来的 prompt 可以直 接送进模型续写回答。
            add_generation_prompt=True
        )
    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = self.create_chat_prompt(sample['conversations'])

        #answer为空，模型会根据提示词在线生成回答，再由reward model评分
        return {
            'prompt': prompt,
            'answer': ""
        }

class AgentRLDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.samples)

    def parse_conversations(self, conversations):
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                # 如果 `tools` 是字符串，就先通过 `json.loads()` 转换成 Python 对象。
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            messages.append(message)
            # 移除 conversations 的最后一条消息
        return messages[:-1], tools

    #返回 初始对话上下文，可调用工具的定义，和Groudtruth(标准答案)
    def __getitem__(self, index):
        sample = self.samples[index]
        messages, tools = self.parse_conversations(sample['conversations'])
        return {'messages': messages, 'tools': tools, 'gt': sample['gt']}


if __name__ == "__main__":
    pass