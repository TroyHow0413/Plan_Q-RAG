import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# ================= 核心配置区 =================
MODEL_ID = "google/gemma-4-E4B-it"
OUTPUT_DIR = "./gemma_planner_musique_cleaned"

LOCAL_DATASET_PATH = "./musique_ans_v1.0_train_clean.jsonl"

SYSTEM_PROMPT = (
    "You are an expert multi-hop question planner.\n"
    "Your task is to decompose a complex, multi-step question into a logical sequence of simple, self-contained sub-questions.\n\n"
    "Guidelines:\n"
    "1. Break the question into the minimum number of steps required to find the final answer.\n"
    "2. Use '#N' (e.g., #1, #2) to refer to the answer of a previous step. \n"
    "3. Strictly use the '#N' placeholder instead of repeating the full entity name once it has been identified.\n"
    "4. Each sub-question must be a complete, searchable sentence.\n"
    "5. Output format: \n"
    "   Step 1: [Sub-question]\n"
    "   Step 2: [Sub-question]\n"
    "   ...\n"
    "6. For multi-step reasoning (3+ hops), ensure each step logically leads to the next by correctly incrementing the '#N' reference.\n"
)
# ============================================


def format_musique_to_chat(example):
    """
    将 MuSiQue 的一条样本转换为 Qwen Chat 所需的 messages 格式。

    MuSiQue question_decomposition 字段结构：
    [
        {"id": 1, "question": "...", "answer": "...", "paragraph_support_idx": 0},
        {"id": 2, "question": "Where was #1 born?", "answer": "...", ...},
        ...
    ]
    只取 question 字段，保留 MuSiQue 原生的 #1/#2 占位符引用风格。
    """
    question = example["question"]
    decomposition = example["question_decomposition"]  # list of dicts

    steps = []
    for i, step in enumerate(decomposition, start=1):
        steps.append(f"Step {i}: {step['question']}")
    assistant_output = "\n".join(steps)

    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": f"Decompose the following question:\n\n{question}"},
        {"role": "assistant", "content": assistant_output},
    ]
    return {"messages": messages}


def main():
    # ── 1. Tokenizer ──────────────────────────────────────────────
    print("[1/5] 正在加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # ✅ 直接复用 eos_token 作为 pad_token，词表大小完全不变
    # 不需要 add_special_tokens / resize_token_embeddings
    # 避免产生词表 size mismatch（151643 vs 151680 vs 152064 的问题根源）
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"   vocab size      : {len(tokenizer)}")
    print(f"   pad_token       : {tokenizer.pad_token!r}")
    print(f"   pad_token_id    : {tokenizer.pad_token_id}")
    print(f"   eos_token_id    : {tokenizer.eos_token_id}")

    # ── 2. 基础模型 ────────────────────────────────────────────────
    print("[2/5] 正在加载基础模型 (bfloat16 + sdpa)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", # "sdpa",
        device_map="auto", #自动分配到可用 GPU
    )

    # ✅ 词表没变，只需同步 pad_token_id 到 model config 即可
    model.config.pad_token_id = tokenizer.pad_token_id
    print(f"   model vocab size : {model.config.vocab_size}  (未改变)")

    # ── 3. LoRA 配置 ───────────────────────────────────────────────
    print("[3/5] 注入 LoRA 适配器...")
    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── 4. 加载 & 处理 MuSiQue 数据集 ─────────────────────────────
    print("[4/5] 加载并格式化 MuSiQue 数据集...")
    raw_dataset = load_dataset("json", data_files=LOCAL_DATASET_PATH, split="train")

    # ❶ 只保留有答案的样本
    raw_dataset = raw_dataset.filter(
        lambda x: x["answerable"] is True,
        num_proc=4,
    )
    print(f"   过滤后可用样本数：{len(raw_dataset)}")

    # ❷ 去掉分解步数 < 2 的单跳伪多跳问题
    raw_dataset = raw_dataset.filter(
        lambda x: len(x["question_decomposition"]) >= 2,
        num_proc=4,
    )
    print(f"   过滤单跳后样本数：{len(raw_dataset)}")

    # ❸ 转换为 messages 格式
    dataset = raw_dataset.map(
        format_musique_to_chat,
        num_proc=4,
        remove_columns=raw_dataset.column_names,
    )

    # ❹ 应用 Qwen Chat Template，生成训练用的纯文本 text 字段
    def apply_chat_template(example):
        example["text"] = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return example

    dataset = dataset.map(apply_chat_template, num_proc=4)

    # 打印一条样本确认格式
    print("\n   ── 样本预览 ──")
    print(dataset[0]["text"][:500])
    print("   ──────────────\n")

    # ── 5. SFT Trainer ────────────────────────────────────────────
    print("[5/5] 配置 SFT Trainer 并启动训练...")
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        max_length=2048,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,      # 等效 batch size = 16（单卡）
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_args,
        processing_class=tokenizer,
    )

    print("开始训练！")
    trainer.train()

    print("保存最终的 LoRA 权重...")
    trainer.model.save_pretrained(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
    print("训练完成！")


if __name__ == "__main__":
    main()