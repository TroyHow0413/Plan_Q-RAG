import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# ================= 核心配置区 =================
MODEL_ID = "google/gemma-4-E2B-it"
OUTPUT_DIR = "./test_gemma_planner/gemma_planner_musique_cleaned"

LOCAL_DATASET_PATH = "./test_gemma_planner/musique_ans_v1.0_train_clean.jsonl"

# [改动] Gemma-it 系列对 system role 的支持在不同版本有差异。
# gemma-3/4-it 支持 system message，但若训练时报 "system role not supported"，
# 可将 system prompt 合并到第一条 user message 中（见下方 format 函数注释）。
SYSTEM_PROMPT = (
    "You are a multi-hop question planner. "
    "Given a complex question that requires multiple reasoning steps, "
    "decompose it into a sequence of simple, self-contained sub-questions. "
    "Each sub-question should be answerable independently or by referring to "
    "the answer of a previous step (use '#1', '#2', ... as placeholders). "
    "Output each sub-question on a new line, prefixed with 'Step N:'."
)
# ============================================


def format_musique_to_chat(example):
    question = example["question"]
    decomposition = example["question_decomposition"]

    steps = []
    for i, step in enumerate(decomposition, start=1):
        steps.append(f"Step {i}: {step['question']}")
    assistant_output = "\n".join(steps)

    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": f"Decompose the following question:\n\n{question}"},
        {"role": "assistant", "content": assistant_output},
    ]

    # [备用方案] 若 apply_chat_template 报 system role 错误，改用下面这段：
    # messages = [
    #     {"role": "user", "content": f"{SYSTEM_PROMPT}\n\nDecompose the following question:\n\n{question}"},
    #     {"role": "assistant", "content": assistant_output},
    # ]

    return {"messages": messages}


def main():
    # ── 1. Tokenizer ──────────────────────────────────────────────
    print("[1/5] Loading Tokenizer...")

    # [改动] Gemma 是标准 HF 模型，不需要 trust_remote_code=True
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # [改动 + Bug修复] Gemma tokenizer 通常已内置 <pad> token（id=0）。
    # 原 Qwen 代码直接赋值 eos_token，对 Gemma 可能覆盖已有的 pad 设置。
    # 改为：只在 pad_token 确实缺失时才 fallback 到 eos_token。
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("   [warning] pad_token is None, fallback to eos_token")

    tokenizer.padding_side = "right"

    print(f"   vocab size      : {len(tokenizer)}")
    print(f"   pad_token       : {tokenizer.pad_token!r}")
    print(f"   pad_token_id    : {tokenizer.pad_token_id}")
    print(f"   eos_token_id    : {tokenizer.eos_token_id}")

    # ── 2. 基础模型 ────────────────────────────────────────────────
    print("[2/5] Loading base model (bfloat16 + sdpa)...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,           # [修复] torch_dtype → dtype
        attn_implementation="sdpa",
        device_map={"": 0},
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    # [修复] Gemma4Config 是多模态嵌套 config，vocab_size 在 text_config 子节点下
    # 用 getattr 做兼容，同时 fallback 到 len(tokenizer)
    vocab_size = getattr(
        getattr(model.config, "text_config", model.config),
        "vocab_size",
        len(tokenizer),
    )
    print(f"   model vocab size : {vocab_size}")

    model.enable_input_require_grads()

    # ── 3. LoRA 配置 ───────────────────────────────────────────────
    print("[3/5] Injecting LoRA adapters...")

    # [修复] 直接枚举语言模型层的完整路径，PEFT 支持精确全路径匹配
    # 这样完全绕开 vision_tower / audio_tower 的 Gemma4ClippableLinear 问题
    LORA_TARGET_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj")

    target_modules = [
        name for name, module in model.named_modules()
        if "language_model" in name
        and type(module).__name__ == "Linear"          # 只要标准 nn.Linear
        and name.endswith(LORA_TARGET_SUFFIXES)
    ]
    print(f"   LoRA target layers: {len(target_modules)} modules")
    print(f"   前3个: {target_modules[:3]}")
    
    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=target_modules,
        # [修复] 直接排除掉视觉和音频塔，PEFT >= 0.10 支持
        exclude_modules=["vision_tower", "audio_tower"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── 4. 加载 & 处理 MuSiQue 数据集 ─────────────────────────────
    print("[4/5] Loading and formatting MuSiQue dataset...")
    raw_dataset = load_dataset("json", data_files=LOCAL_DATASET_PATH, split="train")

    raw_dataset = raw_dataset.filter(
        lambda x: x["answerable"] is True,
        num_proc=4,
    )
    print(f"   Available samples after filtering: {len(raw_dataset)}")

    raw_dataset = raw_dataset.filter(
        lambda x: len(x["question_decomposition"]) >= 2,
        num_proc=4,
    )
    print(f"   Available samples after filtering: {len(raw_dataset)}")

    dataset = raw_dataset.map(
        format_musique_to_chat,
        num_proc=4,
        remove_columns=raw_dataset.column_names,
    )

    def apply_chat_template(example):
        example["text"] = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return example

    dataset = dataset.map(apply_chat_template, num_proc=4)

    print("\n   ── Sample Preview ──")
    print(dataset[0]["text"][:500])
    print("   ──────────────\n")

    # ── 5. SFT Trainer ────────────────────────────────────────────
    print("[5/5] Configuring SFT Trainer and starting training...")
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        max_length=2048,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        # [Bug修复] 原代码缺少此项。use_reentrant=True（默认值）在 LoRA + gradient
        # checkpointing 下会因 non-reentrant autograd 图断裂导致训练崩溃或静默出错。
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_args,
        processing_class=tokenizer,
    )
    
    print("Starting training!")
    trainer.train()

    print("Saving final LoRA weights...")
    trainer.model.save_pretrained(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
    print("Training completed!")


if __name__ == "__main__":
    main()