"""
Stage 2: Progressive Context Training (32K-64K)
- 2 epochs
- ~3K samples (medium-context: multi-file examples, API implementations)
- Loads LoRA adapter from Stage 1 and continues training
- Expected: 3-6 hours on L40S

Usage:
  python train_stage2.py --data ../data/formatted/stage2_train.jsonl --adapter ./outputs/stage1/lora_adapter
"""

import argparse
from pathlib import Path

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# ── Stage 2 Configuration ────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3.5-27B"
MAX_SEQ_LENGTH = 65536  # Stage 2: 64K context
LOAD_IN_4BIT = False

LORA_R = 16
LORA_ALPHA = 32
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8
EPOCHS = 2
LEARNING_RATE = 1e-4  # Lower LR for continuation
WARMUP_STEPS = 50
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "cosine"
SEED = 3407
OUTPUT_DIR = "./outputs/stage2"


def main(data_path: str, adapter_path: str = None, val_path: str = None):
    # ── Load Model (with Stage 1 adapter if provided) ────────────────
    if adapter_path and Path(adapter_path).exists():
        print(f"Loading base model with Stage 1 adapter from {adapter_path}...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=adapter_path,  # Loads base + adapter
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=LOAD_IN_4BIT,
        )
    else:
        print("No adapter provided, loading base model...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=LOAD_IN_4BIT,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_R,
            target_modules=TARGET_MODULES,
            lora_alpha=LORA_ALPHA,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=SEED,
            max_seq_length=MAX_SEQ_LENGTH,
        )

    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    def formatting_func(examples):
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        return {"text": texts}

    dataset = load_dataset("json", data_files=data_path, split="train")
    dataset = dataset.map(formatting_func, batched=True)

    eval_dataset = None
    if val_path and Path(val_path).exists():
        eval_dataset = load_dataset("json", data_files=val_path, split="train")
        eval_dataset = eval_dataset.map(formatting_func, batched=True)

    print(f"Training samples: {len(dataset)}")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER,
        bf16=True,
        fp16=False,
        logging_steps=5,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=100 if eval_dataset else None,
        seed=SEED,
        report_to="none",
        gradient_checkpointing=True,
        optim="adamw_8bit",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,  # No packing at longer sequences
    )

    print(f"Starting Stage 2 training (max_seq_length={MAX_SEQ_LENGTH})...")
    trainer.train()

    model.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
    print(f"Stage 2 complete. LoRA adapter saved to {OUTPUT_DIR}/lora_adapter")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--val", default=None)
    parser.add_argument("--adapter", default=None, help="Stage 1 LoRA adapter path")
    args = parser.parse_args()
    main(args.data, args.adapter, args.val)
