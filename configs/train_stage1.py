"""
Stage 1: Progressive Context Training (8K-16K)
- 2-3 epochs, fast convergence
- ~8K samples (short-context: F# core, library examples, Svelte components)
- Expected: 2-4 hours on L40S

Usage:
  python train_stage1.py --data ../data/formatted/stage1_train.jsonl --val ../data/formatted/stage1_val.jsonl
"""

import argparse
from pathlib import Path

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# ── Model Configuration ──────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3.5-27B"
MAX_SEQ_LENGTH = 16384  # Stage 1: 16K context
LOAD_IN_4BIT = False  # Train in BF16, NOT QLoRA
DTYPE = None  # Auto-detect (will use BF16 on supported GPUs)

# ── LoRA Configuration ───────────────────────────────────────────────
LORA_R = 16
LORA_ALPHA = 32  # alpha/r ratio of 2
LORA_DROPOUT = 0.0
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# ── Training Hyperparameters ─────────────────────────────────────────
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8  # Effective batch size = 8
EPOCHS = 3
LEARNING_RATE = 2e-4
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "cosine"
SEED = 3407
OUTPUT_DIR = "./outputs/stage1"


def main(data_path: str, val_path: str = None):
    # ── Load Model ───────────────────────────────────────────────────
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
        dtype=DTYPE,
    )

    # ── Apply LoRA ───────────────────────────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Critical for VRAM savings
        random_state=SEED,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    # ── Chat Template ────────────────────────────────────────────────
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    def formatting_func(examples):
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        return {"text": texts}

    # ── Load Dataset ─────────────────────────────────────────────────
    dataset = load_dataset("json", data_files=data_path, split="train")
    dataset = dataset.map(formatting_func, batched=True)

    eval_dataset = None
    if val_path and Path(val_path).exists():
        eval_dataset = load_dataset("json", data_files=val_path, split="train")
        eval_dataset = eval_dataset.map(formatting_func, batched=True)

    print(f"Training samples: {len(dataset)}")
    if eval_dataset:
        print(f"Validation samples: {len(eval_dataset)}")

    # ── Training Arguments ───────────────────────────────────────────
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
        logging_steps=10,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=3,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=250 if eval_dataset else None,
        seed=SEED,
        report_to="none",  # Set to "wandb" if using Weights & Biases
        gradient_checkpointing=True,
        optim="adamw_8bit",
    )

    # ── Trainer ──────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=True,  # Pack short sequences together for efficiency
    )

    # ── Train ────────────────────────────────────────────────────────
    print(f"Starting Stage 1 training (max_seq_length={MAX_SEQ_LENGTH})...")
    trainer.train()

    # ── Save ─────────────────────────────────────────────────────────
    model.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
    print(f"Stage 1 complete. LoRA adapter saved to {OUTPUT_DIR}/lora_adapter")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Training data JSONL")
    parser.add_argument("--val", default=None, help="Validation data JSONL")
    args = parser.parse_args()
    main(args.data, args.val)
