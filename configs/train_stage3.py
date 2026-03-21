"""
Stage 3: Progressive Context Training (128K)
- 1-2 epochs
- ~1.5K samples (long-context: full project implementations, long docs)
- Loads LoRA adapter from Stage 2
- Expected: 12-24 hours on L40S

Usage:
  python train_stage3.py --data ../data/formatted/stage3_train.jsonl --adapter ./outputs/stage2/lora_adapter
"""

import argparse
from pathlib import Path

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

MODEL_NAME = "Qwen/Qwen3.5-27B"
MAX_SEQ_LENGTH = 131072  # Stage 3: 128K context
LOAD_IN_4BIT = False

BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 4  # Smaller effective batch at long context
EPOCHS = 2
LEARNING_RATE = 5e-5  # Even lower LR
WARMUP_STEPS = 25
SEED = 3407
OUTPUT_DIR = "./outputs/stage3"


def main(data_path: str, adapter_path: str = None, val_path: str = None):
    if adapter_path and Path(adapter_path).exists():
        print(f"Loading with Stage 2 adapter from {adapter_path}...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=adapter_path,
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
            r=16,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=32,
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
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        bf16=True,
        fp16=False,
        logging_steps=1,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=50 if eval_dataset else None,
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
        packing=False,
    )

    print(f"Starting Stage 3 training (max_seq_length={MAX_SEQ_LENGTH})...")
    print("WARNING: This stage is slow (~2-4 min/step). Expected: 12-24 hours.")
    trainer.train()

    model.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
    print(f"Stage 3 complete. LoRA adapter saved to {OUTPUT_DIR}/lora_adapter")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--val", default=None)
    parser.add_argument("--adapter", default=None, help="Stage 2 LoRA adapter path")
    args = parser.parse_args()
    main(args.data, args.adapter, args.val)
