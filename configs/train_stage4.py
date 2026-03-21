"""
Stage 4: Progressive Context Training (256K / 204800)
- 1 epoch
- ~500 samples (max-context: full project walkthroughs, long-context examples)
- Loads LoRA adapter from Stage 3
- Expected: 10-20 hours on L40S
- Use 204800 if training locally on 32GB VRAM

Usage:
  python train_stage4.py --data ../data/formatted/stage4_train.jsonl --adapter ./outputs/stage3/lora_adapter
  python train_stage4.py --data ../data/formatted/stage4_train.jsonl --adapter ./outputs/stage3/lora_adapter --local
"""

import argparse
from pathlib import Path

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

MODEL_NAME = "Qwen/Qwen3.5-27B"
MAX_SEQ_LENGTH_CLOUD = 262144  # 256K for cloud (48-80GB VRAM)
MAX_SEQ_LENGTH_LOCAL = 204800  # 204800 for local (32GB VRAM)

BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 4
EPOCHS = 1
LEARNING_RATE = 2e-5  # Very low LR for final stage
WARMUP_STEPS = 10
SEED = 3407
OUTPUT_DIR = "./outputs/stage4"


def main(
    data_path: str, adapter_path: str = None, val_path: str = None, local: bool = False
):
    max_seq_length = MAX_SEQ_LENGTH_LOCAL if local else MAX_SEQ_LENGTH_CLOUD
    load_in_4bit = local  # Fall back to QLoRA if 32GB VRAM

    print(
        f"Stage 4: max_seq_length={max_seq_length}, 4bit={'yes' if load_in_4bit else 'no'}"
    )
    if local:
        print("WARNING: Local mode (32GB VRAM). Using QLoRA and 204800 context.")

    if adapter_path and Path(adapter_path).exists():
        print(f"Loading with Stage 3 adapter from {adapter_path}...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=adapter_path,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
        )
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
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
            max_seq_length=max_seq_length,
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
        save_steps=25,
        save_total_limit=2,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=25 if eval_dataset else None,
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
        max_seq_length=max_seq_length,
        packing=False,
    )

    print(f"Starting Stage 4 training (max_seq_length={max_seq_length})...")
    print("WARNING: This is the slowest stage (~5-10 min/step). Expected: 10-20 hours.")
    trainer.train()

    model.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
    print(f"Stage 4 complete. LoRA adapter saved to {OUTPUT_DIR}/lora_adapter")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--val", default=None)
    parser.add_argument("--adapter", default=None, help="Stage 3 LoRA adapter path")
    parser.add_argument(
        "--local", action="store_true", help="Use 204800 context + QLoRA for 32GB VRAM"
    )
    args = parser.parse_args()
    main(args.data, args.adapter, args.val, args.local)
