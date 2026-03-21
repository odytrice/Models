"""
Merge LoRA Adapter & Export to GGUF/GPTQ

After all 4 training stages complete, this script:
1. Merges the final LoRA adapter into the base model
2. Saves the full BF16 merged model
3. Exports to GGUF (Q4_K_M, Q5_K_M, Q8_0) for llama.cpp / Ollama
4. Exports to GPTQ-Int4 for vLLM / HuggingFace Transformers

Usage:
  python merge_and_export.py --adapter ./outputs/stage4/lora_adapter --output ./outputs/merged
  python merge_and_export.py --adapter ./outputs/stage4/lora_adapter --output ./outputs/merged --gguf-only
"""

import argparse
from pathlib import Path

from unsloth import FastLanguageModel


def main(adapter_path: str, output_dir: str, gguf_only: bool = False):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print(f"Loading adapter from {adapter_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=262144,
        load_in_4bit=False,
    )

    # ── Save merged BF16 model ───────────────────────────────────────
    bf16_path = output / "bf16"
    print(f"Saving merged BF16 model to {bf16_path}...")
    model.save_pretrained_merged(str(bf16_path), tokenizer, save_method="merged_16bit")
    print(f"BF16 model saved ({bf16_path})")

    # ── Export GGUF variants ─────────────────────────────────────────
    gguf_quants = ["q4_k_m", "q5_k_m", "q8_0"]
    for quant in gguf_quants:
        gguf_path = output / f"gguf-{quant}"
        print(f"Exporting GGUF {quant} to {gguf_path}...")
        model.save_pretrained_gguf(
            str(gguf_path),
            tokenizer,
            quantization_method=quant,
        )
        print(f"GGUF {quant} saved ({gguf_path})")

    # ── Export GPTQ-Int4 (optional) ──────────────────────────────────
    if not gguf_only:
        gptq_path = output / "gptq-int4"
        print(f"Exporting GPTQ-Int4 to {gptq_path}...")
        model.save_pretrained_merged(
            str(gptq_path),
            tokenizer,
            save_method="merged_4bit",
        )
        print(f"GPTQ-Int4 saved ({gptq_path})")

    print("=" * 60)
    print("EXPORT COMPLETE")
    print(f"  BF16:      {bf16_path}")
    for quant in gguf_quants:
        print(f"  GGUF {quant}: {output / f'gguf-{quant}'}")
    if not gguf_only:
        print(f"  GPTQ-Int4: {output / 'gptq-int4'}")
    print()
    print("To use with Ollama, create a Modelfile pointing to the GGUF:")
    print(f"  FROM {output / 'gguf-q4_k_m' / 'unsloth.Q4_K_M.gguf'}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter", required=True, help="Final stage LoRA adapter path"
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for merged/exported models"
    )
    parser.add_argument("--gguf-only", action="store_true", help="Skip GPTQ export")
    args = parser.parse_args()
    main(args.adapter, args.output, args.gguf_only)
