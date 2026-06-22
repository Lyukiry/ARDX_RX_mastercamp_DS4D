"""MedGemma 4B adaptation with PEFT / QLoRA (advanced medical option).

Runnable on a GPU machine, **only after** a validated prompting baseline (COULD
level in the brief). It trains a QLoRA adapter on top of MedGemma 4B so the model
follows the project's JSON contract while staying inside the cautious,
non-clinical scope.

Uses Hugging Face ``transformers`` + ``peft`` + ``trl`` directly (no Unsloth),
which is the closest path to Google's official MedGemma recipes. All heavy
imports are lazy so the file compiles on a torch-free CI runner.

    pip install -r requirements-gpu.txt
    # accept the MedGemma license on Hugging Face, then `huggingface-cli login`

Usage
-----
    python finetuning/medgemma_peft_qlora.py \
        --cases data/rsna/cases.csv --split dev \
        --model google/medgemma-4b-it --out models/medgemma_qlora --epochs 1
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from finetuning.sft_dataset import build_examples  # noqa: E402

DEFAULT_MODEL = "google/medgemma-4b-it"


def _build_collate_fn(processor):
    """Return a collator that tokenises (image, instruction, answer) chats."""
    from PIL import Image

    def collate(examples):
        texts, images = [], []
        for ex in examples:
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": ex["instruction"]},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": ex["answer"]}]},
            ]
            texts.append(processor.apply_chat_template(messages, tokenize=False).strip())
            images.append([Image.open(ex["image_path"]).convert("RGB")])

        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        # mask image tokens so loss is computed on text only
        image_token_id = getattr(processor.tokenizer, "image_token_id", None)
        if image_token_id is not None:
            labels[labels == image_token_id] = -100
        batch["labels"] = labels
        return batch

    return collate


def train(args) -> None:
    import torch
    from transformers import (AutoProcessor, AutoModelForImageTextToText,
                              BitsAndBytesConfig, Trainer, TrainingArguments)
    from peft import LoraConfig, get_peft_model

    examples = build_examples(args.cases, mode=args.mode, split=args.split)
    if not examples:
        raise SystemExit(f"No examples found in {args.cases} (split={args.split}).")
    print(f"Loaded {len(examples)} SFT examples.")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, quantization_config=bnb_config, torch_dtype=torch.bfloat16, device_map="auto",
    )
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        train_dataset=examples,
        data_collator=_build_collate_fn(processor),
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            warmup_ratio=0.03,
            logging_steps=1,
            optim="paged_adamw_8bit",
            bf16=True,
            seed=args.seed,
            output_dir=str(Path(args.out) / "trainer"),
            report_to="none",
            remove_unused_columns=False,
        ),
    )
    trainer.train()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    processor.save_pretrained(str(out))
    print(f"Saved QLoRA adapter to {out}")
    print("Evaluate on the held-out 'final' split before trusting any gain.")


def main() -> None:
    parser = argparse.ArgumentParser(description="MedGemma 4B QLoRA (PEFT)")
    parser.add_argument("--cases", default="data/rsna/cases.csv")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--mode", default="improved", choices=["baseline", "improved"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", default="models/medgemma_qlora")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=13)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
