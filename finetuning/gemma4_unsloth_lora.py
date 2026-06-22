"""Gemma 4 (E2B / E4B) multimodal LoRA fine-tuning with Unsloth.

Runnable on a GPU machine, **only after** the prompting baseline, the dataset
and the metrics are stable (COULD level in the brief). It trains a LoRA adapter
to make the model emit the project's JSON contract more reliably.

Design choices follow the brief (section "Voie rapide de fine-tuning"):
- multimodal E2B / E4B variant,
- image content placed before the text instruction,
- LoRA (not full fine-tuning) to save VRAM,
- ``finetune_vision_layers=False`` at first.

All heavy imports (``unsloth``, ``torch``, ``datasets``, ``trl``) are lazy so the
file compiles on a torch-free CI runner. Install the GPU stack first:

    pip install -r requirements-gpu.txt

Usage
-----
    python finetuning/gemma4_unsloth_lora.py \
        --cases data/rsna/cases.csv --split dev \
        --model unsloth/gemma-3-4b-it --out models/gemma4_lora --epochs 1
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from finetuning.sft_dataset import build_examples  # noqa: E402

DEFAULT_MODEL = "unsloth/gemma-3-4b-it"


def _to_hf_dataset(examples):
    """Convert SFT examples to a Hugging Face dataset of chat conversations."""
    from datasets import Dataset
    from PIL import Image

    def to_conversation(ex):
        image = Image.open(ex["image_path"]).convert("RGB")
        return {"messages": [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": ex["instruction"]},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": ex["answer"]}]},
        ]}

    return Dataset.from_list([to_conversation(ex) for ex in examples])


def train(args) -> None:
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTConfig, SFTTrainer

    examples = build_examples(args.cases, mode=args.mode, split=args.split)
    if not examples:
        raise SystemExit(f"No examples found in {args.cases} (split={args.split}).")
    print(f"Loaded {len(examples)} SFT examples.")

    model, processor = FastVisionModel.from_pretrained(
        args.model, load_in_4bit=args.load_in_4bit, use_gradient_checkpointing="unsloth",
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,        # start with text-only adaptation
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none",
        random_state=args.seed,
    )

    dataset = _to_hf_dataset(examples)
    FastVisionModel.for_training(model)
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=processor.tokenizer,
        data_collator=UnslothVisionDataCollator(model, processor),
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_steps=5,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=args.seed,
            output_dir=str(Path(args.out) / "trainer"),
            report_to="none",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_seq_length=2048,
        ),
    )
    trainer.train()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    processor.save_pretrained(str(out))
    print(f"Saved LoRA adapter to {out}")
    print("Next: evaluate with src.vlm_inference (RADIO_VLM_MODEL pointing at the merged model).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemma 4 multimodal LoRA (Unsloth)")
    parser.add_argument("--cases", default="data/rsna/cases.csv")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--mode", default="improved", choices=["baseline", "improved"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", default="models/gemma4_lora")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=13)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
