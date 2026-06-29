"""Fine-tuning MedGemma 4B en PEFT/QLoRA — PC GPU.

À lancer UNIQUEMENT après une baseline prompting validée (cahier des charges §3.1,
§6). Vérifier l'accès au modèle (licence Hugging Face), la VRAM et la recette
officielle Google/HF avant exécution. Imports lourds paresseux.

Prérequis (sur le PC GPU) :
    pip install transformers peft bitsandbytes accelerate datasets trl
    huggingface-cli login            # accès à google/medgemma-4b-it
    python finetuning/prepare_dataset.py --split final --out finetuning/data/train.jsonl
    python finetuning/medgemma_peft_qlora.py --dataset finetuning/data/train.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_MODEL = "google/medgemma-4b-it"


def load_records(dataset: Path) -> list[dict]:
    with dataset.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=Path("finetuning/outputs/medgemma_qlora"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lora-rank", type=int, default=16)
    args = parser.parse_args()

    # --- Imports lourds paresseux (GPU requis) -----------------------------
    import torch
    from datasets import Dataset
    from PIL import Image
    from peft import LoraConfig
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, quantization_config=quant_config, torch_dtype=torch.bfloat16, device_map="auto",
    )

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    records = load_records(args.dataset)
    dataset = Dataset.from_list(records)

    def collate(batch: list[dict]) -> dict:
        """Image AVANT texte, réponse = sortie JSON cible (masquée sur le prompt)."""
        images, texts = [], []
        for example in batch:
            images.append(Image.open(example["image_path"]).convert("RGB"))
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": example["prompt"]},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": example["response"]}]},
            ]
            texts.append(processor.apply_chat_template(messages, tokenize=False))
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        data_collator=collate,
        args=SFTConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_train_epochs=args.epochs,
            learning_rate=1e-4,
            warmup_ratio=0.03,
            logging_steps=1,
            optim="paged_adamw_8bit",
            output_dir=str(args.output),
            remove_unused_columns=False,
            dataset_kwargs={"skip_prepare_dataset": True},
            gradient_checkpointing=True,
        ),
    )
    trainer.train()
    trainer.save_model(str(args.output))
    print(f"Adaptateur QLoRA MedGemma sauvegardé dans {args.output}")
    print("Étape suivante : ré-évaluer (JSON, hallucinations, sensibilité, spécificité) avant toute décision.")


if __name__ == "__main__":
    main()
