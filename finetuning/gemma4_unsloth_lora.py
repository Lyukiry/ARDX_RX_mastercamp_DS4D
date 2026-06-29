"""Fine-tuning LoRA multimodal Gemma (Unsloth FastVisionModel) — PC GPU.

Phase P1 du cahier des charges. À lancer UNIQUEMENT après une baseline validée.
Imports lourds paresseux : `import unsloth` / `torch` se font dans `main()`, donc
ce fichier se compile en CI sans GPU.

Règles imposées (cahier des charges §6) :
- LoRA d'abord, pas de full fine-tuning (FFT ≈ 4× plus de VRAM).
- Image AVANT le texte dans le format multimodal.
- `finetune_vision_layers = False` au départ (n'entraîner que langage + attention).
- Valider JSON, hallucinations, sensibilité, spécificité APRÈS entraînement.

Prérequis (sur le PC GPU) :
    pip install unsloth
    python finetuning/prepare_dataset.py --split final --out finetuning/data/train.jsonl
    python finetuning/gemma4_unsloth_lora.py --dataset finetuning/data/train.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_MODEL = "unsloth/gemma-3-4b-it"  # variante E4B multimodale (adapter si besoin)


def load_records(dataset: Path) -> list[dict]:
    with dataset.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def to_conversation(record: dict) -> dict:
    """Convertit un exemple {image_path, prompt, response} au format chat multimodal.

    L'image est placée AVANT l'instruction (recommandation Gemma/Unsloth).
    """
    from PIL import Image

    image = Image.open(record["image_path"]).convert("RGB")
    return {
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": record["prompt"]},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": record["response"]}]},
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=Path("finetuning/outputs/gemma_lora"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lora-rank", type=int, default=16)
    args = parser.parse_args()

    # --- Imports lourds paresseux (GPU requis) -----------------------------
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTConfig, SFTTrainer

    model, processor = FastVisionModel.from_pretrained(
        args.model,
        load_in_4bit=True,  # QLoRA : empreinte VRAM réduite
        use_gradient_checkpointing="unsloth",
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,   # cahier des charges : False au départ
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        lora_dropout=0.0,
        bias="none",
        random_state=3407,
    )

    dataset = [to_conversation(r) for r in load_records(args.dataset)]
    FastVisionModel.for_training(model)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=processor.tokenizer,
        data_collator=UnslothVisionDataCollator(model, processor),
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=4,
            num_train_epochs=args.epochs,
            learning_rate=2e-4,
            warmup_ratio=0.03,
            logging_steps=1,
            optim="adamw_8bit",
            output_dir=str(args.output),
            remove_unused_columns=False,
            dataset_kwargs={"skip_prepare_dataset": True},
            max_seq_length=2048,
        ),
    )
    trainer.train()
    model.save_pretrained(str(args.output))
    processor.save_pretrained(str(args.output))
    print(f"Adaptateur LoRA sauvegardé dans {args.output}")
    print("Étape suivante : ré-évaluer (JSON, hallucinations, sensibilité, spécificité) avant toute décision.")


if __name__ == "__main__":
    main()
