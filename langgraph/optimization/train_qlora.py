#!/usr/bin/env python3
"""QLoRA fine-tuning of Qwen 2.5 7B for crypto trade direction prediction.

Self-contained script: exports training data, fine-tunes with Unsloth,
evaluates on test set, and saves results in the standard optimization format.

Usage:
    python optimization/train_qlora.py
    python optimization/train_qlora.py --config optimization/configs/qlora.yaml
    python optimization/train_qlora.py --lr 0.00002 --rank 16 --epochs 3

Requirements (GPU machine only):
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    pip install --no-deps trl peft accelerate bitsandbytes
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.export import export_training_data, build_training_example
from backtest.models.features import _load_dataset, _temporal_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

DATASET_PATH = str(Path(__file__).resolve().parent.parent / "backtest" / "data" / "labeled" / "dataset.jsonl")
TRAINING_DATA_DIR = Path(__file__).resolve().parent.parent / "training_data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
MODELS_DIR = Path(__file__).resolve().parent.parent / "backtest" / "data" / "models"


class QLoRATrainer:
    """Fine-tunes Qwen 2.5 7B with QLoRA 4-bit for trade direction prediction.

    Handles the full pipeline: data preparation, training with Unsloth/TRL,
    evaluation on the held-out test set, and result serialization.
    """

    DEFAULT_CONFIG = {
        "model_name": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        "max_seq_length": 2048,
        "learning_rate": 0.00002,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "epochs": 3,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "warmup_steps": 50,
        "weight_decay": 0.01,
        "seed": 42,
        "mode": "FUTURES",
        "max_eval_samples": None,
        "output_dir": str(MODELS_DIR / "qlora_qwen25_7b"),
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.cfg = {**self.DEFAULT_CONFIG, **(config or {})}
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def prepare_data(self) -> Dict[str, int]:
        """Export labeled dataset to chat-format JSONL splits for SFT."""
        log.info(f"Exporting training data from {DATASET_PATH}")
        log.info(f"  Output: {TRAINING_DATA_DIR}/")

        counts = export_training_data(
            dataset_path=DATASET_PATH,
            output_dir=str(TRAINING_DATA_DIR),
            mode=self.cfg["mode"],
        )
        log.info(f"  Train: {counts['train']}, Val: {counts['val']}, Test: {counts['test']}")
        return counts

    def load_model(self):
        """Load the base model with 4-bit quantization and apply LoRA adapters."""
        from unsloth import FastLanguageModel

        log.info(f"Loading model: {self.cfg['model_name']}")
        log.info(f"  LoRA rank={self.cfg['lora_rank']}, alpha={self.cfg['lora_alpha']}")

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.cfg["model_name"],
            max_seq_length=self.cfg["max_seq_length"],
            dtype=None,
            load_in_4bit=True,
        )

        # Apply LoRA adapters to attention + MLP layers
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.cfg["lora_rank"],
            lora_alpha=self.cfg["lora_alpha"],
            lora_dropout=self.cfg["lora_dropout"],
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=self.cfg["seed"],
        )

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        log.info(f"  Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    def _load_chat_dataset(self, split: str):
        """Load a JSONL split into HuggingFace Dataset format."""
        from datasets import Dataset

        path = TRAINING_DATA_DIR / f"{split}.jsonl"
        examples = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))

        # Convert chat messages to the format expected by SFTTrainer
        texts = []
        for ex in examples:
            text = self.tokenizer.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False
            )
            texts.append(text)

        return Dataset.from_dict({"text": texts})

    def train(self) -> Dict[str, Any]:
        """Run SFT training with early stopping based on validation loss."""
        from trl import SFTTrainer
        from transformers import TrainingArguments

        log.info("Loading datasets...")
        train_dataset = self._load_chat_dataset("train")
        val_dataset = self._load_chat_dataset("val")
        log.info(f"  Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

        effective_batch = self.cfg["batch_size"] * self.cfg["gradient_accumulation_steps"]
        total_steps = (len(train_dataset) // effective_batch) * self.cfg["epochs"]
        log.info(f"  Effective batch size: {effective_batch}")
        log.info(f"  Total training steps: ~{total_steps}")

        output_dir = self.cfg["output_dir"]
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.cfg["epochs"],
            per_device_train_batch_size=self.cfg["batch_size"],
            gradient_accumulation_steps=self.cfg["gradient_accumulation_steps"],
            learning_rate=self.cfg["learning_rate"],
            weight_decay=self.cfg["weight_decay"],
            warmup_steps=self.cfg["warmup_steps"],
            lr_scheduler_type="cosine",
            fp16=True,
            logging_steps=25,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            seed=self.cfg["seed"],
            report_to="none",
        )

        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            args=training_args,
            dataset_text_field="text",
            max_seq_length=self.cfg["max_seq_length"],
            packing=False,
        )

        log.info("Starting training...")
        t0 = time.time()
        train_result = self.trainer.train()
        elapsed = time.time() - t0

        log.info(f"  Training completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
        log.info(f"  Final train loss: {train_result.training_loss:.4f}")

        return {
            "train_loss": train_result.training_loss,
            "train_runtime_seconds": elapsed,
            "total_steps": train_result.global_step,
        }

    def save_model(self):
        """Save the fine-tuned LoRA adapters."""
        output_dir = self.cfg["output_dir"]
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        log.info(f"Saving LoRA adapters to {output_dir}/")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Save merged GGUF for Ollama deployment (optional)
        gguf_path = Path(output_dir) / "gguf"
        log.info(f"Saving GGUF (Q4_K_M) to {gguf_path}/")
        self.model.save_pretrained_gguf(
            str(gguf_path), self.tokenizer, quantization_method="q4_k_m"
        )

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the fine-tuned model on the test split.

        Generates predictions for each test sample and computes
        direction accuracy, matching the MLBacktestRunner output format.
        """
        from unsloth import FastLanguageModel

        log.info("Evaluating on test set...")
        FastLanguageModel.for_inference(self.model)

        test_path = TRAINING_DATA_DIR / "test.jsonl"
        test_examples = []
        with open(test_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    test_examples.append(json.loads(line))

        max_samples = self.cfg.get("max_eval_samples")
        if max_samples and max_samples < len(test_examples):
            test_examples = test_examples[:max_samples]
            log.info(f"  Evaluating on first {max_samples} samples (of {len(test_examples)} total)")

        correct = 0
        total = 0
        predictions = []
        actuals = []
        errors = 0

        for i, ex in enumerate(test_examples):
            if (i + 1) % 100 == 0:
                log.info(f"  Progress: {i + 1}/{len(test_examples)} ({correct}/{total} correct)")

            messages = ex["messages"]
            actual_response = json.loads(messages[-1]["content"])
            actual_bias = actual_response["bias"]

            # Build input (system + user only)
            input_messages = messages[:2]
            input_text = self.tokenizer.apply_chat_template(
                input_messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

            with __import__("torch").no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.1,
                    do_sample=False,
                )

            # Decode only the generated tokens
            generated = self.tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            # Parse the prediction
            predicted_bias = self._parse_bias(generated)
            if predicted_bias is None:
                errors += 1
                continue

            total += 1
            predictions.append(predicted_bias)
            actuals.append(actual_bias)
            if predicted_bias == actual_bias:
                correct += 1

        accuracy = correct / total if total > 0 else 0.0
        log.info(f"  Test accuracy: {accuracy:.4f} ({correct}/{total})")
        log.info(f"  Parse errors: {errors}")

        # Compute detailed metrics
        from sklearn.metrics import f1_score, precision_score, recall_score
        y_true = [1 if b == "LONG" else 0 for b in actuals]
        y_pred = [1 if b == "LONG" else 0 for b in predictions]

        return {
            "accuracy": accuracy,
            "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro")),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro")),
            "total_evaluated": total,
            "errors": errors,
            "correct": correct,
        }

    def _parse_bias(self, text: str) -> Optional[str]:
        """Extract bias (LONG/SHORT) from model output."""
        try:
            # Try JSON parse first
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(text[start:end])
                bias = parsed.get("bias", "").upper()
                if bias in ("LONG", "SHORT"):
                    return bias
        except (json.JSONDecodeError, KeyError):
            pass

        # Fallback: look for keywords
        text_upper = text.upper()
        if "LONG" in text_upper and "SHORT" not in text_upper:
            return "LONG"
        if "SHORT" in text_upper and "LONG" not in text_upper:
            return "SHORT"

        return None

    def run(self) -> Dict[str, Any]:
        """Execute the full pipeline: prepare → load → train → evaluate → save."""
        log.info("=" * 60)
        log.info("  QLoRA Fine-Tuning — Qwen 2.5 7B")
        log.info(f"  Config: lr={self.cfg['learning_rate']}, rank={self.cfg['lora_rank']}, "
                 f"alpha={self.cfg['lora_alpha']}, epochs={self.cfg['epochs']}, "
                 f"batch={self.cfg['batch_size']}")
        log.info("=" * 60)

        t0 = time.time()

        # Step 1: Prepare data
        counts = self.prepare_data()

        # Step 2: Load model + LoRA
        self.load_model()

        # Step 3: Train
        train_info = self.train()

        # Step 4: Save model
        self.save_model()

        # Step 5: Evaluate on test set
        test_metrics = self.evaluate()

        elapsed = time.time() - t0

        # Assemble final result in standard format
        result = {
            "model": "qlora_qwen25_7b",
            "ensemble_type": "single",
            "strategy": "fine_tuning",
            "best_score": test_metrics["accuracy"],
            "best_params": {
                "model_name": self.cfg["model_name"],
                "learning_rate": self.cfg["learning_rate"],
                "lora_rank": self.cfg["lora_rank"],
                "lora_alpha": self.cfg["lora_alpha"],
                "epochs": self.cfg["epochs"],
                "batch_size": self.cfg["batch_size"],
                "gradient_accumulation_steps": self.cfg["gradient_accumulation_steps"],
                "max_seq_length": self.cfg["max_seq_length"],
            },
            "val_metrics": {"train_loss": train_info["train_loss"]},
            "test_metrics": test_metrics,
            "data_counts": counts,
            "train_info": train_info,
            "elapsed_seconds": round(elapsed, 1),
            "timestamp": datetime.now().isoformat(),
        }

        # Save result
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / "qlora_optimization.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        log.info(f"\nResult saved: {out_path}")
        log.info(f"Model saved: {self.cfg['output_dir']}")
        log.info(f"Total time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")

        return result


def load_config_from_yaml(path: str) -> Dict[str, Any]:
    """Load training config from YAML, mapping to trainer params."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    # Use first recommended config if available
    configs = raw.get("recommended_configs", [])
    if configs:
        cfg = configs[0]
        return {
            "learning_rate": cfg.get("learning_rate"),
            "lora_rank": cfg.get("lora_rank"),
            "lora_alpha": cfg.get("lora_alpha"),
            "epochs": cfg.get("epochs"),
            "batch_size": cfg.get("batch_size"),
        }
    return {}


def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for trade prediction")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--rank", type=int, help="LoRA rank")
    parser.add_argument("--alpha", type=int, help="LoRA alpha")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Per-device batch size")
    parser.add_argument("--max-eval", type=int, help="Max samples to evaluate (for quick testing)")
    parser.add_argument("--model", type=str, help="Model name/path (default: Qwen2.5-7B-Instruct-bnb-4bit)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Build config from YAML + CLI overrides
    config = {}
    if args.config:
        config = load_config_from_yaml(args.config)

    # CLI args override YAML
    if args.lr:
        config["learning_rate"] = args.lr
    if args.rank:
        config["lora_rank"] = args.rank
    if args.alpha:
        config["lora_alpha"] = args.alpha
    if args.epochs:
        config["epochs"] = args.epochs
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.max_eval:
        config["max_eval_samples"] = args.max_eval
    if args.model:
        config["model_name"] = args.model

    # Run training
    trainer = QLoRATrainer(config)
    result = trainer.run()

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  RESULT: Test Accuracy = {result['test_metrics']['accuracy']:.4f}")
    print(f"  F1 Macro = {result['test_metrics']['f1_macro']:.4f}")
    print(f"  Time = {result['elapsed_seconds']:.0f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
