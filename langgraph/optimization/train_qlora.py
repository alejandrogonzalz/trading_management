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
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.prompts import build_system_prompt, build_user_prompt
from backtest.evaluation.metrics import compute_all_metrics
from backtest.evaluation.runner import CANDLES_DIR, _load_candles_map
from backtest.evaluation.simulate import _parse_prediction, simulate_trade
from backtest.export import export_training_data
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

    def __init__(self, config: dict[str, Any] | None = None):
        self.cfg = {**self.DEFAULT_CONFIG, **(config or {})}
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def prepare_data(self) -> dict[str, int]:
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
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
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
            text = self.tokenizer.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)
            texts.append(text)

        return Dataset.from_dict({"text": texts})

    def train(self) -> dict[str, Any]:
        """Run SFT training with early stopping based on validation loss."""
        from transformers import TrainingArguments
        from trl import SFTTrainer

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
            fp16=False,
            bf16=True,
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

        # Pull the best eval_loss from the trainer's log history (load_best_model_at_end
        # restores the checkpoint with the lowest eval_loss).
        eval_losses = [rec["eval_loss"] for rec in self.trainer.state.log_history if "eval_loss" in rec]
        best_eval_loss = min(eval_losses) if eval_losses else None

        log.info(f"  Training completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
        log.info(f"  Final train loss: {train_result.training_loss:.4f}")
        if best_eval_loss is not None:
            log.info(f"  Best eval loss: {best_eval_loss:.4f}")

        return {
            "train_loss": train_result.training_loss,
            "eval_loss": best_eval_loss,
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
        self.model.save_pretrained_gguf(str(gguf_path), self.tokenizer, quantization_method="q4_k_m")

    def evaluate(self) -> dict[str, Any]:
        """Evaluate the fine-tuned model on the test split.

        Mirrors LLMBacktestRunner exactly so the result is comparable to the
        other models: same temporal test split, same prompts, full prediction
        parsing (bias + entry/tp/sl), trade simulation against future candles,
        and the same metric set (direction accuracy, win rate, profit factor,
        Sharpe, max drawdown).

        Critically, it emits one prediction per test sample (a fallback on
        parse failure rather than skipping) and stores ``sample_keys`` so the
        paired McNemar / t-test can align this model against the others.
        """
        import torch
        from unsloth import FastLanguageModel

        log.info("Evaluating on test set...")
        FastLanguageModel.for_inference(self.model)

        # Use the SAME labeled dataset + temporal split convention as the ML/LLM
        # runners (no sort) so the test set — and therefore the paired stats —
        # are aligned with LSTM/XGBoost/zero-shot results.
        all_samples = _load_dataset(DATASET_PATH)
        _train, _val, test = _temporal_split(all_samples)

        max_samples = self.cfg.get("max_eval_samples")
        if max_samples and max_samples < len(test):
            test = test[:max_samples]
            log.info(f"  Evaluating on first {max_samples} test samples")
        log.info(f"  Test samples: {len(test)}")

        candles_map, ts_idx_map = _load_candles_map(CANDLES_DIR)
        system_prompt = build_system_prompt(self.cfg["mode"])

        predictions: list[dict[str, Any]] = []
        actuals: list[dict[str, Any]] = []
        trade_results: list[dict[str, Any]] = []
        sample_keys: list[str] = []
        parse_errors = 0

        for i, sample in enumerate(test):
            if (i + 1) % 100 == 0:
                log.info(f"  Progress: {i + 1}/{len(test)} ({parse_errors} parse errors)")

            symbol = sample.get("symbol", "BTCUSDT")
            label = sample["label"]
            indicators = sample["indicators"]
            if indicators and not isinstance(next(iter(indicators.values())), dict):
                indicators = {"1h": indicators}

            user_prompt = build_user_prompt(symbol, indicators)
            input_text = self.tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,  # greedy decoding — deterministic, reproducible
                )
            generated = self.tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )

            # Full prediction parse (bias + entry/tp/sl). On failure, fall back
            # to a bias-only prediction so the sample still counts toward the
            # paired comparison (its trade simulates as ERROR, pnl 0).
            prediction = _parse_prediction(generated)
            if prediction is None:
                parse_errors += 1
                bias_kw = self._parse_bias(generated)
                prediction = {"bias": bias_kw or "LONG", "confidence": 0}
            prediction.setdefault("confidence", 5)

            predictions.append(prediction)
            actuals.append(label)
            sample_keys.append(f"{symbol}@{sample['timestamp']}")

            candle_idx = ts_idx_map.get(symbol, {}).get(sample["timestamp"])
            if candle_idx is not None and candle_idx + 1 < len(candles_map.get(symbol, [])):
                future = candles_map[symbol][candle_idx + 1 : candle_idx + 25]
                trade_results.append(simulate_trade(prediction, future))
            else:
                trade_results.append({"outcome": "ERROR", "pnl_pct": 0, "hold_bars": 0})

        metrics = compute_all_metrics(predictions, actuals, trade_results) if predictions else {}
        accuracy = metrics.get("direction_accuracy", 0.0)
        log.info(
            f"  Test accuracy: {accuracy:.4f}  win_rate={metrics.get('win_rate', 0):.4f}  "
            f"profit_factor={metrics.get('profit_factor', 0):.3f}"
        )
        log.info(f"  Parse errors: {parse_errors}")

        return {
            "accuracy": accuracy,
            "metrics": metrics,
            "total_evaluated": len(predictions),
            "errors": parse_errors,
            # Per-sample arrays — required for the paired McNemar + t-test.
            # Same schema as LLMBacktestRunner (predictions are dicts with bias).
            "predictions": predictions,
            "actuals": actuals,
            "trade_results": trade_results,
            "sample_keys": sample_keys,
        }

    def _parse_bias(self, text: str) -> str | None:
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

    def run(self) -> dict[str, Any]:
        """Execute the full pipeline: prepare → load → train → evaluate → save."""
        log.info("=" * 60)
        log.info("  QLoRA Fine-Tuning — Qwen 2.5 7B")
        log.info(
            f"  Config: lr={self.cfg['learning_rate']}, rank={self.cfg['lora_rank']}, "
            f"alpha={self.cfg['lora_alpha']}, epochs={self.cfg['epochs']}, "
            f"batch={self.cfg['batch_size']}"
        )
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

        # Assemble final result. Per-sample arrays (predictions/actuals/
        # trade_results/sample_keys) live at the TOP level — the same schema as
        # LLMBacktestRunner — so stats_tests.compare() and analyze_results can
        # read them directly for the paired McNemar / t-test.
        result = {
            "model": "qlora_qwen25_7b",
            "tag": "qlora_qwen25_7b",
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
            "val_metrics": {
                "train_loss": train_info["train_loss"],
                "eval_loss": train_info["eval_loss"],
            },
            "test_metrics": {
                "accuracy": test_metrics["accuracy"],
                "total_evaluated": test_metrics["total_evaluated"],
                "errors": test_metrics["errors"],
                **test_metrics["metrics"],
            },
            "metrics": test_metrics["metrics"],
            "predictions": test_metrics["predictions"],
            "actuals": test_metrics["actuals"],
            "trade_results": test_metrics["trade_results"],
            "sample_keys": test_metrics["sample_keys"],
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


def load_config_from_yaml(path: str) -> dict[str, Any]:
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
    m = result["metrics"]
    print(f"\n{'=' * 60}")
    print(f"  RESULT: Test Accuracy = {result['test_metrics']['accuracy']:.4f}")
    print(f"  Win Rate = {m.get('win_rate', 0):.4f}  Profit Factor = {m.get('profit_factor', 0):.3f}")
    print(f"  Sharpe = {m.get('sharpe_ratio', 0):.3f}  Max Drawdown = {m.get('max_drawdown', 0):.2f}%")
    print(f"  Time = {result['elapsed_seconds']:.0f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
