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
from typing import Any

import yaml

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

LANGGRAPH_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(LANGGRAPH_ROOT))

from agent.prompts import build_system_prompt, build_user_prompt
from backtest.evaluation.metrics import compute_all_metrics, direction_accuracy
from backtest.evaluation.runner import CANDLES_DIR, _load_candles_map
from backtest.evaluation.simulate import _parse_prediction, simulate_trade
from backtest.export import export_training_data
from backtest.models.features import _load_dataset, _temporal_split
from optimization.io.plots import save_loss_curve_plot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

DATASET_PATH = str(LANGGRAPH_ROOT / "backtest" / "data" / "labeled" / "dataset.jsonl")
TRAINING_DATA_DIR = LANGGRAPH_ROOT / "training_data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
MODELS_DIR = LANGGRAPH_ROOT / "backtest" / "data" / "models"


class QLoRATrainer:
    """Fine-tunes Qwen 2.5 7B with QLoRA 4-bit for trade direction prediction.

    Handles the full pipeline: data preparation, training with Unsloth/TRL,
    evaluation on the held-out test set, and result serialization.
    """

    DEFAULT_CONFIG = {
        "model_name": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        # Full chat-templated examples (prompt + assistant JSON) span 877-933
        # tokens across the dataset — the prompt alone is ~820. 1024 fits the
        # longest sample with headroom and causes ZERO truncation, which is what
        # avoids Unsloth's padding-free fused-CE crash (it truncates input_ids
        # but not labels when a sequence exceeds max_seq_length). Do not lower
        # this below 1024. batch_size=1 keeps 1024 within the 16GB RTX 5070 Ti.
        "max_seq_length": 1024,
        "learning_rate": 0.00002,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "epochs": 3,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
        "warmup_steps": 50,
        "weight_decay": 0.01,
        "seed": 42,
        "mode": "FUTURES",
        "max_steps": None,  # cap optimizer steps (smoke tests); None = full epochs
        "max_eval_samples": None,
        # Subset size used to measure train/val direction accuracy for the
        # overfitting gap. The full test split is always evaluated; train/val are
        # capped because generation is expensive and a few hundred samples are
        # enough to estimate the gap. Set to 0 to skip the gap diagnostics.
        "diagnostic_samples": 500,
        "resume": False,  # resume from latest checkpoint-N in output_dir if present
        "tag": "qlora_qwen25_7b",  # names result/loss-curve files and the model id
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
        """Load a JSONL split into HuggingFace Dataset format.

        Any example whose tokenized length exceeds ``max_seq_length`` is dropped
        rather than fed to the trainer. Unsloth's padding-free batching truncates
        an over-long sequence's ``input_ids`` but leaves its ``labels`` intact,
        which crashes the fused cross-entropy loss with a batch-size mismatch
        (the original failure mode). Filtering here makes that impossible for any
        dataset, so the run is safe to leave unattended. With max_seq_length=1024
        this drops 0 of the current samples (longest is 933 tokens).
        """
        from datasets import Dataset

        path = TRAINING_DATA_DIR / f"{split}.jsonl"
        examples = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))

        max_len = self.cfg["max_seq_length"]
        texts = []
        dropped = 0
        for ex in examples:
            text = self.tokenizer.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)
            n_tokens = len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
            if n_tokens > max_len:
                dropped += 1
                continue
            texts.append(text)

        if dropped:
            log.warning(f"  {split}: dropped {dropped}/{len(examples)} examples exceeding max_seq_length={max_len}")

        return Dataset.from_dict({"text": texts})

    def _resume_checkpoint(self) -> str | None:
        """Return the latest checkpoint-N dir to resume from, or None for a fresh run.

        Honors cfg['resume']: checkpoints save every save_steps (250), so an
        interrupted long run can pick up from the last one instead of restarting
        at step 0. Falls back to a fresh start (with a warning) if --resume was
        passed but no checkpoint exists yet.
        """
        if not self.cfg.get("resume"):
            return None
        from transformers.trainer_utils import get_last_checkpoint

        last = get_last_checkpoint(self.cfg["output_dir"]) if Path(self.cfg["output_dir"]).is_dir() else None
        if last:
            log.info(f"  Resuming from checkpoint: {last}")
            return last
        log.warning("  --resume set but no checkpoint found in output_dir; starting fresh")
        return None

    def train(self) -> dict[str, Any]:
        """Run SFT training with early stopping based on validation loss."""
        import trl
        from transformers import EarlyStoppingCallback
        from trl import SFTTrainer

        log.info("Loading datasets...")
        train_dataset = self._load_chat_dataset("train")
        val_dataset = self._load_chat_dataset("val")
        # Cap the in-training eval set. A full 8.4k-sample eval at every eval_steps
        # would add hours of forward passes to an already long run; a fixed 1000
        # subset is enough to track eval_loss for best-checkpoint selection. The
        # FINAL test metrics (evaluate()) still use the complete held-out split.
        if len(val_dataset) > 1000:
            val_dataset = val_dataset.select(range(1000))
        log.info(f"  Train samples: {len(train_dataset)}, Val (eval subset): {len(val_dataset)}")

        effective_batch = self.cfg["batch_size"] * self.cfg["gradient_accumulation_steps"]
        total_steps = (len(train_dataset) // effective_batch) * self.cfg["epochs"]
        log.info(f"  Effective batch size: {effective_batch}")
        log.info(f"  Total training steps: ~{total_steps}")

        output_dir = self.cfg["output_dir"]
        max_steps = self.cfg.get("max_steps") or -1  # -1 = honor num_train_epochs

        trl_major = int(trl.__version__.split(".")[1]) if trl.__version__.startswith("0.") else 99
        log.info(f"  TRL version: {trl.__version__} (using {'new' if trl_major >= 12 else 'legacy'} API)")

        if trl_major >= 12:
            # TRL >= 0.12: SFTConfig replaces TrainingArguments, params moved into config
            from trl import SFTConfig
            sft_config = SFTConfig(
                output_dir=output_dir,
                num_train_epochs=self.cfg["epochs"],
                max_steps=max_steps,
                per_device_train_batch_size=self.cfg["batch_size"],
                gradient_accumulation_steps=self.cfg["gradient_accumulation_steps"],
                learning_rate=self.cfg["learning_rate"],
                weight_decay=self.cfg["weight_decay"],
                warmup_steps=self.cfg["warmup_steps"],
                lr_scheduler_type="cosine",
                fp16=False,
                bf16=True,
                logging_steps=25,
                per_device_eval_batch_size=1,
                eval_strategy="steps",
                eval_steps=250,
                save_strategy="steps",
                save_steps=250,
                save_total_limit=3,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                seed=self.cfg["seed"],
                report_to="none",
                dataset_text_field="text",
                max_seq_length=self.cfg["max_seq_length"],
                packing=False,
            )
            self.trainer = SFTTrainer(
                model=self.model,
                processing_class=self.tokenizer,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                args=sft_config,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            )
        else:
            # TRL < 0.12: legacy API with TrainingArguments + params in SFTTrainer
            from transformers import TrainingArguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=self.cfg["epochs"],
                max_steps=max_steps,
                per_device_train_batch_size=self.cfg["batch_size"],
                gradient_accumulation_steps=self.cfg["gradient_accumulation_steps"],
                learning_rate=self.cfg["learning_rate"],
                weight_decay=self.cfg["weight_decay"],
                warmup_steps=self.cfg["warmup_steps"],
                lr_scheduler_type="cosine",
                fp16=False,
                bf16=True,
                logging_steps=25,
                per_device_eval_batch_size=1,
                eval_strategy="steps",
                eval_steps=250,
                save_strategy="steps",
                save_steps=250,
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
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            )

        log.info("Starting training...")
        t0 = time.time()
        train_result = self.trainer.train(resume_from_checkpoint=self._resume_checkpoint())
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
        """Save the fine-tuned LoRA adapters (and optionally GGUF)."""
        output_dir = self.cfg["output_dir"]
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        log.info(f"Saving LoRA adapters to {output_dir}/")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def save_gguf(self):
        """Export merged GGUF for Ollama deployment. Non-fatal on failure."""
        output_dir = self.cfg["output_dir"]
        gguf_path = Path(output_dir) / "gguf"
        log.info(f"Saving GGUF (Q4_K_M) to {gguf_path}/")
        try:
            self.model.save_pretrained_gguf(str(gguf_path), self.tokenizer, quantization_method="q4_k_m")
        except Exception as e:
            log.warning(f"  GGUF export failed (non-fatal): {e}")
            log.warning("  Adapters are saved — convert to GGUF manually later if needed.")

    def _predict_split(
        self,
        samples: list[dict[str, Any]],
        candles_map: dict,
        ts_idx_map: dict,
        system_prompt: str,
        simulate: bool = True,
        label: str = "test",
    ) -> dict[str, Any]:
        """Generate one prediction per sample — the per-sample backtest core.

        Shared by the full TEST backtest and the capped TRAIN/VAL accuracy
        probes. Emits exactly one prediction per sample (parse failures fall back
        to a bias-only prediction rather than being dropped) plus ``sample_keys``
        for paired stats. Trade simulation against future candles runs only when
        ``simulate=True`` (TEST); the gap probes need direction accuracy only.
        """
        import torch

        predictions: list[dict[str, Any]] = []
        actuals: list[dict[str, Any]] = []
        trade_results: list[dict[str, Any]] = []
        sample_keys: list[str] = []
        parse_errors = 0

        for i, sample in enumerate(samples):
            if (i + 1) % 100 == 0:
                log.info(f"  [{label}] {i + 1}/{len(samples)} ({parse_errors} parse errors)")

            symbol = sample.get("symbol", "BTCUSDT")
            label_obj = sample["label"]
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
            actuals.append(label_obj)
            sample_keys.append(f"{symbol}@{sample['timestamp']}")

            if simulate:
                candle_idx = ts_idx_map.get(symbol, {}).get(sample["timestamp"])
                if candle_idx is not None and candle_idx + 1 < len(candles_map.get(symbol, [])):
                    future = candles_map[symbol][candle_idx + 1 : candle_idx + 25]
                    trade_results.append(simulate_trade(prediction, future))
                else:
                    trade_results.append({"outcome": "ERROR", "pnl_pct": 0, "hold_bars": 0})

        return {
            "predictions": predictions,
            "actuals": actuals,
            "trade_results": trade_results,
            "sample_keys": sample_keys,
            "parse_errors": parse_errors,
        }

    @staticmethod
    def _heuristic_bias(indicators: dict[str, Any]) -> str:
        """Cheap indicator rule used as an honest comparison baseline.

        Majority-class accuracy is only ~51%, so the meaningful contrast is
        against a simple heatmap/MACD heuristic: if the fine-tuned model barely
        beats this, the task is easy and a high accuracy says little; if it
        clearly beats it, the model learned something non-trivial.
        """
        if indicators and not isinstance(next(iter(indicators.values())), dict):
            base = indicators
        elif indicators:
            base = indicators.get("1h", next(iter(indicators.values())))
        else:
            base = {}
        hm = base.get("heatmap", "NEUTRAL")
        if "BULLISH" in hm:
            return "LONG"
        if "BEARISH" in hm:
            return "SHORT"
        return "LONG" if base.get("macd_hist", 0) >= 0 else "SHORT"

    def evaluate(self) -> dict[str, Any]:
        """Backtest the fine-tuned model and measure the overfitting gap.

        Runs the LLMBacktestRunner-compatible per-sample backtest on the full
        temporal TEST split — same prompts, full prediction parse, trade
        simulation, full metric set, and ``sample_keys`` for paired stats.

        Adds the diagnostics the audit flagged as missing: direction accuracy on
        capped TRAIN and VAL subsets → ``gap = train_acc - test_acc`` (the direct
        memorization signal), and a heuristic indicator baseline on the same test
        set so the headline accuracy is read against ~51% majority-class honestly.
        """
        import torch  # noqa: F401  (kept here so a missing GPU stack fails fast)
        from unsloth import FastLanguageModel

        log.info("Evaluating on test set...")
        FastLanguageModel.for_inference(self.model)

        # SAME labeled dataset + temporal split as the ML/LLM runners (now a
        # strict temporal holdout) so the test set — and the paired stats — align
        # with LSTM/XGBoost/zero-shot results.
        all_samples = _load_dataset(DATASET_PATH)
        train, val, test = _temporal_split(all_samples)

        empty = {
            "accuracy": 0, "metrics": {}, "total_evaluated": 0, "errors": 0,
            "predictions": [], "actuals": [], "trade_results": [], "sample_keys": [],
            "train_acc": None, "val_acc": None, "test_acc": 0, "gap": None,
            "baseline_metrics": {},
        }
        max_samples = self.cfg.get("max_eval_samples")
        if max_samples is not None and max_samples < len(test):
            if max_samples == 0:
                log.info("  --max-eval 0: skipping evaluation")
                return empty
            test = test[:max_samples]
            log.info(f"  Evaluating on first {max_samples} test samples")
        log.info(f"  Test samples: {len(test)}")

        candles_map, ts_idx_map = _load_candles_map(CANDLES_DIR)
        system_prompt = build_system_prompt(self.cfg["mode"])

        # --- Full TEST backtest (with trade simulation) ---
        test_out = self._predict_split(test, candles_map, ts_idx_map, system_prompt, simulate=True, label="test")
        predictions = test_out["predictions"]
        actuals = test_out["actuals"]
        trade_results = test_out["trade_results"]
        sample_keys = test_out["sample_keys"]
        parse_errors = test_out["parse_errors"]

        metrics = compute_all_metrics(predictions, actuals, trade_results) if predictions else {}
        test_acc = metrics.get("direction_accuracy", 0.0)

        # --- Overfitting gap: TRAIN/VAL direction accuracy on capped subsets ---
        n_diag = self.cfg.get("diagnostic_samples") or 0
        train_acc = val_acc = gap = None
        if n_diag > 0:
            train_sub, val_sub = train[:n_diag], val[:n_diag]
            log.info(f"  Gap diagnostics: train subset={len(train_sub)}, val subset={len(val_sub)}")
            if train_sub:
                t = self._predict_split(train_sub, candles_map, ts_idx_map, system_prompt, simulate=False, label="train")
                train_acc = direction_accuracy(t["predictions"], t["actuals"])
            if val_sub:
                v = self._predict_split(val_sub, candles_map, ts_idx_map, system_prompt, simulate=False, label="val")
                val_acc = direction_accuracy(v["predictions"], v["actuals"])
            if train_acc is not None:
                gap = train_acc - test_acc

        # --- Heuristic indicator baseline on the SAME test set ---
        baseline_preds = [{"bias": self._heuristic_bias(s["indicators"]), "confidence": 5} for s in test]
        baseline_acc = direction_accuracy(baseline_preds, actuals) if actuals else 0.0
        baseline_metrics = {"direction_accuracy": baseline_acc, "kind": "indicator_heuristic"}

        log.info(
            f"  Test accuracy: {test_acc:.4f}  win_rate={metrics.get('win_rate', 0):.4f}  "
            f"profit_factor={metrics.get('profit_factor', 0):.3f}"
        )
        if gap is not None:
            log.info(f"  Train acc: {train_acc:.4f}  Val acc: {val_acc:.4f}  Gap(train-test): {gap:+.4f}")
        log.info(f"  Heuristic baseline accuracy: {baseline_acc:.4f}")
        log.info(f"  Parse errors: {parse_errors}")

        return {
            "accuracy": test_acc,
            "metrics": metrics,
            "total_evaluated": len(predictions),
            "errors": parse_errors,
            # Overfitting diagnostics (None when --diagnostic-samples 0).
            "train_acc": train_acc,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "gap": gap,
            "baseline_metrics": baseline_metrics,
            # Per-sample arrays — required for the paired McNemar + t-test.
            # Same schema as LLMBacktestRunner (predictions are dicts with bias).
            "predictions": predictions,
            "actuals": actuals,
            "trade_results": trade_results,
            "sample_keys": sample_keys,
        }

    def save_loss_curve(self):
        """Persist per-step train/eval loss (JSON + PNG) for overfitting analysis.

        ``train_loss`` reported at the end is a run average and not directly
        comparable to ``eval_loss``; the per-step curve shows where (if) eval loss
        starts rising while train loss keeps falling — the visual overfitting
        signal for the thesis.
        """
        if self.trainer is None:
            return
        tag = self.cfg["tag"]
        history = self.trainer.state.log_history
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        json_path = RESULTS_DIR / f"{tag}_loss_curve.json"
        with open(json_path, "w") as f:
            json.dump(history, f, indent=2)
        log.info(f"  Loss curve data: {json_path}")
        try:
            save_loss_curve_plot(history, RESULTS_DIR / f"{tag}_loss_curve.png", title=f"{tag} — train vs eval loss")
        except Exception as e:  # plotting is best-effort; never fail the run over a chart
            log.warning(f"  Loss curve plot failed (non-fatal): {e}")

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

        # Step 3b: Persist the loss curve (JSON + PNG) before anything can fail
        self.save_loss_curve()

        # Step 4: Save LoRA adapters
        self.save_model()

        # Step 5: Evaluate on test set (before GGUF so results are saved even if GGUF fails)
        test_metrics = self.evaluate()

        elapsed = time.time() - t0

        # Assemble final result. Per-sample arrays (predictions/actuals/
        # trade_results/sample_keys) live at the TOP level — the same schema as
        # LLMBacktestRunner — so stats_tests.compare() and analyze_results can
        # read them directly for the paired McNemar / t-test.
        tag = self.cfg["tag"]
        result = {
            "model": tag,
            "tag": tag,
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
            # Overfitting diagnostics (Apéndice A): train/val/test accuracy + gap.
            "overfitting": {
                "train_acc": test_metrics["train_acc"],
                "val_acc": test_metrics["val_acc"],
                "test_acc": test_metrics["test_acc"],
                "gap": test_metrics["gap"],
            },
            "baseline_metrics": test_metrics["baseline_metrics"],
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

        # Save result JSON (before GGUF — ensures results survive even if GGUF fails).
        # Write a per-tag file (so single-model runs don't clobber each other) plus a
        # canonical qlora_optimization.json that `compare-stats` reads by default.
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / f"qlora_{tag}.json"
        canonical = RESULTS_DIR / "qlora_optimization.json"
        for path in (out_path, canonical):
            with open(path, "w") as f:
                json.dump(result, f, indent=2)
        log.info(f"\nResult saved: {out_path} (canonical copy: {canonical})")

        # Step 6: GGUF export (optional, non-fatal)
        self.save_gguf()

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
    parser.add_argument(
        "--diagnostic-samples",
        type=int,
        help="Train/val subset size for the overfitting gap (default 500; 0 disables)",
    )
    parser.add_argument("--tag", type=str, help="Run tag — names result/loss-curve files and the model id")
    parser.add_argument("--max-steps", type=int, help="Cap optimizer steps (smoke test the training loop)")
    parser.add_argument("--grad-accum", type=int, help="Gradient accumulation steps (default: 16)")
    parser.add_argument("--output-dir", type=str, help="Output directory for model checkpoints/adapters")
    parser.add_argument(
        "--resume", action="store_true", help="Resume from the latest checkpoint-N in output_dir if present"
    )
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
    if args.max_eval is not None:
        config["max_eval_samples"] = args.max_eval
    if args.diagnostic_samples is not None:
        config["diagnostic_samples"] = args.diagnostic_samples
    if args.tag:
        config["tag"] = args.tag
    if args.max_steps:
        config["max_steps"] = args.max_steps
    if args.grad_accum:
        config["gradient_accumulation_steps"] = args.grad_accum
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.resume:
        config["resume"] = True
    if args.model:
        config["model_name"] = args.model

    # Run training
    trainer = QLoRATrainer(config)
    result = trainer.run()

    # Print summary
    m = result["metrics"]
    ov = result["overfitting"]
    base = result.get("baseline_metrics", {}).get("direction_accuracy")
    print(f"\n{'=' * 60}")
    print(f"  RESULT: Test Accuracy = {result['test_metrics']['accuracy']:.4f}")
    print(f"  Win Rate = {m.get('win_rate', 0):.4f}  Profit Factor = {m.get('profit_factor', 0):.3f}")
    print(f"  Sharpe = {m.get('sharpe_ratio', 0):.3f}  Max Drawdown = {m.get('max_drawdown', 0):.2f}%")
    if ov.get("gap") is not None:
        print(f"  Train = {ov['train_acc']:.4f}  Val = {ov['val_acc']:.4f}  Gap(train-test) = {ov['gap']:+.4f}")
    if base is not None:
        print(f"  Heuristic baseline = {base:.4f}")
    print(f"  Time = {result['elapsed_seconds']:.0f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
