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
import random
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
from backtest.evaluation.simulate import _parse_prediction, simulate_trade, simulate_trade_atr
from backtest.export import export_training_data
from backtest.models.features import _load_dataset, _temporal_split
from optimization.io.plots import save_loss_curve_plot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Suppress the noisy "max_new_tokens vs max_length" warning that fires on every
# generate() call (one per eval sample — thousands of lines of useless spam).
logging.getLogger("transformers.generation.configuration_utils").setLevel(logging.ERROR)

_LABELED_DIR = LANGGRAPH_ROOT / "backtest" / "data" / "labeled"
DATASET_PATH = str(_LABELED_DIR / "dataset.jsonl")
# Unfiltered dataset produced by `cli.py prepare-dataset --no-drawdown-filter`
# (the experiment/no-drawdown-filter ablation — see docs/qlora/AUDIT_QLORA_88PCT.md §2).
DATASET_PATH_NO_FILTER = str(_LABELED_DIR / "dataset_no_drawdown_filter.jsonl")
TRAINING_DATA_DIR = LANGGRAPH_ROOT / "training_data"
# Separate export dir for the no-filter run so its chat-format train/val/test
# splits never overwrite the filtered run's (the two experiments stay isolated).
TRAINING_DATA_DIR_NO_FILTER = LANGGRAPH_ROOT / "training_data_no_filter"
# Maps --dataset-type → (labeled dataset path, chat-export dir).
DATASET_TYPES = {
    "filtered": (DATASET_PATH, str(TRAINING_DATA_DIR)),
    "no_filter": (DATASET_PATH_NO_FILTER, str(TRAINING_DATA_DIR_NO_FILTER)),
}
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
        "lora_dropout": 0.05,
        "epochs": 3,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
        "warmup_steps": 50,
        "weight_decay": 0.01,
        "seed": 42,
        "mode": "FUTURES",
        "max_steps": None,  # cap optimizer steps (smoke tests); None = full epochs
        "max_eval_samples": None,
        # Per-batch size for evaluation generation. 1 = the original exact
        # per-sample greedy path. >1 batches prompts through a single padded
        # generate() call — the throughput lever that turns an hours-long serial
        # test eval into minutes. Safe on >=48GB GPUs at 16; try 32 on 80GB.
        # Independent of the training batch_size (that's VRAM-bound on activations
        # + gradients; eval has no gradients so it can go much wider).
        "eval_batch_size": 1,
        # Subset size used to measure train/val direction accuracy for the
        # overfitting gap. The full test split is always evaluated; train/val are
        # capped because generation is expensive and a few hundred samples are
        # enough to estimate the gap. Set to 0 to skip the gap diagnostics.
        "diagnostic_samples": 500,
        "resume": False,  # resume from latest checkpoint-N in output_dir if present
        "tag": "qlora_qwen25_7b",  # names result/loss-curve files and the model id
        "output_dir": str(MODELS_DIR / "qlora_qwen25_7b"),
        # Labeled dataset used for BOTH training-data export and test evaluation,
        # plus the chat-format export dir. Defaults reproduce the original run
        # exactly. --dataset-type no_filter (or --dataset PATH) swaps in the
        # unfiltered dataset for the drawdown-filter ablation.
        "dataset_path": DATASET_PATH,
        "training_data_dir": str(TRAINING_DATA_DIR),
        "use_atr_tp_sl": False,
        # Feature-occlusion probe (AUDIT_QLORA_88PCT.md §10): drop these keys from
        # every timeframe's indicator dict before building the prompt. Applied at
        # BOTH train-data export (prepare_data()) and eval (_predict_split()), so
        # a retrain with this set actually never sees the field, not just the eval
        # pass. With --eval-only, only eval is affected (no retrain happens).
        "strip_fields": [],
        # Replace the symbol name with a generic placeholder ("ASSET") in the
        # prompt — same dual train+eval scope as strip_fields. Tests reliance on
        # symbol-specific pretrained associations (e.g. "BTC").
        "anonymize_symbol": False,
    }

    def __init__(self, config: dict[str, Any] | None = None):
        self.cfg = {**self.DEFAULT_CONFIG, **(config or {})}
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def prepare_data(self) -> dict[str, int]:
        """Export labeled dataset to chat-format JSONL splits for SFT."""
        dataset_path = self.cfg["dataset_path"]
        training_data_dir = self.cfg["training_data_dir"]
        strip_fields = self.cfg.get("strip_fields") or []
        anonymize_symbol = self.cfg.get("anonymize_symbol", False)
        if strip_fields or anonymize_symbol:
            # Separate export dir — export_training_data() always overwrites
            # train/val/test.jsonl unconditionally, so an occluded export must
            # not land in the shared dir other (non-occluded) runs read from.
            # Written back into self.cfg so _load_chat_dataset() (called later,
            # from train()) reads the occluded files too, not the defaults.
            training_data_dir = str(
                Path(training_data_dir).parent / f"{Path(training_data_dir).name}_{self.cfg['tag']}"
            )
            self.cfg["training_data_dir"] = training_data_dir
        log.info(f"Exporting training data from {dataset_path}")
        log.info(f"  Output: {training_data_dir}/")

        if not Path(dataset_path).exists():
            raise FileNotFoundError(
                f"Labeled dataset not found: {dataset_path}\n"
                "  For --dataset-type no_filter, generate it first with:\n"
                "    python -m cli prepare-dataset --no-drawdown-filter\n"
                "  (and `dvc pull backtest/data/candles` if candles are missing)."
            )

        counts = export_training_data(
            dataset_path=dataset_path,
            output_dir=training_data_dir,
            mode=self.cfg["mode"],
            use_atr_tp_sl=self.cfg.get("use_atr_tp_sl", False),
            anonymize_symbol=anonymize_symbol,
            strip_fields=strip_fields,
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

        path = Path(self.cfg["training_data_dir"]) / f"{split}.jsonl"
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
                eval_steps=150,
                save_strategy="steps",
                save_steps=150,
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
                callbacks=[EarlyStoppingCallback(early_stopping_patience=4, early_stopping_threshold=0.001)],
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
                eval_steps=150,
                save_strategy="steps",
                save_steps=150,
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
                callbacks=[EarlyStoppingCallback(early_stopping_patience=4, early_stopping_threshold=0.001)],
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

    def _occlude(self, symbol: str, indicators: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Feature-occlusion probe: optionally strip fields / anonymize the symbol
        before the prompt is built. No-op unless cfg['strip_fields'] or
        cfg['anonymize_symbol'] is set — default behavior is unchanged.
        """
        strip_fields = self.cfg.get("strip_fields") or []
        if strip_fields:
            indicators = {
                tf: {k: v for k, v in tf_ind.items() if k not in strip_fields} for tf, tf_ind in indicators.items()
            }
        if self.cfg.get("anonymize_symbol"):
            symbol = "ASSET"
        return symbol, indicators

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
        predictions: list[dict[str, Any]] = []
        actuals: list[dict[str, Any]] = []
        trade_results: list[dict[str, Any]] = []
        atr_trade_results: list[dict[str, Any]] = []
        sample_keys: list[str] = []
        parse_errors = 0

        total = len(samples)
        batch_size = max(1, int(self.cfg.get("eval_batch_size", 1) or 1))

        # Pre-build the chat-templated prompt for every sample (strings only, so
        # this is cheap) up front, so generation can run in batches. Normalize a
        # single-TF indicator dict to the {"1h": {...}} shape the prompt builder
        # expects — same guard the original per-sample loop applied.
        prompts: list[str] = []
        for sample in samples:
            symbol = sample.get("symbol", "BTCUSDT")
            indicators = sample["indicators"]
            if indicators and not isinstance(next(iter(indicators.values())), dict):
                indicators = {"1h": indicators}
            symbol, indicators = self._occlude(symbol, indicators)
            user_prompt = build_user_prompt(symbol, indicators)
            prompts.append(
                self.tokenizer.apply_chat_template(
                    [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        # Batched generation is the throughput lever. A serial greedy decode is
        # latency-bound (~5-17s/sample depending on GPU), so the full test set can
        # take many hours one-at-a-time. Decoding `batch_size` prompts in one
        # padded generate() call is near-free until compute-bound, cutting eval to
        # minutes. batch_size=1 reproduces the original exact per-sample path. If a
        # batched generate ever fails (e.g. an Unsloth fast-path quirk on padded
        # input), we fall back to the proven serial path for THAT batch only, so
        # results are never wrong or dropped — only slower.
        n_batches = (total + batch_size - 1) // batch_size
        log_every = 1 if n_batches <= 20 else min(50, max(1, n_batches // 20))
        start_t = time.time()
        log.info(
            f"  [{label}] generating predictions for {total} samples (batch_size={batch_size}, {n_batches} batches)"
        )

        for b in range(n_batches):
            b_start = b * batch_size
            b_end = min(b_start + batch_size, total)
            batch_samples = samples[b_start:b_end]
            batch_prompts = prompts[b_start:b_end]

            if batch_size == 1:
                generated_texts = [self._generate_single(batch_prompts[0])]
            else:
                try:
                    generated_texts = self._generate_batch(batch_prompts)
                except Exception as exc:  # noqa: BLE001 — degrade gracefully, never drop samples
                    log.warning(
                        f"  [{label}] batched generate failed ({exc}); "
                        f"falling back to per-sample for batch {b + 1}/{n_batches}"
                    )
                    generated_texts = [self._generate_single(p) for p in batch_prompts]

            for sample, generated in zip(batch_samples, generated_texts):
                symbol = sample.get("symbol", "BTCUSDT")
                label_obj = sample["label"]

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
                    atr_raw = sample.get("atr_raw", 0)
                    candle_idx = ts_idx_map.get(symbol, {}).get(sample["timestamp"])
                    if candle_idx is not None and candle_idx + 1 < len(candles_map.get(symbol, [])):
                        future = candles_map[symbol][candle_idx + 1 : candle_idx + 25]
                        trade_results.append(simulate_trade(prediction, future))
                        atr_trade_results.append(simulate_trade_atr(prediction, atr_raw, future))
                    else:
                        trade_results.append({"outcome": "ERROR", "pnl_pct": 0, "hold_bars": 0})
                        atr_trade_results.append({"outcome": "ERROR", "pnl_pct": 0, "hold_bars": 0})

            if (b + 1) % log_every == 0 or (b + 1) == n_batches:
                done = b_end
                elapsed = time.time() - start_t
                rate = done / elapsed if elapsed > 0 else 0.0
                eta = (total - done) / rate if rate > 0 else 0.0
                log.info(
                    f"  [{label}] {done}/{total} ({parse_errors} parse errors, {rate:.2f} samples/s, ETA {eta:.0f}s)"
                )

        return {
            "predictions": predictions,
            "actuals": actuals,
            "trade_results": trade_results,
            "atr_trade_results": atr_trade_results,
            "sample_keys": sample_keys,
            "parse_errors": parse_errors,
        }

    def _generate_single(self, input_text: str) -> str:
        """Greedy-decode one prompt — the original, proven per-sample path.

        Used for eval_batch_size=1 and as the safe fallback when a batched
        generate fails, so a fast-path quirk never costs correctness.
        """
        import torch

        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        return self.tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    def _generate_batch(self, input_texts: list[str]) -> list[str]:
        """Greedy-decode a batch of prompts with LEFT padding.

        Decoder-only generation requires left padding so every sequence's newly
        generated tokens begin at the same column (the shared, padded input
        width); right padding would splice pad tokens into the continuation and
        corrupt the output. Greedy + left padding makes each row's result
        identical to decoding it alone. Padding side is saved and restored so no
        other state is disturbed.
        """
        import torch

        tok = self.tokenizer
        if tok.pad_token_id is None:  # Qwen2.5 ships a pad token, but be safe
            tok.pad_token = tok.eos_token
        prev_side = tok.padding_side
        tok.padding_side = "left"
        try:
            enc = tok(input_texts, return_tensors="pt", padding=True).to(self.model.device)
            with torch.no_grad():
                output_ids = self.model.generate(
                    **enc,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
            # Left padding makes the prompt width uniform across the batch, so the
            # generated continuation for every row starts at the same column.
            gen = output_ids[:, enc["input_ids"].shape[1] :]
            return tok.batch_decode(gen, skip_special_tokens=True)
        finally:
            tok.padding_side = prev_side

    @staticmethod
    def _heuristic_bias(indicators: dict[str, Any]) -> str:
        """Multi-TF majority-vote heuristic used as a comparison baseline.

        Majority-class accuracy is only ~50%, so the meaningful contrast is
        against an indicator heuristic: if the fine-tuned model barely beats
        this, the task is easy; if it clearly beats it, the model learned
        something non-trivial. Uses all available timeframes (not just 1h)
        for a fairer ~53% baseline on the strict temporal test set.
        """
        if indicators and not isinstance(next(iter(indicators.values())), dict):
            indicators = {"1h": indicators}
        bullish = sum(1 for v in indicators.values() if isinstance(v, dict) and "BULLISH" in v.get("heatmap", ""))
        bearish = sum(1 for v in indicators.values() if isinstance(v, dict) and "BEARISH" in v.get("heatmap", ""))
        if bullish > bearish:
            return "LONG"
        if bearish > bullish:
            return "SHORT"
        base = indicators.get("1h", next(iter(indicators.values()), {}))
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
        all_samples = _load_dataset(self.cfg["dataset_path"])
        train, val, test = _temporal_split(all_samples)

        empty = {
            "accuracy": 0,
            "metrics": {},
            "metrics_atr": {},
            "total_evaluated": 0,
            "errors": 0,
            "predictions": [],
            "actuals": [],
            "trade_results": [],
            "atr_trade_results": [],
            "sample_keys": [],
            "train_acc": None,
            "val_acc": None,
            "test_acc": 0,
            "gap": None,
            "baseline_metrics": {},
        }
        max_samples = self.cfg.get("max_eval_samples")
        if max_samples is not None and max_samples < len(test):
            if max_samples == 0:
                log.info("  --max-eval 0: skipping evaluation")
                return empty
            # Evenly STRIDE across the full (timestamp-sorted) test split instead
            # of taking a prefix. test[:N] would only cover the EARLIEST slice of
            # the holdout window — one market regime, possibly skewed by symbol —
            # so a bigger N still wouldn't be representative. Striding picks
            # roughly every k-th sample, spanning the whole test period and all
            # symbols, so a sub-sample stays a faithful mini-test set. Greedy decode
            # is deterministic, so this selection is reproducible across runs.
            full_test_n = len(test)
            stride = full_test_n / max_samples
            test = [test[int(i * stride)] for i in range(max_samples)]
            log.info(
                f"  Evaluating on {len(test)} test samples strided across the full "
                f"holdout (every ~{stride:.1f}th of {full_test_n})"
            )
        log.info(f"  Test samples: {len(test)}")

        candles_map, ts_idx_map = _load_candles_map(CANDLES_DIR)
        system_prompt = build_system_prompt(self.cfg["mode"])

        # --- Full TEST backtest (with trade simulation) ---
        test_out = self._predict_split(test, candles_map, ts_idx_map, system_prompt, simulate=True, label="test")
        predictions = test_out["predictions"]
        actuals = test_out["actuals"]
        trade_results = test_out["trade_results"]
        atr_trade_results = test_out["atr_trade_results"]
        sample_keys = test_out["sample_keys"]
        parse_errors = test_out["parse_errors"]

        metrics = compute_all_metrics(predictions, actuals, trade_results) if predictions else {}
        atr_metrics = compute_all_metrics(predictions, actuals, atr_trade_results) if predictions else {}
        test_acc = metrics.get("direction_accuracy", 0.0)

        # --- Overfitting gap: TRAIN/VAL direction accuracy on capped subsets ---
        n_diag = self.cfg.get("diagnostic_samples") or 0
        train_acc = val_acc = gap = None
        if n_diag > 0:
            # Sample randomly from across the full training period (not just oldest).
            # train[:n_diag] would bias toward the earliest market regime (2023).
            rng = random.Random(42)
            train_sub = rng.sample(train, min(n_diag, len(train)))
            val_sub = val[:n_diag]
            log.info(f"  Gap diagnostics: train subset={len(train_sub)}, val subset={len(val_sub)}")
            if train_sub:
                t = self._predict_split(
                    train_sub, candles_map, ts_idx_map, system_prompt, simulate=False, label="train"
                )
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
        log.info(
            f"  ATR sim: win_rate={atr_metrics.get('win_rate', 0):.4f}  "
            f"profit_factor={atr_metrics.get('profit_factor', 0):.3f}"
        )
        if gap is not None:
            log.info(f"  Train acc: {train_acc:.4f}  Val acc: {val_acc:.4f}  Gap(train-test): {gap:+.4f}")
        log.info(f"  Heuristic baseline accuracy: {baseline_acc:.4f}")
        log.info(f"  Parse errors: {parse_errors}")

        return {
            "accuracy": test_acc,
            "metrics": metrics,
            "metrics_atr": atr_metrics,
            "total_evaluated": len(predictions),
            "errors": parse_errors,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "gap": gap,
            "baseline_metrics": baseline_metrics,
            "predictions": predictions,
            "actuals": actuals,
            "trade_results": trade_results,
            "atr_trade_results": atr_trade_results,
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
        run_dir = RESULTS_DIR / tag
        run_dir.mkdir(parents=True, exist_ok=True)
        json_path = run_dir / "loss_curve.json"
        with open(json_path, "w") as f:
            json.dump(history, f, indent=2)
        log.info(f"  Loss curve data: {json_path}")
        try:
            save_loss_curve_plot(history, run_dir / "loss_curve.png", title=f"{tag} — train vs eval loss")
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

    def load_for_inference(self, adapters_dir: str) -> None:
        """Load saved LoRA adapters directly — no training scaffolding."""
        from unsloth import FastLanguageModel

        log.info(f"Loading saved adapters for inference: {adapters_dir}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(adapters_dir),
            max_seq_length=self.cfg["max_seq_length"],
            dtype=None,
            load_in_4bit=True,
        )
        log.info("  Adapters loaded (inference mode set inside evaluate())")

    def run_eval_only(self, adapters_dir: str) -> dict[str, Any]:
        """Skip training — load saved adapters and evaluate on the test set."""
        log.info("=" * 60)
        log.info("  QLoRA Eval-Only — loading saved adapters")
        log.info(f"  Adapters: {adapters_dir}")
        log.info("=" * 60)

        t0 = time.time()
        self.load_for_inference(adapters_dir)
        test_metrics = self.evaluate()
        elapsed = time.time() - t0

        tag = self.cfg["tag"]
        result = {
            "model": tag,
            "tag": tag,
            "ensemble_type": "single",
            "strategy": "fine_tuning_eval_only",
            "eval_only": True,
            "adapters_dir": str(adapters_dir),
            "best_score": test_metrics["accuracy"],
            "best_params": {"adapters_dir": str(adapters_dir)},
            "val_metrics": {},
            "test_metrics": {
                "accuracy": test_metrics["accuracy"],
                "total_evaluated": test_metrics["total_evaluated"],
                "errors": test_metrics["errors"],
                **test_metrics["metrics"],
            },
            "overfitting": {
                "train_acc": test_metrics["train_acc"],
                "val_acc": test_metrics["val_acc"],
                "test_acc": test_metrics["test_acc"],
                "gap": test_metrics["gap"],
            },
            "baseline_metrics": test_metrics["baseline_metrics"],
            "metrics": test_metrics["metrics"],
            "metrics_atr": test_metrics.get("metrics_atr", {}),
            "predictions": test_metrics["predictions"],
            "actuals": test_metrics["actuals"],
            "trade_results": test_metrics["trade_results"],
            "atr_trade_results": test_metrics.get("atr_trade_results", []),
            "sample_keys": test_metrics["sample_keys"],
            "elapsed_seconds": round(elapsed, 1),
            "timestamp": datetime.now().isoformat(),
        }

        run_dir = RESULTS_DIR / tag
        run_dir.mkdir(parents=True, exist_ok=True)
        out_path = run_dir / "result.json"
        canonical = RESULTS_DIR / "qlora_optimization.json"
        for path in (out_path, canonical):
            with open(path, "w") as f:
                json.dump(result, f, indent=2)
        log.info(f"\nResult saved: {out_path} (canonical: {canonical})")
        log.info(f"Total time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
        return result

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
            "metrics_atr": test_metrics.get("metrics_atr", {}),
            "predictions": test_metrics["predictions"],
            "actuals": test_metrics["actuals"],
            "trade_results": test_metrics["trade_results"],
            "atr_trade_results": test_metrics.get("atr_trade_results", []),
            "sample_keys": test_metrics["sample_keys"],
            "data_counts": counts,
            "train_info": train_info,
            "elapsed_seconds": round(elapsed, 1),
            "timestamp": datetime.now().isoformat(),
        }

        # Save result JSON (before GGUF — ensures results survive even if GGUF fails).
        # Per-run folder: results/<tag>/result.json + loss_curve.{json,png}.
        # Also writes results/qlora_optimization.json as the "latest" canonical so
        # `compare-stats --a optimization/qlora/results/qlora_optimization.json` keeps working.
        run_dir = RESULTS_DIR / tag
        run_dir.mkdir(parents=True, exist_ok=True)
        out_path = run_dir / "result.json"
        canonical = RESULTS_DIR / "qlora_optimization.json"
        for path in (out_path, canonical):
            with open(path, "w") as f:
                json.dump(result, f, indent=2)
        log.info(f"\nResult saved: {out_path} (canonical: {canonical})")

        # Step 6: GGUF export (optional, non-fatal; skipped with --no-gguf)
        if not self.cfg.get("skip_gguf", False):
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
    parser.add_argument(
        "--max-eval",
        type=int,
        help="Cap test-eval samples, strided evenly across the full holdout (representative). Omit = full test.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        help="Prompts per generate() call during eval. 1=serial (default); 16-32 on big GPUs cuts eval to minutes.",
    )
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
    parser.add_argument(
        "--dataset-type",
        choices=["filtered", "no_filter"],
        default="filtered",
        help=(
            "Which labeled dataset to train + evaluate on. 'filtered' (default) = "
            "the production dataset.jsonl. 'no_filter' = dataset_no_drawdown_filter.jsonl "
            "(the drawdown-filter ablation; generate it first with "
            "`cli prepare-dataset --no-drawdown-filter`). See docs/qlora/AUDIT_QLORA_88PCT.md §2."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Explicit labeled dataset path. Overrides --dataset-type if given.",
    )
    parser.add_argument(
        "--no-gguf",
        action="store_true",
        help="Skip GGUF export (useful for smoke tests — adapters are always saved regardless)",
    )
    parser.add_argument(
        "--use-atr-tp-sl",
        action="store_true",
        help="Use ATR-based forward-looking TP/SL in training labels instead of hindsight-derived values. "
        "Breaks the circular dependency where the model learns to replicate price targets derived from future data.",
    )
    parser.add_argument(
        "--eval-only",
        type=str,
        metavar="ADAPTERS_DIR",
        help="Skip training — load saved adapters from this path and run eval only",
    )
    parser.add_argument(
        "--strip-fields",
        type=str,
        help="Comma-separated indicator field names to drop from the prompt (e.g. 'heatmap,structure'). "
        "Feature-occlusion probe — applied to training data export AND eval. With --eval-only, only eval is affected.",
    )
    parser.add_argument(
        "--anonymize-symbol",
        action="store_true",
        help="Replace the symbol name with a generic placeholder ('ASSET') in the prompt. "
        "Feature-occlusion probe — applied to training data export AND eval. With --eval-only, only eval is affected.",
    )
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
    if args.eval_batch_size is not None:
        config["eval_batch_size"] = args.eval_batch_size
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

    # Dataset selection: --dataset-type picks a known (dataset, export-dir) pair;
    # an explicit --dataset path overrides just the dataset (export dir still
    # follows the type, so no_filter exports stay in training_data_no_filter/).
    ds_path, ds_export_dir = DATASET_TYPES[args.dataset_type]
    config["dataset_path"] = args.dataset or ds_path
    config["training_data_dir"] = ds_export_dir

    if args.use_atr_tp_sl:
        config["use_atr_tp_sl"] = True

    if args.no_gguf:
        config["skip_gguf"] = True

    if args.strip_fields:
        config["strip_fields"] = [f.strip() for f in args.strip_fields.split(",") if f.strip()]

    if args.anonymize_symbol:
        config["anonymize_symbol"] = True

    trainer = QLoRATrainer(config)
    if args.eval_only:
        result = trainer.run_eval_only(args.eval_only)
    else:
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
