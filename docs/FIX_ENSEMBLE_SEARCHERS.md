# Fix Guide: Ensemble Searchers

Diagnosis completed 2026-06-15. Five confirmed problems in priority order.
Apply in the sequence listed at the bottom — earlier fixes are prerequisites for later ones.

---

## Why the Ensemble Gets 68% Instead of ~81%

The `LSTMSearcher` 81.4% score was measured on a **file-order val split** (last 2 symbols in
`dataset.jsonl`, e.g. MATICUSDT/NEARUSDT). The ensemble 68.4% is measured on the
**temporal test split** (most recent 3 months, all 12 symbols). These measure different
populations — the apparent drop is partly an apples-vs-oranges comparison, not pure
regression. The real temporal performance of a correctly configured LSTM is unknown until
the fixes below are applied.

The dominant causes of the gap (in order):
1. Suboptimal hardcoded hyperparameters (`lr=0.001` vs best `0.0005`, `batch=32` vs `64`, `dropout=0.2` vs `0.3`)
2. File-order split bug in `LSTMPredictor.train()` — trains on wrong data
3. No bootstrap sampling — all bags are identical
4. In-sample weight calibration in VotingSearcher
5. LSTM ignores fold/slice boundaries in Stacking and Blending

---

## Fix 1 — Extend `LSTMPredictor` to expose all hyperparameters

**File**: `langgraph/backtest/models/lstm.py`
**Impact**: Highest. Unlocks the grid-search winner params for every ensemble that uses LSTM.

### 1a. Add `dropout` parameter to `_LSTMNet` (around line 185)

**Before:**
```python
class _LSTMNet:
    def __new__(cls, input_size, hidden_size, num_layers):
        ...
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=0.2 if num_layers > 1 else 0.0,
                    batch_first=True,
                )
```

**After:**
```python
class _LSTMNet:
    def __new__(cls, input_size, hidden_size, num_layers, dropout=0.2):
        ...
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0.0,
                    batch_first=True,
                )
```

### 1b. Add `learning_rate`, `dropout`, `batch_size` to `LSTMPredictor.__init__` (line ~13)

**Before:**
```python
def __init__(self, sequence_length=10, hidden_size=64, num_layers=2):
    self.sequence_length = sequence_length
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self._mean = None
    self._std = None
    self._timeframes = None
    self.input_size = None
    self.model = None
```

**After:**
```python
def __init__(self, sequence_length=10, hidden_size=64, num_layers=2,
             learning_rate=0.001, dropout=0.2, batch_size=32):
    self.sequence_length = sequence_length
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.learning_rate = learning_rate
    self.dropout = dropout
    self.batch_size = batch_size
    self._mean = None
    self._std = None
    self._timeframes = None
    self.input_size = None
    self.model = None
```

### 1c. Wire the new params into `train()` (lines ~59–63)

**Before:**
```python
    self.input_size = X_all.shape[1]
    self.model = _LSTMNet(self.input_size, self.hidden_size, self.num_layers)
    optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
    self.model.train()
    batch_size = 32
```

**After:**
```python
    self.input_size = X_all.shape[1]
    self.model = _LSTMNet(self.input_size, self.hidden_size, self.num_layers, self.dropout)
    optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    self.model.train()
    batch_size = self.batch_size
```

---

## Fix 2 — Fix file-order vs temporal-order slice in `LSTMPredictor.train()`

**File**: `langgraph/backtest/models/lstm.py`
**Lines**: ~40–56

`_temporal_split` sorts its argument and returns a new list — it does **not** mutate the
caller's `samples` variable. So `X_all` built from `samples` is in file order (symbol-grouped),
but `n_train` comes from the temporal sort. `X_all[:n_train]` is the first `n_train` rows by
file position — roughly all BTCUSDT + ETHUSDT rows — not the temporally-earliest rows.

**Before:**
```python
    samples = _load_dataset(dataset_path)
    self._timeframes = _infer_timeframes(samples)
    train_s, val_s, _test_s = _temporal_split(samples, 0.70, val_split)

    X_all = np.array(
        [extract_features(s["indicators"], self._timeframes) for s in samples],
        dtype=np.float32,
    )
    y_all = np.array(
        [1 if s["label"]["bias"] == "LONG" else 0 for s in samples],
        dtype=np.int32,
    )

    n_train = len(train_s)
    n_val = len(val_s)

    self._mean = X_all[:n_train].mean(axis=0)
    self._std = X_all[:n_train].std(axis=0) + 1e-8
```

**After** (sort `samples` before building `X_all`, pass `sort=False` to avoid redundant sort):
```python
    raw = _load_dataset(dataset_path)
    self._timeframes = _infer_timeframes(raw)
    samples = sorted(raw, key=lambda s: s.get("timestamp", 0))
    train_s, val_s, _test_s = _temporal_split(samples, 0.70, val_split, sort=False)

    X_all = np.array(
        [extract_features(s["indicators"], self._timeframes) for s in samples],
        dtype=np.float32,
    )
    y_all = np.array(
        [1 if s["label"]["bias"] == "LONG" else 0 for s in samples],
        dtype=np.int32,
    )

    n_train = len(train_s)
    n_val = len(val_s)

    self._mean = X_all[:n_train].mean(axis=0)
    self._std = X_all[:n_train].std(axis=0) + 1e-8
```

---

## Fix 3 — Load best params from optimization JSONs in all ensemble searchers

**File**: `langgraph/optimization/searchers/ensemble_searcher.py`

Add this helper near the top of the file, after the imports:

```python
import json as _json
import pathlib as _pathlib

def _load_best_params(model_type: str) -> dict:
    """Return best_params from the model's optimization result JSON, or {} if missing."""
    path = _pathlib.Path(__file__).parent.parent / "results" / f"{model_type}_optimization.json"
    if path.exists():
        return _json.loads(path.read_text()).get("best_params", {})
    return {}
```

Then in **each** of the four searchers' `search()` methods, replace the hardcoded defaults:

**Before** (same pattern in all four, e.g. `BaggingLSTMSearcher` line ~102):
```python
    lstm_params = cfg.get("lstm_params", {"hidden_size": 32, "num_layers": 3, "sequence_length": 5})
    xgb_params = cfg.get("xgb_params", {})
```

**After:**
```python
    _lstm_best = _load_best_params("lstm")
    lstm_defaults = {
        "hidden_size":    _lstm_best.get("hidden_size", 32),
        "num_layers":     _lstm_best.get("num_layers", 2),
        "sequence_length":_lstm_best.get("sequence_length", 5),
        "learning_rate":  _lstm_best.get("learning_rate", 0.001),
        "dropout":        _lstm_best.get("dropout", 0.2),
        "batch_size":     _lstm_best.get("batch_size", 32),
    }
    lstm_params = {**lstm_defaults, **cfg.get("lstm_params", {})}

    _xgb_best = _load_best_params("xgboost")
    xgb_params = {**_xgb_best, **cfg.get("xgb_params", {})}
```

Apply the same substitution in all four searchers: `BaggingLSTMSearcher`, `VotingSearcher`,
`StackingSearcher`, `BlendingSearcher`.

**Quick verification** — the `lstm_optimization.json` best_params should resolve to:
```
hidden_size=32, num_layers=2, sequence_length=5,
learning_rate=0.0005, dropout=0.3, batch_size=64
```
If the path is wrong, `_load_best_params` returns `{}` and falls back to defaults — add a
`print(f"[ensemble] lstm best_params: {_lstm_best}")` temporarily to confirm the path resolves.

---

## Fix 4 — Add bootstrap sampling to `BaggingLSTMSearcher`

True bagging requires each bag to train on a **different random subset** (with replacement)
of the training data. Currently every bag calls `predictor.train(dataset_path)` on the
identical full dataset → identical training data → all bags converge to the same local minimum
regardless of different weight initializations.

This fix requires a new method on `LSTMPredictor` that accepts pre-built tensors instead of
a dataset path, so we can pass per-bag bootstrapped data.

### 4a. Add `train_from_sequences()` to `LSTMPredictor`

**File**: `langgraph/backtest/models/lstm.py` — add after the existing `train()` method.

```python
def train_from_sequences(
    self,
    X_train_seq: "torch.Tensor",
    y_train: "torch.Tensor",
    X_val_seq: "torch.Tensor",
    y_val: "torch.Tensor",
    max_epochs: int = 50,
    patience: int = 10,
) -> dict:
    """Train directly on pre-built sequence tensors (for bootstrap bagging).

    Caller is responsible for normalization and sequence construction.
    `self.input_size`, `self._mean`, `self._std`, `self._timeframes` must be set
    before calling this method.
    """
    import torch
    import torch.nn as nn

    self.model = _LSTMNet(self.input_size, self.hidden_size, self.num_layers, self.dropout)
    optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    criterion = nn.CrossEntropyLoss()
    best_val, no_improve = float("inf"), 0
    last_train_loss = float("inf")

    for epoch in range(max_epochs):
        self.model.train()
        perm = torch.randperm(len(X_train_seq))
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(X_train_seq), self.batch_size):
            idx = perm[i: i + self.batch_size]
            optimizer.zero_grad()
            out = self.model(X_train_seq[idx])
            loss = criterion(out, y_train[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        last_train_loss = epoch_loss / max(1, n_batches)

        self.model.eval()
        with torch.no_grad():
            val_loss = criterion(self.model(X_val_seq), y_val).item()

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return {"train_loss": last_train_loss, "val_loss": best_val}
```

### 4b. Update `BaggingLSTMSearcher.search()` to bootstrap-sample per bag

**File**: `langgraph/optimization/searchers/ensemble_searcher.py`

Replace the data-loading block and the `for bag in range(n_bags)` loop
(approximately lines 115–145) with the version below.

The key invariant: **all bags share the same normalization** (computed from the temporal
train split once), so their probability outputs are on the same scale.

```python
    # ── load and split once ────────────────────────────────────────────────────
    all_data = self._load_all(dataset_path)      # _DataMixin: temporal split, normalized
    sl = lstm_params.get("sequence_length", 5)
    n_train = len(all_data["y_train"])
    n_tv    = len(all_data["y_trainval"])

    # Build sequence tensors from the temporal trainval block
    X_tv_norm = all_data["X_trainval"]           # already normalized by _DataMixin
    X_all_seqs = self._build_lstm_sequences(X_tv_norm, sl, 0, n_train)
    y_all_t    = torch.tensor(all_data["y_train"], dtype=torch.long)

    X_val_seqs = self._build_lstm_sequences(X_tv_norm, sl, n_train, n_tv)
    X_val_seqs = X_val_seqs[: len(all_data["y_val"])]
    y_val_t    = torch.tensor(all_data["y_val"], dtype=torch.long)

    # ── train bags with bootstrap sampling ────────────────────────────────────
    bag_preds_val  = []
    bag_preds_test = []

    # Build test sequences once (temporal test split)
    X_test_norm = all_data["X_test"]             # already normalized
    # Pad with the last sl-1 rows of trainval for sequence context
    X_context = np.vstack([X_tv_norm[-(sl - 1):], X_test_norm]) if sl > 1 else X_test_norm
    X_test_seqs = self._build_lstm_sequences(X_context, sl, sl - 1 if sl > 1 else 0,
                                             len(X_context))
    X_test_seqs = X_test_seqs[: len(all_data["y_test"])]

    for bag in range(n_bags):
        seed = base_seed + bag
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed)

        # Bootstrap sample from training indices (with replacement — true bagging)
        boot_idx = rng.choice(n_train, size=n_train, replace=True)
        X_boot = X_all_seqs[boot_idx]
        y_boot = y_all_t[boot_idx]

        # Set up predictor with shared normalization (already applied to X_tv_norm)
        predictor = LSTMPredictor(**lstm_params)
        predictor.input_size  = X_tv_norm.shape[1]
        predictor._mean       = np.zeros(predictor.input_size, dtype=np.float32)
        predictor._std        = np.ones(predictor.input_size, dtype=np.float32)
        predictor._timeframes = all_data["timeframes"]

        predictor.train_from_sequences(X_boot, y_boot, X_val_seqs, y_val_t)

        predictor.model.eval()
        with torch.no_grad():
            bag_preds_val.append(
                torch.softmax(predictor.model(X_val_seqs), dim=1)[:, 1].numpy()
            )
            bag_preds_test.append(
                torch.softmax(predictor.model(X_test_seqs), dim=1)[:, 1].numpy()
            )

    # ── aggregate ─────────────────────────────────────────────────────────────
    ensemble_val  = np.mean(bag_preds_val, axis=0)
    ensemble_test = np.mean(bag_preds_test, axis=0)
    # Continue with evaluation using all_data["y_val"] and all_data["y_test"] ...
```

> **Note on `_DataMixin._load_all` keys**: verify the exact dict keys it returns
> (`X_train`, `X_val`, `X_test`, `X_trainval`, `y_train`, `y_val`, `y_test`, `y_trainval`,
> `timeframes`). The above assumes those names — grep `_load_all` in `ensemble_searcher.py`
> to confirm. If normalization is stored separately (mean/std in the dict), use those
> directly instead of zeros/ones.

---

## Fix 5 — Fix VotingSearcher weight calibration

**File**: `langgraph/optimization/searchers/ensemble_searcher.py`
**Lines**: ~271–288

XGB and SVM are trained on `X_trainval` (train + val combined), then their "val accuracy"
is computed on `X_val` which they already trained on. This inflates their weights.
LSTM's val accuracy is genuinely out-of-sample (it trains internally on train only).

**Before:**
```python
    xgb_model.fit(data["X_trainval"], data["y_trainval"])
    xgb_val = xgb_model.predict_proba(data["X_val"])[:, 1]

    svm.fit(data["X_trainval_s"], data["y_trainval"])
    svm_val = svm.predict_proba(data["X_val_s"])[:, 1]
```

**After** (calibrate weights using train-only fits; keep the trainval fits for test inference):
```python
    # ── weight calibration: train-only fits to get fair val accuracy ──────────
    xgb_cal = xgb.XGBClassifier(
        **xgb_params, use_label_encoder=False, eval_metric="logloss", verbosity=0
    )
    xgb_cal.fit(data["X_train"], data["y_train"])
    xgb_val = xgb_cal.predict_proba(data["X_val"])[:, 1]

    svm_scaler_cal = StandardScaler().fit(data["X_train"])
    svm_cal = SVC(kernel="rbf", probability=True, random_state=42)
    svm_cal.fit(svm_scaler_cal.transform(data["X_train"]), data["y_train"])
    svm_val = svm_cal.predict_proba(svm_scaler_cal.transform(data["X_val"]))[:, 1]

    # ── production fits: trainval for actual test inference ───────────────────
    xgb_model.fit(data["X_trainval"], data["y_trainval"])
    svm.fit(data["X_trainval_s"], data["y_trainval"])
```

The weight computation block below this is unchanged — it uses `xgb_val` and `svm_val`
(now from fair hold-out evaluation) and the existing `lstm_val`.

---

## Fix 6 — Fix StackingSearcher LSTM OOF contamination

**File**: `langgraph/optimization/searchers/ensemble_searcher.py`
**Lines**: ~374–395

`lstm_f.train(dataset_path)` in the OOF loop ignores `tr_idx` entirely — the LSTM trains
on the same full dataset in every fold. The OOF predictions for `vl_idx` positions come
from a model that already saw those rows during training.

**Option A (recommended — simpler, defensible for thesis)**: Train LSTM once on
the full trainval split, fill OOF predictions from that single model. This is not
strictly out-of-fold, but is far cleaner than the current contaminated version.
Acknowledge the limitation in one sentence in the thesis ("LSTM OOF approximated by
single model trained on full training window to avoid fold boundary ambiguity").

**Before (inside the OOF loop):**
```python
    for fold, (tr_idx, vl_idx) in enumerate(tscv.split(X_tv)):
        ...
        lstm_f = LSTMPredictor(**lstm_params)
        lstm_f.train(dataset_path)
        X_tv_norm = (X_tv - lstm_f._mean) / lstm_f._std
        ...
        fold_seqs = self._build_lstm_sequences(X_tv_norm, sl, vl_idx[0], vl_idx[-1] + 1)
        oof[vl_idx, 3] = torch.softmax(lstm_f.model(fold_seqs), dim=1)[:, 1].numpy()
```

**After** (train LSTM once, before the fold loop):
```python
    # Train LSTM once on the temporal training split (Option A)
    lstm_single = LSTMPredictor(**lstm_params)
    lstm_single.train(dataset_path)
    X_tv_norm = (X_tv - lstm_single._mean) / lstm_single._std

    for fold, (tr_idx, vl_idx) in enumerate(tscv.split(X_tv)):
        ...
        # XGB / RF OOF blocks are unchanged (fold-aware)
        ...

        # LSTM OOF: single model's predictions for this fold's val indices
        fold_seqs = self._build_lstm_sequences(
            X_tv_norm, sl, int(vl_idx[0]), int(vl_idx[-1]) + 1
        )
        lstm_single.model.eval()
        with torch.no_grad():
            fold_probs = torch.softmax(lstm_single.model(fold_seqs), dim=1)[:, 1].numpy()
        # vl_idx may have gaps if TimeSeriesSplit skips rows — map by position
        for j, vi in enumerate(vl_idx):
            local = vi - int(vl_idx[0])
            if local < len(fold_probs):
                oof[vi, 3] = fold_probs[local]
```

**Option B** (more correct — requires Fix 4a): For each fold, call
`lstm_f.train_from_sequences(X_boot[tr_idx], y_tv[tr_idx], X_boot[vl_idx], y_tv[vl_idx])`
so the LSTM is excluded from each fold's val data. This is the proper OOF implementation
but requires the sequence pre-building infrastructure from Fix 4b.

---

## Fix 7 — Fix BlendingSearcher LSTM asymmetric data

**File**: `langgraph/optimization/searchers/ensemble_searcher.py`
**Lines**: ~527–528

XGB, SVM, and MLP train on `X_base = X_trainval[:split_idx]` (~70% of trainval, ~33K samples).
LSTM calls `train(dataset_path)` and internally trains on the full dataset's first ~70% in
file order (~39K samples, different slice). LSTM sees more and different training data →
blend-set predictions may include future information relative to the base learners.

**Requires Fix 4a** (`train_from_sequences`).

**Before:**
```python
    lstm_model = LSTMPredictor(**lstm_params)
    lstm_model.train(dataset_path)
```

**After** (train LSTM on the same `X_base` slice as the sklearn models):
```python
    sl = lstm_params.get("sequence_length", 5)

    # Shared normalization from _DataMixin (already applied to X_trainval)
    # Build sequences from X_base (same slice as XGB/SVM/MLP)
    X_base_seqs  = self._build_lstm_sequences(data["X_trainval"], sl, 0, split_idx)
    X_blend_seqs = self._build_lstm_sequences(data["X_trainval"], sl, split_idx, len(data["X_trainval"]))
    y_base_t  = torch.tensor(data["y_trainval"][:split_idx], dtype=torch.long)
    y_blend_t = torch.tensor(data["y_trainval"][split_idx:], dtype=torch.long)

    lstm_model = LSTMPredictor(**lstm_params)
    lstm_model.input_size  = data["X_trainval"].shape[1]
    lstm_model._mean       = np.zeros(lstm_model.input_size, dtype=np.float32)
    lstm_model._std        = np.ones(lstm_model.input_size, dtype=np.float32)
    lstm_model._timeframes = data["timeframes"]
    lstm_model.train_from_sequences(X_base_seqs, y_base_t, X_blend_seqs, y_blend_t)
```

Replace the downstream LSTM blend prediction:
```python
    # Before:  lstm_blend = torch.softmax(lstm_model.model(...), dim=1)[:, 1].numpy()
    # After:
    lstm_model.model.eval()
    with torch.no_grad():
        lstm_blend = torch.softmax(lstm_model.model(X_blend_seqs), dim=1)[:, 1].numpy()
```

---

## Implementation Order

Apply in this sequence — each fix depends on the ones above it.

```
Step 1  →  Fix 1 (lstm.py: __init__ signature + _LSTMNet dropout param)
Step 2  →  Fix 2 (lstm.py: sort samples before building X_all)
Step 3  →  Fix 3 (ensemble_searcher.py: _load_best_params helper + wire into all 4 searchers)
            ── CHECKPOINT: re-run BaggingLSTMSearcher, verify accuracy rises above 70% ──
Step 4  →  Fix 5 (VotingSearcher weight calibration) — independent, no prereqs
Step 5  →  Fix 4a (lstm.py: add train_from_sequences method)
Step 6  →  Fix 4b (BaggingLSTMSearcher: bootstrap sampling — requires 4a)
Step 7  →  Fix 6 (StackingSearcher Option A OOF — can be done without 4a)
Step 8  →  Fix 7 (BlendingSearcher symmetric slice — requires 4a)
            ── FINAL: re-run all ensembles ──
```

---

## Checkpoint Verification (after Steps 1–3)

```bash
cd langgraph && source .venv/bin/activate

# Confirm best_params are loading correctly
python - <<'EOF'
import json, pathlib
p = pathlib.Path("optimization/results/lstm_optimization.json")
print(json.loads(p.read_text())["best_params"])
EOF
# Expected: hidden_size=32, num_layers=2, sequence_length=5,
#           learning_rate=0.0005, dropout=0.3, batch_size=64

# Re-run bagging-lstm only (~10 min)
OMP_NUM_THREADS=1 python optimization/optimize.py \
  --model bagging_lstm \
  --config optimization/configs/bagging_lstm.yaml \
  --dataset backtest/data/labeled/dataset.jsonl
```

**Expected**: test accuracy > 70% (up from 68.4%). Bags should show different loss curves
(no longer all 0.3946/0.4344 for every bag). If accuracy is still ~68%, check that
`lstm_params` inside `BaggingLSTMSearcher` is picking up the loaded `best_params` —
add a temporary `print(f"[bag] lstm_params={lstm_params}")` at the top of `search()`.

---

## Final Verification (after all 8 steps)

```bash
OMP_NUM_THREADS=1 python optimization/run_ensembles.py \
  --dataset backtest/data/labeled/dataset.jsonl \
  2>&1 | tee logs/ensembles_fixed.log
```

Then update `NEXT_STEPS.md` Step 3 ("Re-train ML/ensembles") with the new result JSON tags,
and run the comparison:
```bash
python -m cli compare-stats \
  --a optimization/results/bagging_lstm_optimization.json \
  --b optimization/qlora/results/qlora_optimization.json
```

---

## Files Modified

| File | Fixes applied |
|------|--------------|
| `langgraph/backtest/models/lstm.py` | Fix 1, Fix 2, Fix 4a |
| `langgraph/optimization/searchers/ensemble_searcher.py` | Fix 3, Fix 4b, Fix 5, Fix 6, Fix 7 |
