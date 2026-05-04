#!/usr/bin/env python3
"""Optimización de hiperparámetros para modelos de predicción de señales de trading.

Uso:
    python optimize.py --model xgboost --config configs/xgboost.yaml
    python optimize.py --model lstm --config configs/lstm.yaml --dataset /ruta/a/data.jsonl
    nohup python optimize.py --model xgboost --config configs/xgboost.yaml > logs/xgb.log 2>&1 &
"""

import argparse
import itertools
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

# --- Imports del proyecto ---
# optimization/ está dentro de langgraph/, al mismo nivel que backtest/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backtest.ml_models import (  # noqa: E402
    _load_dataset,
    _samples_to_xy,
    _temporal_split,
    extract_features,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_DATASET = Path(__file__).resolve().parent.parent / "backtest" / "data" / "labeled" / "dataset.jsonl"


# ---------------------------------------------------------------------------
# Sklearn (XGBoost, Random Forest)
# ---------------------------------------------------------------------------

def optimize_sklearn(model_name: str, cfg: dict, X: np.ndarray, y: np.ndarray) -> dict:
    """GridSearchCV o RandomizedSearchCV para modelos sklearn."""
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit

    if model_name == "xgboost":
        import xgboost as xgb
        base = xgb.XGBClassifier(
            eval_metric=cfg.get("fixed_params", {}).get("eval_metric", "logloss"),
            random_state=cfg.get("fixed_params", {}).get("random_state", 42),
            use_label_encoder=False,
        )
    elif model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        fixed = cfg.get("fixed_params", {})
        param_grid = dict(cfg["param_grid"])
        if "max_depth" in param_grid:
            param_grid["max_depth"] = [None if v == -1 else v for v in param_grid["max_depth"]]
        cfg["param_grid"] = param_grid
        base = RandomForestClassifier(
            random_state=fixed.get("random_state", 42),
            n_jobs=fixed.get("n_jobs", -1),
        )
    else:
        raise ValueError(f"Modelo sklearn desconocido: {model_name}")

    cv = TimeSeriesSplit(n_splits=cfg.get("cv_splits", 3))
    method = cfg.get("search_method", "grid")

    log.info(f"Iniciando búsqueda {method} para {model_name}")
    log.info(f"Param grid: {cfg['param_grid']}")

    if method == "grid":
        search = GridSearchCV(
            base, cfg["param_grid"], cv=cv, scoring=cfg.get("scoring", "accuracy"),
            verbose=1, n_jobs=-1, return_train_score=True,
        )
    else:
        search = RandomizedSearchCV(
            base, cfg["param_grid"], n_iter=cfg.get("n_random_iter", 100), cv=cv,
            scoring=cfg.get("scoring", "accuracy"), verbose=1, n_jobs=-1,
            random_state=42, return_train_score=True,
        )

    t0 = time.time()
    search.fit(X, y)
    elapsed = time.time() - t0

    results = []
    for i in range(len(search.cv_results_["params"])):
        results.append({
            "params": search.cv_results_["params"][i],
            "mean_score": float(search.cv_results_["mean_test_score"][i]),
            "std_score": float(search.cv_results_["std_test_score"][i]),
            "mean_train_score": float(search.cv_results_["mean_train_score"][i]),
            "rank": int(search.cv_results_["rank_test_score"][i]),
        })
    results.sort(key=lambda r: r["rank"])

    log.info(f"Mejor score: {search.best_score_:.4f}")
    log.info(f"Mejores params: {search.best_params_}")
    log.info(f"Tiempo: {elapsed:.1f}s")

    return {
        "model": model_name,
        "search_method": method,
        "best_params": {k: _serialize(v) for k, v in search.best_params_.items()},
        "best_score": float(search.best_score_),
        "all_results": results[:50],
        "total_fits": len(results),
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": datetime.now().isoformat(),
        "dataset_size": len(X),
        "cv_splits": cfg.get("cv_splits", 3),
    }


# ---------------------------------------------------------------------------
# LSTM (loop manual)
# ---------------------------------------------------------------------------

def optimize_lstm(cfg: dict, dataset_path: str) -> dict:
    """Búsqueda manual de hiperparámetros para LSTM."""
    import torch
    import torch.nn as nn

    samples = _load_dataset(dataset_path)
    train_s, val_s, _test_s = _temporal_split(samples, 0.70, 0.15)

    X_all = np.array([extract_features(s["indicators"]) for s in samples], dtype=np.float32)
    y_all = np.array([1 if s["label"]["bias"] == "LONG" else 0 for s in samples], dtype=np.int32)

    n_train, n_val = len(train_s), len(val_s)
    mean = X_all[:n_train].mean(axis=0)
    std = X_all[:n_train].std(axis=0) + 1e-8
    X_all = (X_all - mean) / std

    input_size = X_all.shape[1]
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    log.info(f"LSTM device: {device}")

    grid = cfg["param_grid"]
    all_combos = list(itertools.product(*grid.values()))
    keys = list(grid.keys())
    n_iter = cfg.get("n_random_iter", len(all_combos))
    if n_iter < len(all_combos):
        random.seed(42)
        all_combos = random.sample(all_combos, n_iter)

    log.info(f"LSTM: probando {len(all_combos)} configuraciones")
    max_epochs = cfg.get("max_epochs", 50)
    patience = cfg.get("patience", 10)
    results = []
    best_score = -1
    best_params = None
    t0 = time.time()

    for combo in tqdm(all_combos, desc="LSTM configs"):
        params = dict(zip(keys, combo))
        seq_len = params.get("sequence_length", 10)
        hidden = params.get("hidden_size", 64)
        layers = params.get("num_layers", 2)
        lr = params.get("learning_rate", 0.001)
        drop = params.get("dropout", 0.2)
        bs = params.get("batch_size", 32)

        def build_seq(features, labels, sl):
            n, d = features.shape
            seqs = np.zeros((n, sl, d), dtype=np.float32)
            for i in range(n):
                start = max(0, i - sl + 1)
                s = features[start:i + 1]
                seqs[i, sl - len(s):] = s
            return torch.tensor(seqs, device=device), torch.tensor(labels, dtype=torch.long, device=device)

        X_tr, y_tr = build_seq(X_all[:n_train], y_all[:n_train], seq_len)
        X_va, y_va = build_seq(X_all[:n_train + n_val], y_all[:n_train + n_val], seq_len)
        X_va, y_va = X_va[n_train:], y_va[n_train:]

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden, layers, dropout=drop if layers > 1 else 0, batch_first=True)
                self.fc = nn.Linear(hidden, 2)
            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])

        model = Net().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        wait = 0
        best_state = None

        for epoch in range(max_epochs):
            model.train()
            perm = torch.randperm(len(X_tr), device=device)
            for i in range(0, len(perm), bs):
                idx = perm[i:i + bs]
                loss = criterion(model(X_tr[idx]), y_tr[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                vl = criterion(model(X_va), y_va).item()
            if vl < best_val_loss:
                best_val_loss = vl
                wait = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    break

        if best_state:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        model.eval()
        with torch.no_grad():
            preds = model(X_va).argmax(dim=1).cpu().numpy()
        acc = float((preds == y_va.cpu().numpy()).mean())

        results.append({"params": params, "mean_score": acc, "epochs_trained": epoch + 1})
        if acc > best_score:
            best_score = acc
            best_params = params
            log.info(f"  Nuevo mejor: {acc:.4f} con {params}")

    elapsed = time.time() - t0
    results.sort(key=lambda r: r["mean_score"], reverse=True)

    log.info(f"Mejor LSTM score: {best_score:.4f}")
    log.info(f"Mejores params: {best_params}")

    return {
        "model": "lstm",
        "search_method": "random",
        "best_params": best_params,
        "best_score": best_score,
        "all_results": results[:50],
        "total_fits": len(results),
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": datetime.now().isoformat(),
        "dataset_size": n_train + n_val,
        "cv_splits": 1,
    }


# ---------------------------------------------------------------------------
# QLoRA (placeholder — imprime configs para entrenamiento manual)
# ---------------------------------------------------------------------------

def optimize_qlora(cfg: dict) -> dict:
    """Imprime configuraciones recomendadas para entrenamiento manual con Unsloth."""
    recommended = cfg.get("recommended_configs", [])
    if not recommended:
        grid = cfg["param_grid"]
        all_combos = list(itertools.product(*grid.values()))
        keys = list(grid.keys())
        random.seed(42)
        recommended = [dict(zip(keys, c)) for c in random.sample(all_combos, min(5, len(all_combos)))]

    print("\n" + "=" * 70)
    print("QLoRA — Configuraciones recomendadas para Unsloth")
    print("=" * 70)
    for i, c in enumerate(recommended, 1):
        print(f"\nConfig {i}:")
        for k, v in c.items():
            print(f"  {k}: {v}")
    print("\n" + "=" * 70)
    print("Ejecutar cada config manualmente con Unsloth y registrar accuracy.")
    print("Guardar resultados en results/qlora_optimization.json con el mismo formato.")
    print("=" * 70 + "\n")

    return {
        "model": "qlora",
        "search_method": "manual",
        "recommended_configs": recommended,
        "best_params": None,
        "best_score": None,
        "all_results": [],
        "timestamp": datetime.now().isoformat(),
        "note": "Ejecutar cada config manualmente con Unsloth. Actualizar este archivo con resultados.",
    }


# ---------------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------------

def save_plot(result: dict, output_path: Path):
    """Genera gráfica resumen de la optimización."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_res = result.get("all_results", [])
    if not all_res or all_res[0].get("mean_score") is None:
        log.info("Sin resultados graficables, omitiendo gráfica.")
        return

    scores = [r["mean_score"] for r in all_res[:30]]
    labels = [f"#{i+1}" for i in range(len(scores))]

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(scores))]
    ax.barh(labels[::-1], scores[::-1], color=colors[::-1])
    ax.set_xlabel("Accuracy")
    ax.set_title(f"{result['model'].upper()} — Top {len(scores)} Configuraciones")
    ax.axvline(x=result["best_score"], color="red", linestyle="--", alpha=0.7, label=f"Mejor: {result['best_score']:.4f}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info(f"Gráfica guardada en {output_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialize(v):
    """Hace un valor serializable a JSON."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Optimización de hiperparámetros")
    parser.add_argument("--model", required=True, choices=["xgboost", "random_forest", "lstm", "qlora"])
    parser.add_argument("--config", required=True, help="Ruta al archivo YAML de configuración")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Ruta al dataset JSONL")
    parser.add_argument("--output-dir", default=str(RESULTS_DIR), help="Directorio de resultados")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Modelo: {args.model}")
    log.info(f"Config: {args.config}")
    log.info(f"Dataset: {args.dataset}")

    if args.model in ("xgboost", "random_forest"):
        samples = _load_dataset(args.dataset)
        train_s, val_s, _ = _temporal_split(samples, 0.70, 0.15)
        combined = train_s + val_s
        X, y = _samples_to_xy(combined)
        log.info(f"Cargadas {len(X)} muestras para CV (train+val)")
        result = optimize_sklearn(args.model, cfg, X, y)
    elif args.model == "lstm":
        result = optimize_lstm(cfg, args.dataset)
    elif args.model == "qlora":
        result = optimize_qlora(cfg)

    # Guardar resultados
    out_file = out_dir / f"{args.model}_optimization.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2, default=_serialize)
    log.info(f"Resultados guardados en {out_file}")

    # Guardar gráfica
    if result.get("all_results"):
        save_plot(result, out_dir / f"{args.model}_optimization.png")

    # Resumen
    print(f"\n{'='*50}")
    print(f"  {args.model.upper()} — Optimización Completa")
    print(f"{'='*50}")
    if result.get("best_score") is not None:
        print(f"  Mejor score: {result['best_score']:.4f}")
        print(f"  Mejores params: {json.dumps(result['best_params'], indent=4)}")
    if result.get("elapsed_seconds"):
        m, s = divmod(int(result["elapsed_seconds"]), 60)
        print(f"  Tiempo: {m}m {s}s")
    print(f"  Resultados: {out_file}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
