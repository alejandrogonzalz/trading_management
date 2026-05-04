# Optimización de Hiperparámetros — Predicción de Señales de Trading

Infraestructura para optimizar los hiperparámetros de los 4 modelos de predicción de señales de trading crypto.

## Modelos

| Modelo | Método | Tiempo estimado | Infraestructura |
|--------|--------|-----------------|-----------------|
| XGBoost | Grid/Random Search | 30-90 min | CPU local |
| Random Forest | Grid/Random Search | 20-60 min | CPU local |
| LSTM | Loop manual + early stopping | 2-6 horas | GPU recomendado |
| QLoRA (Qwen 2.5 7B) | Manual (Unsloth) | 4-12 horas | GPU (16GB+ VRAM) |

**Dataset**: ~56K muestras, split temporal 70/15/15.

## Prerrequisitos

```bash
pip install tqdm pyyaml matplotlib scikit-learn numpy
# Para XGBoost:
pip install xgboost
# Para LSTM:
pip install torch
```

## Cómo ejecutar

### Optimización individual

```bash
cd langgraph/optimization

# XGBoost — grid search completo (~30-90 min)
python optimize.py --model xgboost --config configs/xgboost.yaml

# Random Forest — random search 100 iters (~20-60 min)
python optimize.py --model random_forest --config configs/random_forest.yaml

# LSTM — loop manual 30 configs (~2-6 horas)
python optimize.py --model lstm --config configs/lstm.yaml

# QLoRA — imprime configs para entrenamiento manual con Unsloth
python optimize.py --model qlora --config configs/qlora.yaml
```

### Dataset personalizado

```bash
python optimize.py --model xgboost --config configs/xgboost.yaml --dataset /ruta/a/mi/dataset.jsonl
```

Por defecto usa `../backtest/data/labeled/dataset.jsonl`.

### Ejecución en segundo plano (sesiones largas)

```bash
# Con nohup — sigue corriendo al cerrar la terminal
nohup python optimize.py --model xgboost --config configs/xgboost.yaml > logs/xgboost.log 2>&1 &

nohup python optimize.py --model lstm --config configs/lstm.yaml > logs/lstm.log 2>&1 &

# Monitorear progreso
tail -f logs/xgboost.log
```

### Análisis de resultados

```bash
# Después de que terminen las optimizaciones
python analyze_results.py
```

Genera:
- Tabla comparativa en terminal
- `results/comparison.png` — gráfica de barras + heatmaps

## Salida esperada

Cada optimización genera en `results/`:

- `{modelo}_optimization.json` — Resultados completos (mejores params, todos los scores, metadata)
- `{modelo}_optimization.png` — Gráfica de barras con top 30 configuraciones

Formato del JSON:

```json
{
  "model": "xgboost",
  "best_params": {"n_estimators": 300, "max_depth": 8, "...": "..."},
  "best_score": 0.623,
  "all_results": [{"params": {}, "mean_score": 0.61, "rank": 2}],
  "elapsed_seconds": 1234,
  "dataset_size": 47661
}
```

## Flujo de trabajo completo

```bash
# 1. Optimizar cada modelo (pueden correr en paralelo con nohup)
nohup python optimize.py --model xgboost --config configs/xgboost.yaml > logs/xgboost.log 2>&1 &
nohup python optimize.py --model random_forest --config configs/random_forest.yaml > logs/rf.log 2>&1 &
nohup python optimize.py --model lstm --config configs/lstm.yaml > logs/lstm.log 2>&1 &
python optimize.py --model qlora --config configs/qlora.yaml

# 2. Esperar a que terminen (revisar logs)
tail -f logs/xgboost.log

# 3. Comparar resultados
python analyze_results.py

# 4. Usar los mejores params en backtest/ml_models.py
cat results/xgboost_optimization.json | python -c "import sys,json; print(json.load(sys.stdin)['best_params'])"
```

## Tiempos estimados por modelo

| Modelo | Grid completo | Random 100 iters | Random 30 iters |
|--------|--------------|-------------------|-----------------|
| XGBoost | ~90 min (576 combos) | ~15-20 min | ~5-10 min |
| Random Forest | ~60 min (720 combos) | ~10-15 min | ~5 min |
| LSTM (CPU) | N/A | N/A | ~3-6 horas |
| LSTM (GPU/MPS) | N/A | N/A | ~30-90 min |

## Configuración

Los archivos YAML en `configs/` definen los grids de búsqueda. Editar para ajustar rangos:

- `configs/xgboost.yaml` — n_estimators, max_depth, learning_rate, subsample, colsample_bytree
- `configs/random_forest.yaml` — n_estimators, max_depth, min_samples_split/leaf, max_features
- `configs/lstm.yaml` — hidden_size, num_layers, sequence_length, learning_rate, dropout, batch_size
- `configs/qlora.yaml` — learning_rate, lora_rank, lora_alpha, epochs, batch_size

## Estructura

```
optimization/
├── README.md              ← Este archivo
├── __init__.py
├── optimize.py            ← Script principal de optimización
├── analyze_results.py     ← Comparación entre modelos
├── configs/
│   ├── xgboost.yaml
│   ├── random_forest.yaml
│   ├── lstm.yaml
│   └── qlora.yaml
├── results/               ← JSON + PNG generados
│   └── .gitkeep
└── logs/                  ← Logs de nohup
    └── .gitkeep
```
