# Guía de implementación — Fix de leakage + instrumentación anti-overfitting (QLoRA)

> **Propósito:** guía precisa para que **otro agente implemente** los cambios. Cada
> tarea dice *qué*, *dónde*, *por qué* y *cómo verificar*. NO implementar al leer esto;
> es la especificación.
>
> **Contexto:** ver `docs/AUDITORIA_QLORA_LEAKAGE_OVERFITTING.md` (hallazgos) y
> `docs/FLUJO_DE_DATOS_QLORA.md` (flujo de datos). Decisión tomada: re-entrenar **un
> modelo en la nube** + **opcional uno local** (RTX 5070 Ti). NO se hace sweep de configs.

---

## Orden de ejecución sugerido
1. Tarea 1 (split) → 2. Tarea 2 (test del split) → 3. Tarea 3 (instrumentación) →
4. Tarea 4 (scripts/outputs) → 5. Tarea 5 (docs) → 6. Tarea 6 (re-ejecución operativa).

---

## Tarea 1 — Split temporal estricto (el fix de leakage)
**Archivo:** `langgraph/backtest/models/features.py` → `_temporal_split`

**Por qué:** `dataset.jsonl` está agrupado por símbolo (`pipeline.py::build_dataset`
itera `for sym in self.symbols`), y `_temporal_split` corta por posición → split por
símbolo, no por tiempo. Como TODOS los runners (LSTM/XGBoost/RF/ensembles/QLoRA y
`export.py`) usan esta única función, corregirla aquí propaga el fix a entrenamiento y
backtest a la vez.

**Cambio (corte global por timestamp + embargo por tiempo):**
```python
def _temporal_split(samples, train_frac=0.70, val_frac=0.15,
                    embargo_bars=24, base_tf_minutes=60, sort=True):
    """Holdout temporal real: ordena por timestamp y purga el solape de lookahead.

    El labeler mira `embargo_bars` velas al futuro; toda muestra de train cuyo
    horizonte de etiquetado cruce el inicio de val/test filtra info futura. Se purga
    por TIEMPO (no por conteo) porque, tras el orden global, cerca de cada frontera
    coexisten muestras de los 12 símbolos.
    """
    if sort:
        samples = sorted(samples, key=lambda s: s.get("timestamp", 0))
    n = len(samples)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    embargo_ms = embargo_bars * base_tf_minutes * 60 * 1000  # asume timestamps en ms

    val_start_ts  = samples[train_end]["timestamp"] if train_end < n else None
    test_start_ts = samples[val_end]["timestamp"]   if val_end   < n else None

    train = samples[:train_end]
    val   = samples[train_end:val_end]
    test  = samples[val_end:]

    if val_start_ts is not None:
        train = [s for s in train if s.get("timestamp", 0) + embargo_ms <= val_start_ts]
    if test_start_ts is not None:
        val = [s for s in val if s.get("timestamp", 0) + embargo_ms <= test_start_ts]
    return train, val, test
```

**Obligatorio verificar antes de fijar:**
- **Unidad del timestamp.** Binance klines = milisegundos (13 dígitos, ~1.7e12). Si
  fueran segundos, ajustar `embargo_ms`. Confirmar con una muestra real.
- `base_tf="1h"` → `base_tf_minutes=60`. Parametrizar si cambia el base TF.

**Limpieza relacionada:** en `backtest/export.py`, **eliminar** la función
`temporal_split` marcada DEPRECATED (ordenaba por timestamp pero sin embargo; ya no
aporta y confunde). `export_training_data` y `train_qlora.py::evaluate` heredan el fix
porque ambos llaman `_temporal_split` — **no tocar esos dos**.

**Alternativa (documentar, no usar como primaria):** split temporal *por símbolo* +
embargo → test balanceado entre activos pero retiene solape de régimen cross-asset. El
corte global es más defendible para "el agente nunca ve datos futuros".

---

## Tarea 2 — Test de regresión del split
**Archivo:** `langgraph/tests/backtest/test_backtest.py`

**Por qué:** evitar volver al split posicional sin darse cuenta.

**Qué asertar** (dataset sintético con timestamps entrelazados por símbolo):
- `max(ts in train) < min(ts in test)` (orden temporal real).
- Ninguna muestra de train cumple `timestamp + embargo_ms > test_start_ts` (embargo OK).
- Las 3 particiones son disjuntas y suman ≤ total (por el embargo se descartan algunas).

---

## Tarea 3 — Instrumentación anti-overfitting
**Archivo:** `langgraph/optimization/qlora/train_qlora.py`

**3.1 Early stopping real.** Hoy hay `load_best_model_at_end=True` pero NINGÚN
`EarlyStoppingCallback` (el docstring miente). Añadir en ambas ramas (SFTConfig y legacy):
```python
from transformers import EarlyStoppingCallback
self.trainer = SFTTrainer(..., callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
```
*Por qué:* con `eval_strategy="steps"` ya activo, corta cuando `eval_loss` deja de
mejorar → evita epochs de sobreajuste y ahorra GPU/$.

**3.2 Accuracy train/val/test + gap.** Refactorizar el loop de generación de
`evaluate()` a un helper `_predict_split(samples, n_max=None)` que devuelva
`predictions/actuals/sample_keys`. Llamarlo sobre subconjunto de **train** (~500) y
**val** (~500) además del **test** completo. Reportar `train_acc, val_acc, test_acc` y
`gap = train_acc - test_acc` en el JSON de resultado.
*Por qué:* `eval_loss` no mide accuracy de la tarea; el **gap train−test** es la señal
directa de memorización vs generalización. Es lo que permite *cerrar* la pregunta
"¿overfitted?" (hoy no se puede).

**3.3 Persistir curva de loss.** Volcar `self.trainer.state.log_history` a
`results/<tag>_loss_curve.json` + PNG (train_loss vs eval_loss por step) reutilizando
`optimization/io/plots.py`.
*Por qué:* `train_loss` reportado es el promedio de la corrida (no comparable directo a
`eval_loss`); la curva por step muestra el punto de divergencia → evidencia visual de
overfitting para la tesis.

**3.4 Baseline heurístico.** Sobre el MISMO test:
```python
def _heuristic_bias(indicators):
    base = indicators.get("1h", next(iter(indicators.values())))
    hm = base.get("heatmap", "NEUTRAL")
    if "BULLISH" in hm: return "LONG"
    if "BEARISH" in hm: return "SHORT"
    return "LONG" if base.get("macd_hist", 0) >= 0 else "SHORT"
```
Guardar su accuracy como `baseline_metrics`.
*Por qué:* el baseline de clase mayoritaria es solo 51.1%; el contraste útil es contra
un heurístico de indicadores. Si FT ≫ heurístico → aporta poder real; si ≈ → la tarea
es fácil y el número dice poco. Contextualiza el resultado honestamente.

**3.5 (sin código)** McNemar/t-test ya soportado (`stats_tests.py` alinea por
`sample_keys`, que el script ya emite). Solo recordar correr zero-shot sobre el test
corregido (Tarea 6).

---

## Tarea 4 — Scripts de modelo único + limpieza de outputs
**Por qué:** la corrida de 3 configs fue mala; la tesis solo necesita 1 nube + quizá 1
local. Simplificar y evitar que se cite el 92% viejo.

**4.1 Retirar el sweep.**
- Eliminar `optimization/qlora/run_3configs.sh` y `optimization/qlora/run_qlora_search.sh`.
- Crear `optimization/qlora/run_cloud.sh` (SageMaker): mejor config
  `lr=2e-5, rank=16, alpha=32, epochs=2-3, batch=2, grad_accum=8`, **sin `--max-eval`**
  (test completo).
- Crear `optimization/qlora/run_local.sh` (RTX 5070 Ti, opcional):
  `batch=1, grad_accum=16, epochs=1, max_seq_length=1024`.
- *Por qué quitar `--max-eval 2000`:* fue lo que limitó el backtest a 1 solo símbolo
  (LINK). Sin el cap, el test es completo y representativo.

**4.2 Archivar resultados inválidos.**
- Mover `optimization/qlora/results/qlora_optimization.json` y `qlora_config1.json` a
  `optimization/qlora/results/archive/` con un `README.md`: "Entrenados con split
  contaminado + eval parcial (2000 LINK) + métricas financieras 0 por falta de dvc pull
  — NO usar para la tesis."
- Revisar `backtest/data/models/qlora_config1.dvc` (si el modelo viejo estaba trackeado).

**4.3 Nombrado de salida.** Que `train_qlora.py` escriba a `results/qlora_<tag>.json`
(por flag/`--output-dir`) en vez de sobrescribir siempre `qlora_optimization.json`;
mantener una copia canónica para `compare-stats`.

---

## Tarea 5 — Sincronizar documentación
- `.claude/steering-langgraph.md` (~líneas 302-304): "same temporal split (… **no sort**)"
  → "global temporal split **con orden por timestamp + embargo**".
- `.claude/rules/ml-conventions.md`: aclarar que "temporal split" exige orden global por
  timestamp + embargo, y que **orden de archivo ≠ orden temporal**.
- `.claude/steering-qlora.md`, `.claude/rules/qlora-training.md`, `docs/QLORA_FINETUNING.md`:
  reflejar 1 nube + 1 local (no sweep), nuevas métricas (gap, baseline, curva), y marcar
  el 92% viejo como inválido.

---

## Tarea 6 — Re-ejecución operativa (post-cambios)
1. `dvc pull` de `backtest/data/labeled/dataset.jsonl` **y** `backtest/data/candles/`
   (sin las velas, el backtest da win_rate/PF/Sharpe = 0).
2. Re-exportar training data (hereda el split nuevo) — lo hace `train_qlora.py` solo.
3. Re-entrenar barato en CPU: LSTM / XGBoost / RF (minutos–~1h) → resultados nuevos.
4. Re-entrenar **1 QLoRA en SageMaker** (`run_cloud.sh`); opcional **1 local**
   (`run_local.sh`).
5. Re-correr **zero-shot** sobre el test nuevo (`LLMBacktestRunner`, provider Groq/Ollama).
6. `compare-stats` (McNemar + t-test) alineado por `sample_keys`.

---

## Verificación global
- Test del split (Tarea 2) en verde.
- Corrida corta `train_qlora.py --epochs 1 --max-eval <pequeño>` que loguee
  `train/val/test acc + gap`, escriba `*_loss_curve.json` y `baseline_metrics`.
- Confirmar que el test ya **no** es de un solo símbolo: contar símbolos únicos en
  `sample_keys` del resultado nuevo (deben aparecer varios activos).
- `compare-stats` corre sin error con el resultado nuevo + un resultado ML.

---

## Resumen de archivos a tocar
| Archivo | Tarea |
|---------|-------|
| `langgraph/backtest/models/features.py` | 1 — `_temporal_split` (núcleo del fix) |
| `langgraph/backtest/export.py` | 1 — borrar `temporal_split` DEPRECATED |
| `langgraph/tests/backtest/test_backtest.py` | 2 — test del split |
| `langgraph/optimization/qlora/train_qlora.py` | 3 — early stop, accuracies+gap, curva, baseline |
| `langgraph/optimization/qlora/run_3configs.sh` / `run_qlora_search.sh` | 4 — eliminar |
| `langgraph/optimization/qlora/run_cloud.sh` / `run_local.sh` | 4 — crear |
| `langgraph/optimization/qlora/results/` (+ `archive/`) | 4 — archivar resultados viejos |
| `.claude/steering-langgraph.md`, `.claude/rules/ml-conventions.md`, etc. | 5 — docs |
