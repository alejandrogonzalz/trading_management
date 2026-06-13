# Auditoría del pipeline QLoRA — Data Leakage & Overfitting

> **Branch auditada:** `feat/qlora-fine-tunning`
> **Fecha:** 2026-06-13
> **Alcance:** flujo de datos fine-tuning ↔ backtest, y validación de generalización
> del modelo Qwen 2.5 7B fine-tuneado (tesis: LLM zero-shot vs fine-tuned vs LSTM/XGBoost/RF).

---

## Resumen ejecutivo

| # | Hallazgo | Severidad |
|---|----------|-----------|
| 1 | El "split temporal" **no es temporal**: corta por posición de archivo sobre un dataset **agrupado por símbolo**. | 🔴 Alta |
| 2 | El test set evaluado fue **100% LINKUSDT** (1 solo activo), en una ventana de calendario que **solapa** con el periodo de entrenamiento de los activos correlacionados → **leakage de régimen de mercado**. | 🔴 Alta |
| 3 | El 92% de accuracy se midió sobre **2000 muestras** (no las 8 425 del test) y con **métricas financieras = 0** (sin velas, por no hacer `dvc pull`). No es representativo. | 🟠 Media |
| 4 | El 92% **no es demostrablemente overfitting ni memorización**: la generalización está **sin verificar** porque el test está contaminado por el leakage (#1, #2). | 🟠 Media |
| 5 | No hay **early stopping real**, ni accuracy de train/val, ni gap train/test, ni curvas de loss persistidas, ni baseline heurístico → la pregunta "¿overfitted?" no se puede cerrar con los datos actuales. | 🟠 Media |

**Conclusión rápida:** No hay solapamiento *literal* de muestras entre el conjunto
de fine-tuning y el de evaluación (son slices disjuntos), y **no hay evidencia de
overfitting** (la `eval_loss` no diverge de la `train_loss`). El problema vinculante
es otro: el particionamiento **no es temporal**, así que no garantiza que "el agente
nunca vea datos futuros" y el 92% **no es creíble como medida de generalización**.
Overfitting y leakage son problemas distintos — aquí lo que falla es el **leakage**.

---

## 1. Data Leakage

### 1.1 Cómo se construye el dataset (la causa raíz)

`langgraph/backtest/pipeline.py` → `DataPipeline.build_dataset()`:

```python
for sym in self.symbols:          # DEFAULT_SYMBOLS en orden fijo
    ...
    all_labeled.extend(labeled)   # se concatena símbolo tras símbolo
save_labeled_dataset(all_labeled, self.output_path)
```

Con `DEFAULT_SYMBOLS = [BTC, ETH, BNB, SOL, XRP, ADA, AVAX, DOT, DOGE, LINK, MATIC, NEAR]`,
el archivo `dataset.jsonl` queda **agrupado por símbolo** (primero todas las muestras
de BTC en orden de tiempo, luego todas las de ETH, etc.). **No está ordenado
globalmente por timestamp.**

### 1.2 Cómo se parte (el error)

`langgraph/backtest/models/features.py` → `_temporal_split()`:

```python
def _temporal_split(samples, train_frac=0.70, val_frac=0.15):
    n = len(samples)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return samples[:train_end], samples[train_end:val_end], samples[val_end:]
```

Parte por **posición en el archivo**, sin ordenar por tiempo. Como el archivo está
agrupado por símbolo, el resultado es un split **por símbolo**, no por tiempo:

- `train` (70%) ≈ los primeros ~8 símbolos.
- `test` (último 15%) ≈ los últimos símbolos del archivo.

Esta misma función la usan **todos** los runners (LSTM, XGBoost, RF, ensembles,
QLoRA y `export.py`), por lo que el sesgo es idéntico en toda la comparación de la
tesis (consistente, pero consistentemente mal).

> Nota: en `export.py` existe `temporal_split()` que **sí** ordena por timestamp,
> pero está marcada `DEPRECATED` y reemplazada por la versión sin ordenar para
> "ser consistente con los runners". Es decir, la versión más correcta fue
> descartada.

### 1.3 Evidencia empírica (resultado real)

De `optimization/qlora/results/qlora_optimization.json`, los `sample_keys`
evaluados:

```
Total evaluado: 2000
Distribución por símbolo:  LINKUSDT: 2000   (100%)
Rango temporal:  2025-06-09  →  2026-01-01
```

El "test set" evaluado es **un único activo (LINK)**. Esto ocurre porque:
1. El test es el último 15% por posición → cae en los símbolos finales del archivo.
2. `run_3configs.sh` pasa `--max-eval 2000`, así que solo se evaluaron las
   primeras 2000 muestras del slice de test (todas LINK).

### 1.4 Por qué esto es leakage

La ventana de LINK (jun-2025 → ene-2026) está **dentro** del rango de calendario
con el que se entrenó el modelo para los demás activos (BTC, ETH, SOL… altamente
correlacionados con LINK). El modelo vio el régimen de mercado de ese periodo a
través de los activos de entrenamiento y se evalúa sobre LINK en **el mismo
periodo**. En cripto, donde la correlación entre activos es muy alta, esto es
**leakage de régimen de mercado**: el holdout no está aislado en el tiempo, solo
cambia el ticker.

La afirmación de la tesis "el agente nunca ve datos futuros" **no se cumple** con
este particionamiento.

### 1.5 Refactor propuesto (split temporal estricto + embargo)

Un único cambio en la función compartida se propaga a todos los modelos:

```python
# langgraph/backtest/models/features.py
def _temporal_split(samples, train_frac=0.70, val_frac=0.15, embargo=24, sort=True):
    """Split temporal global con embargo.

    Ordena TODAS las muestras por timestamp (entre símbolos) → holdout real en
    el tiempo: test = ventana más reciente, train = la más antigua. Descarta
    `embargo` muestras en cada frontera para eliminar el solape de la ventana
    de lookahead (el labeler mira 24 velas adelante).
    """
    if sort:
        samples = sorted(samples, key=lambda s: s.get("timestamp", 0))
    n = len(samples)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train = samples[: max(0, train_end - embargo)]
    val   = samples[train_end : max(train_end, val_end - embargo)]
    test  = samples[val_end:]
    return train, val, test
```

Cambios complementarios:
- **`run_3configs.sh`**: quitar `--max-eval 2000` (evaluar las 8 425 del test) o
  usar un subconjunto **estratificado por símbolo**.
- **Embargo ideal por *tiempo***: descartar muestras cuya ventana de lookahead
  (`timestamp + 24 × tamaño_de_vela`) cruce el corte. La versión por conteo de
  arriba es la aproximación mínima.
- Alternativa más barata si no se quiere re-entrenar todo: **split temporal por
  símbolo + embargo** (test balanceado entre activos, pero persiste algo de
  solape de régimen entre activos correlacionados).

> ⚠️ **Consecuencia:** corregir el split **invalida los resultados ya
> entrenados** (LSTM/XGBoost/RF/QLoRA). Para que la comparación de la tesis siga
> siendo válida, hay que re-entrenar/re-evaluar todos los modelos con el nuevo
> split.

---

## 2. Overfitting — ¿está sobreajustado el modelo?

**Veredicto: NO hay evidencia de que esté overfitted.** El problema real no es
overfitting, es el leakage de la sección 1. Son conceptos distintos:
- **Overfitting** = el modelo memoriza el train y falla en datos *nuevos*.
- **Leakage** = el "test" no es independiente del train, así que una buena nota de
  test **no es confiable** (aunque el modelo no esté sobreajustado).

Aquí: el modelo *no* muestra overfitting, pero su test está contaminado → el 92% no
prueba generalización.

### 2.1 Qué se mide hoy

`optimization/qlora/train_qlora.py`:
- Trackea `train_loss` y `eval_loss` (loss de validación). Resultado real:
  `train_loss = 0.795`, `eval_loss = 0.787`. La eval **no diverge** de la train →
  sin señal de overfitting. **Matiz:** `train_loss` es el *promedio de toda la
  corrida* (incluye los primeros pasos con loss alta) y `eval_loss` es la del mejor
  checkpoint → no son directamente comparables. Para concluir con rigor hace falta
  la **curva** train-vs-eval por step, que hoy no se persiste.
- `load_best_model_at_end=True` + `metric_for_best_model="eval_loss"`, pero
  **no hay `EarlyStoppingCallback`** → entrena los epochs completos y solo
  restaura el mejor checkpoint. El docstring dice "early stopping" pero no lo es.
- Solo calcula **direction accuracy en test** al final. No hay accuracy de train ni
  de val, ni **gap train−test** (la señal directa de memorización) → por eso la
  pregunta "¿overfitted?" **no se puede cerrar** con los datos actuales.

### 2.2 Sobre el 92% (corrección a una versión previa de este informe)

Una versión anterior afirmaba que el `bias` era "casi una función determinista de
los indicadores del prompt" y que el modelo "memorizaba la regla de etiquetado".
**Eso es incorrecto.** Verificado en `backtest/ingestion/labeler.py::label_candle`:
el `bias` se deriva por **hindsight** — compara `max_up` vs `max_down` sobre las **24
velas FUTURAS**. El modelo NO ve en el prompt la señal que genera el label; predice
dirección futura genuina.

Datos verificados sobre la corrida real:
- **Balance de clases del test:** 1022 LONG / 978 SHORT → un clasificador de clase
  mayoritaria saca solo **51.1%**. El 92% NO es un artefacto de desbalance; es
  discriminación real muy por encima del azar.
- **`errors: 0` parse errors** → el modelo aprendió a generar el JSON estructurado
  correctamente (el *formato/razonamiento estructurado* sí generaliza).

Entonces, ¿por qué el 92% no es creíble? Porque se midió **bajo leakage de régimen**
(sección 1.4) y sobre **un único activo (LINK)**. Es un número *optimista* cuya
generalización a mercado futuro y no visto **está sin verificar** — no es "falso" ni
"memorización", simplemente no es confiable hasta corregir el split.

Cómo *sí* contextualizarlo: dado que el baseline de clase mayoritaria es 51.1%, el
contraste relevante es contra un **baseline heurístico** de indicadores
(p. ej. `bias = LONG si heatmap ∈ {BULLISH, STRONG_BULLISH} else SHORT`). Si el
fine-tuned ≫ heurístico → aporta poder predictivo real; si ≈ heurístico → la tarea
(tras los filtros de calidad del labeler) es fácil y el 92% dice poco.

### 2.3 Instrumentación recomendada (a nivel de código)

1. **Early stopping real**:
   ```python
   from transformers import EarlyStoppingCallback
   SFTTrainer(..., callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
   ```

2. **Accuracy en train / val / test** con el mismo pipeline `generate → _parse_prediction`,
   y reportar `gap = train_acc - test_acc`. Un gap grande = memorización.

3. **Persistir curvas de loss**: volcar `self.trainer.state.log_history` a
   `results/<tag>_loss_curve.json` y graficar train-vs-eval loss por step
   (reusar `optimization/io/plots.py`).

4. **Baseline heurístico** sobre el MISMO test:
   `bias = LONG si heatmap ∈ {BULLISH, STRONG_BULLISH} else SHORT`.
   Comparar contra el fine-tuned: si la brecha es pequeña, el 92% aporta poco poder
   predictivo; si es grande, el FT sí aprendió algo no trivial. (El baseline de clase
   mayoritaria ya se sabe que es 51.1%, por eso el contraste útil es el heurístico.)

5. **Significancia estadística**: `stats_tests.py` / `compare-stats` ya alinea por
   `sample_keys` (el script QLoRA ya emite `predictions`/`actuals`/`sample_keys`).
   Falta correr el **zero-shot sobre el MISMO test corregido** para el McNemar +
   t-test pareado.

6. **Arreglar la simulación de trades** (problema **operativo**, no bug de código):
   `win_rate`/`profit_factor`/`Sharpe` salieron en 0 porque faltaban las velas en
   SageMaker — no se hizo `dvc pull` de `backtest/data/candles/` (están gitignored /
   trackeadas por DVC). Hacer `dvc pull` antes de evaluar y la simulación dará métricas
   reales.

7. **Validación cruzada correcta para series temporales**: no usar K-fold normal;
   usar **walk-forward / expanding-window con embargo** (purged CV estilo López de
   Prado). Para sklearn ya existe `TimeSeriesSplit`; para el LLM, al menos
   reportar val y test con embargo.

---

## 3. Checklist de remediación

- [ ] Reescribir `_temporal_split` con orden global por timestamp + embargo.
- [ ] Quitar `--max-eval 2000` de `run_3configs.sh` (o estratificar por símbolo).
- [ ] Re-entrenar/re-evaluar todos los modelos con el split corregido.
- [ ] Añadir `EarlyStoppingCallback` + accuracy train/val/test + gap.
- [ ] Persistir y graficar curvas de loss.
- [ ] Añadir baseline heurístico de indicadores como punto de comparación.
- [ ] `dvc pull` de velas para que la simulación de trades dé métricas reales.
- [ ] Correr zero-shot sobre el test corregido y ejecutar McNemar + t-test.
- [ ] Documentar el protocolo de split (real, no posicional) en `ml-conventions.md`.

---

## Apéndice — archivos clave

| Archivo | Rol en el problema |
|---------|--------------------|
| `langgraph/backtest/pipeline.py` | Construye el dataset agrupado por símbolo |
| `langgraph/backtest/models/features.py` | `_temporal_split` posicional (núcleo del fix) |
| `langgraph/backtest/export.py` | Exporta splits de fine-tuning con el mismo split |
| `langgraph/optimization/qlora/train_qlora.py` | Entrena + evalúa; falta instrumentación |
| `langgraph/optimization/qlora/run_3configs.sh` | `--max-eval 2000` → test de 1 símbolo |
| `optimization/qlora/results/qlora_optimization.json` | Evidencia: 92% / 2000 / 100% LINK |
