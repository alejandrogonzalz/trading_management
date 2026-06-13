# Flujo de datos: `dataset.jsonl`, `candles/` y el backtest del fine-tuning

> Documento explicativo para entender **qué datos hay, para qué sirve cada uno, y
> cómo los usa el script de fine-tuning**. Es la base conceptual antes del plan de
> cambios de código.

---

## 1. Los dos artefactos de datos

### `backtest/data/candles/` — datos crudos de mercado
- Un archivo JSON por **símbolo × timeframe**: `BTCUSDT_1h.json`, `BTCUSDT_4h.json`, …
- Cada vela (candle) es OHLCV crudo: `{timestamp, open, high, low, close, volume}`.
- Es lo que se descarga de Binance. **No tiene indicadores ni etiquetas.**
- Tamaño: ~21 MB. Trackeado por DVC (gitignored) → requiere `dvc pull`.

### `backtest/data/labeled/dataset.jsonl` — dataset procesado y etiquetado
- Una línea = **una muestra** lista para entrenar/evaluar:
  ```json
  {
    "symbol": "BTCUSDT",
    "timestamp": 1700000000000,
    "indicators": {"1h": {...}, "4h": {...}, "1d": {...}},   // calculados de las velas
    "label": {"bias": "LONG", "entry": ..., "tp": ..., "sl": ..., "quality": ...}
  }
  ```
- Los `indicators` se calculan a partir de las velas (RSI, ADX, MACD, heatmap…).
- El `label` se genera por **hindsight**: se miran las 24 velas FUTURAS para decidir
  si la dirección correcta era LONG o SHORT.
- Tamaño: ~51 MB, 56 161 muestras. Trackeado por DVC (gitignored) → requiere `dvc pull`.

### Relación entre ambos
```
candles/ (OHLCV crudo)
   │  pipeline.py: calcular indicadores + etiquetar (hindsight)
   ▼
dataset.jsonl (indicadores + label por muestra)
```
`dataset.jsonl` se **deriva** de `candles/`. Las velas son la fuente; el dataset es
el producto procesado.

---

## 2. Para qué se usa cada uno (y por qué se necesitan AMBOS en el backtest)

| Paso | Usa `dataset.jsonl` | Usa `candles/` |
|------|:---:|:---:|
| **Entrenar** (fine-tuning) | ✅ (input + label, vía `training_data/*.jsonl`) | ❌ |
| **Backtest / evaluación** | ✅ (input que ve el modelo + label real) | ✅ (simular trades) |

En el backtest pasan dos cosas distintas:
1. **Direction accuracy** → compara la predicción del modelo (`bias`) contra el
   `label` del `dataset.jsonl`. **Solo necesita el dataset.**
2. **Métricas financieras** (win rate, profit factor, Sharpe) → el modelo predice
   `entry/tp/sl`, y `simulate_trade()` **camina las velas siguientes** para ver si
   pegó primero el TP o el SL. **Esto necesita `candles/`.**

> 🔑 Por eso en la corrida mala salió `win_rate = 0`: faltó `dvc pull` de las velas,
> así que la simulación de trades no tuvo datos. La direction accuracy (92%) sí salió
> porque esa solo depende del dataset. **No era un bug, era falta de `dvc pull`.**

---

## 3. Cómo usa estos datos `train_qlora.py`

El método `run()` ejecuta la cadena completa, y **el último paso ES un backtest**:

```
run():
  1. prepare_data()  → export.py parte dataset.jsonl en train/val/test
                       y escribe training_data/{train,val,test}.jsonl (formato chat)
  2. load_model()    → carga Qwen 2.5 7B 4-bit + adapters LoRA
  3. train()         → fine-tunea sobre training_data/train.jsonl
                       (vigila eval_loss sobre val.jsonl)
  4. save_model()    → guarda los adapters LoRA
  5. evaluate()      → ★ EL BACKTEST ★
                       - vuelve a partir dataset.jsonl → toma el split TEST
                       - por cada muestra: arma prompt → el modelo predice
                       - carga candles/ (_load_candles_map) → simula el trade
                       - calcula accuracy + win_rate + profit_factor + Sharpe
                       - guarda predictions/actuals/sample_keys (para McNemar/t-test)
  6. save_gguf()     → exporta GGUF para Ollama (opcional)
```

Conclusiones:
- **No hay un "script de backtest" separado para el modelo fine-tuneado**: el backtest
  vive dentro de `train_qlora.py::evaluate()` y corre automáticamente al terminar de
  entrenar. (Es la misma lógica que `LLMBacktestRunner`, que sí es un runner aparte
  usado para zero-shot y para los modelos ML.)
- Alternativa para backtestear "en frío" (sin re-entrenar): desplegar el GGUF en
  Ollama, apuntar `LLM_MODEL` a ese modelo y correr `LLMBacktestRunner`. Da el mismo
  tipo de resultado.

---

## 4. Qué cambios se necesitan (resumen — el detalle va en el plan)

1. **El corte del split debe ser por TIEMPO, no por posición** (hoy `dataset.jsonl`
   está agrupado por símbolo y `_temporal_split` corta por posición → leakage). Fix
   en `backtest/models/features.py::_temporal_split` (orden global por timestamp +
   embargo). Se propaga a entrenamiento y backtest porque ambos llaman esa función.
2. **`dvc pull` de `candles/` (y `dataset.jsonl`) antes del backtest** para que las
   métricas financieras dejen de salir en 0.
3. **Quitar `--max-eval 2000`** del script de corrida (limitaba el backtest a 1 solo
   símbolo). Evaluar el test completo o estratificado por símbolo.
4. **Instrumentación anti-overfitting** en `train_qlora.py`: early stopping real,
   accuracy train/val/test + gap, curva de loss, baseline heurístico.
5. **Re-entrenar** un modelo en la nube (y quizá uno local) con el split corregido,
   porque el modelo actual aprendió con datos contaminados.

> **No se necesitan datos nuevos.** Son las mismas 56 161 muestras y las mismas velas;
> solo cambia *dónde se corta* el dataset (por tiempo en vez de por símbolo) y que se
> haga `dvc pull` de las velas para el backtest.

---

## 5. Glosario rápido

| Término | Qué es aquí |
|---------|-------------|
| **candle / vela** | Una barra OHLCV de precio (open/high/low/close/volume) en un timeframe |
| **dataset.jsonl** | Muestras etiquetadas (indicadores + bias/entry/tp/sl) derivadas de las velas |
| **label (hindsight)** | La dirección correcta, calculada mirando 24 velas al futuro |
| **split temporal** | Partir el dataset en train/val/test por **fecha**, sin solape |
| **embargo** | Franja descartada entre splits para que el lookahead de 24 velas no "espíe" |
| **backtest** | Correr el modelo sobre el split **test** y simular trades contra velas futuras |
| **direction accuracy** | % de veces que el `bias` predicho coincide con el `label` |
| **DVC** | Versiona los datos grandes en S3; `dvc pull` los baja al disco |
