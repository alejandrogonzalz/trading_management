# Plan Maestro — Avances Finales de Tesis (Avance 6 + 7)

## Contexto
Proyecto de tesis: *Sistema Híbrido de Trading — Comparación LLM vs ML*  
La propuesta Fase 0 comprometió una comparación de 4 enfoques: zero-shot LLM, QLoRA fine-tuned, XGBoost, LSTM.  
Los Avances 1–5 completaron toda la parte ML (Bagging-LSTM = 83.37% test acc, ganador).  
Lo que falta: correr los LLMs, obtener métricas de trading en el ML, hacer la comparación estadística y documentar.  
GPU disponible: NVIDIA RTX 5070 Ti (16 GB VRAM) — el fine-tuning y la inferencia serán locales, no en la nube.

---

## Estado actual — qué existe y qué falta

### ✅ Completado
| Artefacto | Archivo |
|---|---|
| Dataset 56,161 muestras | `backtest/data/labeled/dataset.jsonl` |
| Feature engineering (21 features, 3 TFs) | `backtest/models/features.py` |
| Notebooks Avance 1–5 ejecutados | `langgraph/Avance[1-5].ipynb` |
| Optimización ML individual (LSTM, XGB, RF, SVM, KNN, MLP, LR) | `optimization/results/*_optimization.json` |
| Ensembles A5 (Bagging-LSTM, Stacking, Blending, Voting, AdaBoost) | `optimization/results/bagging_lstm_optimization.json` + otros |
| Métricas clasificación de Bagging-LSTM (acc=83.37%, F1, AUC, per-sample arrays) | `optimization/results/bagging_lstm_optimization.json` |
| Exportación datos fine-tuning | `backtest/export.py` → `training_data/{train,val,test}.jsonl` |
| Script QLoRA trainer | `optimization/qlora/train_qlora.py` |
| Framework estadístico (McNemar + t-test pareado) | `optimization/stats_tests.py` |
| Guía cloud GPU (EC2 + costos) | `optimization/qlora/EC2_GUIDE.md` |
| Infraestructura LLM (7 providers) | `agent/llm_factory.py` |

### ❌ Falta
| Qué | Por qué es crítico |
|---|---|
| Métricas de **trading** para Bagging-LSTM (win rate, profit factor, Sharpe, drawdown) | La propuesta §3.7 las prometió; el Avance 5 solo tiene acc/F1/AUC |
| Backtest **zero-shot LLM** (Qwen 2.5 7B base + Llama 3.3 70B) | Línea base sin fine-tuning |
| Backtest **QLoRA fine-tuned** Qwen 2.5 7B | El corazón de la tesis |
| **Pruebas estadísticas** (McNemar + t-test) entre los 4 modelos | Validación científica requerida |
| **Notebook Avance 6** con tabla comparativa final de los 4 enfoques | Entregable académico |
| **Documento PDF Avance 6** (análisis de modelo, accionables, cloud) | Entregable PDF requerido por la profesora |
| **Avance 7** resumen ejecutivo | Entregable final |

---

## Plan de ejecución — ordenado por dependencia

### PASO 1 — Extender MLBacktestRunner para Bagging-LSTM con métricas de trading
**Por qué primero**: sin métricas de trading en el ML, la tabla comparativa final está incompleta.  
**Qué hacer**:
- En `backtest/evaluation/runner.py`, agregar soporte para `model_type="bagging-lstm"` en `MLBacktestRunner`
- Crear clase `EnsembleLSTMPredictor` en `backtest/models/lstm.py` (o en nuevo archivo) que:
  - Entrena N `LSTMPredictor` con semillas distintas (reutilizar `BaggingLSTMSearcher._train_one_lstm()` de `optimization/searchers/ensemble_searcher.py`)
  - Implementa `.predict(indicators)` promediando probabilidades (soft voting)
- Exponer en CLI: `python -m cli train-ml --model bagging-lstm --serialize --tag ml-bagging-lstm`
- La salida incluirá `win_rate`, `profit_factor`, `sharpe_ratio`, `max_drawdown` igual que hace `MLBacktestRunner` con LSTM individual
- **Resultado**: `backtest/data/results/ml-bagging-lstm.json` con métricas completas

### PASO 2 — Exportar datos de fine-tuning
**Precondición**: dataset en `backtest/data/labeled/dataset.jsonl`  
**Comando**:
```bash
cd langgraph
python -m cli export-training-data \
  --dataset backtest/data/labeled/dataset.jsonl \
  --output training_data/
```
**Resultado**: `training_data/{train,val,test}.jsonl` (~39,312 / 8,424 / 8,425 ejemplos en formato chat Qwen)  
**Archivo de referencia**: `backtest/export.py` — usa `_temporal_split` idéntico al de los runners (garantiza pairing)

### PASO 3 — Fine-tuning QLoRA en RTX 5070 Ti (16 GB VRAM)
**Config recomendada** (de `optimization/configs/qlora.yaml`):
```
model: unsloth/Qwen2.5-7B-Instruct-bnb-4bit
learning_rate: 0.00002
lora_rank: 16
lora_alpha: 32
epochs: 3
batch_size: 4 (effective 16 con gradient_accumulation_steps=4)
```
**Comando**:
```bash
python optimization/qlora/train_qlora.py --lr 0.00002 --rank 16 --epochs 3 --batch-size 4
```
**Duración estimada**: 2–4 horas en RTX 5070 Ti  
**Resultado**:
- Adapters LoRA en `backtest/data/models/qlora_qwen25_7b/`
- GGUF Q4_K_M para Ollama en `backtest/data/models/qlora_qwen25_7b/gguf/`
- Métricas automáticas en `optimization/qlora/results/qlora_optimization.json`

### PASO 4 — Backtest zero-shot (dos modelos)
**4a. Qwen 2.5 7B base (Ollama local)**:
```bash
LLM_PROVIDER=ollama LLM_MODEL=qwen2.5:7b \
  python -m cli run-backtest \
    --dataset backtest/data/labeled/dataset.jsonl \
    --provider ollama --tag zero-shot-qwen7b
```
**4b. Llama 3.3 70B (Groq API)**:
```bash
LLM_PROVIDER=groq LLM_MODEL=llama-3.3-70b-versatile \
  python -m cli run-backtest \
    --dataset backtest/data/labeled/dataset.jsonl \
    --provider groq --tag zero-shot-llama70b
```
**Nota**: Groq tiene rate limit; usar `--max-samples 2000` si necesario para estimar.  
**Resultado**: `backtest/data/results/zero-shot-*.json` con métricas completas incluyendo trading

### PASO 5 — Deploy QLoRA fine-tuned a Ollama y backtest
```bash
# Crear Modelfile para Ollama
ollama create qwen25-ft -f Modelfile   # apunta al GGUF del paso 3

# Backtest del modelo fine-tuneado
LLM_PROVIDER=ollama LLM_MODEL=qwen25-ft \
  python -m cli run-backtest \
    --dataset backtest/data/labeled/dataset.jsonl \
    --provider ollama --tag qlora-qwen7b
```
**Alternativa**: `qlora_optimization.json` ya incluye la evaluación si `train_qlora.py` se corrió completo  
**Archivo de referencia**: `optimization/qlora/train_qlora.py:evaluate()` — mismo split temporal, mismo parser JSON

### PASO 6 — Pruebas estadísticas (McNemar + t-test)
Comparaciones requeridas (pares):
```bash
cd langgraph

# 1. QLoRA fine-tuned vs Bagging-LSTM (pregunta central de la tesis)
python -m cli compare-stats \
  --a optimization/qlora/results/qlora_optimization.json \
  --b backtest/data/results/ml-bagging-lstm.json

# 2. Zero-shot Qwen vs Bagging-LSTM
python -m cli compare-stats \
  --a backtest/data/results/zero-shot-qwen7b.json \
  --b backtest/data/results/ml-bagging-lstm.json

# 3. Zero-shot Llama70B vs Bagging-LSTM
python -m cli compare-stats \
  --a backtest/data/results/zero-shot-llama70b.json \
  --b backtest/data/results/ml-bagging-lstm.json

# 4. QLoRA vs Zero-shot Qwen (efecto del fine-tuning puro)
python -m cli compare-stats \
  --a optimization/qlora/results/qlora_optimization.json \
  --b backtest/data/results/zero-shot-qwen7b.json
```
**Archivos de referencia**: `optimization/stats_tests.py` (McNemar exact binomial + t-test normal approx)  
**Requerimiento**: todos los JSONs deben tener `sample_keys` — el runner lo garantiza

### PASO 7 — Notebook Avance 6 (`Avance6.ipynb`)
Secciones propuestas:
1. **Configuración** — paths, constantes
2. **Tabla comparativa final** — los 4 enfoques con todas las métricas (acc, F1, AUC, win rate, profit factor, Sharpe, max drawdown)
3. **Gráfica comparativa** — barras por métrica, colores por tipo (zero-shot / fine-tuned / ML)
4. **Equity curves** — curvas de capital acumulado por modelo (usar `evaluation/report.py`)
5. **Pruebas estadísticas** — tabla de resultados McNemar (p-value, significancia) para los 4 pares
6. **Análisis por símbolo** — desempeño por par (BTCUSDT, ETHUSDT, etc.) para detectar sesgo
7. **Análisis de errores del QLoRA** — casos donde falla: qué indicadores tienen, qué predijo vs realidad
8. **Calibración de confianza del QLoRA** — diagrama de calibración (ya implementado en `evaluation/metrics.py:confidence_calibration`)
9. **Conclusiones** — responder las 5 preguntas de la propuesta Fase 0 §2.4

**Scripts a reutilizar**:
- `optimization/avance5_report.py` — patrón para tabla comparativa (adaptar para 4 modelos)
- `backtest/evaluation/report.py:plot_equity_curve()` — equity curves
- `backtest/evaluation/metrics.py:confidence_calibration()` — calibración QLoRA

---

## Documento PDF — Avance 6 Conclusiones clave

> Entregable: `Avance6.pdf` (nombre: `Avance6.#Equipo`)

### Estructura del documento (3 secciones → 100 pts)

#### SECCIÓN 1 — Análisis del modelo (50 pts rubrica)
**Referencia a criterios de éxito de Fase 0 §2.3:**
- Meta original: >55% accuracy para QLoRA (cualquier mejora sobre zero-shot justifica el proyecto)
- Resultado real: comparar directamente con cada objetivo específico del proposal

**4 preguntas obligatorias a responder:**

**P1: ¿El rendimiento es suficiente para producción?**
- Bagging-LSTM 83.37% test acc → SÍ para señales direccionales horarias
- QLoRA: depende del resultado; si >55%, cumple el mínimo; si >65%, es competitivo
- Umbral práctico: win rate >55% con profit factor >1.2 es rentable en trading algorítmico

**P2: ¿Hay margen de mejora?**
- ML: sí — incorporar más timeframes (15m, 1w), features de order book, sentiment
- LLM: sí — más épocas, data augmentation, curriculum learning, RLHF con señales de P&L real
- Calibración QLoRA: Platt Scaling para mejorar probabilidades

**P3: Recomendaciones clave para implementación**
- Bagging-LSTM para señales de alta confianza (prob > 0.75)
- QLoRA para explicabilidad cuando se necesita razonamiento
- Pipeline híbrido: LSTM filtra, LLM explica y parametriza (TP/SL)

**P4: Accionables por stakeholder (20 pts rubrica — sección 2)**
| Accionable | Stakeholder | Horizonte |
|---|---|---|
| Integrar Bagging-LSTM en el pipeline de producción via `MLBacktestRunner` serializado | Desarrollador (Alejandro) | 1 semana |
| Configurar endpoint Ollama con modelo QLoRA para el agente LangGraph | Desarrollador (Alejandro) | 3 días |
| Definir umbrales de confianza para activar señales en producción (backtesting de thresholds) | Investigador / Trader | 2 semanas |
| Monitorear drift del modelo cada 3 meses con nuevo dataset Binance | Operaciones | Recurrente |
| Validar señales en paper trading antes de capital real | Trader | 1 mes |

#### SECCIÓN 2 — Análisis de proveedores cloud (30 pts rubrica)
**Nota**: El proyecto se ejecutará localmente en RTX 5070 Ti. El análisis cloud es requerido por la rúbrica y demuestra que la elección local fue informada.

**Factores de comparación** (usar estos mismos para los 4 proveedores):
1. Facilidad de uso / curva de aprendizaje
2. Servicio de fine-tuning de LLMs (managed)
3. Inferencia serverless de modelos custom
4. Costo por hora de GPU (A10G / A100 / H100)
5. Integración con Hugging Face / Unsloth
6. Latencia de inferencia para trading en tiempo real
7. Privacidad de datos (datos financieros propietarios)

**Proveedores a comparar** (todos 4 requeridos):
- **AWS (EC2 g6e.xlarge / SageMaker)** — referencia: `optimization/qlora/EC2_GUIDE.md` ya en el repo; BedrockProvider ya implementado en `agent/llm_factory.py`
- **Azure ML** — Azure OpenAI Service + fine-tuning managed
- **GCP Vertex AI** — Model Garden + TPU access; GoogleProvider ya implementado en `agent/llm_factory.py`
- **IBM Watson** — watsonx.ai; menor integración con Hugging Face

**Conclusión recomendada**: Despliegue local en RTX 5070 Ti por:
- Costo cero marginal (hardware ya adquirido)
- Sin latencia de red (crítico para trading horario)
- Privacidad total de estrategias de trading
- 16 GB VRAM suficiente para Qwen 2.5 7B Q4 en inferencia (<6 GB) y fine-tuning (con batch=4)
- Comparar contra costo estimado cloud: AWS ml.g5.xlarge ~$1.58/hr → ~$10-15 para fine-tuning + $0.10/inference-batch

---

## Gráficas para entregable y presentación final

| Gráfica | Fuente de datos | Script/función |
|---|---|---|
| Tabla comparativa 4 modelos (acc, F1, AUC, win rate, Sharpe) | JSONs de resultados | Adaptar `avance5_report.py` |
| Barras comparativas por métrica | JSONs de resultados | matplotlib en `Avance6.ipynb` |
| Equity curves los 4 modelos superpuestas | `trade_results[*].pnl_pct` | `backtest/evaluation/report.py` |
| Matriz de confusión QLoRA (comparar con Bagging-LSTM) | `test_y_true/pred` del JSON | sklearn `ConfusionMatrixDisplay` |
| Curva ROC los 4 modelos en un solo plot | `test_y_proba` + labels | sklearn `RocCurveDisplay` |
| McNemar heatmap (p-values entre pares) | salida `stats_tests.py` | seaborn heatmap |
| Calibración de confianza QLoRA | `confidence_calibration()` | ya en `Avance5.ipynb` sec 7 (replicar) |
| Desempeño por símbolo (accuracy por par) | `predictions[*].symbol` | groupby + barplot |
| Distribución de probabilidades QLoRA vs Bagging-LSTM | `test_y_proba` | histograma overlapping |

---

## Archivos a crear / modificar

| Archivo | Acción | Descripción |
|---|---|---|
| `backtest/models/lstm.py` | Modificar | Agregar `EnsembleLSTMPredictor` (N bags, soft voting) |
| `backtest/evaluation/runner.py` | Modificar | Agregar `model_type="bagging-lstm"` en `MLBacktestRunner` |
| `cli.py` | Modificar | Agregar `bagging-lstm` a las opciones de `--model` en `train-ml` |
| `langgraph/Avance6.ipynb` | Crear | Notebook con comparativa final, gráficas, pruebas estadísticas |
| `langgraph/AVANCE6.md` | Crear | Documentación del avance (mismo estilo que AVANCE5.md) |
| `langgraph/avance6_report.py` | Crear | Script para generar tabla y gráficas del Avance 6 |
| `Avance6_PDF.md` o directamente Word/PDF | Crear | Documento PDF para entrega a la profesora |

---

## Verificación — cómo saber que todo funciona

1. `bagging_lstm_optimization.json` + `ml-bagging-lstm.json` tienen `win_rate`, `profit_factor`, `sharpe_ratio`, `max_drawdown` — no solo acc/F1/AUC
2. Todos los JSONs de resultados tienen `sample_keys` con el mismo largo (~8,425)
3. `compare-stats` no lanza advertencia de "using positional fallback" (confirma que el pairing es exacto)
4. El notebook `Avance6.ipynb` corre de inicio a fin sin errores (`Kernel → Restart & Run All`)
5. La tabla comparativa muestra exactamente 4 filas: zero-shot-qwen7b, zero-shot-llama70b, qlora-qwen7b, bagging-lstm

---

## Orden de ejecución en la máquina local (resumen)

```
1. git pull origin feature/avance5-ensembles
2. [código] Extender runner para bagging-lstm
3. python -m cli train-ml --model bagging-lstm --serialize --tag ml-bagging-lstm
4. python -m cli export-training-data ...
5. python optimization/qlora/train_qlora.py --lr 0.00002 --rank 16 --epochs 3
6. python -m cli run-backtest --provider ollama --model qwen2.5:7b --tag zero-shot-qwen7b
7. python -m cli run-backtest --provider groq --tag zero-shot-llama70b
8. python -m cli run-backtest --provider ollama --model qwen25-ft --tag qlora-qwen7b
9. python -m cli compare-stats (4 pares)
10. Ejecutar Avance6.ipynb completo
11. Redactar Avance6 PDF
```
