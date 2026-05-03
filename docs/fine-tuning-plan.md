> ⚠️ **SUPERSEDED** — This document has been replaced by [fine-tuning-strategy.md](./fine-tuning-strategy.md). Kept for historical reference.

# Fine-Tuning Plan — Trading Management System

Plan para fine-tunear Qwen 2.5 con datos reales de trading y medir la mejora vs el modelo base.

---

## Objetivo

Demostrar cuantitativamente que un modelo fine-tuneado con señales de trading históricas produce mejores predicciones que el modelo base, medido por accuracy, profit factor y confidence calibration.

---

## Plataforma: Together AI

| Aspecto | Detalle |
|---|---|
| **Proveedor** | [Together AI](https://www.together.ai/) |
| **Modelo base** | Qwen 2.5 7B o 14B (mismo family que usamos) |
| **Formato** | API OpenAI-compatible (mismo que inferencia) |
| **Costo estimado** | ~$5-15 por sesión de fine-tuning |
| **Presupuesto trimestre** | ~$30-50 total |

Together soporta fine-tuning de modelos Qwen directamente. Subes un JSONL, entrenan en sus GPUs, te dan un model ID custom que usas con el mismo endpoint.

---

## Fase 1 — Recolección de datos (Semanas 1-4)

### Fuente de datos

**Velas históricas de Binance** — datos OHLCV públicos y gratuitos via API. No se requieren trades reales ni cuenta de trading.

Binance provee hasta 1,000 velas por request. Para 6 meses de velas 1h = ~4,320 velas por par. Con 20 pares = ~86,400 puntos de datos brutos.

### Cómo se genera el dataset (sin trades reales)

```
Para cada par (BTCUSDT, ETHUSDT, ...):
  1. Descargar velas históricas de Binance (6-12 meses, timeframe 1h y 4h)
  2. En cada vela T:
     a. Calcular indicadores técnicos (RSI, MACD, BB, ADX, etc.) usando indicator_service
     b. Mirar qué pasó N horas después (T+4h, T+8h, T+24h):
        - Precio subió > 1%  → label = LONG
        - Precio bajó > 1%   → label = SHORT
        - Lateral (< 1%)     → label = NEUTRAL
     c. Calcular TP/SL óptimos con hindsight:
        - TP = máximo alcanzado antes de revertir
        - SL = mínimo alcanzado antes de recuperar
     d. Asignar confidence basado en claridad del movimiento:
        - Movimiento > 3% con volumen alto → confidence 85-95
        - Movimiento 1-3% → confidence 60-80
        - Movimiento ambiguo → descartar ejemplo
  3. Exportar como JSONL
```

Esto genera labels **perfectos** porque usamos hindsight — sabemos exactamente qué pasó. El modelo aprende a reconocer los patrones de indicadores que precedieron movimientos reales.

### Script de generación

```
backend/scripts/generate_training_data.py

1. Descarga velas históricas via Binance API (público, sin API key)
   - GET /api/v3/klines?symbol=BTCUSDT&interval=1h&limit=1000&startTime=...
2. Para cada ventana temporal, calcula indicadores con indicator_service
3. Mira el futuro (hindsight) para generar el label correcto
4. Filtra ejemplos ambiguos (movimiento < 1% o volumen bajo)
5. Exporta a train.jsonl / val.jsonl / test.jsonl (80/10/10 split)
```

### Formato JSONL (requerido por Together AI para fine-tuning)

JSONL = un JSON por línea. Es el formato estándar de fine-tuning para Together AI, OpenAI, y otros. El script lo genera automáticamente.

```jsonl
{"messages":[{"role":"system","content":"You are a crypto trading analyst..."},{"role":"user","content":"BTCUSDT 1h: price=67234.5 rsi=62.3 macd_hist=125.4 adx=35.2 volume_ratio=2.1 atr_ratio=1.35 bb_pos=0.78 heatmap=STRONG_BULLISH structure=BREAKOUT heatmap_multi={1m:BULLISH,5m:BULLISH,15m:STRONG_BULLISH,1h:STRONG_BULLISH,4h:BULLISH,1d:NEUTRAL,1w:BEARISH}"},{"role":"assistant","content":"{\"bias\":\"LONG\",\"confidence\":82,\"entry\":67234.5,\"tp\":68500.0,\"sl\":66800.0,\"reasoning\":\"Strong bullish confluence with breakout structure and high volume. ADX confirms trend. Weekly bearish limits upside.\"}"}]}
{"messages":[{"role":"system","content":"You are a crypto trading analyst..."},{"role":"user","content":"ETHUSDT 4h: price=3450.2 rsi=28.1 ..."},{"role":"assistant","content":"{\"bias\":\"LONG\",\"confidence\":75,...}"}]}
```

### Volumen objetivo

| Métrica | Target |
|---|---|
| Pares | 20+ (los mismos que escanea el turbo-scanner) |
| Timeframes | 1h, 4h |
| Periodo histórico | 6-12 meses |
| Ejemplos brutos | ~80,000+ (20 pares × 4,320 velas × 2 timeframes) |
| Ejemplos filtrados (calidad) | 3,000-10,000 (descartando movimientos ambiguos) |
| Split | 80% train / 10% validation / 10% test |
| Distribución objetivo | ~40% LONG, ~40% SHORT, ~20% NEUTRAL |

---

## Fase 2 — Fine-tuning (Semanas 5-8)

### Proceso

```
1. Dividir dataset: 80% train, 10% validation, 10% test
2. Subir train.jsonl a Together AI
3. Lanzar fine-tuning job via API
4. Monitorear loss/accuracy en validation set
5. Recibir model ID del modelo fine-tuneado
6. Evaluar en test set (nunca visto durante entrenamiento)
```

### Together AI fine-tuning API

```bash
# Subir dataset
curl -X POST https://api.together.xyz/v1/files \
  -H "Authorization: Bearer $TOGETHER_API_KEY" \
  -F "file=@train.jsonl" \
  -F "purpose=fine-tune"

# Lanzar fine-tuning
curl -X POST https://api.together.xyz/v1/fine-tunes \
  -H "Authorization: Bearer $TOGETHER_API_KEY" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "training_file": "file-xxxxx",
    "n_epochs": 3,
    "learning_rate": 1e-5,
    "suffix": "trading-v1"
  }'

# Resultado: modelo custom accesible como
# "alexglz/Qwen2.5-7B-Instruct-trading-v1"
```

### Iteraciones planificadas

| Iteración | Cambio | Objetivo |
|---|---|---|
| v1 | Dataset completo, hiperparámetros default | Baseline fine-tuned |
| v2 | Filtrar ejemplos ambiguos (confidence < 60) | Mejorar precision |
| v3 | Agregar más contexto (multi-timeframe detallado) | Mejorar recall |
| v4 | Ajustar learning rate / epochs | Optimizar convergencia |

---

## Fase 3 — Evaluación (Semanas 9-12)

### Métricas de comparación

| Métrica | Qué mide | Cómo se calcula |
|---|---|---|
| **Direction Accuracy** | ¿Predijo correctamente LONG/SHORT? | Aciertos / Total |
| **Confidence Calibration** | ¿Confidence 80 = 80% acierto real? | Reliability diagram |
| **Profit Factor** | Ganancia bruta / Pérdida bruta en backtest | Simulación con señales |
| **Sharpe Ratio** | Retorno ajustado por riesgo | (Retorno - Risk-free) / Std dev |
| **Max Drawdown** | Peor caída desde un pico | Simulación de equity curve |
| **TP/SL Accuracy** | ¿Los niveles sugeridos fueron alcanzados? | Hit rate de TP vs SL |
| **Latencia** | Tiempo de inferencia | Promedio por request |

### Evaluación A/B

Correr ambos modelos en paralelo sobre los mismos datos de test:

```
Test set (10% del dataset, nunca visto)
    │
    ├── Modelo base (Qwen 2.5 sin fine-tune) → predicciones A
    ├── Modelo fine-tuned (trading-v1)        → predicciones B
    │
    ▼
Comparar contra resultados reales del mercado
```

### Backtest simulado

```
Para cada señal en el test set:
  1. Entrada al precio indicado
  2. Si precio toca TP primero → ganancia
  3. Si precio toca SL primero → pérdida
  4. Si ni TP ni SL en N horas → cierre neutral

Calcular:
  - Win rate
  - Average win / average loss
  - Profit factor
  - Equity curve
```

### Entregables de evaluación

| Entregable | Formato |
|---|---|
| Tabla comparativa base vs fine-tuned | Markdown / LaTeX |
| Reliability diagram (confidence calibration) | Gráfica matplotlib |
| Equity curve (backtest) | Gráfica matplotlib |
| Confusion matrix (LONG/SHORT/NEUTRAL) | Gráfica matplotlib |
| Training loss curve | Gráfica (de Together AI dashboard) |

---

## Integración con la app

El modelo fine-tuneado se usa exactamente igual que el base — solo cambia el model ID:

```bash
# .env.prod
LLM_PROVIDER=together        # o deepseek
LLM_MODEL=alexglz/Qwen2.5-7B-Instruct-trading-v1   # modelo fine-tuned
LLM_API_KEY=xxx
LLM_BASE_URL=https://api.together.xyz/v1
```

El código de `llm_service.py` no cambia — mismo endpoint, mismo formato, diferente modelo.

---

## Timeline

```
Semana 1-2:   Implementar script de recolección de datos
Semana 2-4:   Correr scanner diario, acumular dataset
Semana 5:     Preparar dataset (limpieza, split, formato JSONL)
Semana 6:     Fine-tuning v1 + evaluación inicial
Semana 7:     Iterar (v2, v3) basado en resultados
Semana 8:     Fine-tuning final (v4)
Semana 9-10:  Evaluación completa (backtest, métricas, gráficas)
Semana 11:    Integrar modelo final en app desplegada
Semana 12:    Documentación y presentación final
```

---

## Riesgos y mitigaciones

| Riesgo | Impacto | Mitigación |
|---|---|---|
| Dataset muy pequeño (< 500 ejemplos) | Fine-tune no converge | Aumentar frecuencia de scans, agregar más pares |
| Overfitting al dataset | Bueno en train, malo en test | Early stopping, validation set, regularización |
| Mercado cambia de régimen | Modelo entrenado en bull no funciona en bear | Incluir datos de diferentes condiciones de mercado |
| Together AI cambia precios | Presupuesto excedido | Tener Fireworks AI como backup (misma API) |
| Modelo fine-tuned peor que base | Resultado negativo | Resultado negativo es válido académicamente — documentar por qué |
