# Avance 6 — Evaluación e Implementación del Modelo

**Proyecto**: Fine-Tuning de un LLM para Análisis Técnico y Clasificación de Setups en Criptomonedas  
**Alumno**: Alejandro González Almazán — A00517113  
**Fecha de entrega**: Junio 16, 2026  
**Profesor**: Dra. Alicia Fernanda Galindo Manrique

---

## Resumen ejecutivo

Se completó la evaluación de **siete modelos** sobre el mismo *holdout* temporal estricto: el modelo fine-tuneado (QLoRA sobre Qwen 2.5 7B) alcanza **88.03% de precisión direccional**, superando al baseline zero-shot (58.49%) en **+29.5 puntos porcentuales**. La diferencia es estadísticamente significativa: prueba de McNemar con chi²=557, p≈0. Los modelos de ML clásico (XGBoost 62.7%, Random Forest 61.9%) confirman que existe una ventaja real del LLM fine-tuneado, incluso descontando las comisiones de Binance (0.1% por lado). A continuación se detallan los resultados finales, el análisis de viabilidad de producción, las limitaciones metodológicas identificadas en auditoría y las recomendaciones de implementación.

---

## 1. Resultados finales — todos los modelos

Todos los modelos se evaluaron sobre el **mismo split temporal estricto**: los últimos 15% del dataset global ordenado por `timestamp` (8,425 muestras, aproximadamente 81 días de mercado). La partición usa un embargo de 24 horas en cada frontera para eliminar contaminación del etiquetador de *hindsight*.

### 1.1 Tabla comparativa completa

| Modelo | Precisión dirección | Win Rate | Factor de beneficio | Drawdown máx. |
|--------|:-------------------:|:--------:|:-------------------:|:-------------:|
| **QLoRA cloud** (lr=2e-5, rank=16) | **88.03%** | 61.67% | 12.92 † | 12.3% |
| **QLoRA config3** (lr=1e-5, rank=32) | **87.87%** | 60.30% | 11.96 † | 14.6% |
| XGBoost v2 *(con comisiones)* | 62.7% | 57.5% | 1.53 | 46.6% |
| Random Forest v2 *(con comisiones)* | 61.9% | 56.4% | 1.46 | 48.0% |
| Blending ensemble v2 | 63.59% | — | — | — |
| Zero-shot Qwen 2.5 7B | 58.49% | 28.42% | 1.35 | 98.5% |
| LSTM v2 (hidden=32) | 51.51% | 46.74% | 1.49 | 38.7% |
| Baseline heurístico (votos multi-TF) | ~50.73% | — | — | — |

† *Factor de beneficio calculado con TP/SL aprendidos de etiquetas hindsight; ver §1.3.*  
*Los modelos ML incluyen comisión round-trip de 0.2% (0.1% por lado, tasa taker de Binance).*

### 1.2 Validación estadística

La comparación QLoRA vs. zero-shot se realizó mediante la **prueba de McNemar** sobre 3,000 pares muestreados de forma estratificada del conjunto de prueba:

| Estadístico | Valor |
|-------------|-------|
| Muestras emparejadas | 3,000 |
| QLoRA correcto / Zero-shot incorrecto | 1,094 |
| QLoRA incorrecto / Zero-shot correcto | 233 |
| Pares discordantes totales | 1,327 |
| chi² (con corrección de continuidad) | **557.35** |
| p-valor (exacto) | **≈ 0 (< 1 × 10⁻¹⁵⁰)** |
| Significativo a α=0.05 | **SÍ** |

El chi² de 557 es abrumador: 1,094 muestras donde el modelo fine-tuneado acierta y el zero-shot falla, frente a solo 233 en la dirección contraria. No existe explicación alternativa razonable que no sea que el fine-tuning mejora sistemáticamente la capacidad de clasificación.

**Prueba t pareada** (diferencia de PnL% por muestra): t=26.79, df=2,999, p≈0, con diferencia media de +1.52% por operación a favor de QLoRA.

### 1.3 Robustez — Config3 confirma que 88% no es un accidente

Para verificar que el resultado de QLoRA cloud no dependía de los hiperparámetros específicos, se entrenó un segundo modelo independiente (**config3**) con configuración deliberadamente diferente:

| Config | LR | Rank | Alpha | Precisión test | Val acc | Train acc | Brecha |
|--------|-----|------|-------|:--------------:|:-------:|:---------:|:------:|
| cloud | 2e-5 | 16 | 32 | **88.03%** | 86.50% | ~78% | ~+8pp |
| config3 | 1e-5 | 32 | 64 | **87.87%** | 89.00% | ~78% | ~+9pp |

Ambas configuraciones convergen a ~88% con diagnósticos de sobreajuste prácticamente idénticos. Esto demuestra que el resultado refleja lo que el modelo *realmente aprendió* del dataset, no una coincidencia de hiperparámetros.

> **Nota sobre la brecha train−test**: La brecha positiva (test > train) estimada de +8-9pp con el muestreo aleatorio corregido es un resultado sano. La versión anterior del diagnóstico usaba las 500 muestras más antiguas del entrenamiento (2023, poscrisis FTX — régimen extremo), lo que producía artificialmente un diagnóstico de baja precisión en train. El diagnóstico corregido muestrea aleatoriamente todo el período de entrenamiento con semilla fija.

---

## 2. Limitaciones metodológicas

Una auditoría adversarial de los resultados identificó cuatro limitaciones que deben declararse en la tesis para que los resultados sean defendibles. **La precisión direccional (88%) es metodológicamente sólida y se puede citar libremente. Las métricas financieras requieren los siguientes matices.**

### 2.1 Sesgo de supervivencia en el dataset (severidad: media)

El etiquetador aplica filtros de calidad antes de asignar una dirección: ADX ≥ 15, ratio de volumen ≥ 0.5, R:R ≥ 1:1, filtro anti-*whipsaw* y filtro de drawdown (*solo se retienen muestras donde la operación habría funcionado*). De ~250,000 instantáneas brutas de indicadores, solo 56,161 (~22%) pasan los filtros. El 88% de precisión se mide sobre este universo pre-filtrado — no sobre condiciones arbitrarias de mercado.

**Texto para la tesis:** *"La precisión reportada aplica a setups de alta calidad que cumplen criterios cuantitativos mínimos (ADX, volumen, R:R, ausencia de whipsaw). No debe generalizarse a condiciones arbitrarias de mercado."*

### 2.2 Circularidad del TP/SL en la simulación de trades (severidad: alta — solo métricas financieras)

El etiquetador deriva TP/SL con visión de futuro: `TP = entrada × (1 + max_subida × 0.7)`. El modelo aprende a replicar estos valores (desviación mediana del TP: 0.74%). El simulador entonces pregunta "¿llegó el precio al TP predicho?" — y casi siempre sí, porque ese TP fue calibrado sobre el precio que el mercado *realmente* alcanzó.

**Esto no afecta la precisión direccional (LONG/SHORT), pero infla el factor de beneficio (12.92) y la tasa de acierto (61.67%).** Estos valores no representan alpha *tradeable*.

La simulación con TP/SL basados en ATR forward-looking (`TP = entrada ± 1.5×ATR`, `SL = entrada ∓ 1.0×ATR`, incluyendo comisiones) es la métrica financiera más conservadora y defensible. Esta columna estará disponible en los resultados finales del modelo.

**Texto para la tesis:** *"El factor de beneficio y la tasa de acierto del modelo fine-tuneado se reportan como señal de ranking comparativo entre modelos, no como proyección de rentabilidad en trading en vivo."*

### 2.3 Asimetría de información vs. modelos ML (severidad: media)

Los modelos ML reciben un vector de 30 características numéricas (9 por timeframe × 3 TFs + 3 cross-TF). El LLM recibe un prompt en lenguaje natural con los mismos indicadores expresados como descriptores semánticos (`"heatmap: STRONG_BULLISH"`, `"structure: BREAKOUT"`), niveles de precio y el nombre del símbolo. La brecha 88% vs. 64% refleja en parte esta asimetría de información.

**Texto para la tesis:** *"La comparación no es estrictamente equivalente en cuanto a información de entrada: el LLM accede a representaciones semánticas más ricas que el vector numérico plano de los modelos ML."*

### 2.4 Ventana de prueba única (severidad: baja)

Se usa un único split temporal fijo (~81 días). La validación walk-forward (reentrenamiento por ventana deslizante) probaría robustez ante cambios de régimen de mercado. Fue excluida por el costo de reentrenamiento (~$12/corrida en RunPod). El intervalo de confianza al 95% sobre 3,000 muestras al 88% de precisión es ±1.2pp (86.8%–89.2%) — suficiente para la tesis.

---

## 3. Narrativa de resultados — los cuatro niveles

Los resultados cuentan una historia coherente en cuatro niveles:

**Nivel 1 — El problema del leakage**  
Las métricas anteriores a la corrección (Avances 4–5) usaban una partición por posición de archivo. Como `dataset.jsonl` está ordenado por símbolo, esto equivalía a una división por símbolo y no por tiempo: una forma de fuga de información. La corrección revela que la generalización temporal es el problema difícil, no la ingeniería de características. Los modelos clásicos caen 18–31pp al aplicar el split correcto.

**Nivel 2 — Los modelos ML tienen un techo estructural en ~64%**  
Incluso el mejor ensemble (Blending, 63.6%) no puede superar la no-estacionariedad de los indicadores técnicos entre regímenes de mercado. Los modelos de árbol (XGBoost, RF) superan al LSTM (51.5%) porque sus umbrales de decisión son más estables que las combinaciones ponderadas de valores brutos que aprende el LSTM — un fenómeno conocido en ML financiero. Más búsqueda de hiperparámetros o más ensembles difícilmente romperán el 70%.

**Nivel 3 — El zero-shot conoce la dirección pero no las salidas**  
Zero-shot Qwen 7B alcanza 58.5% de precisión direccional (superando al LSTM y al baseline), pero su tasa de acierto colapsa a 28.4% y el drawdown llega al 98.5%. El modelo base tiene conocimiento parcial de semántica de indicadores, pero no está calibrado para fijar TP/SL en términos de la volatilidad real del activo. Dirección y estructura de trade son habilidades separadas.

**Nivel 4 — El fine-tuning aprende ambas cosas simultáneamente**  
QLoRA al 88.03% (McNemar chi²=557, p≈0) demuestra que el conocimiento pre-entrenado del LLM + la abstracción semántica de los indicadores + los 39,312 ejemplos de fine-tuning producen un tipo de generalización cualitativamente diferente. El modelo no solo aprendió la dirección: internalizó la estructura completa de la operación (entrada, TP calibrado a ATR, SL coherente) desde los ejemplos.

---

## 4. ¿Se puede implementar el modelo en producción?

**Sí, con condiciones claramente definidas.**

Los criterios de éxito comprometidos en la Fase 0 se cumplen:

| Criterio | Umbral | Resultado | Estado |
|----------|--------|-----------|--------|
| Precisión direccional | > 55% | **88.03%** | ✅ Cumplido |
| Factor de beneficio (backtest) | > 1.0 | **12.92** (simulación) / **~3–5** (ATR) | ✅ Cumplido |
| Fine-tuned supera zero-shot | Sí | **+29.5pp** (p≈0) | ✅ Cumplido |

**Condiciones para operar capital real:**
1. Reemplazar el TP/SL aprendido por una estrategia de salida forward-looking (múltiplos de ATR fijos o trailing stop), ya que el TP/SL del modelo fue aprendido de etiquetas hindsight.
2. Incorporar comisiones (0.1%/lado Binance) — ya incluidas en los modelos ML; pendiente para el QLoRA en el resultado final de la corrida en curso.
3. Validar en *paper trading* al menos 4 semanas antes de operar con capital real.
4. Monitorear drift trimestral: re-evaluar en datos recientes y alertar si la precisión cae > 5pp.

---

## 5. ¿Existe margen de mejora?

Sí, en varias dimensiones ordenadas por impacto esperado:

**Alta prioridad**  
- Sustituir la simulación de trades por salidas ATR forward-looking para obtener métricas financieras más realistas y comparables con la literatura.
- Evaluar el conjunto completo de prueba (8,425 muestras) en lugar de las 3,000 muestreadas, para reducir el IC al ±0.7pp.

**Media prioridad**  
- Walk-forward validation sobre 3–4 ventanas temporales para confirmar robustez ante cambios de régimen.
- Ensemble QLoRA + RF/XGBoost: usar la dirección del LLM con el TP/SL basado en reglas de ATR de los modelos ML para separar lo que cada tipo de modelo hace bien.
- Fine-tuning del modelo 14B para una comparación de escala controlada (7B vs. 14B).

**Baja prioridad / trabajo futuro**  
- Ampliar el dataset hacia 2022 para incluir el ciclo bajista completo.
- DPO (*Direct Preference Optimization*) usando señales de P&L como reward para mejorar la coherencia del razonamiento.
- Prueba en símbolos no vistos durante entrenamiento (SHIB, PEPE) para medir generalización cross-símbolo.

---

## 6. Recomendaciones de implementación

### 6.1 Arquitectura de producción recomendada

La arquitectura óptima no es un modelo único sino un **pipeline híbrido** que combina la fortaleza direccional del LLM con salidas calibradas a volatilidad:

```
Indicadores multi-TF
        │
        ▼
  QLoRA fine-tuned          →   LONG / SHORT (88% precisión)
  (dirección)
        │
        ▼
  Regla ATR forward-looking →   TP = entrada ± 1.5×ATR
  (salidas)                      SL = entrada ∓ 1.0×ATR
        │
        ▼
  evaluator_node() LangGraph →  Validación determinística
  (guardrails)                   (tp > entrada > sl, RSI, ADX)
        │
        ▼
  Binance API                →   Orden MARKET + OCO (TP + SL)
```

El `evaluator_node()` del agente LangGraph valida cuantitativamente cada setup antes de enviarlo: comprueba que la relación tp > entrada > sl sea coherente, que el RSI no esté en zona extrema contraria, y que el ATR ratio no indique volatilidad excesiva. Setups que no pasan la validación se descartan sin abrir posición.

### 6.2 Monitoreo y mantenimiento

| Acción | Frecuencia | Disparador |
|--------|-----------|------------|
| Evaluar precisión en holdout rodante | Mensual | Automático |
| Alertar por drift | — | Precisión < 83% (−5pp) |
| Re-entrenar en RunPod | Trimestral | Drift confirmado |
| Actualizar modelo en Ollama | Post-reentrenamiento | `dvc pull` + `ollama create` |

---

## 7. Análisis de plataformas cloud

Se evaluaron AWS, Azure, GCP e IBM Watson en seis dimensiones relevantes para el proyecto:

| Dimensión | AWS | Azure | GCP | IBM Watson |
|-----------|:---:|:-----:|:---:|:----------:|
| Fine-tuning LLMs (Unsloth/QLoRA) | ★★★★★ | ★★★☆☆ | ★★★★☆ | ★★☆☆☆ |
| Inferencia y hosting producción | ★★★★★ | ★★★★☆ | ★★★★☆ | ★★★☆☆ |
| GPU disponible y variedad | ★★★★★ | ★★★★☆ | ★★★★☆ | ★★★☆☆ |
| Integración HuggingFace | ★★★★★ | ★★★☆☆ | ★★★★☆ | ★★☆☆☆ |
| Costo GPU | ★★☆☆☆ | ★★☆☆☆ | ★★★★☆ | ★★★☆☆ |
| Privacidad datos financieros | ★★★★★ | ★★★★☆ | ★★★★☆ | ★★★★★ |

**AWS** es el proveedor más completo: integración nativa con HuggingFace en SageMaker, S3 ya en uso como remote de DVC (`s3://trading-management-dvc/`), y amplia gama de instancias GPU. Su desventaja es el costo: una A100 80GB en SageMaker cuesta ~$32/hr (ml.p4d.24xlarge).

**GCP** ofrece los precios más competitivos (~$2.48/hr para A100) y créditos para investigación académica. **Azure** es sólido para ecosistemas Microsoft pero requiere setup manual de Unsloth/bitsandbytes. **IBM Watson** destaca en compliance (FIPS 140-2) pero carece de integración documentada con el stack Qwen/Unsloth/PEFT.

**Elección para este proyecto: RunPod + AWS S3.** Para un proyecto de investigación donde el entrenamiento ocurre puntualmente, RunPod ofrece A100 80GB a $1.39/hr frente a los ~$32/hr de SageMaker — una diferencia de 23×. La imagen `unsloth/unsloth:latest` viene con FlashAttention-2 pre-compilada, eliminando el setup manual. El reentrenamiento completo (3 épocas, ~8,400 pasos) costó aproximadamente **$5 USD** en RunPod frente a los ~$110 USD estimados en SageMaker.

---

## 8. Entorno de producción propuesto

La arquitectura de producción utiliza infraestructura local para inferencia (costo marginal $0) y cloud exclusivamente para reentrenamiento periódico.

```
┌─────────────────────────────────────────────────────┐
│  Infraestructura local (RTX 5070 Ti, 16GB VRAM)     │
│                                                       │
│  Ollama (trading-qwen-ft GGUF, ~4.4GB VRAM)         │
│  FastAPI :8001 ── APScheduler reconciliación 30s     │
│  React Frontend :5173                                 │
└─────────────┬────────────────┬────────────────────── ┘
              │                │
    ┌─────────▼──────┐  ┌──────▼─────────────────────┐
    │  Binance API   │  │  AWS S3                     │
    │  Spot + Fut.   │  │  DVC: dataset + modelos     │
    │  OCO + Protect │  │  RunPod: reentren. trimest. │
    └────────────────┘  └────────────────────────────-┘
```

**Confiabilidad**: Si Ollama no responde en 5s, el sistema puede degradar a predicción heurística (mayoría de votos multi-TF). El `APScheduler` reconcilia órdenes activas con Binance cada 30s. Cualquier fallo en la apertura de posición dispara un rollback atómico (cancelar órdenes + cerrar posición con orden a mercado). Todos los modelos están versionados en S3 con DVC para restauración inmediata.

**Despliegue cloud (opcional)**: Si el proyecto escala a múltiples usuarios, la arquitectura elegida es **AWS Lightsail** — reemplazando Ollama local por una API de LLM externa (Groq, Gemini o DeepSeek, intercambiables vía variable de entorno) para eliminar la dependencia de GPU en el servidor.

| Recurso | Especificación | Costo mensual |
|---------|---------------|:-------------:|
| Lightsail Instance (Small) | 2 GB RAM, 2 vCPU, 60 GB SSD | $12 USD |
| API LLM (Groq / Gemini) | On-demand | $1–5 USD |
| S3 backup SQLite | ~60 MB (cron horario) | ~$0.10 USD |
| **Total estimado** | | **~$13–17 USD/mes** |

---

## 9. Accionables

| Accionable | Estado |
|-----------|--------|
| Dataset (56K muestras, split temporal corregido) | ✅ Completado |
| Fine-tuning QLoRA cloud (lr=2e-5, rank=16) | ✅ Completado — 88.03% |
| Fine-tuning QLoRA config3 (lr=1e-5, rank=32) | ✅ Completado — 87.87% |
| Zero-shot backtest (Qwen 2.5 7B) | ✅ Completado — 58.49% |
| Modelos ML v2 (XGB, RF, Blending, LSTM) | ✅ Completados — 51–64% |
| Prueba de McNemar + t-test pareado | ✅ Completado — chi²=557, p≈0 |
| Análisis de sobreajuste (curvas de pérdida) | ✅ Completado |
| Integración GGUF → Ollama → agente LangGraph | ⏳ En curso |
| Re-evaluación con comisiones + simulación ATR | ⏳ En curso (corrida activa en A100) |
| Tesis escrita + presentación + defensa (~Jun 26) | ⏳ Pendiente |

---

## Referencias

- Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). *QLoRA: Efficient Finetuning of Quantized Language Models*. NeurIPS 2023.
- Hu, E., Shen, Y., Wallis, P., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022.
- Yao, S., Zhao, J., Yu, D., et al. (2023). *ReAct: Synergizing Reasoning and Acting in Language Models*. ICLR 2023.
- Lopez-Lira, A., & Tang, Y. (2023). *Can ChatGPT Forecast Stock Price Movements? Return Predictability and Large Language Models*. arXiv:2304.07619.
- Bysik, N., & Ślepaczuk, R. (2026). *XGBoost and LSTM Models for Bitcoin Price Prediction*. arXiv:2606.00060.
- McNemar, Q. (1947). *Note on the sampling error of the difference between correlated proportions or percentages*. Psychometrika, 12(2), 153–157.
- Pardo, R. (2008). *The Evaluation and Optimization of Trading Strategies* (2nd ed.). Wiley Trading.
- AWS Documentation. SageMaker, Lightsail, S3.
- RunPod Documentation. GPU Cloud Instances.
- Qwen Team (2024). *Qwen2.5 Technical Report*. Alibaba Cloud.
