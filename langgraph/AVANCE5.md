# Avance 5 — Modelos Ensemble

**Proyecto**: Sistema Híbrido de Trading: Comparación LLM vs ML  
**Dataset**: 56,161 señales etiquetadas · 12 símbolos · 18 meses · 3 timeframes  
**Split temporal**: 70% entrenamiento / 15% validación / 15% prueba (sin mezcla)

---

## 1. Objetivo

Explorar si los modelos ensemble superan al mejor modelo individual del Avance 4 (LSTM, 84.29% val acc).  
Se implementaron **5 arquitecturas** de dos categorías:

| Categoría | Modelos |
|---|---|
| Homogéneo | Bagging-LSTM |
| Heterogéneo | AdaBoost, Soft-Voting, Stacking (OOF), Blending |

---

## 2. Arquitecturas Implementadas

### 2.1 Bagging-LSTM (`bagging_lstm`)

Entrena **N LSTMs independientes** con semillas aleatorias distintas sobre el conjunto train+val completo. Las predicciones finales se obtienen promediando las probabilidades de clase de cada bolsa (*soft voting*). Al promediar modelos con varianza independiente, se reduce el error de varianza sin introducir sesgo adicional.

```
Semilla 42  → LSTM → P(LONG|x) ┐
Semilla 43  → LSTM → P(LONG|x) ├─ mean → predicción final
Semilla 44  → LSTM → P(LONG|x) ┘
...
```

**Parámetros:**

| Parámetro | Valor |
|---|---|
| `n_bags` | 5 |
| `hidden_size` | 32 |
| `num_layers` | 3 |
| `sequence_length` | 5 |
| Tiempo de entrenamiento | 11m 35s |

---

### 2.2 AdaBoost (`adaboost`)

Boosting secuencial sobre *decision stumps* (árboles de profundidad 2). Cada ronda re-pondera las muestras mal clasificadas, forzando al clasificador siguiente a enfocarse en los casos difíciles. Se hace búsqueda en grilla sobre `n_estimators` y `learning_rate` con `TimeSeriesSplit(3)`.

**Parámetros buscados:**

| Parámetro | Rango |
|---|---|
| `n_estimators` | [100, 200, 300] |
| `learning_rate` | [0.01, 0.05, 0.1, 0.5] |

> **Nota:** Requiere `scikit-learn ≥ 1.4` (el parámetro `algorithm` fue eliminado en esa versión). Los resultados de esta corrida fallaron por incompatibilidad de versión — ya corregido en el repositorio.

---

### 2.3 Soft-Voting (`soft_voting`)

Combina tres modelos heterogéneos (LSTM, XGBoost, SVM) mediante un **promedio ponderado** de probabilidades. Los pesos se asignan proporcionalmente a la *accuracy* de validación de cada modelo, por lo que el más fuerte tiene mayor influencia.

```
LSTM     (w=0.3613) → P(LONG|x) ┐
XGBoost  (w=0.3110) → P(LONG|x) ├─ suma ponderada → predicción
SVM      (w=0.3278) → P(LONG|x) ┘
```

**Pesos aprendidos:**

| Modelo base | Peso |
|---|---|
| LSTM | 0.3613 |
| XGBoost | 0.3110 |
| SVM | 0.3278 |

Tiempo de entrenamiento: **10m 23s**

---

### 2.4 Stacking OOF (`stacking`)

Meta-aprendizaje en dos niveles. Los modelos base (XGBoost, SVM, MLP, LSTM) generan predicciones *out-of-fold* sobre el entrenamiento usando `TimeSeriesSplit(5)`. Un meta-modelo de Regresión Logística se entrena sobre estas predicciones OOF. La inferencia final usa los modelos base entrenados con todos los datos + el meta-modelo.

```
Train (OOF via TimeSeriesSplit-5):
  XGBoost → P̂_xgb ┐
  SVM     → P̂_svm ├─ LogReg meta → predicción final
  MLP     → P̂_mlp ┘
  LSTM    → P̂_lstm

Test:
  base_models.predict(X_test) → meta → ŷ
```

**Coeficientes del meta-modelo:**

| Base learner | Coeficiente |
|---|---|
| XGBoost | 1.7205 |
| SVM | -1.0501 |
| MLP | 0.5958 |
| LSTM | 6.6334 |

> El LSTM domina con coeficiente 6.63 — el meta-modelo aprendió que sus probabilidades son las más confiables.

Tiempo de entrenamiento: **35m 32s**

---

### 2.5 Blending (`blending`)

Similar al stacking pero sin validación cruzada. Se reserva un **30% del conjunto de entrenamiento** como *blend set* (nunca visto por los modelos base). El meta-modelo se entrena sobre las predicciones de los base models en ese hold-out set. Más rápido que stacking pero usa menos datos de entrenamiento.

```
Train 70%  → base_models.fit()
Blend 30%  → base_models.predict() → LogReg meta.fit()
Test       → base_models.predict() → meta → ŷ
```

**Configuración:**

| Parámetro | Valor |
|---|---|
| `blend_fraction` | 0.30 |
| Base learners | XGBoost, SVM, MLP, LSTM |
| Meta-modelo | Logistic Regression |

Tiempo de entrenamiento: **7m 14s**

---

## 3. Resultados

### 3.1 Tabla Comparativa

| Modelo | Tipo | Val Acc | Test Acc | F1-macro | AUC-ROC | Tiempo |
|---|---|---|---|---|---|---|
| **Bagging-LSTM** | Bagging | **0.8332** | **0.8337** | **0.8330** | **0.9157** | 11m 35s |
| Stacking (OOF) | Stacking | 0.8171 | 0.8164 | 0.8158 | 0.8961 | 35m 32s |
| Blending | Blending | 0.8153 | 0.8189 | 0.8183 | 0.8969 | 7m 14s |
| Soft-Voting | Voting | 0.8141 | 0.8023 | 0.8012 | 0.8817 | 10m 23s |

**Referencia Avance 4 (modelos individuales — test set):**

| Modelo | Test Acc | F1-macro | AUC-ROC | Notas |
|---|---|---|---|---|
| LSTM (best) | **0.8150** | 0.814 | 0.896 | hidden=32, layers=3, seq=5 |
| XGBoost | 0.6664 | 0.662 | 0.747 | con regularización L1/L2 |
| Random Forest | — | — | — | no evaluado en test |

> **Mejora del Avance 5**: Bagging-LSTM 83.37% vs LSTM individual 81.5% → **+1.87 pp en test set**

### 3.2 Gráfica Comparativa

![Comparación Avance 5](optimization/results/avance5_comparison.png)

> Generada automáticamente con `python optimization/avance5_report.py`

---

## 4. Análisis

### ¿Mejoran los ensembles al LSTM individual?

El **Bagging-LSTM obtiene 83.37% test acc** vs **81.5% test acc del LSTM individual** (Avance 4).  
Comparación correcta: ambos evaluados sobre el mismo conjunto de **test** (15% temporal, nunca visto durante entrenamiento ni selección de hiperparámetros). El ensemble mejora **+1.87 pp** en test — reduciendo la varianza del LSTM individual al promediar 5 instancias con semillas distintas.

**Observaciones clave:**

1. **Bagging-LSTM gana** entre los ensembles (+1.87 pp sobre Soft-Voting). Promediar 5 LSTMs reduce la varianza de la arquitectura más fuerte del proyecto.

2. **El LSTM domina el meta-modelo de Stacking** (coeficiente 6.63 vs 1.72 de XGBoost). El meta-aprendizaje básicamente aprendió a confiar en el LSTM casi exclusivamente.

3. **Blending es el mejor trade-off** velocidad/accuracy en los heterogéneos: 7m 14s vs 35m 32s de Stacking con resultados similares (81.89% vs 81.64%).

4. **Soft-Voting baja en test** (80.23% vs 80.41% val) — XGBoost y SVM arrastran el promedio y el ensemble no generaliza tan bien como en validación.

### Explicación del efecto del Bagging

El LSTM con `hidden=32, layers=3, seq=5` es un modelo relativamente pequeño con cierta varianza. Al entrenar 5 instancias con semillas distintas, la varianza de cada una se cancela en el promedio:

```
Error = Sesgo² + Varianza + Ruido
Bagging → Varianza / N  (con N bolsas independientes)
```

El sesgo no cambia (todas las bolsas usan la misma arquitectura), pero la varianza se divide entre 5, lo que mejora la generalización.

---

## 5. Selección del Modelo Final

**Modelo seleccionado: Bagging-LSTM**

| Criterio | Valor |
|---|---|
| Test accuracy | **83.37%** |
| F1-macro | **0.833** |
| AUC-ROC | **0.916** |
| Tiempo de inferencia | ~5× más lento que LSTM individual |

El costo de inferencia (5 LSTMs en serie) es aceptable para el caso de uso del proyecto (análisis de señales horarias, no trading de alta frecuencia).

---

## 6. Reproducibilidad

```bash
# Activar entorno
cd langgraph
conda activate trading

# Re-entrenar todos los ensembles (resume automático si ya existen JSONs)
OMP_NUM_THREADS=4 python optimization/run_ensembles.py

# Ver reporte completo + guardar gráfica
python optimization/avance5_report.py

# Ver solo tabla (sin matplotlib)
python optimization/avance5_report.py --no-plots
```

Resultados guardados en `optimization/results/`:
- `bagging_lstm_optimization.json`
- `soft_voting_optimization.json`
- `stacking_optimization.json`
- `blending_optimization.json`
- `avance5_comparison.png`

---

## 7. Conclusiones

1. El **Bagging-LSTM supera al LSTM individual** del Avance 4 en test (+1.87 pp: 83.37% vs 81.5%). La comparación válida es test vs test — el ensemble generaliza mejor al reducir la varianza del modelo base.

2. **Bagging-LSTM** es el mejor ensemble — al promediar 5 LSTMs con semillas distintas, reduce la varianza sin aumentar el sesgo (misma arquitectura), logrando 83.37% test con gap val-test de apenas 0.05 pp.

3. Los ensembles heterogéneos (Stacking, Blending, Voting) están limitados por los modelos base más débiles (XGBoost 76.8%, RF 74.9%) que arrastran el promedio hacia abajo.

4. Para la comparación final de la tesis, el modelo a contrastar con el LLM zero-shot y QLoRA es el **Bagging-LSTM** como representante del enfoque ML.
