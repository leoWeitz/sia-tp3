# 02 · Requerimientos y trazabilidad

Fuente: *Enunciado TP3 — Perceptrón Simple y Multicapa (SIA 2026)*. Cada requerimiento tiene un ID que se usa en las specs de fases, en los nombres de experimentos y en las slides.

## 1. Requerimientos funcionales (herramientas)

| ID | Requerimiento | Fase |
|---|---|---|
| RF-01 | Perceptrón simple **escalón** (regla de Rosenblatt) | ✅ motor |
| RF-02 | Perceptrón simple **lineal** (ADALINE) | ✅ motor |
| RF-03 | Perceptrón simple **no lineal** (tanh y logística con β; ReLU opcional) | ✅ motor |
| RF-04 | **Perceptrón multicapa** con arquitectura configurable y backpropagation | ✅ motor |
| RF-05 | Modos de entrenamiento online / minibatch / batch | ✅ motor |
| RF-06 | Optimizadores: GD, momentum, η adaptativo, RMSProp, Adam (AdaGrad opcional) | F04 |
| RF-07 | Regularización: early stopping, L2 / weight decay, data augmentation (ruido gaussiano; traslaciones si los dígitos son imágenes) | F04 |
| RF-08 | Normalización: min-max a [a,b], z-score, unit length | F05 |
| RF-09 | Particiones: holdout estratificado, k-fold (estratificado) | F05 |
| RF-10 | Métricas: matriz de confusión (binaria y multiclase), accuracy, precision, recall, F1, TPR, FPR; MSE/MAE para regresión | F06 |
| RF-11 | Barrido de umbral, curvas ROC y Precision-Recall, AUC por trapecios | F06 |

## 2. Requerimientos no funcionales (recomendaciones del enunciado)

| ID | Requerimiento | Fase |
|---|---|---|
| RNF-01 | Operaciones **matriciales** para performance | ✅ motor |
| RNF-02 | **Reportar progreso** durante el entrenamiento | F07 |
| RNF-03 | **Configuración extensible** y almacenable | F07 |
| RNF-04 | **Guardar y levantar modelos** para seguir entrenando (junto con su config) | F07 |
| RNF-05 | **Separar** registro de experimentos (loss por época, hiperparámetros, tiempos) del **análisis** (gráficos, tablas) | F07, F14 |
| RNF-06 | Reproducibilidad (semillas, config resuelta, commit) | F07 |

## 3. Requerimientos de estudio (preguntas del enunciado)

### Validación (no se presenta, pero es obligatorio para nosotros)

| ID | Qué | Evidencia | Fase |
|---|---|---|---|
| V-01 | AND con perceptrón escalón | Converge a error 0; recta de decisión graficada | ✅ tests (F08: papel y figuras) |
| V-02 | Lineal ajusta 50 muestras de y = x | MSE → ~0, pesos ≈ (1, 0) | ✅ tests (F08: papel y figuras) |
| V-03 | No lineal ajusta 50 muestras de y = tanh(x) | MSE → ~0 con tanh, β = 1 | ✅ tests (F08: papel y figuras) |
| V-04 | XOR con MLP [2,2,1] y [2,3,2,1]; comparación con escalón | MLP 100 % accuracy; escalón no converge | ✅ tests (F08: papel y figuras) |
| V-05 | Cálculo a mano de un paso de backprop | Fixture numérico coincide con el código | ✅ tests (F08: papel y figuras) |

### Ejercicio 1 — Fraude (Knowledge Distillation, "TinyModel")

| ID | Pregunta | Experimento(s) | Figura / tabla | Fase |
|---|---|---|---|---|
| E1-A-a | ¿Underfitting? (lineal vs no lineal, **todas las muestras**) | `ej1_learning` | Curvas de error de entrenamiento vs época (media ± desvío) para ambos | F10 |
| E1-A-b | ¿Saturación de las capacidades? | `ej1_learning`, `ej1_long`, `ej1_beta`, `ej1_scaling` | Meseta del error; distribución de salidas / fracción de neuronas saturadas; salidas del lineal fuera de [0,1] | F10 |
| E1-A-c | ¿Cuál elegimos para generalizar y por qué? | Síntesis de anteriores | Tabla comparativa final de error de entrenamiento | F10 |
| E1-B-a | ¿Qué métricas y por qué? | `ej1_cv` | Tabla de métricas + justificación (desbalance) | F11 |
| E1-B-b | ¿Estrategia de partición? ¿Cómo se elige el mejor conjunto de entrenamiento? | `ej1_split_strategy`, `ej1_best_fold` | Varianza entre folds; estratificado vs aleatorio; holdout vs k-fold | F11 |
| E1-B-c | Mejor modelo + **umbral recomendado** | `ej1_threshold`, `ej1_final` (`--final-eval`) | Curvas P/R/F1 vs umbral, ROC, PR; matriz de confusión en test al umbral elegido | F11 |
| E1-OPT | ReLU / features nuevos / calibración | `ej1_relu`, teórico | — | F16 |

### Ejercicio 2 — Dígitos con MLP

| ID | Pregunta | Experimento(s) | Figura / tabla | Fase |
|---|---|---|---|---|
| E2-a | ¿Cómo evalúo el desempeño? | Protocolo de evaluación | Esquema de partición; accuracy, macro-F1, matriz 10×10, recall por clase | F12 |
| E2-b1 | Variantes de **tasa de aprendizaje** | `ej2_lr` | Curvas de loss y accuracy de validación por η | F12 |
| E2-b2 | Variantes de **arquitectura** | `ej2_arch_width`, `ej2_arch_depth` | Accuracy validación vs cantidad de parámetros; curvas | F12 |
| E2-b3 | Variantes de **optimizador** | `ej2_opt` | Curvas por optimizador (épocas y tiempo de pared) | F12 |
| E2-b4 | Otros hiperparámetros (batch, activación, init, β, salida) | `ej2_extra` | Tabla resumen | F12 |
| E2-final | Resultado en "producción" | `ej2_final` (`--final-eval` sobre `digits_test.csv`) | Matriz de confusión + métricas de test | F12 |

### Ejercicio 3 — More digits (objetivo accuracy ≥ 98 %)

| ID | Pregunta | Experimento(s) | Figura / tabla | Fase |
|---|---|---|---|---|
| E3-a | Mejor resultado con el nuevo dataset | `ej3_best`, `ej3_final` (`--final-eval`) | Accuracy test (media ± desvío) + mejor modelo guardado | F13 |
| E3-b | Técnicas usadas para mejorar | `ej3_ablation` | Tabla de ablación (agregar/quitar cada técnica) | F13 |
| E3-c | Otros factores que influyeron | `ej3_data_factors` | Curva de aprendizaje vs cantidad de datos; comparación de distribuciones; duplicados | F13 |
| E3-OPT | Robustez al ruido / interpretabilidad | `ej3_noise`, `ej3_attrib` | Accuracy vs σ de ruido; mapas de atribución | F16 |

## 4. Restricciones explícitas del enunciado

- R-01: En Ej1, el estudio de aprendizaje (lineal vs no lineal) usa **todas las muestras** del dataset.
- R-02: En Ej2 y Ej3, `digits.csv` / `more_digits.csv` se usan para ajustar parámetros **e hiperparámetros**; `digits_test.csv` equivale a producción (no se toca para decidir).
- R-03: No arrancar opcionales antes de tener todo lo obligatorio.
- R-04: Explorar los datos antes de modelar (documentación de columnas, rangos, composición, limpieza).
- R-05: La salida del Ej1 se interpreta como probabilidad de fraude en [0, 1] → la función de activación del no lineal debe tener imagen en (0,1) (logística) o re-escalarse explícitamente.