# Ej1 · Resultados

**Cómo leer este informe.**
- **Estudio principal: las 9 columnas del CSV** (todas menos el target y `flagged_fraud`). Las secciones A y B lo cubren.
- **Control: las 6 columnas elegidas en la EDA de F09.** La sección C lo documenta completo, con la comparación entre las dos versiones.
- **Modelo recomendado a CompanyX: el de 6 columnas** (ver B-c). Las dos versiones rinden igual en la práctica, y la de 6 es más chica y no depende del `timestamp`.

## A · Aprendizaje: perceptrón lineal vs no lineal (F10)

**Setup común**
- Todas las muestras (7500, `split: none`, R-01).
- Las 9 columnas, crudas y con z-score: `timestamp`, `amount_usd`, `quantity_purchased`, `session_duration_seconds`, `days_since_last_purchase`, `account_age_days`, `device_screen_resolution`, `time_since_last_login_s` e `items_viewed_before_purchase`.
- Target `big_model_fraud_probability` con MSE. 5 semillas (0–4) en cada config.
- Lineal: `["auto", 1]` con salida `identity`. No lineal: logística de la cátedra $\theta(h) = 1/(1+e^{-2\beta h})$.

Configs en `experiments/configs/ej1/all_features/{learning,long,beta,scaling}.json`. Resultados en `results/ej1_all_features/ej1_{learning,long,beta,scaling}/summary.csv`. Figuras, `tabla.csv` y `resumen.json` en `figures/ej1/all_features/learning/`, generados por `python -m analysis.ej1_learning --results-dir results/ej1_all_features --out-dir figures/ej1/all_features/learning`. Todo número de esta sección sale de esos archivos. Los valores son media ± desvío entre semillas.

**Las corridas se hicieron en paralelo** (8 y 4 procesos), así que los s/época de esta sección no se comparan con los de C. Las comparaciones de velocidad salen del control (C.1), que corrió en serie.

**Referencias** (calculadas en el análisis; no son modelos):
- Predictor constante (la media del target): MSE = **0.09148**.
- Mínimos cuadrados con bias sobre las mismas 9 columnas (`np.linalg.lstsq`), que es la cota exacta del perceptrón lineal: MSE = **0.02606**.

**Barrido `ej1_learning`:** η ∈ {1e-4, 1e-3, 1e-2, 1e-1} × {online, batch}, 500 épocas, la misma grilla para los dos modelos (figura `E1-A_lr_sensitivity`). Mejor config de cada modelo, elegida entre las que convergieron en las 5 semillas:

| Modelo | Mejor config | MSE train | Mejora vs constante | MSE − mínimos cuadrados | Fuera de [0, 1] | Neurona saturada |
|---|---|---|---|---|---|---|
| Lineal | GD batch, η = 0.1 | 0.02606 ± 3e-18 | 71.5 % | 0 (a la precisión impresa) | 6.0 % | — |
| Logística (β = 1) | GD online, η = 1e-4 | **0.01087** ± 7e-10 | **88.1 %** | −0.0152 | 0.0 % | 6.9 % |

Fuente: `figures/ej1/all_features/learning/tabla.csv` y `E1-A-c_summary_table`.

Otros resultados del barrido:
- **Lineal online con η = 0.1:** diverge en las 5 semillas.
- **Batch con η chico:** con 500 épocas no llega a converger. El lineal con η = 1e-4 queda en 1.68, peor que el constante. La logística batch solo llega cerca del piso con η = 0.1 (0.01098); el lineal ya llega con η = 0.01.
- **Online:** con η ≤ 1e-3 queda a ~1 % del mejor error de cada modelo.

### a) ¿Observan underfitting?

**Lineal: sí, por su capacidad.**
- Llega **exactamente** a la solución de mínimos cuadrados: 0.02606 contra 0.02606 (figura `E1-A-a_train_mse`). Ningún perceptrón lineal puede bajar de ahí con estas columnas.
- Mejora 71.5 % al predictor constante, pero el error que queda es un RMSE de 0.16 en unidades de probabilidad.

**Logística: aprende bastante más, pero también tiene un piso.**
- 0.01087: 88.1 % mejor que el constante y **58 % por debajo del mejor lineal posible**.
- El error no baja de ahí con ningún η, modo, optimizador ni β ≥ 1 (ver b). RMSE de 0.104.
- Con el criterio de la spec (subajusta si apenas mejora al constante), ninguno de los dos subajusta en forma severa. Pero ninguno imita a BigModel: los dos tienen un error de entrenamiento que no se va a cero, y en el lineal es 2.4 veces mayor.

**Dónde está el error de la logística** (`resumen.json › error_por_clase`, igual en las 5 semillas):
- MSE en los no fraudes: 0.01204. MSE en los fraudes: 0.00192.
- Los fraudes aportan solo el **2 %** del error total. En el lineal aportan el 21 % (MSE en fraudes 0.0470).
- **Esto contradice la hipótesis de F09**, que esperaba el mayor error cerca del corte 0.85. La logística ajusta bien los fraudes, que caen en su zona saturada cerca de 1. Lo que no puede reproducir es la forma de la probabilidad entre 0 y 0.85 de los no fraudes.

**Dato secundario** (umbral 0.5 contra `flagged_fraud`, que es la probabilidad umbralizada en 0.85): el lineal tiene F1 0.582 y la logística 0.499. No sirve para comparar los modelos: el umbral se estudia en B.

### b) ¿Observan saturación de las capacidades?

**Sí, en los dos.** Figura `E1-A-b_plateau`.

- **Meseta.** El criterio es una mejora relativa < 0.1 % en el último 10 % de las épocas. Con GD, en 500 épocas, la mejora es 0 en el lineal y 1.9e-7 en la logística. El lineal queda a menos de 1 % de su error final antes de la época ~30 y la logística online entre la 40 y la 80, según la semilla (`E1-A-a_train_mse`).
- **Más épocas y otro optimizador no cambian nada** (`ej1_long`: Adam, mini-batch 32, 2500 épocas):
  - lineal 0.02623 ± 5e-05, logística 0.01089 ± 8e-06;
  - en el último 10 % el error solo oscila por el ruido del mini-batch (mejora −0.31 % y −0.07 %).
- **El lineal agotó su capacidad**: alcanza la cota exacta de mínimos cuadrados.
- **La logística llega al mismo piso por caminos distintos**, que tomamos como su mejor error alcanzable:

  | Camino | MSE |
  |---|---|
  | GD online, η = 1e-4 | 0.010868 |
  | GD online, η = 1e-3 | 0.010870 |
  | Adam, 5× épocas | 0.01089 |
  | GD batch, β = 2 | 0.010870 |
  | GD batch, β = 4 | 0.010868 |

- **Salidas no probabilísticas del lineal:** 6.0 % de las salidas caen fuera de [0, 1], entre −0.42 y 2.06 (`E1-A-b_linear_outputs`). La logística, por construcción, nunca sale de (0, 1).

**Saturación de la neurona.** Figura `E1-A-b_saturation_beta`; mide la fracción de muestras con θ′ < 5 % del máximo.

- **Con z-score:** entre 0.7 % (β = 0.25) y 6.9 % (β = 4). Con β = 1, 6.4 % ± 0.3 %. Son sobre todo fraudes con salida cerca de 1, que es lo correcto.
- **Sin normalizar:**
  - el **100 %** de las muestras queda saturado con cualquier β, porque `timestamp` crudo vale ~1.7·10⁹ y domina el puntaje;
  - la neurona no aprende nada: el MSE queda en **0.363 ± 0.085** con todos los β, peor que el predictor constante (0.091);
  - la dispersión entre semillas es grande, porque cada inicialización queda trabada donde cae.
- **β:** con z-score, un β chico no llega a converger en 500 épocas con GD batch (β = 0.25: 0.0262; β = 0.5: 0.0144). Con β ≥ 1 llega al piso (0.01087–0.01098). En el código, β multiplica el gradiente, así que en la práctica también cambia el paso.

**Normalización** (`E1-A-b_scaling`; GD batch, η = 0.1 y 500 épocas para ambos modelos):

| | Sin normalizar | Min-max | Z-score |
|---|---|---|---|
| Lineal | diverge (5/5) | 0.0301 ± 0.0019 | **0.02606** |
| Logística | 0.363 ± 0.085 (saturada) | 0.0395 ± 0.0028 | **0.01098** ± 8e-5 |

Z-score es el mejor para los dos. Confirma la decisión de F09.

**Desvío del plan.** La spec pide correr `ej1_beta` y `ej1_scaling` con el mejor η de `ej1_learning`. Para la logística ese es online, y costaba ~1.5 h de cómputo en serie (0.78 s/época en C.1). Se usó **GD batch con η = 0.1**: es la mejor config batch de los dos modelos, con 0.01098 contra el piso de 0.01087 (1 % peor). Además usa el mismo optimizador para ambos modelos, que es lo que pide la comparación justa.

### c) ¿Cuál seleccionarían para el estudio de generalización?

**La logística (perceptrón simple no lineal).** Motivos:

1. **Más potencial de aprendizaje.** Su piso de error (0.01087) está 58 % por debajo del mejor lineal *posible* (0.02606). El lineal ya agotó su capacidad y no puede mejorar.
2. **La salida es una probabilidad** (R-05). Tiene imagen en (0, 1), como el target. El lineal devuelve valores fuera de [0, 1] en el 6.0 % de las transacciones, que no se pueden presentar al cliente como probabilidad sin recortarlos.
3. **Es igual de chico que el lineal** (10 parámetros con 9 columnas). El costo extra es solo la activación.

**Hiperparámetros de partida para B:**
- **Modelo:** `["auto", 1]`, salida `sigmoid`, β = 1, z-score, MSE.
- **Optimizador:** Adam con η = 1e-3 y mini-batch 32.
  - Queda a 1 % del piso desde la época ~10 (entre la 6 y la 12, según la semilla). Online con η = 1e-3 llega a lo mismo, pero es ~30 veces más lento por época (C.1: 0.78 contra 0.026 s/época).
  - Si el equipo prefiere quedarse con el algoritmo de la cátedra, GD online con η = 1e-3 da el mismo error.
- **Épocas:** ~300 con early stopping sobre `val_loss`. B lo ajusta con validación.
- β ≥ 2 da prácticamente lo mismo que β = 1. En B se prueban β ∈ {1, 2} solo para confirmarlo.

## B · Generalización y umbral (F11)

**Setup común**
- **TEST**: 20 % estratificado (1500 filas, `holdout_test.seed = 0`). Los índices están en `results/ej1_split/test_idx.npy`, con sha256 `9e711725e86c18460503685baaf37c87ad5c46e742a5971da5fa33a489943bcc`.
  - Se leyó **dos veces**, siempre con `--final-eval`: en `results/ej1_final` (control de 6 columnas, C.2) y en `results/ej1_all_features/ej1_final` (esta sección).
  - La segunda lectura fue una decisión explícita del equipo para comparar las dos versiones. Ninguna decisión del informe sale del TEST: ni la config, ni las épocas, ni el umbral, ni la elección entre 6 y 9 columnas (ver c).
- **DESARROLLO**: las otras 6000 filas. Todas las decisiones salen de acá.
- **Estratificación**: el runner estratifica por deciles de `big_model_fraud_probability`, no por `flagged_fraud`. Igual conserva la tasa de fraude: 11.73 % en TEST, 11.55 % en DESARROLLO y 11.55 ± 0.24 % en los 15 folds de `ej1_cv`.
- **Modelo**: el de A (logística `["auto", 1]`, las 9 columnas con z-score, MSE contra la probabilidad de BigModel), con Adam, mini-batch 32 y early stopping sobre `val_loss` (patience 30).
- **Etiqueta**: las métricas binarias usan `flagged_fraud`, que se lee por índice de fila y nunca entra al entrenamiento.

Configs en `experiments/configs/ej1/all_features/{cv,split_strategy,best_fold,final}.json`; el experimento del mejor fold lo corre `experiments/ej1_best_fold.py`. Figuras, `cv.csv`, `criterios.csv`, `costos.csv` y `resumen.json` en `figures/ej1/all_features/generalization/`, generados por `python -m analysis.ej1_generalization --results-dir results/ej1_all_features --out-dir figures/ej1/all_features/generalization`. Todo número de esta sección sale de esos archivos. Los valores son media ± desvío.

### a) Métricas elegidas

| Métrica | Rol | Por qué |
|---|---|---|
| **Average Precision** (área PR) | Principal, no depende del umbral | Con 11.6 % de positivos, PR mide el desempeño sobre la clase rara. La ROC-AUC da 0.99 y se ve optimista, porque la dominan los 7.6 negativos por cada positivo |
| **Recall** y **precision** en el umbral elegido | Principales operativas | Recall = fraudes detectados (el FN es el error caro). Precision = qué fracción de las alertas es fraude de verdad (el costo de revisar) |
| **F2** | Resumen en un número | Pondera el recall 4 veces más que la precision |
| **Cuentas por cada 1000 transacciones** | Para CompanyX | Traducen las métricas a detectados, no detectados y alertas falsas |
| MSE vs BigModel | Calidad de la destilación | Es lo que optimiza el entrenamiento |
| ROC-AUC, FPR | Secundarias | Comparabilidad |
| Accuracy | Con advertencia | Un modelo que nunca marca fraude tiene accuracy 0.884 |

Selección en `ej1_cv` (5-fold estratificado × 3 semillas; η ∈ {3e-4, 1e-3, 3e-3} × β ∈ {1, 2}; figura `E1-B-a_cv_table`):
- **Las 6 configs son equivalentes.** El val MSE va de 0.010989 a 0.011026. Esa diferencia de 4e-5 es 15 veces menor que el desvío entre folds (5.5e-4). El AP OOF está entre 0.9534 y 0.9539 en todas.
- Por la regla fijada de antemano (menor val MSE media) se eligió **η = 1e-3, β = 2** (`results/ej1_all_features/ej1_cv/544d6eed`), con val MSE **0.01099 ± 0.00055**, AP OOF 0.954 y ROC-AUC OOF 0.992. Converge en una mediana de 25 épocas.
- **No hay sobreajuste:** el train MSE (0.01102) es igual al val MSE. Es esperable con 10 parámetros para 4800 filas.

### b) Estrategia de partición y "el mejor conjunto de entrenamiento"

**Estrategia:** holdout estratificado para TEST y 5-fold estratificado dentro de DESARROLLO. La evidencia sale de `ej1_split_strategy`: 20 semillas por estrategia, todo dentro de DESARROLLO (figuras `E1-B-b_split_strategy` y `E1-B-b_train_fraction`).

| Estrategia (80/20 o 5-fold) | % de fraude en validación | AP estimado | MSE estimado |
|---|---|---|---|
| Holdout aleatorio | 11.9 ± **0.72** (10.7–13.6) | 0.954 ± **0.0095** | 0.01117 ± 5.5e-4 |
| Holdout estratificado | 11.6 ± **0.28** | 0.955 ± **0.0091** | 0.01080 ± 4.6e-4 |
| 5-fold estratificado (media de los folds) | 11.6 ± 0.32 | 0.954 ± **0.0007** | 0.01098 ± 1e-5 |

- **Estratificar** hace que la validación sea representativa: la tasa de fraude varía 2.6 veces menos. En cambio, reduce poco la varianza de la estimación: el desvío del MSE de validación entre semillas pasa de 5.5e-4 a 4.6e-4.
- **El k-fold es lo que más estabiliza la estimación.** Promedia 5 validaciones que cubren todo DESARROLLO, y el desvío del MSE estimado entre semillas baja de ~5e-4 con un holdout 80/20 a ~1e-5.
- **Fracción de entrenamiento** (holdout estratificado, 50 %–90 %): el val MSE (0.0108–0.0110) y el AP (0.954–0.955) no cambian. Con 10 parámetros, 3000 filas ya alcanzan. Lo que empeora con más entrenamiento es la **incertidumbre de la validación**: el desvío del MSE estimado sube de 3.1e-4 (validación del 50 %) a 7.4e-4 (validación del 10 %).

**¿Cómo se elige el mejor conjunto de entrenamiento?** Tiene que ser uno *representativo* de la distribución: estratificado y lo más grande posible. No el que casualmente da mejor puntaje. Evidencia de `ej1_best_fold` (figura `E1-B-b_best_fold`):
- **Método:** 5 particiones. En cada una se separa un pseudo-test del 20 % de DESARROLLO y se hace 5-fold sobre el resto. Se elige el fold de menor val MSE y se lo compara con un modelo entrenado con todo el resto (durante la mediana de `best_epoch`).
- **El mejor fold promete, en su propia validación, un MSE de 0.01019 ± 0.00033.** En el pseudo-test da **0.01129 ± 0.00037**: la estimación era ~11 % optimista.
- **No generaliza mejor que el resto.** Todo el resto da 0.01133 ± 0.00041 en el pseudo-test y el promedio de los 5 folds, 0.01128. En AP da lo mismo: 0.949 contra 0.949.
- **Conclusión:** el "mejor fold" solo tuvo una validación fácil. Elegir la partición por su puntaje sesga la estimación hacia arriba y no mejora el modelo. Por eso el modelo final se reentrena con todo DESARROLLO.

### c) Mejor modelo y umbral recomendado

**Modelo final del estudio:** perceptrón simple logístico `[9, 1]` (10 parámetros), β = 2, z-score y MSE contra la probabilidad de BigModel. Se entrenó con Adam (η = 1e-3, mini-batch 32) sobre todo DESARROLLO durante 25 épocas (la mediana de `best_epoch`), con 3 semillas: `results/ej1_all_features/ej1_final/final_s{0,1,2}`.

**Umbral: se eligió en validación, sobre las predicciones OOF de `ej1_cv`** (3 semillas × 6000 filas; `criterios.csv`, figuras `E1-B-c_threshold_metrics` y `E1-B-c_pr_roc`):

| Criterio | Umbral | Precision | Recall | Detectados / 1000 | No detectados / 1000 | Falsas alarmas / 1000 |
|---|---|---|---|---|---|---|
| F1 | 0.89 | 0.89 | 0.86 | 99.7 | 15.8 | 12.0 |
| **F2 (recomendado)** | **0.81** | **0.76** | **0.95** | **109.5** | **6.0** | **33.7** |
| Youden | 0.77 | 0.71 | 0.97 | 111.7 | 3.8 | 46.6 |
| Costo mínimo, c_FN/c_FP = 5 | 0.80 | 0.75 | 0.95 | 110.1 | 5.4 | 36.4 |
| Precision máxima con recall ≥ 0.8 | 0.92 | 0.92 | 0.81 | 93.9 | 21.6 | 7.9 |
| Precision máxima con recall ≥ 0.9 | 0.84 | 0.80 | 0.91 | 105.2 | 10.3 | 26.6 |

**Recomendación: umbral 0.81 (criterio F2).**
1. El error caro es dejar pasar un fraude; una alerta falsa es una revisión. F2 prioriza el recall sin necesidad de conocer los costos exactos.
2. Coincide con el umbral de costo mínimo para c_FN/c_FP = 5 (0.80). O sea, equivale a suponer que un fraude cuesta unas 5 alertas falsas.
3. Es estable: elegido en cada uno de los 15 folds da 0.80 ± 0.022 (entre 0.76 y 0.83).
4. Youden (0.77) se descarta: ignora la prevalencia y suma un 38 % más de alertas falsas para atrapar 2 fraudes más cada 1000.

**Sensibilidad al costo** (`costos.csv`, figura `E1-B-c_cost_threshold`), con c_FP = 1:

| c_FN/c_FP | Umbral |
|---|---|
| 1–2 | 0.89 |
| 5 | 0.80 |
| 10 | 0.76 |
| 20 | 0.72 |

Si CompanyX conoce su razón de costos, elige el umbral de esa curva. Todo sale de validación, así que moverse sobre ella no toca el TEST.

**Resultado en TEST** (1500 filas, con el umbral 0.81 fijado de antemano, 3 modelos; figura `E1-B-c_test_confusion`, `results/ej1_all_features/ej1_final/final_eval.json`):

| Métrica | TinyModel (9 col.) | BigModel (umbral 0.85) |
|---|---|---|
| Precision | 0.767 ± 0.004 | 1.000 |
| Recall | 0.938 ± 0.006 | 1.000 |
| F2 | 0.898 ± 0.005 | 1.000 |
| FPR | 0.038 ± 0.001 | 0.000 |
| AP | 0.943 ± 0.001 | 1.000 |
| ROC-AUC | 0.991 ± 0.0002 | 1.000 |
| MSE vs BigModel | 0.01046 ± 3e-5 | — |

- **Matriz de confusión** (media de los 3 modelos): de 176 fraudes, se detectan **165.0** y se escapan **11.0**. De 1324 legítimas, hay **50.0** alertas falsas.
- **Por cada 1000 transacciones:** 117 fraudes, de los cuales se detectan 110 y se escapan 7, con 33 alertas falsas (143 alertas en total).
- **El test confirma lo que se estimó en validación.** El recall pasó de 0.948 (OOF) a 0.938 y la precision de 0.765 a 0.767.

**¿Alcanza la performance de BigModel?** No. BigModel separa las clases perfecto, porque `flagged_fraud` es exactamente su probabilidad > 0.85. El TinyModel reproduce bien la probabilidad (MSE 0.0105) pero no el corte: con un solo producto escalar no puede imitar las reglas duras de la EDA (cantidad > 9, cuenta < 30 días, ...). Sus errores se concentran cerca del corte: el **90 % ± 0.3 %** de los FN y FP en TEST tienen probabilidad de BigModel entre 0.7 y 0.95, una franja donde cae solo el 14 % de las transacciones (`resumen.json › test.mean.errores_cerca_del_corte`). Esto confirma la hipótesis de F09 sobre la zona del corte, aunque A mostró que el mayor error *cuadrático* está entre los no fraudes.

**¿Es tiny?** Figura `E1-B-c_tiny`:
- 10 parámetros.
- `model.npz` de 12.6 KiB, que es casi todo metadata (config y scaler); los pesos son 10 floats.
- Infiere 100 000 transacciones en unos 7 ms en una notebook. El tiempo exacto varía entre corridas y máquinas; la figura muestra la mediana de 7 repeticiones.

#### Modelo recomendado a CompanyX: el de 6 columnas

Se entrega el modelo del control (C.2): logística `[6, 1]`, 7 parámetros, β = 1, η = 3e-3, 26 épocas (`results/ej1_final/final_s{0,1,2}`), con el mismo umbral 0.81.

**Por qué el de 6 y no el de 9.** La decisión sale de validación y de criterios de diseño, nunca del TEST:
1. **En validación rinden igual.** Con la misma config y los mismos folds, las 9 columnas bajan el val MSE solo un 0.6 % (C.0). Con el umbral 0.81, el recall OOF es 0.948 contra 0.947 y las alertas falsas son 33.7 cada 1000 en los dos.
2. **Las 3 columnas extra no aportan.** El modelo de 9 les da un peso ~0 (C.0), que es lo que anticipaba la EDA de F09.
3. **Es más chico:** 7 parámetros contra 10.
4. **No depende del `timestamp`.** Ni el TEST ni los folds miden su riesgo: se arman al azar, así que todas las fechas de validación caen dentro del rango de entrenamiento. En producción, las transacciones nuevas son posteriores y el z-score del `timestamp` crece sin límite.

**El TEST favorece levemente al de 9 columnas, y no se usó para decidir.** Con 9 se escapan 11.0 fraudes de 176, contra 14.3 con 6. Son 3 fraudes, dentro de la variación de un conjunto de 1500 filas, y en validación (6000 filas × 3 semillas) la diferencia no aparece.

**Resultado del modelo recomendado en TEST** (C.2): precision 0.772 ± 0.011, recall 0.919 ± 0.012, F2 0.885 ± 0.011 y AP 0.944 ± 0.003. Por cada 1000 transacciones, detecta 108 de 117 fraudes, se le escapan 10 y genera 32 alertas falsas.

## C · Control: las 6 columnas elegidas en F09

En la EDA de F09 se descartaron `timestamp`, `device_screen_resolution` y `time_since_last_login_s`, porque no tenían relación con el target (|ρ| < 0.03). Fue una decisión del grupo, no de la consigna. Esta sección documenta completo el estudio hecho con esas 6 columnas: C.1 es el aprendizaje y C.2 la generalización. Antes, C.0 lo compara con el estudio principal.

- Configs en `experiments/configs/ej1/*.json`. Resultados en `results/ej1_*`. Figuras en `figures/ej1/{learning,generalization}/`, generadas con los mismos scripts sin `--results-dir`.
- Las configs de las dos versiones son idénticas salvo `features` y `drop`. Comparten la grilla, las semillas, los folds y el TEST.
- El control corrió en serie, así que sus s/época son los que valen para comparar velocidades.

### C.0 · 9 columnas contra 6

**Aprendizaje (A contra C.1)**

| Modelo | MSE train, 9 col. | MSE train, 6 col. | Diferencia |
|---|---|---|---|
| Lineal: mínimos cuadrados (cota exacta) | 0.02606 | 0.02612 | −0.2 % |
| Logística: GD online, η = 1e-4 | 0.01087 | 0.01095 | −0.7 % |
| Logística: Adam, 2500 épocas | 0.01089 | 0.01097 | −0.7 % |

- **Las conclusiones de aprendizaje no cambian.**
  - Las mejores configs son las mismas, y el lineal llega exactamente a mínimos cuadrados.
  - Las salidas fuera de [0, 1] del lineal (6.0 % contra 5.9 %) y la neurona saturada de la logística (6.9 % contra 6.8 %) son casi iguales.
  - Divergen las mismas 5 corridas.
- **Sin normalizar, las 9 columnas son peores.** La logística queda en 0.363 ± 0.085, contra 0.289 ± 0.13, y satura el 100 % de las muestras, contra 99 %–99.99 %. El `timestamp` crudo vale ~1.7·10⁹.
- Agregar columnas solo puede bajar el error de entrenamiento: la cota de mínimos cuadrados con 9 columnas incluye a la de 6. Lo relevante es la validación.

**Generalización: misma config, mismos folds** (comparación apareada en `ej1_cv`, 5 folds × 3 semillas)

| Config | Val MSE, 9 − 6 col. | Relativo | Folds donde 9 col. es mejor |
|---|---|---|---|
| η = 3e-4, β = 1 | −6.2e-5 ± 5.2e-5 | −0.56 % | 14/15 |
| η = 1e-3, β = 1 | −6.8e-5 ± 5.4e-5 | −0.61 % | 14/15 |
| η = 1e-3, β = 2 | −7.3e-5 ± 5.6e-5 | −0.66 % | 13/15 |
| η = 3e-3, β = 1 | −6.7e-5 ± 5.4e-5 | −0.60 % | 13/15 |

- **Hay una mejora consistente, pero de 0.6 %.** Es 8 veces menor que la variación del val MSE entre folds (5.5e-4).
- **Selección.** Con la misma regla, cada versión elige otra config: η = 1e-3 y β = 2 con 9 columnas, η = 3e-3 y β = 1 con 6. En las dos, las 6 configs del barrido son equivalentes.
- **Umbral.** F2 da **0.81** en las dos versiones, con el mismo desempeño en OOF: recall 0.948 contra 0.947 y 33.7 alertas falsas cada 1000 en las dos.
- **Partición y "mejor fold".** Las dos llegan a las mismas conclusiones con los mismos números: k-fold con AP ± 0.0007 contra ± 0.0005, y un mejor fold que es ~10 % optimista.

**TEST** (umbral 0.81, 3 modelos por versión)

| Métrica | 9 col. (B) | 6 col. (C.2) |
|---|---|---|
| MSE vs BigModel | 0.01046 | 0.01055 |
| Precision | 0.767 ± 0.004 | 0.772 ± 0.011 |
| Recall | 0.938 ± 0.006 | 0.919 ± 0.012 |
| F2 | 0.898 ± 0.005 | 0.885 ± 0.011 |
| AP | 0.943 ± 0.001 | 0.944 ± 0.003 |
| Fraudes no detectados (de 176) | 11.0 | 14.3 |
| Alertas falsas (de 1324) | 50.0 | 47.7 |
| Errores cerca del corte (0.7–0.95) | 90 % | 91 % |
| Parámetros | 10 | 7 |

- La diferencia en recall son **3 fraudes de 176**.
- En validación, con el mismo umbral, el recall es idéntico. La diferencia del TEST es la variación de un conjunto de 1500 filas, no una mejora del modelo.
- El AP, que no depende del umbral, es el mismo.
- No se usó para ninguna decisión (ver B-c).

**Qué aprende el modelo de las columnas extra**

Pesos del modelo final de 9 columnas (media de 3 semillas, entradas en z-score). Ese modelo usa β = 2, así que su peso efectivo es el doble del que se muestra. Comparadas así, las 6 columnas compartidas aprenden lo mismo que el modelo de 6 (por ejemplo, `amount_usd`: 2 × 0.393 = 0.786, contra 0.783).

| Columna | Peso (9 col.) |
|---|---|
| `amount_usd` | +0.393 |
| `quantity_purchased` | +0.199 |
| `account_age_days` | −0.158 |
| `days_since_last_purchase` | −0.136 |
| `session_duration_seconds` | −0.110 |
| `items_viewed_before_purchase` | −0.032 |
| `device_screen_resolution` | +0.015 ± 0.004 |
| `timestamp` | −0.003 ± 0.002 |
| `time_since_last_login_s` | +0.001 ± 0.003 |

- **El modelo ignora `timestamp` y `time_since_last_login_s`:** sus pesos no se distinguen de cero.
- **`device_screen_resolution` recibe un peso chico pero estable**, 10 a 25 veces menor que el de las features útiles. Es la única candidata a explicar la mejora de 0.6 %. Es coherente con la EDA, donde la tasa de fraude va de 10.5 % a 12.9 % según la resolución. No se verificó con un experimento.

**Conclusión de la comparación:** las 9 columnas no cambian ninguna conclusión de A ni de B. El modelo confirma por sí solo lo que mostró la EDA, y por eso se recomienda el de 6 (B-c).

### C.1 · Aprendizaje con 6 columnas (F10)

**Setup común**
- Todas las muestras (7500, `split: none`, R-01).
- Las 6 features de F09 con z-score y target `big_model_fraud_probability`, MSE.
- 5 semillas (0–4) en cada config.
- Lineal: `["auto", 1]` con salida `identity`. No lineal: logística de la cátedra $\theta(h) = 1/(1+e^{-2\beta h})$.

Configs en `experiments/configs/ej1/{learning,long,beta,scaling}.json`. Resultados en `results/ej1_{learning,long,beta,scaling}/summary.csv`. Figuras, `tabla.csv` y `resumen.json` en `figures/ej1/learning/`, generados por `python -m analysis.ej1_learning`. Todo número de esta sección sale de esos archivos. Los valores son media ± desvío entre semillas.

**Referencias** (calculadas en el análisis; no son modelos):
- Predictor constante (la media del target): MSE = **0.09148**.
- Mínimos cuadrados con bias sobre las mismas features (`np.linalg.lstsq`), que es la cota exacta del perceptrón lineal: MSE = **0.02612**.

**Barrido `ej1_learning`:** η ∈ {1e-4, 1e-3, 1e-2, 1e-1} × {online, batch}, 500 épocas, la misma grilla para los dos modelos (figura `E1-A_lr_sensitivity`). Mejor config de cada modelo, elegida entre las que convergieron en las 5 semillas:

| Modelo | Mejor config | MSE train | Mejora vs constante | MSE − mínimos cuadrados | Fuera de [0, 1] | Neurona saturada | s/época |
|---|---|---|---|---|---|---|---|
| Lineal | GD batch, η = 0.1 | 0.02612 ± 3e-18 | 71.4 % | 0 (a la precisión impresa) | 5.9 % | — | 0.0011 |
| Logística (β = 1) | GD online, η = 1e-4 | **0.01095** ± 4e-10 | **88.0 %** | −0.0152 | 0.0 % | 6.8 % | 0.78 |

Fuente: `figures/ej1/learning/tabla.csv` y `E1-A-c_summary_table`.

Otros resultados del barrido:
- **Lineal online con η = 0.1:** diverge en las 5 semillas.
- **Batch con η chico:** con 500 épocas no llega a converger (lineal con η = 1e-4: 1.69, peor que el constante). Solo con η = 0.1 alcanza a los mejores.
- **Online:** con η ≤ 1e-3 queda a menos de 1 % del mejor error de cada modelo.

#### a) ¿Observan underfitting?

**Lineal: sí, por su capacidad.**
- Llega **exactamente** a la solución de mínimos cuadrados: 0.02612 contra 0.02612 (figura `E1-A-a_train_mse`). Ningún perceptrón lineal puede bajar de ahí con estas features.
- Mejora 71 % al predictor constante, pero el error que queda es un RMSE de 0.16 en unidades de probabilidad.

**Logística: aprende bastante más, pero también tiene un piso.**
- 0.01095: 88 % mejor que el constante y **58 % por debajo del mejor lineal posible**.
- El error no baja de ahí con ningún η, modo, optimizador ni β ≥ 1 (ver b). RMSE de 0.105.
- Con el criterio de la spec (subajusta si apenas mejora al constante), ninguno de los dos subajusta en forma severa. Pero ninguno imita a BigModel: los dos tienen un error de entrenamiento que no se va a cero, y en el lineal es 2.4 veces mayor.

**Dónde está el error de la logística** (`resumen.json › error_por_clase`, igual en las 5 semillas):
- MSE en los no fraudes: 0.01214. MSE en los fraudes: 0.00192.
- Los fraudes aportan solo el **2 %** del error total. En el lineal aportan el 21 % (MSE en fraudes 0.0471).
- **Esto contradice la hipótesis de F09**, que esperaba el mayor error cerca del corte 0.85. La logística ajusta bien los fraudes, que caen en su zona saturada cerca de 1. Lo que no puede reproducir es la forma de la probabilidad entre 0 y 0.85 de los no fraudes.

**Dato secundario** (umbral 0.5 contra `flagged_fraud`, que es la probabilidad umbralizada en 0.85): el lineal tiene F1 0.586 y la logística 0.498. No sirve para comparar los modelos: el umbral se estudia en F11.

#### b) ¿Observan saturación de las capacidades?

**Sí, en los dos.** Figura `E1-A-b_plateau`.

- **Meseta.** El criterio es una mejora relativa < 0.1 % en el último 10 % de las épocas. Con GD, en 500 épocas, la mejora es 0 en el lineal y 1.2e-7 en la logística: los dos se planchan antes de la época ~50 (`E1-A-a_train_mse`).
- **Más épocas y otro optimizador no cambian nada** (`ej1_long`: Adam, mini-batch 32, 2500 épocas):
  - lineal 0.02623 ± 8e-05, logística 0.01097 ± 7e-06;
  - en el último 10 % el error solo oscila ±0.3 % por el ruido del mini-batch (mejora −0.18 % y −0.05 %).
- **El lineal agotó su capacidad**: alcanza la cota exacta de mínimos cuadrados.
- **La logística llega al mismo piso por caminos distintos**, que tomamos como su mejor error alcanzable:

  | Camino | MSE |
  |---|---|
  | GD online, η = 1e-4 | 0.010955 |
  | GD online, η = 1e-3 | 0.010955 |
  | Adam, 5× épocas | 0.01097 |
  | GD batch, β = 2 | 0.01096 |
  | GD batch, β = 4 | 0.01096 |

- **Salidas no probabilísticas del lineal:** 5.9 % de las salidas caen fuera de [0, 1], entre −0.44 y 2.07 (`E1-A-b_linear_outputs`). La logística, por construcción, nunca sale de (0, 1).

**Saturación de la neurona.** Figura `E1-A-b_saturation_beta`; mide la fracción de muestras con θ′ < 5 % del máximo.

- **Con z-score:** entre 0.7 % (β = 0.25) y 7 % (β = 4). Con β = 1, 6.5 % ± 0.5 %. Son sobre todo fraudes con salida cerca de 1, que es lo correcto.
- **Sin normalizar:**
  - **99 %–99.99 %** de las muestras quedan saturadas con cualquier β;
  - la neurona deja de aprender: el MSE queda en **0.28–0.29**, peor que el predictor constante (0.091);
  - la dispersión entre semillas es grande (± 0.13), porque cada inicialización queda trabada donde cae.
- **β:** con z-score, un β chico no llega a converger en 500 épocas con GD batch (β = 0.25: 0.0261; β = 0.5: 0.0145). Con β ≥ 1 llega al piso (0.01096–0.01107). En el código, β multiplica el gradiente, así que en la práctica también cambia el paso.

**Normalización** (`E1-A-b_scaling`; GD batch, η = 0.1 y 500 épocas para ambos modelos):

| | Sin normalizar | Min-max | Z-score |
|---|---|---|---|
| Lineal | diverge (5/5) | 0.0297 ± 0.0025 | **0.02612** |
| Logística | 0.289 ± 0.13 (saturada) | 0.0399 ± 0.0043 | **0.01107** ± 1e-4 |

Z-score es el mejor para los dos. Confirma la decisión de F09.

**Desvío del plan.** La spec pide correr `ej1_beta` y `ej1_scaling` con el mejor η de `ej1_learning`. Para la logística ese es online (0.78 s/época), y costaba ~1.5 h de cómputo. Se usó **GD batch con η = 0.1**: es la mejor config batch de los dos modelos, con 0.01107 contra el piso de 0.01095 (1 % peor), y tarda 0.001 s/época. Además usa el mismo optimizador para ambos modelos, que es lo que pide la comparación justa.

#### c) ¿Cuál seleccionarían para el estudio de generalización?

**La logística (perceptrón simple no lineal).** Motivos:

1. **Más potencial de aprendizaje.** Su piso de error (0.01095) está 58 % por debajo del mejor lineal *posible* (0.02612). El lineal ya agotó su capacidad y no puede mejorar.
2. **La salida es una probabilidad** (R-05). Tiene imagen en (0, 1), como el target. El lineal devuelve valores fuera de [0, 1] en el 5.9 % de las transacciones, que no se pueden presentar al cliente como probabilidad sin recortarlos.
3. **Es igual de chico que el lineal** (7 parámetros). El costo extra es solo la activación.

**Hiperparámetros de partida para F11** (propuesta, a revisar con el equipo):
- **Modelo:** `["auto", 1]`, salida `sigmoid`, β = 1, z-score, MSE.
- **Optimizador:** Adam con η = 1e-3 y mini-batch 32.
  - Queda a 1 % del piso (0.01097) desde la época 12, a 0.026 s/época. Online con η = 1e-3 llega a lo mismo, pero a 0.78 s/época.
  - Con k-fold y varias semillas en F11, esa diferencia es de ~30×.
  - Si el equipo prefiere quedarse con el algoritmo de la cátedra, GD online con η = 1e-3 da el mismo error.
- **Épocas:** ~300 con early stopping sobre `val_loss`. F11 lo ajusta con validación.
- β ≥ 2 da prácticamente lo mismo que β = 1 (0.01096). No hace falta barrerlo de nuevo.

### C.2 · Generalización y umbral con 6 columnas (F11)

**Setup común**
- **TEST**: 20 % estratificado (1500 filas, `holdout_test.seed = 0`). Los índices están en `results/ej1_split/test_idx.npy`, con sha256 `9e711725e86c18460503685baaf37c87ad5c46e742a5971da5fa33a489943bcc`. Con 6 columnas se leyó **una vez**, en `ej1_final`. La segunda y última lectura es la del estudio principal con 9 columnas (`results/ej1_all_features/ej1_final`, sección B).
- **DESARROLLO**: las otras 6000 filas. Todas las decisiones (config, épocas, umbral) salen de acá.
- **Estratificación**: el runner estratifica por deciles de `big_model_fraud_probability`, no por `flagged_fraud`. Igual conserva la tasa de fraude: 11.73 % en TEST, 11.55 % en DESARROLLO y 11.55 ± 0.24 % en los 15 folds de `ej1_cv`.
- **Modelo**: el de F10 (logística `["auto", 1]`, z-score, MSE contra la probabilidad de BigModel), con Adam, mini-batch 32 y early stopping sobre `val_loss` (patience 30).
- **Etiqueta**: las métricas binarias usan `flagged_fraud`, que se lee por índice de fila y nunca entra al entrenamiento.

Configs en `experiments/configs/ej1/{cv,split_strategy,best_fold,final}.json`; el experimento del mejor fold lo corre `experiments/ej1_best_fold.py`. Figuras, `cv.csv`, `criterios.csv`, `costos.csv` y `resumen.json` en `figures/ej1/generalization/`, generados por `python -m analysis.ej1_generalization`. Todo número de esta sección sale de esos archivos. Los valores son media ± desvío.

#### a) Métricas elegidas

| Métrica | Rol | Por qué |
|---|---|---|
| **Average Precision** (área PR) | Principal, no depende del umbral | Con 11.6 % de positivos, PR mide el desempeño sobre la clase rara. La ROC-AUC da 0.99 y se ve optimista, porque la dominan los 7.6 negativos por cada positivo |
| **Recall** y **precision** en el umbral elegido | Principales operativas | Recall = fraudes detectados (el FN es el error caro). Precision = qué fracción de las alertas es fraude de verdad (el costo de revisar) |
| **F2** | Resumen en un número | Pondera el recall 4 veces más que la precision |
| **Cuentas por cada 1000 transacciones** | Para CompanyX | Traducen las métricas a detectados, no detectados y alertas falsas |
| MSE vs BigModel | Calidad de la destilación | Es lo que optimiza el entrenamiento |
| ROC-AUC, FPR | Secundarias | Comparabilidad |
| Accuracy | Con advertencia | Un modelo que nunca marca fraude tiene accuracy 0.884 |

Selección en `ej1_cv` (5-fold estratificado × 3 semillas; η ∈ {3e-4, 1e-3, 3e-3} × β ∈ {1, 2}; figura `E1-B-a_cv_table`):
- **Las 6 configs son equivalentes.** El val MSE va de 0.011058 a 0.011088. Esa diferencia de 3e-5 es 20 veces menor que el desvío entre folds (5.7e-4). El AP OOF está entre 0.9534 y 0.9538 en todas.
- Por la regla fijada de antemano (menor val MSE media) se eligió **η = 3e-3, β = 1** (`results/ej1_cv/be30e672`), con val MSE **0.01106 ± 0.00057**, AP OOF 0.953 y ROC-AUC OOF 0.992. Es la que converge más rápido (mediana de 26 épocas).
- **No hay sobreajuste:** el train MSE (0.01110) es igual al val MSE. Es esperable con 7 parámetros para 4800 filas.

#### b) Estrategia de partición y "el mejor conjunto de entrenamiento"

**Estrategia:** holdout estratificado para TEST y 5-fold estratificado dentro de DESARROLLO. La evidencia sale de `ej1_split_strategy`: 20 semillas por estrategia, todo dentro de DESARROLLO (figuras `E1-B-b_split_strategy` y `E1-B-b_train_fraction`).

| Estrategia (80/20 o 5-fold) | % de fraude en validación | AP estimado | MSE estimado |
|---|---|---|---|
| Holdout aleatorio | 11.9 ± **0.72** (10.7–13.6) | 0.954 ± **0.0089** | 0.01124 ± 5.7e-4 |
| Holdout estratificado | 11.6 ± **0.28** | 0.955 ± **0.0092** | 0.01088 ± 4.9e-4 |
| 5-fold estratificado (media de los folds) | 11.6 ± 0.32 | 0.954 ± **0.0005** | 0.01105 ± 1e-5 |

- **Estratificar** hace que la validación sea representativa: la tasa de fraude varía 2.6 veces menos. En cambio, reduce poco la varianza de la estimación: el desvío del MSE de validación entre semillas pasa de 5.7e-4 a 4.9e-4.
- **El k-fold es lo que más estabiliza la estimación.** Promedia 5 validaciones que cubren todo DESARROLLO, y el desvío del MSE estimado entre semillas baja de ~5e-4 con un holdout 80/20 a ~1e-5. Con un solo holdout 80/20, el MSE estimado puede salir entre 0.0102 y 0.0125 según la semilla.
- **Fracción de entrenamiento** (holdout estratificado, 50 %–90 %): el val MSE (0.0109–0.0111) y el AP (0.954–0.955) no cambian. Con 7 parámetros, 2400 filas ya alcanzan. Lo que empeora con más entrenamiento es la **incertidumbre de la validación**: el desvío del MSE estimado sube de 3.1e-4 (validación del 50 %) a 7.5e-4 (validación del 10 %).

**¿Cómo se elige el mejor conjunto de entrenamiento?** Tiene que ser uno *representativo* de la distribución: estratificado y lo más grande posible. No el que casualmente da mejor puntaje. Evidencia de `ej1_best_fold` (figura `E1-B-b_best_fold`):
- **Método:** 5 particiones. En cada una se separa un pseudo-test del 20 % de DESARROLLO y se hace 5-fold sobre el resto. Se elige el fold de menor val MSE y se lo compara con un modelo entrenado con todo el resto (durante la mediana de `best_epoch`).
- **El mejor fold promete, en su propia validación, un MSE de 0.01026 ± 0.00033.** En el pseudo-test da **0.01131 ± 0.00036**: la estimación era ~10 % optimista.
- **No generaliza mejor que el resto.** Todo el resto da 0.01135 ± 0.00037 en el pseudo-test y el promedio de los 5 folds, 0.01129. En AP da lo mismo: 0.949 contra 0.948.
- **Conclusión:** el "mejor fold" solo tuvo una validación fácil. Elegir la partición por su puntaje sesga la estimación hacia arriba y no mejora el modelo. Por eso el modelo final se reentrena con todo DESARROLLO.

#### c) Mejor modelo y umbral recomendado

**Modelo final:** perceptrón simple logístico `[6, 1]` (7 parámetros), β = 1, z-score y MSE contra la probabilidad de BigModel. Se entrenó con Adam (η = 3e-3, mini-batch 32) sobre todo DESARROLLO durante 26 épocas (la mediana de `best_epoch`), con 3 semillas: `results/ej1_final/final_s{0,1,2}`.

**Umbral: se eligió en validación, sobre las predicciones OOF de `ej1_cv`** (3 semillas × 6000 filas; `criterios.csv`, figuras `E1-B-c_threshold_metrics` y `E1-B-c_pr_roc`):

| Criterio | Umbral | Precision | Recall | Detectados / 1000 | No detectados / 1000 | Falsas alarmas / 1000 |
|---|---|---|---|---|---|---|
| F1 | 0.89 | 0.89 | 0.86 | 99.5 | 16.0 | 11.9 |
| **F2 (recomendado)** | **0.81** | **0.76** | **0.95** | **109.4** | **6.1** | **33.7** |
| Youden | 0.73 | 0.65 | 0.98 | 113.7 | 1.8 | 62.4 |
| Costo mínimo, c_FN/c_FP = 5 | 0.80 | 0.75 | 0.95 | 109.9 | 5.6 | 36.1 |
| Precision máxima con recall ≥ 0.8 | 0.92 | 0.92 | 0.81 | 93.4 | 22.1 | 8.2 |
| Precision máxima con recall ≥ 0.9 | 0.84 | 0.80 | 0.91 | 105.3 | 10.2 | 26.9 |

**Recomendación: umbral 0.81 (criterio F2).**
1. El error caro es dejar pasar un fraude; una alerta falsa es una revisión. F2 prioriza el recall sin necesidad de conocer los costos exactos.
2. Coincide con el umbral de costo mínimo para c_FN/c_FP = 5 (0.80). O sea, equivale a suponer que un fraude cuesta unas 5 alertas falsas.
3. Es estable: elegido en cada uno de los 15 folds da 0.80 ± 0.025 (entre 0.74 y 0.83).
4. Youden (0.73) se descarta: ignora la prevalencia y casi duplica las alertas falsas.

**Sensibilidad al costo** (`costos.csv`, figura `E1-B-c_cost_threshold`), con c_FP = 1:

| c_FN/c_FP | Umbral |
|---|---|
| 1–2 | 0.89 |
| 5 | 0.80 |
| 10–20 | 0.73 |

Si CompanyX conoce su razón de costos, elige el umbral de esa curva. Todo sale de validación, así que moverse sobre ella no toca el TEST.

**Resultado en TEST** (1500 filas, con el umbral 0.81 fijado de antemano, 3 modelos; figura `E1-B-c_test_confusion`, `results/ej1_final/final_eval.json`):

| Métrica | TinyModel | BigModel (umbral 0.85) |
|---|---|---|
| Precision | 0.772 ± 0.011 | 1.000 |
| Recall | 0.919 ± 0.012 | 1.000 |
| F2 | 0.885 ± 0.011 | 1.000 |
| FPR | 0.036 ± 0.002 | 0.000 |
| AP | 0.944 ± 0.003 | 1.000 |
| ROC-AUC | 0.991 ± 0.0005 | 1.000 |
| MSE vs BigModel | 0.01055 ± 3e-5 | — |

- **Matriz de confusión** (media de los 3 modelos): de 176 fraudes, se detectan **161.7** y se escapan **14.3**. De 1324 legítimas, hay **47.7** alertas falsas.
- **Por cada 1000 transacciones:** 117 fraudes, de los cuales se detectan 108 y se escapan 10, con 32 alertas falsas (140 alertas en total).
- **El test confirma lo que se estimó en validación.** El recall bajó de 0.947 (OOF) a 0.919: equivale a ~5 de los 176 fraudes del test, dentro de lo esperable para un conjunto de ese tamaño. La precision se mantuvo (0.765 → 0.772).

**¿Alcanza la performance de BigModel?** No. BigModel separa las clases perfecto, porque `flagged_fraud` es exactamente su probabilidad > 0.85. El TinyModel reproduce bien la probabilidad (MSE 0.0106) pero no el corte: con un solo producto escalar no puede imitar las reglas duras de la EDA (cantidad > 9, cuenta < 30 días, ...). Sus errores se concentran cerca del corte: el **91 % ± 0.3 %** de los FN y FP en TEST tienen probabilidad de BigModel entre 0.7 y 0.95, una franja donde cae solo el 14 % de las transacciones (`resumen.json › test.mean.errores_cerca_del_corte`). Esto confirma la hipótesis de F09 sobre la zona del corte, aunque F10 mostró que el mayor error *cuadrático* está entre los no fraudes.

**¿Es tiny?** Figura `E1-B-c_tiny`:
- 7 parámetros.
- `model.npz` de 12.4 KiB, que es casi todo metadata (config y scaler); los pesos son 7 floats.
- Infiere 100 000 transacciones en unos 4 ms en una notebook. El tiempo exacto varía entre corridas y máquinas; la figura muestra la mediana de 7 repeticiones.

