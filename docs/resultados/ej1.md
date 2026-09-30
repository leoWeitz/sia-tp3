# Ej1 · Resultados

## A · Aprendizaje: perceptrón lineal vs no lineal (F10)

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

### a) ¿Observan underfitting?

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

### b) ¿Observan saturación de las capacidades?

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

### c) ¿Cuál seleccionarían para el estudio de generalización?

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

## B · Generalización y umbral (F11)

*Pendiente (F11).*
