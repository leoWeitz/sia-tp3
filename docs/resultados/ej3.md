# Ej3 · More Digits: Superando la Meta del 98 %

## (a) Resultado Oficial en Test (`digits_test.csv`)

### 1. Cumplimiento de la Meta

**Sí, se superó el objetivo de exactitud $\ge 98.0\%$ en el conjunto de test sagrado.**

La evaluación final oficial (`--final-eval` sobre `digits_test.csv`, 2 497 muestras, evaluada una sola vez con 5 semillas independientes) arrojó:

| Métrica Oficial en Test | Media $\pm$ Desvío (5 semillas) | Mejor Semilla (s1) |
| :--- | :---: | :---: |
| **Exactitud (*Accuracy*)** | **98.09% $\pm$ 0.22%** | **98.36%** |
| **Macro F1-Score** | **0.9806 $\pm$ 0.0023** | **0.9833** |
| **Pérdida (MSE)** | **0.003366 $\pm$ 0.000257** | **0.002985** |

Detalle de exactitud por semilla en test:
- **Semilla 0:** 97.96%
- **Semilla 1:** **98.36%** (Mejor modelo guardado en `models/ej3_best/`)
- **Semilla 2:** 98.12%
- **Semilla 3:** 97.80%
- **Semilla 4:** 98.24%

### 2. Configuración Completa del Modelo Ganador

El modelo se mantiene **100% fiel al marco canónico de la cátedra** (sin Softmax y sin Cross-Entropy):

- **Arquitectura:** `[784, 256, 128, 10]` (Perceptrón Multicapa de 2 capas ocultas).
- **Activación Capas Ocultas:** `ReLU` ($f(z) = \max(0, z)$).
- **Inicializador de Pesos:** `He` normal ($\sigma = \sqrt{2 / n_{in}}$), específico para ReLU.
- **Capa de Salida:** `Sigmoide` logística ($\beta = 1.0$).
- **Función de Costo:** `MSE` (error cuadrático medio).
- **Optimizador:** `Adam` ($\eta = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$).
- **Tamaño de Lote:** Mini-batch de 32 muestras.
- **Data Augmentation:** `RandomShift` ($\pm 1$ píxel en $x$ e $y$).
- **Épocas de Reentrenamiento:** 45 épocas (mediana de parada temprana de validación).
- **Dataset de Entrenamiento:** Unión deduplicada `digits_union.csv` (24 501 muestras).

Modelo persistido en: `models/ej3_best/` (`model.npz` + `config.json`).

### 3. Desempeño por Clase en Test

Promedio sobre las 5 semillas en `digits_test.csv`:

| Dígito | Muestras Test | Recall en Test | Precisión en Test | F1-Score |
| :---: | :---: | :---: | :---: | :---: |
| **0** | 245 | **99.51%** | 97.95% | 0.9871 |
| **1** | 283 | **99.93%** | 99.30% | 0.9961 |
| **2** | 258 | **98.22%** | 98.75% | 0.9848 |
| **3** | 252 | **99.37%** | 96.54% | 0.9793 |
| **4** | 245 | **98.61%** | 99.02% | 0.9881 |
| **5** | 223 | **96.50%** | 98.02% | 0.9724 |
| **6** | 239 | **98.66%** | 98.17% | 0.9841 |
| **7** | 257 | **97.98%** | 96.70% | 0.9733 |
| **8** | 243 | **94.07%** | **99.56%** | **0.9674** |
| **9** | 252 | **97.62%** | 97.08% | 0.9735 |

Todas las clases superan el 94% de sensibilidad, y el dígito 8 pasó de 0.00% en Ej2 a **94.07% de recall con 99.56% de precisión**.

Figuras de referencia: `figures/ej3/E3-01_accuracy_progression.png` y `figures/ej3/E3-04_test_confusion_matrix.png`.

---

## (b) Estudio de Ablación: Contribución de Cada Técnica

Sobre la unión deduplicada se barrió red $\times$ data augmentation (`results/ej3_search/summary.csv`). Exactitud de **validación**, media $\pm$ desvío entre 3 semillas (Figura `figures/ej3/E3-02_ablation_ranking.png`):

| Red (capas ocultas, inicialización) | Sin shift | RandomShift $\pm 1$ px | RandomShift $\pm 2$ px |
| :--- | :---: | :---: | :---: |
| **tanh `[128]`** (Xavier) | 96.92% $\pm$ 0.44% | 98.11% $\pm$ 0.43% | 97.95% $\pm$ 0.31% |
| **ReLU `[128]`** (He) | 97.19% $\pm$ 0.57% | 98.30% $\pm$ 0.36% | 98.01% $\pm$ 0.47% |
| **ReLU `[256, 128]`** (He) | 97.65% $\pm$ 0.52% | **98.48% $\pm$ 0.24%** | 98.41% $\pm$ 0.19% |

La celda de arriba a la izquierda es la línea base (`results/ej3_baseline`); la resaltada es la configuración elegida, que se volvió a correr con 5 semillas (`results/ej3_best`: 98.53% $\pm$ 0.22%). El barrido incluye una cuarta red, ReLU `[128, 64, 32]`, que no entra en la grilla porque fue inestable entre semillas (89.99% $\pm$ 6.48% sin shift, 94.29% $\pm$ 6.26% con $\pm 2$ px).

### Camino de la Línea Base a la Configuración Elegida

Un cambio por vez; las ganancias están en puntos porcentuales de exactitud de validación:

| Paso | Val Accuracy | Ganancia del paso | Acumulado |
| :--- | :---: | :---: | :---: |
| **Base:** tanh `[128]`, sin shift | 96.92% | — | — |
| **1.** tanh $\rightarrow$ ReLU (con He) | 97.19% | +0.27 | +0.27 |
| **2.** `[128]` $\rightarrow$ `[256, 128]` | 97.65% | +0.46 | +0.73 |
| **3.** + RandomShift $\pm 1$ px | **98.48%** | +0.83 | **+1.56** |

### Lectura de la Grilla

1. **RandomShift $\pm 1$ px es el factor de mayor efecto en las tres redes:** +1.19 (tanh `[128]`), +1.11 (ReLU `[128]`) y +0.83 puntos (ReLU `[256, 128]`). Es el único cuyo efecto supera con claridad el desvío entre semillas (0.2 a 0.6 puntos). Pasar a $\pm 2$ px no mejora en ninguna red ($-0.16$, $-0.29$ y $-0.07$).
2. **Mayor capacidad (`[128]` $\rightarrow$ `[256, 128]`, con ReLU):** +0.46 sin shift y +0.18 con $\pm 1$ px.
3. **ReLU en lugar de $\tanh$ (en `[128]`):** +0.27 sin shift y +0.19 con $\pm 1$ px. Cambia junto con la inicialización (Xavier $\rightarrow$ He), así que el efecto es del par.
4. **Los aportes no son aditivos.** Sumar los efectos individuales medidos sobre la base ($1.19 + 0.46 + 0.27 = 1.92$) sobreestima la mejora real (+1.56): el shift aporta menos cuanto mejor es la red.
5. **Con 3 semillas, las diferencias entre redes (0.2 a 0.5 puntos) son del orden de un desvío:** alcanzan para elegir una configuración, no para ordenar las arquitecturas con certeza.

---

## (c) Factores Externos: Dataset vs Técnicas (E3-c)

### 1. Composición y Calidad de los Datasets

| Dataset | Filas | Dígito 8 | Dígito 5 | Duplicados con Test |
| :--- | :---: | :---: | :---: | :---: |
| `digits.csv` | 12 449 | **0 (0.0%)** | 271 (2.2%) | 0 |
| `more_digits.csv` | 15 741 | 585 (3.7%) | 542 (3.4%) | 0 |
| **Unión Deduplicada** | **24 501** | **585 (2.4%)** | **785 (3.2%)** | **0** |
| `digits_test.csv` | 2 497 | 243 (9.7%) | 223 (8.9%) | — |

- **Deduplicación:** Se identificaron exactamente **3 689 imágenes idénticas** compartidas entre `digits.csv` y `more_digits.csv`. Su deduplicación fue crítica para evitar que la misma imagen cayera simultáneamente en entrenamiento y validación.
- **Cero Fuga hacia Test:** Se verificó formalmente mediante hashes SHA-256 de píxeles que **ninguna imagen de `digits_test.csv` existe en los conjuntos de desarrollo**.

### 2. Cuánto de la Mejora es "Dato" vs "Técnica"

En `digits_test.csv` la exactitud pasó de **88.31%** (Ej2 Final, entrenado con `digits.csv`) a **98.09%** (Ej3 Final, entrenado con la unión): **+9.79 puntos**. Como la exactitud es el promedio de los recalls por clase ponderado por la cantidad de muestras, esa diferencia se descompone de forma exacta por dígito:

$$\Delta\text{acc} = \sum_c \frac{n_c}{N}\,\bigl(\text{recall}_c^{\text{Ej3}} - \text{recall}_c^{\text{Ej2}}\bigr)$$

con $N = 2\,497$ y los recalls de `results/ej2_final/final_eval.json` y `results/ej3_final/final_eval.json`, promediados entre semillas (`analysis.ej3.decompose_test_gain`):

| Dígito | $n_c$ | Recall Ej2 | Recall Ej3 | Aporte a $\Delta$acc (puntos) |
| :---: | :---: | :---: | :---: | :---: |
| 0 | 245 | 99.73% | 99.51% | $-0.02$ |
| 1 | 283 | 99.18% | 99.93% | +0.09 |
| 2 | 258 | 98.19% | 98.22% | +0.00 |
| 3 | 252 | 98.54% | 99.37% | +0.08 |
| 4 | 245 | 99.05% | 98.61% | $-0.04$ |
| 5 | 223 | 92.68% | 96.50% | +0.34 |
| 6 | 239 | 98.05% | 98.66% | +0.06 |
| 7 | 257 | 98.05% | 97.98% | $-0.01$ |
| **8** | 243 | **0.00%** | **94.07%** | **+9.15** |
| 9 | 252 | 96.30% | 97.62% | +0.13 |
| **Dígito 8** | 243 | | | **+9.15** |
| **Los otros nueve** | 2 254 | | | **+0.63** |
| **Total** (exactitud) | 2 497 | 88.31% | 98.09% | **+9.79** |

Los aportes están redondeados a dos decimales; sin redondear son +9.155 y +0.633, que suman +9.788.

- **Factor dato, medido:** +9.15 de los +9.79 puntos vienen del dígito 8, que `digits.csv` no tiene y `more_digits.csv` sí (585 muestras). Ninguna técnica podía recuperarlos: en Ej2 el recall del 8 es 0 por construcción y el techo era 90.27%.
- **El resto, +0.63 puntos, mezcla dato y técnica.** Entre Ej2 Final y Ej3 Final cambian a la vez los datos de los otros nueve dígitos (más muestras: el 5 pasa de 271 a 785 y es el que más aporta, +0.34) y el modelo (ReLU `[256, 128]` con shift $\pm 1$ px en lugar de $\tanh$ `[128]` con shift $\pm 2$ px). No se evaluó en test un modelo intermedio, así que esos +0.63 no se pueden repartir entre las dos causas.
- **Lo que sí se midió de la técnica está en validación** (sección b): sobre la unión, la configuración elegida supera a la línea base por +1.56 puntos (96.92% $\rightarrow$ 98.48%), y la línea base queda por debajo del 98%.

Figura de referencia: `figures/ej3/E3-03_per_class_recall_comparison.png`.
Ejemplos difíciles analizados: `figures/ej3/E3-05_hard_examples.png`.
