# Ej3 · More Digits: Superando la Meta del 98 %

El Ej3 se hizo en dos etapas sobre la misma unión deduplicada (`datasets/digits_union.csv`, sección c.1):

1. **Búsqueda anterior** (sección b.2): red × augmentation con Adam $\eta = 10^{-3}$ fijo. Su ganadora se evaluó en test (1ª evaluación, 98.09%).
2. **Grilla nueva** (sección b.1), diseñada con lo aprendido en la grilla completa del Ej2 (`resultados/ej2.md` §b.4–b.5): activación × arquitectura × $\eta$ con el shift fijo en $\pm 1$ px, que la búsqueda anterior ya había elegido sobre estos datos. Su ganadora es la **configuración oficial** (2ª evaluación, 98.33%).

## (a) Resultado Oficial en Test (`digits_test.csv`)

### 1. Cumplimiento de la Meta

**Sí, se superó el objetivo de exactitud $\ge 98.0\%$, en las 5 semillas.**

Las dos evaluaciones finales usan la configuración elegida solo por validación en su momento. El test no participó de ninguna decisión: la 2ª evaluación se hizo porque la grilla nueva eligió, por validación, otra configuración.

| | 1ª evaluación (búsqueda anterior) | **2ª evaluación (grilla nueva) — oficial** |
| :--- | :---: | :---: |
| Configuración | ReLU `[784, 256, 128, 10]`, Adam $\eta = 10^{-3}$, shift $\pm 1$ | **ReLU `[784, 256, 10]`, Adam $\eta = 10^{-3}$, shift $\pm 1$** |
| Elegida desde | `results/ej3_best/21ef8017` | `results/ej3v2_grid/6a80e9d4` |
| Val Accuracy (selección) | 98.53% $\pm$ 0.22% (5 semillas) | 98.65% $\pm$ 0.32% (3 semillas) |
| Épocas de reentrenamiento (mediana de `best_epoch`) | 45 | 43 |
| **Test Accuracy** | 98.09% $\pm$ 0.22% | **98.33% $\pm$ 0.23%** |
| Test por semilla (s0 … s4) | 97.96, 98.36, 98.12, 97.80, 98.24 | 98.52, 98.16, 98.28, 98.60, 98.08 |
| Semillas con test $\ge 98\%$ | 3 de 5 | **5 de 5** |
| Test Macro F1 | 0.9806 $\pm$ 0.0023 | 0.9830 $\pm$ 0.0023 |
| Test MSE | 0.003366 $\pm$ 0.000257 | 0.003724 $\pm$ 0.000177 |
| Reporte | `results/ej3_final/final_eval.json` | `results/ej3v2_final/final_eval.json` |

En ambos casos: salida logística ($\beta = 1$) + MSE, ReLU con inicialización He, lote 32, reentrenamiento con el 100% de la unión (24 501 imágenes) sin validación ni early stopping, y **una sola** lectura de `digits_test.csv` (2 497 muestras), con 5 semillas.

### 2. Configuración Oficial

El modelo se mantiene en el marco de la cátedra (salida logística + MSE, sin softmax ni entropía cruzada); ReLU y He se reportan como variantes (constitución C3):

- **Arquitectura:** `[784, 256, 10]` (una capa oculta de 256 neuronas, 203 530 parámetros).
- **Activación oculta:** ReLU, con inicialización He ($\sigma = \sqrt{2 / n_{in}}$).
- **Salida:** logística ($\beta = 1$). **Pérdida:** MSE.
- **Optimizador:** Adam ($\eta = 10^{-3}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$), lote 32.
- **Data augmentation:** `RandomShift` $\pm 1$ px.
- **Épocas de reentrenamiento:** 43 (mediana de `best_epoch` de las 3 corridas de validación: 43, 69 y 36).
- **Config:** `experiments/configs/ej3/final_v2.json`.

`models/ej3_best/` **corresponde todavía a la 1ª evaluación**; falta reemplazarlo por `results/ej3v2_final/final_s3/model.npz` (la mejor semilla en test, 98.60%) o por la semilla 0.

### 3. Desempeño por Clase en Test (2ª evaluación, media de 5 semillas)

| Dígito | Muestras Test | Recall en Test | Precisión en Test | F1-Score |
| :---: | :---: | :---: | :---: | :---: |
| **0** | 245 | **99.43%** | 98.07% | 0.9874 |
| **1** | 283 | **99.93%** | 99.02% | 0.9947 |
| **2** | 258 | **98.60%** | 98.99% | 0.9880 |
| **3** | 252 | **99.52%** | 97.39% | 0.9844 |
| **4** | 245 | **98.53%** | 99.59% | 0.9906 |
| **5** | 223 | **96.05%** | 98.35% | 0.9718 |
| **6** | 239 | **98.66%** | 98.25% | 0.9846 |
| **7** | 257 | **98.44%** | 97.39% | 0.9791 |
| **8** | 243 | **95.14%** | **99.32%** | **0.9718** |
| **9** | 252 | **98.41%** | 97.03% | 0.9772 |

Todas las clases superan el 95% de recall; el dígito 8 pasa de 0% en Ej2 a **95.14% de recall con 99.32% de precisión** (94.07% en la 1ª evaluación).

Figuras (`python -m analysis.ej3_grid`): `figures/ej3/grid/E3-G6_test_confusion.png` (matriz media de los 5 modelos) y `figures/ej3/grid/E3-G8_hard_examples.png` (errores de la mejor semilla). Las de la 1ª evaluación siguen en `figures/ej3/`.

---

## (b) Búsqueda del Modelo

### 1. Grilla Nueva: Activación × Arquitectura × $\eta$

Diseño (`experiments/configs/ej3/grid.json`, `results/ej3v2_grid/`), a partir de la grilla completa del Ej2:

- **Activación oculta (2):** $\tanh$ + Xavier, ReLU + He.
- **Arquitectura (4):** `[128]`, `[256]`, `[128, 64, 32]`, `[256, 128]`.
- **$\eta$ de Adam (2):** $3\cdot10^{-4}$, $10^{-3}$. El Ej2 mostró que el optimizador mueve ≤ 0.2 puntos con su mejor $\eta$, así que se fija Adam.
- **Fijo:** `RandomShift` $\pm 1$ px (elegido sobre estos datos por la búsqueda anterior, b.2), salida logística + MSE, lote 32, early stopping (paciencia 20, máximo 300 épocas), holdout estratificado 80/20.

Son 16 configuraciones × 3 semillas = **48 corridas, todas `ok`**. Por plazo, la grilla se recortó durante la corrida: se sacaron $\eta = 10^{-4}$ (la más lenta, nunca la mejor en el Ej2), "sin shift" y $\pm 2$ px. Las corridas de esos niveles que llegaron a terminar quedan en `results/ej3v2_grid/`, pero no forman configuraciones completas y `analysis.ej3_grid` las descarta.

**Exactitud de validación** (media $\pm$ desvío entre 3 semillas; Figuras `E3-G1_ranking` y `E3-G4_arch_activation`):

| Red | $\tanh$, $\eta = 3\cdot10^{-4}$ | $\tanh$, $\eta = 10^{-3}$ | ReLU, $\eta = 3\cdot10^{-4}$ | ReLU, $\eta = 10^{-3}$ |
| :--- | :---: | :---: | :---: | :---: |
| `[128]` | 98.27 $\pm$ 0.27 | 98.27 $\pm$ 0.27 | 98.29 $\pm$ 0.44 | 98.24 $\pm$ 0.24 |
| `[128, 64, 32]` | 98.28 $\pm$ 0.42 | 98.10 $\pm$ 0.16 | 98.43 $\pm$ 0.27 | 98.27 $\pm$ 0.09 |
| `[256]` | 98.62 $\pm$ 0.26 | 98.38 $\pm$ 0.22 | 98.59 $\pm$ 0.28 | **98.65 $\pm$ 0.32** |
| `[256, 128]` | 98.55 $\pm$ 0.39 | 98.34 $\pm$ 0.22 | 98.63 $\pm$ 0.06 | 98.56 $\pm$ 0.22 |

**Cuánto mueve cada factor** (para cada nivel, la mejor configuración optimizando el resto; Figura `E3-G2_factor_effect`):

| Factor | Peor nivel | Mejor nivel | Diferencia |
| :--- | :--- | :--- | :---: |
| Arquitectura | `[128]` (98.29%) | `[256]` (98.65%) | **0.36 puntos** |
| Activación oculta | $\tanh$ (98.62%) | ReLU (98.65%) | 0.03 puntos |
| $\eta$ de Adam | $3\cdot10^{-4}$ (98.63%) | $10^{-3}$ (98.65%) | 0.01 puntos |

**Lectura:**

1. **Las 16 configuraciones superan el 98% en validación**, y las 6 mejores están dentro de 0.10 puntos, menos que el desvío entre semillas: empate técnico.
2. **Lo único que pesa es el tamaño de la red:** una capa de 256 neuronas rinde ≈ 0.36 puntos más que una de 128; agregar una segunda capa no suma.
3. **Activación y $\eta$ no cambian el resultado** con el resto en su mejor valor, igual que en el Ej2. El efecto de la activación en una red dada depende de $\eta$ (Figura `E3-G3_activation_lr`): con `[256]`, $\tanh$ rinde mejor con $\eta = 3\cdot10^{-4}$ y ReLU con $10^{-3}$.
4. **La red `[128, 64, 32]` ya no colapsa.** En la búsqueda anterior perdía una clase entera en 3 de 9 corridas (b.2); con shift $\pm 1$ y cualquiera de los dos $\eta$ es estable (desvío entre 0.09 y 0.42).

**El modelo del Ej2 con los datos nuevos.** Como referencia, se entrenó la configuración oficial del Ej2 sin cambios ($\tanh$ `[256, 128]`, Adam $\eta = 3\cdot10^{-4}$, shift $\pm 2$ px; `experiments/configs/ej3/ej2_config.json`, `results/ej3v2_ej2_config/`) sobre la unión, con las mismas particiones:

| | Validación en la unión | Por semilla (s0, s1, s2) | Recall del 8 (val) |
| :--- | :---: | :---: | :---: |
| Modelo del Ej2, sin cambios | **98.69% $\pm$ 0.25%** | 98.47, 98.65, 98.96 | 94.9% |
| Ganadora de la grilla | 98.65% $\pm$ 0.32% | 98.29, 98.76, 98.90 | 93.7% |

La diferencia (0.04 puntos) es mucho menor que el desvío, y semilla a semilla cada modelo gana en alguna: **empatan**. El modelo del Ej2 ya estaba en el techo con los datos nuevos; ninguna combinación de la grilla lo supera. Esta corrida se hizo **después** de elegir la ganadora y no participó de la selección (no estaba en la grilla, que fija shift $\pm 1$); por validación habría empatado.

**Referencia en test.** Para medir el efecto del dato en test con el mismo modelo, se evaluó también esta configuración (`experiments/configs/ej3/final_ej2_config.json`, 96 épocas, 5 semillas; `results/ej3v2_final_ej2_config/final_eval.json`). Es una **referencia**, no un candidato: la configuración oficial ya estaba elegida por validación y no se cambia por este resultado.

| Test (`digits_test.csv`, 5 semillas) | Accuracy | Por semilla | Recall del 8 |
| :--- | :---: | :---: | :---: |
| Modelo del Ej2 en `digits.csv` (Ej2 oficial) | 88.68% $\pm$ 0.10% | 88.79, 88.59, 88.67 (3 semillas) | 0.00% |
| **Mismo modelo, entrenado con la unión** | **98.25% $\pm$ 0.16%** | 98.28, 98.40, 98.40, 98.16, 98.04 | 95.56% |
| Ganadora de la grilla (Ej3 oficial) | 98.33% $\pm$ 0.23% | 98.52, 98.16, 98.28, 98.60, 98.08 | 95.14% |

Con el mismo modelo, solo cambiar los datos lleva el test de 88.68% a 98.25% (+9.57 puntos, las 5 semillas sobre 98%). La grilla del Ej3 suma +0.07 puntos más, menos que el desvío entre semillas.

- **Limitaciones:** con 3 semillas, la grilla alcanza para elegir, no para ordenar las mejores configuraciones. La configuración `21ef8017` (ganadora de la búsqueda anterior) dio 98.56% en esta grilla y 98.48% en la anterior, con las mismas semillas: la diferencia probablemente viene de la cantidad de hilos de NumPy o de la máquina, y queda dentro del desvío.

### 2. Búsqueda Anterior: Red × Augmentation

Sobre la unión deduplicada se barrió red $\times$ data augmentation con Adam $\eta = 10^{-3}$ (`results/ej3_search/summary.csv`). Exactitud de **validación**, media $\pm$ desvío entre 3 semillas (Figura `figures/ej3/E3-02_ablation_ranking.png`):

| Red (capas ocultas, inicialización) | Sin shift | RandomShift $\pm 1$ px | RandomShift $\pm 2$ px |
| :--- | :---: | :---: | :---: |
| **tanh `[128]`** (Xavier) | 96.92% $\pm$ 0.44% | 98.11% $\pm$ 0.43% | 97.95% $\pm$ 0.31% |
| **ReLU `[128]`** (He) | 97.19% $\pm$ 0.57% | 98.30% $\pm$ 0.36% | 98.01% $\pm$ 0.47% |
| **ReLU `[256, 128]`** (He) | 97.65% $\pm$ 0.52% | **98.48% $\pm$ 0.24%** | 98.41% $\pm$ 0.19% |

La celda de arriba a la izquierda es la línea base (`results/ej3_baseline`); la resaltada es la configuración elegida, que se volvió a correr con 5 semillas (`results/ej3_best`: 98.53% $\pm$ 0.22%) y dio la 1ª evaluación. El barrido incluye una cuarta red, ReLU `[128, 64, 32]`, que fue inestable entre semillas (89.99% $\pm$ 6.48% sin shift, 94.29% $\pm$ 6.26% con $\pm 2$ px): en 3 de sus 9 corridas una salida logística quedó saturada en 0 y la red no predice nunca una clase (por ejemplo, el dígito 2 en `5dceb290_s0`: recall 0% en validación y exactitud de train de solo 88%). Es una falla de optimización, no de capacidad; con shift $\pm 1$ no aparece (b.1).

**El augmentation como regularización** (ReLU `[256, 128]`, $\eta = 10^{-3}$; Figura `figures/ej3/grid/E3-G5_augmentation.png`):

| Augmentation | Train Accuracy | Val Accuracy | Brecha train − val | Recall del 5 (val) | Recall del 8 (val) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Sin shift | 99.87% | 97.65% | 2.22 | 95.1% | 88.3% |
| **Shift $\pm 1$ px** | 99.69% | **98.48%** | 1.21 | 94.5% | **94.3%** |
| Shift $\pm 2$ px | 99.23% | 98.41% | 0.82 | 95.5% | 92.0% |

$\pm 2$ px achica más la brecha, pero no mejora la validación y le baja el recall al 8; por eso la grilla nueva fija $\pm 1$.

**Lectura de la búsqueda anterior:**

1. **RandomShift $\pm 1$ px es el factor de mayor efecto en las tres redes:** +1.19 (tanh `[128]`), +1.11 (ReLU `[128]`) y +0.83 puntos (ReLU `[256, 128]`). Es el único cuyo efecto supera con claridad el desvío entre semillas (0.2 a 0.6 puntos). Pasar a $\pm 2$ px no mejora en ninguna red ($-0.16$, $-0.29$ y $-0.07$).
2. **Mayor capacidad (`[128]` $\rightarrow$ `[256, 128]`, con ReLU):** +0.46 sin shift y +0.18 con $\pm 1$ px.
3. **ReLU en lugar de $\tanh$ (en `[128]`):** +0.27 sin shift y +0.19 con $\pm 1$ px. Cambia junto con la inicialización (Xavier $\rightarrow$ He), así que el efecto es del par. La grilla nueva lo mide en las cuatro redes y con su mejor $\eta$: la diferencia desaparece (0.03 puntos).
4. **Los aportes no son aditivos.** Sumar los efectos individuales medidos sobre la base ($1.19 + 0.46 + 0.27 = 1.92$) sobreestima la mejora real (+1.56): el shift aporta menos cuanto mejor es la red.

---

## (c) Factores Externos: Dataset vs Técnicas (E3-c)

### 1. Composición y Calidad de los Datasets

| Dataset | Filas | Dígito 8 | Dígito 5 | Duplicados con Test |
| :--- | :---: | :---: | :---: | :---: |
| `digits.csv` | 12 449 | **0 (0.0%)** | 271 (2.2%) | 0 |
| `more_digits.csv` | 15 741 | 585 (3.7%) | 542 (3.4%) | 0 |
| **Unión Deduplicada** | **24 501** | **585 (2.4%)** | **785 (3.2%)** | **0** |
| `digits_test.csv` | 2 497 | 243 (9.7%) | 223 (8.9%) | — |

- **Deduplicación:** Se identificaron exactamente **3 689 imágenes idénticas** compartidas entre `digits.csv` y `more_digits.csv`. Su deduplicación fue crítica para evitar que la misma imagen cayera simultáneamente en entrenamiento y validación. La unión se regenera con `python -m data.build_digits_union`, que verifica que el archivo sea idéntico byte a byte al usado en las corridas.
- **Cero Fuga hacia Test:** Se verificó formalmente mediante hashes SHA-256 de píxeles que **ninguna imagen de `digits_test.csv` existe en los conjuntos de desarrollo**.

### 2. Cuánto de la Mejora es "Dato" vs "Técnica"

En `digits_test.csv` la exactitud pasó de **88.68%** (Ej2 oficial, `results/ej2_final_grid`, entrenado con `digits.csv`) a **98.33%** (Ej3 oficial, `results/ej3v2_final`, entrenado con la unión): **+9.65 puntos**. Como la exactitud es el promedio de los recalls por clase ponderado por la cantidad de muestras, esa diferencia se descompone de forma exacta por dígito:

$$\Delta\text{acc} = \sum_c \frac{n_c}{N}\,\bigl(\text{recall}_c^{\text{Ej3}} - \text{recall}_c^{\text{Ej2}}\bigr)$$

con $N = 2\,497$ y los recalls promediados entre semillas (`analysis.ej3_grid.decompose_test_gain`, tabla en `figures/ej3/grid/descomposicion.csv`; Figura `E3-G7_per_class_test`):

| Dígito | $n_c$ | Recall Ej2 | Recall Ej3 | Aporte a $\Delta$acc (puntos) |
| :---: | :---: | :---: | :---: | :---: |
| 0 | 245 | 99.32% | 99.43% | +0.01 |
| 1 | 283 | 99.65% | 99.93% | +0.03 |
| 2 | 258 | 97.80% | 98.60% | +0.08 |
| 3 | 252 | 99.47% | 99.52% | +0.01 |
| 4 | 245 | 98.78% | 98.53% | $-0.02$ |
| 5 | 223 | 95.37% | 96.05% | +0.06 |
| 6 | 239 | 98.19% | 98.66% | +0.05 |
| 7 | 257 | 98.31% | 98.44% | +0.01 |
| **8** | 243 | **0.00%** | **95.14%** | **+9.26** |
| 9 | 252 | 96.83% | 98.41% | +0.16 |
| **Dígito 8** | 243 | | | **+9.26** |
| **Los otros nueve** | 2 254 | | | **+0.39** |
| **Total** (exactitud) | 2 497 | 88.68% | 98.33% | **+9.65** |

- **Factor dato, medido:** +9.26 de los +9.65 puntos vienen del dígito 8, que `digits.csv` no tiene y `more_digits.csv` sí (585 muestras). Ninguna técnica podía recuperarlos: en Ej2 el recall del 8 es 0 por construcción y el techo era 90.27%.
- **El resto, +0.39 puntos, mezcla dato y técnica:** entre los dos modelos oficiales cambian a la vez los datos de los otros nueve dígitos y la configuración.
- **Separación medida en test, con el modelo intermedio** (el modelo del Ej2 entrenado con la unión, b.1):

  | Paso | Test | Δ | Del dígito 8 | Del resto |
  | :--- | :---: | :---: | :---: | :---: |
  | Ej2 oficial (`digits.csv`) | 88.68% | — | — | — |
  | **Solo dato:** mismo modelo, entrenado con la unión | 98.25% | **+9.57** | +9.30 | +0.27 (sobre todo 5: +0.16 y 9: +0.12) |
  | **Solo técnica:** grilla del Ej3, mismos datos | 98.33% | +0.07 | −0.04 | +0.11 |

  **Casi toda la mejora es el dato** (+9.57 de +9.65 puntos). Los ochos solos llevan el test a 97.98%, apenas debajo del 98%; lo que cruza el 98% son las mejoras chicas en los otros dígitos, que también vienen de los datos nuevos (el 5 pasa de 271 a 785 ejemplos). El cambio de configuración del Ej3 no suma más que el desvío entre semillas, igual que en validación (98.69% contra 98.65%).
- **El desplazamiento de la imagen** es lo que pone a este tipo de modelo en el nivel del 98%, pero ya venía del Ej2: sobre la unión, la misma red da 97.65% de validación sin shift y 98.48% con $\pm 1$ (b.2).

Con la 1ª evaluación (Ej2 88.31% → Ej3 98.09%), la misma descomposición daba +9.15 puntos del dígito 8 y +0.63 del resto (`figures/ej3/E3-03_per_class_recall_comparison.png`).
