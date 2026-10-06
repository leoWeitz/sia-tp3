# Ej2 · Clasificación de Dígitos con MLP

## (a) Protocolo de Evaluación y Métricas

### 1. Protocolo Experimental (Constitución C5, R-02)

Para garantizar mediciones insesgadas y prevenir fugas de información (*data leakage*), el conjunto de datos se divide rigurosamente en tres etapas:

```
digits.csv (12 449 muestras)
   ├── 80% (9 959) ──► TRAIN  (ajuste de pesos sinápticos vía Backpropagation)
   └── 20% (2 490) ──► VAL    (selección de hiperparámetros, early stopping)

digits_test.csv (2 497 muestras)
   └── AISLADO ──────► TEST   (evaluación final de producción, leída una sola vez)
```

- **Estratificación por clase:** Debido al fuerte desbalance en `digits.csv` (el dígito 5 tiene solo 271 muestras mientras que otros tienen ~1 600), la partición 80/20 se realiza de forma **estratificada** por clase.
- **Reproducibilidad:** Cada experimento se evalúa sobre al menos **3 semillas independientes** (`seeds: [0, 1, 2]`), reportando siempre `media ± desvío estándar`.
- **Test Sagrado:** `digits_test.csv` nunca interviene en ninguna decisión de diseño, ajuste de gradientes ni parada temprana; se evalúa exclusivamente al final del estudio mediante `--final-eval`.

Figura de referencia: `figures/ej2/E2-01_evaluation_protocol.png`.

### 2. Métricas Seleccionadas

1. **Exactitud Global (*Accuracy*):** Métrica principal exigida para el problema.
2. **Macro F1-Score:** Promedio no ponderado de F1 entre las 10 clases ($0 \dots 9$). Es la métrica decisiva para detectar si una clase minoritaria está siendo ignorada. En validación queda en $\approx 0.85$ porque el dígito 8, que no existe en `digits.csv`, entra al promedio con F1 $= 0$: sobre las 9 clases presentes equivale a $\tfrac{10}{9}$ de ese valor ($0.853 \rightarrow 0.948$).
3. **Sensibilidad (*Recall*) por Clase y Matriz de Confusión:** Permite diagnosticar qué dígitos específicos se confunden y cuantificar el impacto de clases escasas o ausentes.
4. **Pérdida de Validación (MSE):** Monitoreo del costo cuadrático medio $L = \frac{1}{n\,m}\sum_{\mu}\sum_k (y_k^\mu - \hat{y}_k^\mu)^2$ (promedio sobre las $n$ muestras y las $m = 10$ salidas, sin el $\tfrac12$ de la teórica: `docs/04-matematica.md` §2.4) para activar *Early Stopping* con paciencia de 20 épocas y restaurar los mejores pesos.

---

## (b) Estudio de Variantes e Hiperparámetros (OFAT)

El estudio se hizo en dos etapas:

1. **Barrido por bloques** (secciones 1–3): optimizador × $\eta$ con la red fija, después arquitectura y después augmentation, partiendo de la configuración canónica de la cátedra (`[784, 64, 10]`, $\tanh$ oculta, sigmoide de salida + MSE, Xavier init, batch 32).
2. **Grilla completa** (sección 4): optimizador × $\eta$ × arquitectura × augmentation cruzados, para medir las interacciones que el barrido por bloques no ve. La sección 5 justifica fijar $\tanh$ en la grilla. **La configuración final sale de la grilla** (sección c).

### 1. Tasa de Aprendizaje y Optimizadores

Barrido de $\eta$ para SGD y Adam en `results/ej2_lr/summary.csv` (Figura `figures/ej2/E2-02_lr_optimizer.png`); mini-barridos de Momentum y RMSProp en `results/ej2_opt/summary.csv` y de SGD con $\eta$ adaptativo en `results/ej2_eta_adapt/summary.csv`. Todos comparten red, datos, lote, early stopping y semillas; solo cambia el mecanismo de optimización.

Mejor configuración de cada mecanismo (mayor exactitud media de validación entre los valores probados), Figura `figures/ej2/E2-02b_optimizers.png`:

| Optimizador | Mejor $\eta$ (valores probados) | Val Accuracy | Val Loss (MSE) | Macro F1 | Mejor época | Épocas hasta early stopping |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SGD Mini-batch (32)** | $0.3$ (8) | 95.65% $\pm$ 0.13% | $0.00738 \pm 0.00044$ | $0.8511 \pm 0.0017$ | $191 \pm 20$ | $211 \pm 20$ |
| **Momentum ($\alpha = 0.9$)** | $0.1$ (3) | 95.90% $\pm$ 0.18% | $0.00720 \pm 0.00034$ | $0.8534 \pm 0.0044$ | $86 \pm 23$ | $106 \pm 23$ |
| **RMSProp** | $0.001$ (3) | **96.20% $\pm$ 0.22%** | **$0.00675 \pm 0.00026$** | **$0.8554 \pm 0.0008$** | **$42 \pm 6$** | **$62 \pm 6$** |
| **Adam** | $0.001$ (8) | 96.08% $\pm$ 0.30% | $0.00690 \pm 0.00021$ | $0.8532 \pm 0.0037$ | $46 \pm 8$ | $66 \pm 8$ |
| **SGD + $\eta$ adaptativo** | $\eta_0 = 0.1$; $a = 0.05$, $b = 0.5$, $k = 3$, $k' = 2$ (2) | 95.66% $\pm$ 0.04% | $0.00732 \pm 0.00043$ | $0.8515 \pm 0.0018$ | $115 \pm 6$ | $135 \pm 6$ |

Media $\pm$ desvío entre 3 semillas. "Épocas hasta early stopping" es la época en la que cortó el entrenamiento (paciencia 20, máximo 300); "Mejor época" es la de menor pérdida de validación, cuyos pesos se restauran.

Detalle de los mini-barridos (exactitud de validación; épocas hasta el corte):

- **Momentum ($\alpha = 0.9$):** $\eta = 0.01$ → 95.54% $\pm$ 0.18% (300, sin corte); $\eta = 0.03$ → 95.88% $\pm$ 0.09% ($212 \pm 18$); $\eta = 0.1$ → 95.90% $\pm$ 0.18% ($106 \pm 23$).
- **RMSProp:** $\eta = 0.0001$ → 95.89% $\pm$ 0.21% ($206 \pm 38$); $\eta = 0.0003$ → 96.01% $\pm$ 0.08% ($98 \pm 14$); $\eta = 0.001$ → 96.20% $\pm$ 0.22% ($62 \pm 6$).
- **SGD + $\eta$ adaptativo ($\eta_0 = 0.1$):** $a = 0.01$, $b = 0.1$, $k = 5$, $k' = 3$ → 95.65% $\pm$ 0.10% ($247 \pm 21$), $\eta$ final entre 0.47 y 0.58; $a = 0.05$, $b = 0.5$, $k = 3$, $k' = 2$ → 95.66% $\pm$ 0.04% ($135 \pm 6$), $\eta$ final entre 1.18 y 1.35. Con el mismo $\eta = 0.1$ fijo, SGD llega a 95.48% $\pm$ 0.10% sin cortar en 300 épocas.

- **Conclusión:** El mecanismo de optimización cambia sobre todo la **velocidad de convergencia** y poco la exactitud final: los cinco quedan en un rango de 0.55 puntos (95.65%–96.20%). Los métodos con tasa adaptativa por parámetro (RMSProp y Adam) cortan en **≈ 3 veces menos épocas** que SGD (62–66 vs 211) y obtienen la mayor exactitud; entre ellos la diferencia (0.12 puntos) es menor que un desvío, así que con 3 semillas no se distinguen. Momentum reduce las épocas de SGD a la mitad (106 vs 211). El $\eta$ adaptativo de la cátedra iguala al mejor $\eta$ fijo de SGD (95.66% vs 95.65%) en un 36% menos de épocas (135 vs 211) sin haber ajustado $\eta$ a mano: partiendo de $\eta_0 = 0.1$ lo sube hasta $\approx 1.25$ en promedio. Para SGD con MSE escalada a 10 salidas, tasas pequeñas ($\eta \le 0.01$) resultaron demasiado lentas.
- **Limitaciones:** El mejor $\eta$ de SGD, Momentum y RMSProp es el mayor de su grilla (y el $\eta$ adaptativo termina por encima de 0.3), así que su óptimo puede estar más arriba; solo el de Adam es interior. Los tiempos de pared de `ej2_opt` y `ej2_eta_adapt` no son comparables con los de `ej2_lr` (otra máquina, y dentro de `ej2_opt` el s/época varió entre 0.5 y 8.6), por eso se comparan épocas y no segundos.

### 2. Variación de Arquitectura: Ancho vs Profundidad

Resultados en `results/ej2_arch_width/` y `results/ej2_arch_depth/`, Figura `figures/ej2/E2-03_architecture.png`:

| Tipo | Configuración | Parámetros | Val Accuracy | Val Loss | Macro F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Ancho | [784, 16, 10] | 12 730 | 93.92% $\pm$ 0.22% | 0.01042 | 0.8314 |
| Ancho | [784, 32, 10] | 25 450 | 95.26% $\pm$ 0.28% | 0.00802 | 0.8446 |
| Ancho (Base) | [784, 64, 10] | 50 890 | 96.08% $\pm$ 0.30% | 0.00690 | 0.8532 |
| Ancho | [784, 128, 10] | 101 770 | 96.44% $\pm$ 0.06% | 0.00618 | 0.8602 |
| Ancho | [784, 256, 10] | 203 530 | 96.83% $\pm$ 0.28% | 0.00556 | 0.8625 |
| Profundidad | [784, 64, 64, 10] | 55 050 | 96.53% $\pm$ 0.26% | 0.00582 | 0.8607 |
| Profundidad | [784, 128, 64, 10] | 109 386 | 96.72% $\pm$ 0.14% | 0.00545 | 0.8637 |
| Profundidad | [784, 256, 128, 10] | 235 146 | 96.79% $\pm$ 0.14% | 0.00525 | 0.8619 |
| Profundidad | **[784, 128, 64, 32, 10]** | 111 146 | **96.92% $\pm$ 0.16%** | **0.00523** | **0.8654** |

- **Conclusión:** Aumentar la capacidad mejora monótonamente el ajuste. La pirámide profunda de 4 capas `[784, 128, 64, 32, 10]` superó a una red ancha simple de más de 200k parámetros, demostrando la eficiencia del aprendizaje jerárquico de representaciones.

### 3. Técnicas Extras: Data Augmentation

Resultados en `results/ej2_extra_augmentation/summary.csv`, Figura `figures/ej2/E2-04_augmentation_impact.png`:

| Variante | Val Accuracy | Val Loss | Recall Dígito 5 (Escaso) | Recall Dígito 0 |
| :--- | :---: | :---: | :---: | :---: |
| **Sin Augmentation (Base [128])** | 96.44% $\pm$ 0.06% | 0.00618 | 81.48% | 98.09% |
| **Ruido Gaussiano ($\sigma=0.05$)** | 96.73% $\pm$ 0.13% | 0.00583 | 82.10% | 97.97% |
| **RandomShift ($\pm 1$ px)** | 97.88% $\pm$ 0.06% | 0.00375 | 91.36% | 98.99% |
| **RandomShift ($\pm 2$ px)** | **98.22% $\pm$ 0.35%** | **0.00377** | **93.21%** | **99.10%** |

- **Conclusión:** La invariancia traslacional sintética mediante `RandomShift` fue la técnica individual más impactante de todo el estudio: **redujo el error global a la mitad ($-50\%$ relativo contra la base `[128]` sin augmentation: de 3.56% a 1.78%)** y elevó el recall de la clase minoritaria (dígito 5) en **+11.7 puntos porcentuales** (de 81.48% a 93.21%).

### 4. Grilla Completa: Optimizador × $\eta$ × Arquitectura × Augmentation

El barrido por bloques supone que los factores no interactúan, y fijó cada bloque con lo mejor del anterior (por ejemplo, el augmentation solo se probó sobre `[128]`). Para medir las interacciones se cruzaron todos (`experiments/configs/ej2/combo.json`, `results/ej2_combo/summary.csv`):

- **Optimizador y $\eta$ (12):** SGD $\{0.1, 0.3, 1\}$, Momentum ($\alpha = 0.9$) $\{0.03, 0.1, 0.3\}$, RMSProp $\{3\cdot10^{-4}, 10^{-3}, 3\cdot10^{-3}\}$, Adam $\{3\cdot10^{-4}, 10^{-3}, 3\cdot10^{-3}\}$. Cada optimizador con su propia escala de $\eta$.
- **Arquitectura (5):** `[64]`, `[128]`, `[256]`, `[128, 64, 32]`, `[256, 128]`.
- **Augmentation (3):** sin shift, `RandomShift` $\pm 1$ px, $\pm 2$ px.
- **Fijo:** $\tanh$ + Xavier, sigmoide de salida + MSE, lote 32, early stopping (paciencia 20, máximo 300 épocas).

Son 180 configuraciones × 3 semillas = **540 corridas, todas `ok`** (ninguna divergió). Usan los mismos splits que las secciones 1–3: las configuraciones repetidas dan exactamente los mismos números.

Figuras y tablas en `figures/ej2/grid/`, generadas por `python -m analysis.ej2_grid` (`ranking.csv` con las 180 configuraciones, `efectos.csv`).

**Las 5 mejores** (exactitud de validación, media $\pm$ desvío entre 3 semillas; Figura `E2-G1_ranking`):

| # | Optimizador | Red | Augmentation | Val Accuracy | Épocas hasta early stopping |
| :---: | :--- | :--- | :--- | :---: | :---: |
| 1 | **Adam $\eta = 3\cdot10^{-4}$** | **`[256, 128]`** | **shift $\pm 2$ px** | **98.69% $\pm$ 0.34%** | 148 |
| 2 | SGD $\eta = 1$ | `[256, 128]` | shift $\pm 2$ px | 98.62% $\pm$ 0.37% | 154 |
| 3 | RMSProp $\eta = 3\cdot10^{-4}$ | `[256, 128]` | shift $\pm 2$ px | 98.59% $\pm$ 0.21% | 152 |
| 4 | RMSProp $\eta = 10^{-3}$ | `[256]` | shift $\pm 1$ px | 98.53% $\pm$ 0.22% | 107 |
| 5 | Momentum $\eta = 0.1$ | `[256, 128]` | shift $\pm 2$ px | 98.49% $\pm$ 0.41% | 139 |

La configuración elegida en el barrido por bloques (Adam $\eta = 10^{-3}$, `[128]`, $\pm 2$ px) da 98.22% $\pm$ 0.35% en la grilla.

**Cuánto mueve la exactitud cada factor** (para cada nivel, la mejor configuración optimizando el resto; Figura `E2-G2_factor_effect`):

| Factor | Peor nivel | Mejor nivel | Diferencia |
| :--- | :--- | :--- | :---: |
| Augmentation | sin shift (97.04%) | shift $\pm 2$ px (98.69%) | **1.65 puntos** |
| Arquitectura | `[64]` (97.97%) | `[256, 128]` (98.69%) | 0.72 puntos |
| Optimizador, con su mejor $\eta$ | Momentum (98.49%) | Adam (98.69%) | 0.20 puntos |
| Activación oculta, con su mejor $\eta$ (sección 5) | $\tanh$ (98.54%) | ReLU (98.63%) | 0.09 puntos |

**Efecto de cada factor con el resto fijo en la ganadora:**

- **Optimizador y $\eta$** (`[256, 128]`, $\pm 2$ px; Figura `E2-G3_optimizer_lr`). Cada optimizador tiene su escala: Adam y RMSProp rinden en $\sim 10^{-3}$, Momentum en $\sim 10^{-1}$ y SGD en $\sim 1$. Con su mejor $\eta$, los cuatro quedan entre 98.49% y 98.69%, y cortan entre 139 y 155 épocas. La ventaja de velocidad de Adam y RMSProp de la sección 1 (≈ 3 veces menos épocas, medida en `[64]` sin augmentation) **no se repite** en esta red con shift: Adam solo corta antes (66 épocas) con $\eta = 3\cdot10^{-3}$, perdiendo 1.4 puntos (97.28%).
- **Arquitectura y augmentation** (Adam $\eta = 3\cdot10^{-4}$; Figura `E2-G4_arch_aug`). Con shift $\pm 2$, pasar de `[64]` a `[256, 128]` suma 0.78 puntos (97.91% → 98.69%). El shift (el mejor de $\pm 1$ y $\pm 2$) suma entre 1.8 y 2.0 puntos en cualquier red: `[64]` con shift (97.91%) supera a `[256, 128]` sin shift (96.83%). $\pm 1$ y $\pm 2$ empatan en casi todas las redes; en `[256, 128]` gana $\pm 2$ (98.69% contra 98.38%).
- **Augmentation como regularización** (Adam $\eta = 3\cdot10^{-4}$, `[256, 128]`; Figura `E2-G6_augmentation`):

  | Augmentation | Train Accuracy | Val Accuracy | Brecha train − val | Recall del 5 (val) |
  | :--- | :---: | :---: | :---: | :---: |
  | Sin shift | 99.82% | 96.83% | 3.00 | 84.0% |
  | Shift $\pm 1$ px | 99.89% | 98.38% | 1.51 | 95.1% |
  | **Shift $\pm 2$ px** | 99.79% | **98.69%** | **1.10** | **96.9%** |

  El train queda en ≈ 99.8% en los tres casos; lo que se achica es la brecha con validación.

- **Conclusión:** el orden de importancia es augmentation ≫ arquitectura > optimizador > activación. La grilla confirma lo del barrido por bloques (el shift es lo que más rinde, el optimizador casi no cambia la exactitud final) y encuentra una configuración ≈ 0.5 puntos mejor, que el barrido por bloques no había probado porque nunca cruzó `[256, 128]` con shift.
- **Limitaciones:** las 10 mejores configuraciones están dentro de 0.3 puntos, del orden del desvío entre semillas (0.29 en promedio en el top 10): la grilla alcanza para elegir, no para ordenarlas con certeza. Al elegir la mejor de 180 sobre la misma validación, su exactitud queda algo optimista. El mejor $\eta$ de SGD (1) es el mayor de su grilla, así que su óptimo puede estar más arriba.

### 5. Función de Activación Oculta

Barrido en `experiments/configs/ej2/activation.json` (`results/ej2_activation/summary.csv`): $\tanh$ (Xavier), sigmoide (Xavier) y ReLU (He) × 6 redes (`[64]`, `[128]`, `[256]`, `[128, 64]`, `[256, 128]`, `[128, 64, 32]`) × Adam $\eta \in \{10^{-4}, 10^{-3}, 10^{-2}\}$, con shift $\pm 2$ px y 3 semillas (162 corridas, todas `ok`). Figura `figures/ej2/grid/E2-G5_activation.png`.

- **Con el mejor $\eta$ de cada una, las tres llegan al mismo techo:** ReLU 98.63%, sigmoide 98.55%, $\tanh$ 98.54% (mejor red de cada una). Red por red, la diferencia máxima entre activaciones es 0.45 puntos (en `[64]`: ReLU 97.93%, sigmoide 97.48%) y siempre queda dentro de las barras de error.
- **Lo que sí cambia es la tolerancia a un $\eta$ grande:** con $\eta = 10^{-2}$ las tres empeoran mucho, y ReLU es la que peor queda (media 45% entre redes, contra 67% de $\tanh$ y 75% de sigmoide).
- **Por eso la grilla de la sección 4 fija $\tanh$.** Solo se probó con Adam y shift $\pm 2$.

---

## (c) Resultado Oficial en Producción (`digits_test.csv`)

**Hubo dos evaluaciones finales**, cada una con la configuración elegida solo por validación en su momento. El test no participó de ninguna decisión: la segunda evaluación se hizo porque la grilla (sección b.4) eligió, por validación, otra configuración.

| | 1ª evaluación (barrido por bloques) | **2ª evaluación (grilla completa) — oficial** |
| :--- | :---: | :---: |
| Configuración | `[784, 128, 10]`, Adam $\eta = 10^{-3}$, shift $\pm 2$ | **`[784, 256, 128, 10]`, Adam $\eta = 3\cdot10^{-4}$, shift $\pm 2$** |
| Elegida desde | `results/ej2_extra_augmentation/f5f0383c` | `results/ej2_combo/fcdee1d5` |
| Val Accuracy (selección) | 98.22% $\pm$ 0.35% | 98.69% $\pm$ 0.34% |
| Épocas de reentrenamiento (mediana de `best_epoch`) | 72 | 141 |
| **Test Accuracy** | 88.31% $\pm$ 0.32% | **88.68% $\pm$ 0.10%** |
| Test por semilla (s0, s1, s2) | 88.63%, 88.31%, 87.99% | 88.79%, 88.59%, 88.67% |
| Test Macro F1 | 0.8366 $\pm$ 0.0028 | 0.8398 $\pm$ 0.0010 |
| Test MSE | 0.01562 $\pm$ 0.00083 | 0.01505 $\pm$ 0.00020 |
| Reporte | `results/ej2_final/final_eval.json` | `results/ej2_final_grid/final_eval.json` |

En ambos casos: $\tanh$ + Xavier, sigmoide + MSE, lote 32, reentrenamiento con el 100% de `digits.csv` sin validación ni early stopping, y **una sola** lectura de `digits_test.csv` (2 497 muestras), con 3 semillas.

### Detalle por Clase en Test (2ª evaluación, media de 3 semillas)

| Dígito | Muestras Test | Recall en Test | Precisión en Test | F1-Score |
| :---: | :---: | :---: | :---: | :---: |
| **0** | 245 | **99.32%** | 88.55% | 0.9361 |
| **1** | 283 | **99.65%** | 97.14% | 0.9837 |
| **2** | 258 | **97.80%** | 87.96% | 0.9261 |
| **3** | 252 | **99.47%** | 75.66% | 0.8594 |
| **4** | 245 | **98.78%** | 95.55% | 0.9713 |
| **5** | 223 | **95.37%** | 88.64% | 0.9187 |
| **6** | 239 | **98.19%** | 92.32% | 0.9515 |
| **7** | 257 | **98.31%** | 95.01% | 0.9663 |
| **8** | 243 | **0.00%** | **0.00%** | **0.0000** |
| **9** | 252 | **96.83%** | 81.46% | 0.8847 |

Frente a la 1ª evaluación, la mejora (+0.37 puntos) está sobre todo en el dígito 5 (recall 92.68% → 95.37%), y el desvío entre semillas baja de 0.32 a 0.10.

### Análisis de la Cota Teórica y Discrepancia con Validación

1. **La ausencia del Dígito 8:**
   - En `digits.csv` hay **cero muestras del dígito 8**. En consecuencia, el modelo nunca aprendió sus rasgos y clasifica erróneamente el 100% de los 8s del test (en la semilla 0, principalmente como 3, 9, 2 y 0: 71, 61, 25 y 24 de 243).
   - Como el test tiene 243 ochos sobre 2 497 muestras ($9.73\%$), **la cota máxima teórica alcanzable en test para cualquier modelo de Ej2 es $90.27\%$**.
   - El modelo alcanzó **88.68%**, lo que representa un **$98.24\%$ de exactitud** sobre las clases que sí vio (97.83% en la 1ª evaluación).
2. **Sobreestimación de la Validación:**
   - En validación se midió $98.69\%$ porque el split de `digits.csv` tampoco contenía ochos. La aparente caída de rendimiento en producción no refleja sobreajuste, sino que **el entrenamiento no es representativo: no tiene ningún 8**, y el test sí (243 de 2 497 muestras).
3. **La grilla mejora poco en test.** Medio punto de validación se tradujo en 0.37 puntos de test: con el 8 ausente, el límite lo ponen los datos, no la configuración.

Figuras de referencia (2ª evaluación): `figures/ej2/grid/E2-05_test_confusion_matrix.png` y `figures/ej2/grid/E2-06_misclassified_test.png`. Las de la 1ª evaluación siguen en `figures/ej2/`.
Modelo guardado para reuso: `models/ej2_best/` (`model.npz` + `config.json`), copia de `results/ej2_final_grid/final_s0` (2ª evaluación, semilla 0: 88.79% en test).
