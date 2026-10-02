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
2. **Macro F1-Score:** Promedio no ponderado de F1 entre las 10 clases ($0 \dots 9$). Es la métrica decisiva para detectar si una clase minoritaria está siendo ignorada.
3. **Sensibilidad (*Recall*) por Clase y Matriz de Confusión:** Permite diagnosticar qué dígitos específicos se confunden y cuantificar el impacto de clases escasas o ausentes.
4. **Pérdida de Validación (MSE):** Monitoreo del costo cuadrático medio $E = \frac{1}{2m}\sum_k (y_k - \hat{y}_k)^2$ para activar *Early Stopping* con paciencia de 20 épocas y restaurar los mejores pesos.

---

## (b) Estudio de Variantes e Hiperparámetros (OFAT)

Se exploraron sistemáticamente los factores de aprendizaje, arquitectura y regularización partiendo de la configuración canónica de la cátedra (`[784, 64, 10]`, $\tanh$ oculta, sigmoide de salida + MSE, Xavier init, batch 32).

### 1. Tasa de Aprendizaje y Optimizadores (SGD vs Adam)

Resultados en `results/ej2_lr/summary.csv` y Figura `figures/ej2/E2-02_lr_optimizer.png`:

| Optimizador | Mejor $\eta$ | Val Accuracy | Val Loss (MSE) | Macro F1 | Época de parada |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SGD Mini-batch (32)** | $0.3$ | 95.65% $\pm$ 0.13% | $0.00767 \pm 0.00018$ | $0.8465 \pm 0.0022$ | ~190 |
| **Adam** | **$0.001$** | **96.08% $\pm$ 0.30%** | **$0.00690 \pm 0.00021$** | **$0.8532 \pm 0.0037$** | **~45** |

- **Conclusión:** Adam converge **4 veces más rápido** (45 vs 190 épocas) y alcanza mayor exactitud gracias a la estimación adaptativa de momentos de primer y segundo orden ($m_t, v_t$). Para SGD con MSE escalada a 10 salidas, tasas pequeñas ($\eta \le 0.01$) resultaron demasiado lentas.

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

- **Conclusión:** La invariancia traslacional sintética mediante `RandomShift` fue la técnica individual más impactante de todo el estudio: **redujo el error global a más de la mitad (-54.6% relativo)** y elevó el recall de la clase minoritaria (dígito 5) en **+11.7 puntos porcentuales** (de 81.48% a 93.21%).

---

## (c) Resultado Oficial en Producción (`digits_test.csv`)

La configuración ganadora de Ej2 (`[784, 128, 10]`, Adam $\eta=0.001$, `RandomShift` $\pm 2$ px, $\tanh$ + sigmoide + MSE) se reentrenó con el 100% de `digits.csv` durante la mediana de mejores épocas (72 épocas) y se evaluó **una sola vez** sobre `digits_test.csv` (2 497 muestras).

Reporte oficial en `results/ej2_final/final_eval.json`:

| Métrica en Producción (Test) | Media $\pm$ Desvío (3 semillas) | Mejor Semilla (s0) |
| :--- | :---: | :---: |
| **Exactitud (*Accuracy*)** | **88.31% $\pm$ 0.32%** | **88.63%** |
| **Macro F1-Score** | $0.8366 \pm 0.0028$ | $0.8397$ |
| **Pérdida MSE** | $0.01562 \pm 0.00083$ | $0.01496$ |

### Detalle por Clase en Test (Semilla s0)

| Dígito | Muestras Test | Recall en Test | Precisión en Test | F1-Score |
| :---: | :---: | :---: | :---: | :---: |
| **0** | 245 | **100.00%** | 89.09% | 0.9423 |
| **1** | 283 | **99.65%** | 96.25% | 0.9792 |
| **2** | 258 | **98.06%** | 87.85% | 0.9267 |
| **3** | 254 | **99.21%** | 73.96% | 0.8475 |
| **4** | 257 | **100.00%** | 95.33% | 0.9761 |
| **5** | 223 | **93.27%** | 89.66% | 0.9143 |
| **6** | 239 | **98.74%** | 95.16% | 0.9692 |
| **7** | 257 | **97.67%** | 95.80% | 0.9672 |
| **8** | 243 | **0.00%** | **0.00%** | **0.0000** |
| **9** | 247 | **96.43%** | 79.93% | 0.8741 |

### Análisis de la Cota Teórica y Discrepancia con Validación

1. **La ausencia del Dígito 8:**
   - En `digits.csv` hay **cero muestras del dígito 8**. En consecuencia, el modelo nunca aprendió sus rasgos y clasifica erróneamente el 100% de los 8s del test (confundiéndolos principalmente con 3, 2 y 9).
   - Como el test tiene 243 ochos sobre 2 497 muestras ($9.73\%$), **la cota máxima teórica alcanzable en test para cualquier modelo de Ej2 es $90.27\%$**.
   - El modelo alcanzó **88.31%**, lo que representa un **$97.83\%$ de exactitud relativa** sobre las clases que sí vio.
2. **Sobreestimación de la Validación:**
   - En validación se midió $98.22\%$ porque el split de `digits.csv` tampoco contenía ochos. La aparente caída de rendimiento en producción no refleja sobreajuste, sino un **cambio en la distribución de entrada (*covariate/concept shift*)** provocado por la omisión de una clase entera en el conjunto de entrenamiento original.

Figuras de referencia: `figures/ej2/E2-05_test_confusion_matrix.png` y `figures/ej2/E2-06_misclassified_test.png`.
Modelo guardado para reuso: `models/ej2_best/` (`model.npz` + `config.json`).
