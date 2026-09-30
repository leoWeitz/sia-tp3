# digits.csv y digits_test.csv — diccionario de datos y exploración

Documento de referencia para el Ejercicio 2 (F12) y Ejercicio 3 (F13). Los números y figuras provienen de la ejecución de `python -m analysis.ej2_eda` y se encuentran respaldados en `figures/ej2/eda/resumen.json`, `class_balance.csv` y `pixel_stats.csv`.

---

## 1. Identificación de los archivos

| Archivo | Filas | Columnas | sha256 | Rol |
|---|---|---|---|---|
| `digits.csv` | 12 449 | `label`, `image` (784 floats) | `11271dbc12a3a8db967f6c2708e55ae1918d69186e81d91c53c24e305fc5d3f4` | Entrenamiento y ajuste de hiperparámetros (desarrollo) |
| `digits_test.csv` | 2 497 | `label`, `image` (784 floats) | `56e2ca2d55a7a646fa8883a0c85b8138bbc2b4536438c60cb6bcab02f57f8e55` | Evaluación final en producción (**sagrado**, solo `--final-eval`) |
| `more_digits.csv` | 15 741 | `label`, `image` (784 floats) | `e0d401943cdf210c03f2bf31fefa299d0c6c7d9dc0bf8c88b65341a75fa2ce22` | Dataset ampliado para F13 (Ej3) |

- **Formato:** Cada fila contiene un entero `label` $\in \{0, \dots, 9\}$ y un texto `image` compuesto por 784 números flotantes en $[0, 1]$ que representan una imagen en escala de grises de $28 \times 28$ píxeles aplanada por filas.
- **Carga:** Se procesa mediante `data.loaders.load_digits_csv`, el cual genera un caché en `.npz` validando el hash sha256 para acelerar lecturas posteriores.

---

## 2. Balance de clases

Distribución de muestras por dígito en cada archivo (`figures/ej2/eda/EDA-01_class_balance.png` y `class_balance.csv`):

| Dígito | `digits.csv` (train) | Proporción | `digits_test.csv` (test) | Proporción | `more_digits.csv` | Proporción |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0** | 1 480 | 11.89 % | 245 | 9.81 % | 1 776 | 11.28 % |
| **1** | 1 685 | 13.54 % | 283 | 11.33 % | 2 022 | 12.85 % |
| **2** | 1 489 | 11.96 % | 258 | 10.33 % | 1 787 | 11.35 % |
| **3** | 1 532 | 12.31 % | 252 | 10.09 % | 1 839 | 11.68 % |
| **4** | 1 460 | 11.73 % | 245 | 9.81 % | 1 752 | 11.13 % |
| **5** | **271** | **2.18 %** | 223 | 8.93 % | 542 | 3.44 % |
| **6** | 1 479 | 11.88 % | 239 | 9.57 % | 1 775 | 11.28 % |
| **7** | 1 566 | 12.58 % | 257 | 10.29 % | 1 879 | 11.94 % |
| **8** | **0** | **0.00 %** | 243 | 9.73 % | 585 | 3.72 % |
| **9** | 1 487 | 11.94 % | 252 | 10.09 % | 1 784 | 11.33 % |
| **Total** | **12 449** | 100.0 % | **2 497** | 100.0 % | **15 741** | 100.0 % |

### Observaciones críticas:
1. **Ausencia total del dígito 8 en train:** `digits.csv` no contiene ningún ejemplo del dígito 8. En cambio, `digits_test.csv` contiene 243 ejemplos de 8 (9.73 % del conjunto de prueba).
   - **Consecuencia teórica ineludible:** Un modelo entrenado exclusivamente con `digits.csv` nunca ha observado la clase 8 ni sus pesos hacia la neurona 8 han recibido gradientes positivos. Predecirá erróneamente todas las muestras de la clase 8 en el test.
   - **Techo de desempeño en producción:** La máxima accuracy alcanzable en `digits_test.csv` está acotada por:
     $$\text{Accuracy}_{\max} = \frac{2497 - 243}{2497} = \frac{2254}{2497} \approx 90.27 \%$$
   - La métrica de validación (obtenida de particionar `digits.csv`, donde no hay 8) sobreestimará la capacidad de acierto del modelo en el test de producción. Esta discrepancia no es un defecto del modelo, sino una propiedad fundamental del dataset requerida por el enunciado para comparar con el Ejercicio 3.
2. **Escasez del dígito 5:** En `digits.csv`, el dígito 5 cuenta con apenas 271 ejemplos (2.18 % del dataset), mientras que las restantes clases presentes promedian 1 500 muestras.
   - Exige que cualquier partición de validación sea **estratificada** para garantizar que los pocos dígitos 5 queden proporcionalmente representados tanto en el conjunto de entrenamiento como en validación.

---

## 3. Características de los píxeles y normalización

Estadísticos globales calculados sobre las 12 449 imágenes de `digits.csv` (`pixel_stats.csv` y `figures/ej2/eda/EDA-04_pixel_variance_heatmaps.png`):
- **Rango:** Mínimo exacto $0.0$, Máximo exacto $1.0$.
- **Media global:** $0.1286$, Desvío estándar global: $0.3064$.
- **Píxeles con varianza nula (bordes muertos):** Exactamente **97 de los 784 píxeles** (12.37 %) tienen varianza $0.0$ a lo largo de todas las imágenes. Corresponden a los márgenes perimetrales exteriores de las imágenes de $28 \times 28$.
- **Decisión de preprocesamiento:**
  - Dado que los píxeles ya vienen normalizados en $[0, 1]$, la configuración default es `dataset.normalize: "none"`.
  - Aplicar `MinMaxScaler` por feature provocaría divisiones por cero en los 97 píxeles perimetrales (lo cual es prevenido por el scaler asignándoles $0$, pero no aporta beneficio al estar ya en $[0, 1]$).
  - En la fase de variantes (`ej2_extra`) se evaluará `zscore` como alternativa.

---

## 4. Análisis de duplicados y solapamiento

Verificación de unicidad mediante hashes binarios exactos (`resumen.json › duplicates`):
- **Duplicados intra-dataset:**
  - `digits.csv`: 0 duplicados.
  - `digits_test.csv`: 0 duplicados.
  - `more_digits.csv`: 0 duplicados.
- **Solapamiento inter-dataset:**
  - `digits.csv` vs `digits_test.csv`: **0 muestras solapadas**. El conjunto de test es estrictamente disjunto del conjunto de desarrollo (integridad de evaluación en producción garantizada).
  - `digits.csv` vs `more_digits.csv`: **3 689 muestras idénticas** compartidas. `more_digits.csv` incorpora nuevos datos pero reutiliza una porción sustancial de `digits.csv`.
  - `digits_test.csv` vs `more_digits.csv`: **0 muestras solapadas**.

---

## 5. Decisiones de diseño para el Ejercicio 2

1. **Protocolo de partición de validación:**
   - **Default:** **Holdout estratificado 80/20** (`dataset.split: {"kind": "holdout", "ratio": 0.8, "stratified": true}`).
     - Train: $\approx 9 959$ muestras (con $\approx 217$ muestras de clase 5).
     - Val: $\approx 2 490$ muestras (con $\approx 54$ muestras de clase 5).
     - Permite barridos rápidos manteniendo representatividad de todas las clases presentes.
   - **Alternativa soportada:** 5-fold cross-validation estratificado (`"kind": "kfold", "k": 5, "stratified": true`).
2. **Semillas y reproducibilidad:** Mínimo 3 semillas (`[0, 1, 2]`) por configuración, reportando siempre media $\pm$ desvío.
3. **Codificación de salida:**
   - One-hot con **10 clases** (`dataset.target_encoding: "onehot"`, `dataset.n_classes: 10`).
   - La red debe poseer 10 neuronas en la capa final ($m = 10$). Aunque en entrenamiento la neurona del dígito 8 nunca reciba target activo, la arquitectura debe coincidir con el espacio de clases real del problema para poder operar en producción.
   - Predicción: $\arg\max_i O_i$.
4. **Métricas obligatorias:**
   - `accuracy` (métrica global).
   - `macro_f1` (ponderación equitativa entre clases, sensible al desbalance de la clase 5).
   - Matriz de confusión $10 \times 10$ y `recall` por clase.
