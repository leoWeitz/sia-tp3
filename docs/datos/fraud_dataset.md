# fraud_dataset.csv — diccionario de datos

- Filas: 7500 · Columnas: 11 · Hash sha256: `0e0d2fdf7e6c294b7b8ea2e80452a270f1fad1de02edee9cc25f6eec2ab4c707`
- Target de entrenamiento: `big_model_fraud_probability` (regresión sobre probabilidad, MSE)
- Etiqueta para evaluar / estratificar / elegir umbral: `flagged_fraud` (**NUNCA para entrenar**, lo exige la documentación) · Proporción de positivos: 869 / 7500 = **11.59 %**

Fuentes: documentación de la cátedra (`fraud_dataset_documentation.pdf`), exploración en `notebooks/F09_eda_fraude.ipynb`, y números y figuras de `python -m analysis.ej1_eda`. Esa corrida escribe `figures/ej1/eda/resumen.json`, `rangos.csv` y las figuras `EDA-01` a `EDA-08`. Todo número de este documento sale de esos archivos.

> La EDA mira el archivo completo, incluidas las filas que F11 va a reservar como test. De acá salen solo decisiones cualitativas: qué columnas usar y qué *tipo* de normalización. No sale ningún parámetro ajustado: el scaler se ajusta con el train de cada partición (CLAUDE.md §2.7), y el umbral del TinyModel se elige en F11 con validación.

| Columna | Tipo | Significado (documentación / inferido) | Rango (min–max) | Decisión | Motivo |
|---|---|---|---|---|---|
| `timestamp` | entera · fecha (época Unix, s) | **Doc.:** momento exacto de la compra ("no preocuparse por UTC"). Va del 2023-11-14 al 2024-11-13 (un año). | 1 700 001 808 – 1 731 534 222 | **descartar** | No tiene relación con el target: ρ = 0.001 con la probabilidad y −0.013 con la etiqueta. Tampoco la tiene derivada, porque la tasa de fraude por hora, día de semana y mes es plana (EDA-06, `resumen.json › tiempo`). Crudo, además, extrapola: una transacción futura cae fuera del rango de entrenamiento. |
| `amount_usd` | continua (USD) | **Doc.:** monto total de la compra online. | 1.00 – 2000.00 | **usar** | Relación fuerte y creciente (ρ = 0.557). Es muy asimétrica (asimetría 4.53, p50 = 63.49, p99 = 858.05), así que es candidata a log (F16). Tiene *spikes*: 100.00 aparece 156 veces con 98.7 % de fraude, y 1.00 aparece 82 veces (probable mínimo recortado, 0 % de fraude). |
| `quantity_purchased` | entera | **Doc.:** cantidad de unidades del mismo artículo. | 1 – 24 | **usar** | ρ = 0.563. **Regla dura:** con más de 9 unidades la compra es siempre fraude (524 casos). |
| `session_duration_seconds` | continua (s) | *Inferido:* duración de la sesión web en la que se hizo la compra (la doc. solo da el tipo). | 5.0 – 726.8 | **usar** | ρ = −0.514: sesiones cortas indican fraude. **Regla dura:** por debajo de 10 s siempre es fraude (74 casos, 64 de ellos exactamente en 5.0 s). El valor 10.0 aparece 53 veces y parece un mínimo recortado. |
| `days_since_last_purchase` | continua (días) | *Inferido:* días desde la compra anterior del mismo usuario. No sabemos si 0 significa "compró el mismo día" o "primera compra". | 0.00 – 142.32 | **usar** | ρ = −0.404, decreciente y suave (EDA-06). Es asimétrica (2.13). Los fraudes están todos por debajo de 37.7 días. |
| `account_age_days` | entera (días) | *Inferido:* antigüedad de la cuenta del comprador. | 1 – 3649 | **usar** | Es la feature más correlacionada (ρ = −0.585). **Regla dura:** con cuentas de menos de 30 días siempre es fraude (114 casos). |
| `device_screen_resolution` | entera (píxeles, ancho × alto) · **casi categórica** | **Doc.:** resolución de la pantalla del dispositivo; valores comunes 1 049 088 (1366×768) y 2 073 600 (1920×1080). | 1 006 733 – 8 310 940 | **descartar** | No son "pocos valores": hay 7209 distintos. Son **5 resoluciones más ruido** de ±5000 px (desvío 5012 px): 1280×800, 1366×768, 1920×1080, 2560×1440 y 3840×2160, cada una con alrededor de 1500 filas. No aporta información: la tasa de fraude va de 10.5 % a 12.9 % según la resolución y ρ = 0.025. Como numérica, además, la escala la dominan los valores de 4K. Como `categorical`, `load_csv` generaría una columna one-hot por valor (7209), porque no agrupa. |
| `time_since_last_login_s` | continua (s) | *Inferido:* segundos desde el último inicio de sesión antes de la compra. | 10.0 – 40 160.8 | **descartar** | Sin relación: ρ = 0.002, y la tasa de fraude es plana en todos los deciles (EDA-06). Es asimétrica (2.06) y tiene el mínimo recortado en 10.0 (18 casos). |
| `items_viewed_before_purchase` | entera | *Inferido:* cantidad de artículos que el usuario miró antes de comprar. | 1 – 29 | **usar** | ρ = 0.334 con la probabilidad y 0.553 con la etiqueta. **Regla dura:** con más de 14 artículos vistos siempre es fraude (490 casos). Por debajo de ese valor, la probabilidad casi no cambia (EDA-06). |
| `big_model_fraud_probability` | continua en [0, 1] | **Doc.:** salida de BigModel; "hay que determinar un umbral" para marcar fraude. | 0.000897 – 1.000000 | **target** | Es el objetivo de la destilación. |
| `flagged_fraud` | binaria 0/1 | **Doc.:** *ground truth* que CompanyX juntó a partir de denuncias de fraude. "MUST NOT be used when training". | 0 – 1 | **drop** (solo evaluar) | Se usa para evaluar, estratificar y elegir el umbral (F11). |

Tabla completa de rangos (p1, p50, p99, media, desvío, asimetría, nº de únicos): `figures/ej1/eda/rangos.csv` y `EDA-08_ranges_table`.

## Respuestas a las preguntas de la fase

1. **Documentación.** El PDF documenta el significado de `timestamp`, `amount_usd`, `quantity_purchased`, `big_model_fraud_probability` y `flagged_fraud`. De `device_screen_resolution` da el formato y los valores comunes. De `account_age_days`, `session_duration_seconds`, `time_since_last_login_s`, `days_since_last_purchase` e `items_viewed_before_purchase` solo da tipo y unidad; en la tabla su significado figura como *inferido* del nombre.
2. **Rangos y tipos.** Están en la tabla de arriba y en `rangos.csv`. Hay cinco columnas enteras, cuatro continuas, el target en [0, 1] y la etiqueta binaria. `timestamp` es una fecha y `device_screen_resolution` es casi categórica.
3. **Composición.**
   - Hay 11.59 % de positivos: es un **desbalance** de 1 a 7.6 (EDA-01).
   - `big_model_fraud_probability` **no es bimodal** (EDA-02). La masa decrece desde 0 hacia 0.8 y hay un pico en 0.95–1.0: 604 fraudes en el último bin.
   - Por clase: los negativos van de 0.0009 a **0.8499** (mediana 0.30) y los positivos de **0.8501** a 1.0 (mediana 0.996).
4. **Limpieza.**
   - No hay NaN, ni filas duplicadas (tampoco sin contar target y etiqueta).
   - No hay valores imposibles: ninguna columna tiene negativos y el target no sale de [0, 1].
   - Hay outliers de monto (p99 = 858, máximo = 2000), pero son valores plausibles y pertenecen a fraudes: **no se eliminan**.
   - Los *spikes* son parte del fenómeno, no errores: 100.00 USD y 5 s de sesión son casi siempre fraude. Los mínimos recortados (1.00 USD, 10 s de sesión, 10 s desde el login) son el piso del generador. **No se aplica ninguna limpieza.**
5. **Features conflictivas.**
   - `timestamp`: se descarta, ni crudo ni transformado tiene señal (hora, día y mes son planos).
   - `device_screen_resolution`: se descarta, porque son 5 resoluciones más ruido y no tienen señal.
   - `time_since_last_login_s`: se descarta, porque no tiene señal.
   - La transformación log de `amount_usd` o `days_since_last_purchase` (asimetrías de 4.5 y 2.1) quedaría bien, pero `load_csv` no admite columnas derivadas. El enunciado deja la construcción de features como opcional teórico, así que se deja para F16 (sección de observaciones).
6. **Escalas.** Van de ~10 (cantidades, artículos vistos) a ~10³ (edad de la cuenta, montos) y ~10⁴ (segundos). Crudas, la entrada de la logística $h = w^\top x$ sería enorme con pesos iniciales de Xavier, y la neurona arrancaría saturada. Es la conexión con E1-A-b, que `ej1_scaling` (F10) mide con `none` / `minmax` / `zscore`. **Se normaliza.**
7. **Relación feature ↔ target** (EDA-05, EDA-06).
   - Seis features tienen relación **monótona** con la probabilidad: `amount` (+), `quantity` (+), `items_viewed` (+), `session` (−), `days_since_last_purchase` (−) y `account_age` (−), con |ρ| entre 0.33 y 0.59. Las otras tres tienen |ρ| < 0.03.
   - La relación **no es lineal** y tiene forma de escalones. El fraude aparece como un **O de reglas duras**: cantidad > 9, artículos vistos > 14, monto > 500, cuenta < 30 días o sesión < 10 s. En la parte alta, la probabilidad salta en vez de subir suave (deciles 9→10 de `quantity` y `items_viewed`).
   - Esas cinco reglas más el *spike* de 100 USD cubren **716 de 869 fraudes** (82 %) y tocan solo 2 negativos.
   - Un perceptrón simple solo puede representar $\theta(w^\top x + b)$, una rampa en una sola dirección. Primera intuición: va a captar la tendencia general, pero va a tener underfitting justo en la zona alta (≈ 0.85), que es donde se decide el umbral. Es una hipótesis para F10, no una conclusión.
8. **BigModel como referencia** (EDA-07).
   - **`flagged_fraud` = 1 si y solo si `big_model_fraud_probability` > 0.85**: el máximo de los negativos es 0.849891, el mínimo de los positivos es 0.850089, y no hay ninguna muestra en el medio. ROC-AUC = AP = **1.000**.
   - Con umbral 0.5, BigModel tiene recall 1.00 y precisión 0.32. Con 0.85, precisión y recall son 1.00.
   - Es la cota que el TinyModel intenta igualar. Como el target contiene toda la información de la etiqueta, lo que importa para F11 es **qué tan bien aproxima el TinyModel la probabilidad cerca de 0.85**.

## Limpieza aplicada (configurable, nunca editando el CSV)

Ninguna: `na_policy: "error"`, que no se dispara porque no hay NaN. No se filtran filas ni outliers. Las columnas que no se usan se sacan por config (`drop`).

## Features finales (nombres exactos para la config)

**Estudio principal: las 9 columnas.** Se usan todas, salvo el target y la etiqueta. Las configs están en `experiments/configs/ej1/all_features/`.

```json
"features": ["timestamp", "amount_usd", "quantity_purchased", "session_duration_seconds",
             "days_since_last_purchase", "account_age_days", "device_screen_resolution",
             "time_since_last_login_s", "items_viewed_before_purchase"],
"drop": ["flagged_fraud"],
"categorical": []
```

**Control: la selección de 6 columnas de esta EDA.** Sigue documentada y corrida completa, y es la del modelo que se recomienda a CompanyX (`docs/resultados/ej1.md`, B-c y C).

```json
"features": ["amount_usd", "quantity_purchased", "session_duration_seconds",
             "days_since_last_purchase", "account_age_days", "items_viewed_before_purchase"],
"drop": ["flagged_fraud", "timestamp", "device_screen_resolution", "time_since_last_login_s"],
"categorical": []
```

> **Nota.** La columna "Decisión" de la tabla de arriba y la respuesta 5 registran la selección original de esta EDA. El análisis sigue vigente: las 3 columnas descartadas no tienen relación con el target. Lo que cambió es que el equipo decidió hacer el estudio principal con las 9 columnas y dejar la selección de 6 como control. El resultado confirma la EDA: el modelo de 9 columnas les da un peso ~0 a esas 3, y las dos versiones rinden igual en validación (`docs/resultados/ej1.md`, C.0).

La config de referencia del control está en `experiments/configs/ej1/base.json`, con `split: none` (R-01: el estudio de aprendizaje usa todas las muestras), salida `sigmoid` (R-05), MSE y `normalize: "zscore"`. Las configs del estudio principal usan los mismos valores.

**Normalización por defecto: z-score.**

- Las features son asimétricas y con cola larga. Con min-max, `amount_usd` queda con la mediana en 0.031 y el p99 en 0.43: casi todo el rango útil comprimido cerca de 0, por el máximo de 2000. Con z-score, cada feature queda centrada y con desvío 1, así que el descenso por gradiente queda mejor condicionado y las entradas son comparables.
- En contra: el monto máximo queda en z ≈ 12, pero es una sola muestra, y lo va a absorber un peso chico.
- F10 compara igual `none`, `minmax` y `zscore` en `ej1_scaling`.

Índices del test del Ej1: `results/ej1_split/test_idx.npy` (F11). Son 1500 filas, 20 % estratificado por deciles de la probabilidad, `holdout_test.seed = 0`, con 11.73 % de fraude. Los genera el runner en la primera corrida con `holdout_test`, de forma determinística. sha256 del `.npy`: `9e711725e86c18460503685baaf37c87ad5c46e742a5971da5fa33a489943bcc`.

## Observaciones para modelado

- **Desbalance** de 11.6 % de positivos. Accuracy sola no sirve (el "nunca fraude" da 88.4 %). En F11 conviene usar PR-AUC, F1, recall y precision al umbral.
- **Estratificación.** El runner estratifica un target continuo por deciles de la probabilidad (`data.splits.stratify_labels`). Como la etiqueta es la probabilidad umbralizada en 0.85:
  - los estratos 0–7 tienen 0 % de fraude;
  - el estrato 8 (0.741–0.891) tiene 15.9 %;
  - el estrato 9 (> 0.891) tiene 100 %.

  Estratificar por la probabilidad **ya conserva la proporción de `flagged_fraud`**, así que F11 no necesita cambiar el runner (`resumen.json › estratos_por_cuantiles`).
- **Escalas y saturación.** Hay que normalizar siempre (ver la pregunta 6). La logística tiene imagen en (0, 1), que coincide con el target. El lineal puede dar salidas fuera de [0, 1] (E1-A-b).
- **Linealidad.** La relación es monótona pero en escalones y con forma de "O de reglas". Un perceptrón simple no puede representar un O de umbrales (no es un semiespacio), así que se espera un piso de error de entrenamiento, sobre todo cerca de 0.85 (E1-A-a).
- **Performance de BigModel:** separa perfecto (AUC = 1). El umbral de BigModel es 0.85. El umbral del TinyModel **no** tiene por qué ser 0.85: se elige en F11 con validación.
- **Para F16 (features nuevas, teórico):**
  - indicadores de las reglas duras: `quantity > 9`, `items_viewed > 14`, `account_age < 30`, `session < 10`, `amount > 500`;
  - indicador de monto redondo (`amount == 100`);
  - log de `amount_usd` y de `days_since_last_purchase`;
  - resolución agrupada en 5 categorías (hoy sin señal).

  Quedan descartadas `timestamp` y `time_since_last_login_s`.
