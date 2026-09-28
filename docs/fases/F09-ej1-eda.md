# F09 · Ej1 — Exploración de `datasets/fraud_dataset.csv`

| | |
|---|---|
| **Objetivo** | Entender el dataset antes de modelar (el enunciado lo exige) y fijar las features del Ej1 |
| **Requerimientos** | R-04, R-05 · alimenta E1-A y E1-B |
| **Depende de** | Nada: solo pandas. **Es lo primero que conviene hacer** (sus decisiones alimentan F05) |
| **Estimación** | 3–4 h |

## Lo que ya sabemos (verificado)

- Documentación oficial de la cátedra: `docs/datos/fraud_dataset_documentation.pdf`. **Leerla antes de empezar.**
- 7500 filas × 11 columnas, todas numéricas, **sin NaN**: `timestamp`, `amount_usd`, `quantity_purchased`, `session_duration_seconds`, `days_since_last_purchase`, `account_age_days`, `device_screen_resolution`, `time_since_last_login_s`, `items_viewed_before_purchase`, `big_model_fraud_probability`, `flagged_fraud`.
- **Target de entrenamiento: `big_model_fraud_probability`** (salida de BigModel, en [0, 1]). Es Knowledge Distillation "pura": el TinyModel aprende a imitar esa probabilidad (regresión con MSE, que es lo que minimizan los perceptrones de la cátedra).
- **`flagged_fraud` (ground truth, 0/1, ≈ 11.6 % de positivos) NUNCA se usa para entrenar** (lo dice explícitamente la documentación): ni como target ni como feature. Solo se usa para **evaluar**, **estratificar particiones** y **elegir el umbral** (F11). También sirve para comparar TinyModel vs BigModel umbralizado.

## Preguntas que tiene que contestar

1. **Documentación de cada columna:** significado según el PDF; si algo no está documentado, inferir y marcarlo como inferido.
2. **Rangos** de cada columna (min, max, percentiles 1/50/99) y tipo (numérica continua, entera, fecha, "casi categórica").
3. **Composición:** balance de `flagged_fraud`; distribución de `big_model_fraud_probability` (histograma, ¿bimodal?) y por clase de `flagged_fraud`.
4. **Limpieza:** duplicados exactos, valores imposibles (montos negativos, probabilidades fuera de [0,1]), outliers.
5. **Features que no conviene usar tal cual** → **decisión de features**, justificada:
   - `timestamp` (época Unix): ¿descartar, o transformar (hora del día, día de la semana)?
   - `device_screen_resolution` (ancho × alto en píxeles, pocos valores comunes): ¿numérica, categórica (one-hot de los valores frecuentes) o descartar?
   - Variables muy asimétricas (p. ej. `amount_usd`, `time_since_last_login_s`): ¿transformar con log?
6. **Escalas:** órdenes de magnitud muy distintos entre features (justifica normalizar: con la logística, entradas grandes saturan la neurona — conecta con E1-A-b).
7. **Relación feature ↔ target:** correlación de cada feature con `big_model_fraud_probability`; boxplots por `flagged_fraud`. Primera intuición de si el problema es "casi lineal".
8. **BigModel como referencia:** qué tan bien separa `big_model_fraud_probability` a `flagged_fraud` (curva PR/ROC rápida en el notebook): es la performance que el TinyModel intenta igualar.

## Especificación

- `notebooks/F09_eda_fraude.ipynb` (exploración, puede ser desprolijo) **y** `analysis/ej1_eda.py` (figuras limpias reproducibles para la presentación).
- Figuras en `figures/ej1/eda/`: balance de clases, histograma de `big_model_fraud_probability` por clase, histogramas/boxplots de features, matriz de correlación, tabla de rangos.
- `docs/datos/fraud_dataset.md` (**entregable principal**) con esta plantilla:

```markdown
# fraud_dataset.csv — diccionario de datos
- Filas: … · Columnas: … · Hash sha256: …
- Target de entrenamiento: big_model_fraud_probability (regresión sobre probabilidad)
- Etiqueta para evaluar / estratificar / elegir umbral: flagged_fraud (NUNCA para entrenar) · Proporción de positivos: …

| Columna | Tipo | Significado (documentación / inferido) | Rango (min–max) | Decisión (usar / descartar / transformar) | Motivo |
|---|---|---|---|---|---|

## Limpieza aplicada (configurable, nunca editando el CSV)
## Features finales (nombres exactos para la config)
## Observaciones para modelado (desbalance, escalas, outliers, relación con el target, performance de BigModel)
```

- Crear `experiments/configs/ej1/base.json` con `dataset.path`, `target` (`big_model_fraud_probability`), `features`, `drop` (incluye `flagged_fraud`), `categorical`, `normalize` por defecto (proponer min-max [0,1] o z-score según la EDA, justificado).

## Criterios de aceptación
- [ ] Las 8 preguntas tienen respuesta explícita en `docs/datos/fraud_dataset.md`.
- [ ] Decisión de features escrita en `docs/datos/fraud_dataset.md` y revisada por el equipo.
- [ ] `flagged_fraud` figura en `drop` de `experiments/configs/ej1/base.json`.
- [ ] `experiments/configs/ej1/base.json` pasa la validación de `load_config` (si F07 ya está) y sus columnas cargan con `load_csv` (F05).
- [ ] Figuras de EDA generadas por script (no desde el notebook).

## Notas para el agente
- **No entrenar nada en esta fase.**
- No modificar el CSV. Toda limpieza es por config.
- Si una columna no se puede interpretar, decirlo; no inventar significados.