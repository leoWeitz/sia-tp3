# F09 · Ej1 — Exploración de `datasets/fraud_dataset.csv`

| | |
|---|---|
| **Objetivo** | Entender el dataset antes de modelar (el enunciado lo exige) y decidir el target y las features del Ej1 |
| **Requerimientos** | R-04, R-05 · alimenta E1-A y E1-B |
| **Depende de** | Nada: solo pandas. **Es lo primero que conviene hacer** (sus decisiones alimentan F05) |
| **Estimación** | 3–4 h |

## Preguntas que tiene que contestar (del enunciado + propias)

1. **Documentación de cada columna:** ¿qué significa? ¿de dónde viene? (buscar en el campus/enunciado si hay diccionario; si no, inferir y marcar como inferido).
2. **Target:** ¿hay una columna con la **probabilidad de BigModel** (soft label, continua en [0,1])? ¿hay una **etiqueta real** de fraude (0/1)? ¿ambas? → **decisión del target**.
   - Si hay probabilidad de BigModel → Knowledge Distillation "pura": el TinyModel aprende a imitar esa probabilidad (regresión con MSE, que es justo lo que minimizan los perceptrones de la cátedra).
   - Si hay etiqueta real → las métricas de clasificación y el umbral se evalúan contra ella.
   - Si hay ambas → entrenar contra la probabilidad de BigModel, evaluar contra la etiqueta real **y** comparar con BigModel umbralizado (¿el TinyModel alcanza la performance del BigModel?).
3. **Rangos** de cada columna (min, max, percentiles 1/50/99), tipo (numérica, categórica, binaria, ID, fecha).
4. **Composición:** cantidad de filas; **balance de clases** (proporción de fraude); distribución de la probabilidad de BigModel (histograma, ¿bimodal?).
5. **Limpieza:** NaN por columna, duplicados exactos, valores imposibles (montos negativos, probabilidades fuera de [0,1]), outliers.
6. **Features no utilizables directamente:** IDs, timestamps, texto, categóricas → decidir descartar / codificar (one-hot) / transformar (log de montos, hora del día) → **decisión de features**.
7. **Escalas:** ¿órdenes de magnitud muy distintos entre features? (justifica normalizar: con logística, entradas grandes saturan la neurona — conecta con E1-A-b).
8. **Relación feature ↔ target:** correlación de cada feature con la probabilidad/etiqueta; boxplots por clase. Primera intuición de si el problema es "casi lineal".

## Especificación

- `notebooks/F09_eda_fraude.ipynb` (exploración, puede ser desprolijo) **y** `analysis/ej1_eda.py` (figuras limpias reproducibles para la presentación).
- Figuras en `figures/ej1/eda/`: balance de clases, histograma de la probabilidad de BigModel (por clase si hay etiqueta), histogramas/boxplots de features, matriz de correlación, tabla de rangos.
- `docs/datos/fraud_dataset.md` (**entregable principal**) con esta plantilla:

```markdown
# fraud_dataset.csv — diccionario de datos
- Filas: … · Columnas: … · Hash sha256: …
- Target elegido: … · Tipo de problema de entrenamiento: regresión sobre probabilidad | clasificación
- Etiqueta para evaluar: … · Proporción de positivos: …

| Columna | Tipo | Significado (fuente / inferido) | Rango (min–max) | NaN | Decisión (usar / descartar / transformar) | Motivo |
|---|---|---|---|---|---|---|

## Limpieza aplicada (configurable, nunca editando el CSV)
## Features finales (nombres exactos para la config)
## Observaciones para modelado (desbalance, escalas, outliers, relación con el target)
```

- Dejar escritas en `docs/datos/fraud_dataset.md` las dos decisiones (target y features) con su justificación: son la referencia para F05, F10 y F11.
- Crear `experiments/configs/ej1/base.json` con `dataset.path`, `target`, `features`, `drop`, `categorical`, `na_policy`, `normalize` por defecto (proponer min-max [0,1] o z-score según la EDA, justificado).

## Criterios de aceptación
- [ ] Las 8 preguntas tienen respuesta explícita en `docs/datos/fraud_dataset.md`.
- [ ] Decisiones de target y features escritas en `docs/datos/fraud_dataset.md` y revisadas por el equipo.
- [ ] `experiments/configs/ej1/base.json` pasa la validación de `load_config` (si F07 ya está) y sus columnas cargan con `load_csv` (F05).
- [ ] Figuras de EDA generadas por script (no desde el notebook).

## Notas para el agente
- **No entrenar nada en esta fase.**
- No modificar el CSV. Toda limpieza es por config.
- Si una columna no se puede interpretar, decirlo; no inventar significados.