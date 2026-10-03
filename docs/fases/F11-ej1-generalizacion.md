# F11 · Ej1 — Generalización, mejor modelo y umbral recomendado

| | |
|---|---|
| **Objetivo** | Responder E1-B-a/b/c: métricas elegidas, estrategia de partición, mejor modelo y umbral de detección para CompanyX |
| **Requerimientos** | E1-B-a, E1-B-b, E1-B-c · RF-09, RF-10, RF-11 · Constitución C5 |
| **Depende de** | F10 (modelo elegido), F06 |
| **Estimación** | 5 h + cómputo |

## Protocolo de evaluación (fijo para toda la fase)

```
fraud_dataset.csv
├── TEST (20 %, estratificado, semilla fija)   ← se toca UNA vez, con --final-eval
└── DESARROLLO (80 %)
    └── k-fold estratificado (k = 5) × ≥ 3 semillas de init
        ├── selección de hiperparámetros (η, β, épocas vía early stopping)
        └── predicciones out-of-fold (OOF) → estudio de umbral
```

- Los índices del test se generan una vez y se guardan en `results/ej1_split/test_idx.npy` (y su hash anotado en `docs/datos/fraud_dataset.md`). Todas las configs los reutilizan.
- Early stopping monitorea la validación de cada fold; el modelo final se entrena con todo DESARROLLO durante la mediana de `best_epoch` de los folds.

## Métricas (E1-B-a) — qué se reporta y por qué

| Métrica | Rol | Por qué |
|---|---|---|
| **Average Precision (área PR)** | Principal, independiente del umbral | Con clases desbalanceadas, PR refleja el desempeño sobre la clase rara (fraude); ROC puede verse optimista |
| **Recall (TPR)** en el umbral elegido | Principal operativa | Cuántos fraudes se detectan (cada FN es un fraude que se escapa) |
| **Precision** en el umbral elegido | Principal operativa | Costo de revisar/bloquear transacciones legítimas (FP) |
| **F1** | Resumen en un número | Media armónica de precision y recall |
| ROC-AUC, FPR | Secundarias | Comparabilidad con la literatura |
| Accuracy | Se reporta **con advertencia** | Un modelo que dice "nunca es fraude" tiene accuracy = 1 − tasa de fraude |
| MSE vs BigModel | Si el target es la probabilidad de BigModel | Mide la calidad de la destilación en sí |

Si existen etiqueta real **y** probabilidad de BigModel: comparar TinyModel vs BigModel en las mismas métricas y sobre el mismo test (¿alcanza la performance?).

## Experimentos

| Nombre | Qué | Responde |
|---|---|---|
| `ej1_cv` | Modelo elegido en F10; barrido corto de η y β alrededor de lo encontrado; 5-fold estratificado × 3 semillas; `save_predictions: true` | E1-B-a, E1-B-c |
| `ej1_split_strategy` | Todo dentro de DESARROLLO. Misma config con: (i) 20 holdouts aleatorios 80/20, (ii) 20 holdouts estratificados, (iii) 5-fold estratificado. Además fracción de entrenamiento ∈ {0.5, 0.6, 0.7, 0.8, 0.9} | E1-B-b |
| `ej1_best_fold` | **Dentro de DESARROLLO** (sin tocar TEST): separar un 20 % estratificado como "pseudo-test", hacer 5-fold sobre el resto, tomar el fold con mejor validación ("el mejor conjunto de entrenamiento") y comparar su desempeño en el pseudo-test contra el modelo entrenado con todo el resto. Repetir con ≥ 5 semillas de partición → muestra el sesgo optimista de elegir una partición por su puntaje | E1-B-b |
| `ej1_threshold` | Análisis sobre OOF (no entrena) | E1-B-c |
| `ej1_final` | `--final-eval`: modelo final + umbral recomendado sobre TEST | E1-B-c |

## Estudio de umbral (E1-B-c)
1. `threshold_sweep` sobre las predicciones OOF concatenadas (todas las semillas).
2. Reportar los umbrales de cada criterio de `04-matematica.md` §7: F1, Youden, costo mínimo, precision@recall ≥ r (r ∈ {0.8, 0.9}).
3. Sensibilidad al costo: razón $c_{FN}/c_{FP}$ ∈ {1, 2, 5, 10, 20} → umbral óptimo para cada una.
4. **Recomendación a CompanyX:** un umbral concreto + el criterio + qué implica en números por cada 1000 transacciones (cuántos fraudes se detectan, cuántas alertas falsas). Presentarlo como decisión de negocio parametrizada por el costo, con nuestra recomendación por defecto.
5. Variabilidad del umbral entre folds (¿es estable?).

## Análisis y figuras (`analysis/ej1_generalization.py` → `figures/ej1/generalization/`)
1. Tabla de métricas CV (media ± desvío) del modelo elegido **(E1-B-a)**
2. Boxplot de la métrica principal por estrategia de split; curva métrica vs fracción de entrenamiento; barra "mejor fold vs todo el resto" en el pseudo-test **(E1-B-b)**
3. Curvas PR y ROC (OOF), con el punto del umbral recomendado
4. Precision, recall, F1 vs umbral, con los umbrales de cada criterio marcados
5. Umbral óptimo vs razón de costos
6. Matriz de confusión en TEST al umbral recomendado + tabla de métricas finales **(E1-B-c)**
7. "Tiny": cantidad de parámetros, tamaño en bytes del `model.npz`, tiempo de inferencia por 10⁵ transacciones

## Plantilla de respuesta (`docs/resultados/ej1.md`, sección B)
- **a)** Métricas elegidas y por qué (desbalance, costos asimétricos).
- **b)** Estrategia (holdout estratificado para test + k-fold estratificado para desarrollo) y respuesta a "¿cómo se elige el mejor conjunto de entrenamiento?": uno **representativo** de la distribución (estratificado, suficientemente grande), no el que casualmente da mejor puntaje; evidencia de `ej1_best_fold` y de la varianza entre splits.
- **c)** Mejor modelo (arquitectura, hiperparámetros, métricas en test) + umbral recomendado y su justificación.

## Criterios de aceptación
- [x] TEST generado una sola vez y nunca usado fuera de `ej1_final` (verificable en los logs).
  - *Nota:* los índices se generaron una sola vez, pero el TEST se leyó **dos veces**, siempre con `--final-eval` y en corridas `ej1_final`: `results/ej1_final` (6 columnas) y `results/ej1_all_features/ej1_final` (9 columnas). La segunda lectura fue una decisión explícita del equipo para comparar las dos versiones, y ninguna decisión sale de ella (`docs/resultados/ej1.md`, B y C.0).
- [x] Figuras 1–7 por script.
- [x] `docs/resultados/ej1.md` sección B completa.
- [x] Criterio de umbral recomendado justificado en `docs/resultados/ej1.md`.