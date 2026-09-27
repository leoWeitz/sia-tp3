# F13 · Ej3 — More digits: llegar a accuracy ≥ 98 %

| | |
|---|---|
| **Objetivo** | Alcanzar (o acercarse con evidencia) a accuracy ≥ 98 % en `digits_test.csv` usando `more_digits.csv`, y explicar qué técnicas y qué factores externos influyeron |
| **Requerimientos** | E3-a, E3-b, E3-c · R-02 · RF-07 |
| **Depende de** | F12 (protocolo, mejor config de Ej2 como punto de partida) |
| **Estimación** | 5 h + cómputo |

## Datos
- Desarrollo: `more_digits.csv` (y, si la cátedra lo permite, también `digits.csv ∪ more_digits.csv`, deduplicado). Mismo protocolo que F12 (split estratificado para validación).
- Producción: `digits_test.csv`, **igual que en Ej2**, para que la comparación Ej2 vs Ej3 sea directa.
- EDA corta (se agrega a `docs/datos/digits.md`): tamaño, balance, distribución de píxeles vs `digits.csv`, duplicados con `digits.csv` y **con `digits_test.csv`** (fuga de información: si hay, se reporta y se decide qué hacer).

## Técnicas candidatas (de las teóricas + las más efectivas para MLP)

| Técnica | Dónde está | Hipótesis |
|---|---|---|
| Más datos | `more_digits.csv` | El factor principal (E3-c) |
| Early stopping | F04 | Evita sobreajuste con redes más grandes |
| L2 / weight decay | F04 | Reduce el gap train-val |
| Data augmentation: ruido gaussiano, traslaciones ±1–2 px (si son imágenes) | F04 | Más robustez; clave si el test tiene variaciones de trazo/posición |
| Arquitectura más grande (más ancho/profundidad) | F03 | Más capacidad ahora que hay más datos |
| Adam + mejor η / η adaptativo | F04 | Converge mejor y más rápido |
| Softmax + entropía cruzada (variante) | F01/F03 | Gradientes más informativos en clasificación multiclase (confirmar con la cátedra que se acepta como variante) |
| Ensamble de semillas (promediar salidas de N redes) | análisis | Técnica de regularización mencionada en la Clase 13 |

## Experimentos

| Nombre | Qué | Responde |
|---|---|---|
| `ej3_baseline` | Mejor config de Ej2 **sin cambios** entrenada con los nuevos datos | E3-c (cuánto aporta solo el dato) |
| `ej3_search` | Barrido OFAT sobre las técnicas de la tabla, partiendo de `ej3_baseline` | E3-b |
| `ej3_best` | Combinación final + ≥ 5 semillas; opcional ensamble | E3-a |
| `ej3_ablation` | Partiendo de `ej3_best`, **quitar una técnica por vez** | E3-b (qué aporta cada una) |
| `ej3_data_factors` | Mejor config con fracciones {10, 25, 50, 75, 100 %} del conjunto de desarrollo (curva de aprendizaje por cantidad de datos); y `digits` vs `more_digits` vs unión | E3-c |
| `ej3_final` | `--final-eval` en `digits_test.csv` | E3-a |

## Análisis y figuras (`analysis/ej3.py` → `figures/ej3/`)
1. Barras: accuracy test Ej2 final vs Ej3 baseline vs Ej3 best, con la línea del 98 % **(E3-a)**
2. Tabla de ablación: accuracy val (media ± desvío) con/sin cada técnica, ordenada por aporte **(E3-b)**
3. Accuracy val y test vs cantidad de datos de entrenamiento **(E3-c)**
4. Comparación de distribuciones (balance de clases, histograma de intensidades, dígito promedio) digits vs more_digits **(E3-c)**
5. Matriz de confusión del mejor modelo y ejemplos que sigue fallando.

## Plantilla de respuesta (`docs/resultados/ej3.md`)
- **(a)** Mejor accuracy en test (media ± desvío y mejor semilla), ¿se alcanzó el 98 %? Config completa y modelo guardado en `models/ej3_best/` (`model.npz` + `config.json`).
- **(b)** Técnicas, en orden de aporte según la ablación.
- **(c)** Factores externos: cantidad de datos (curva), diferencias de distribución/calidad entre datasets, balance, duplicados/fuga; cuánto de la mejora es "dato" vs "técnica" (baseline vs best).

## Criterios de aceptación
- [ ] `digits_test.csv` solo leído en `ej3_final` (y en la reevaluación de `ej2_final` para comparar, que ya existía).
- [ ] Ablación completa con ≥ 3 semillas por fila.
- [ ] Figuras 1–5 por script; `docs/resultados/ej3.md` completo.
- [ ] Si no se llega al 98 %: documentar la mejor cifra, qué se intentó y la hipótesis de por qué (no forzar el número mirando el test).