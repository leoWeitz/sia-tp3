# F12 · Ej2 — Clasificación de dígitos con MLP

| | |
|---|---|
| **Objetivo** | Responder E2-a (cómo evaluamos) y E2-b (qué variantes probamos: tasa de aprendizaje, arquitectura, optimizador + extras) y reportar el desempeño en "producción" |
| **Requerimientos** | E2-a, E2-b1..b4, E2-final · R-02 · Constitución C5 |
| **Depende de** | Parte 1 (EDA): nada, solo pandas · Partes 2–4: F07 (y F08 en verde) |
| **Estimación** | 6 h de configs/análisis + cómputo (el más largo del TP) |

## Lo que ya sabemos (verificado)

| Archivo | Filas | Balance de clases |
|---|---|---|
| `digits.csv` | 12 449 | **No tiene ningún 8** y tiene pocos 5 (271); el resto ≈ 1 460–1 685 por clase |
| `digits_test.csv` | 2 497 | Balanceado, **incluye 8** (243) y 5 (223) |
| `more_digits.csv` (F13) | 15 741 | Incluye 8 (585) y más 5 (542) |

- Formato: columnas `label` e `image` (texto con 784 floats en [0, 1] → imagen 28×28). Se carga con `load_digits_csv` (F05).
- Sin duplicados dentro de cada archivo ni entre `digits.csv` y `digits_test.csv`.
- **Consecuencia clave:** un modelo entrenado solo con `digits.csv` nunca vio un 8, así que en producción falla todos los 8 (≈ 9.7 % del test): la accuracy en test queda acotada en ≈ 90 % haga lo que haga el modelo, y la validación (tomada de `digits.csv`, sin 8) sobreestima la accuracy de producción. Esto no es un bug: es parte de la respuesta del Ej2 y del punto (c) del Ej3.

## Parte 1 — EDA de dígitos (antes de entrenar)

`notebooks/F12_eda_digitos.ipynb` + `analysis/ej2_eda.py` → `docs/datos/digits.md`:
- Confirmar los datos de la tabla anterior y documentarlos en `docs/datos/digits.md`.
- Balance de clases en cada archivo (tabla + barras), marcando la ausencia de 8 y la escasez de 5 en `digits.csv`.
- Grilla de ejemplos por clase y "dígito promedio" por clase.
- Duplicados dentro de cada archivo y **entre** train y test (si hay, reportarlo: afecta la lectura de resultados).
- Decidir el **protocolo de validación** (holdout 80/20 estratificado vs 5-fold) según tamaño del dataset y tiempo por época medido con `--smoke`.

## Parte 2 — Protocolo de evaluación (E2-a)

```
digits.csv ──► split estratificado ──┬──► TRAIN → ajuste de parámetros (pesos)
                                     └──► VAL   → selección de hiperparámetros, early stopping
digits_test.csv ──► SOLO --final-eval con la configuración ya elegida ("producción")
```

- **Métricas:** accuracy (la que pide el cliente) **siempre junto con** macro-F1, recall por clase y matriz de confusión 10×10, porque `digits.csv` está desbalanceado y no tiene 8. Además, curvas de loss/accuracy train vs val (gap → sobreajuste) y tiempo por época.
- Split estratificado por clase (así los pocos 5 quedan repartidos entre train y val).
- Cada configuración: ≥ 3 semillas → media ± desvío.
- Codificación de salida: one-hot con **10 neuronas** aunque en train no haya 8 (el test sí los tiene); predicción = `argmax`.
- Escalado: los píxeles ya vienen en [0, 1] → `normalize: "none"` por defecto (un min-max por píxel dividiría por cero en los bordes siempre negros). Alternativa a probar en `ej2_extra`: z-score.

## Parte 3 — Plan de variantes (E2-b)

**Configuración base** (`experiments/configs/ej2/base.json`): `dataset.format: "digits"`, `normalize: "none"`, `target_encoding: "onehot"`, `[784, 64, 10]`, tanh oculta (β = 1), sigmoide de salida + MSE (variante de la cátedra), init `xavier` (default del motor), minibatch 32, early stopping (patience 20, máx. 300 épocas).

Notas de motor: "SGD" en las tablas = optimizador `"gd"` con `batch_size` chico; la MSE del código divide por `n·m` con `m = 10` salidas, así que los η útiles de GD pueden ser más grandes que los habituales (`04-matematica.md` §2.4) → confirmar el rango con `--smoke` antes del barrido.

Estrategia: **un factor por vez (OFAT)** partiendo de la base, después una grilla chica combinando los mejores. Se estima el costo con `--smoke` antes de lanzar cada barrido.

| Experimento | Config | Barrido | Responde |
|---|---|---|---|
| `ej2_lr` | `experiments/configs/ej2/lr.json` | η ∈ {1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1} × optimizador ∈ {SGD, Adam} | E2-b1 |
| `ej2_eta_adapt` | `experiments/configs/ej2/eta_adapt.json` | SGD con η adaptativo (a, b, k, k′ en 2–3 combinaciones) vs mejor η fijo | E2-b1 |
| `ej2_arch_width` | `experiments/configs/ej2/arch_width.json` | 1 capa oculta de {16, 32, 64, 128, 256} | E2-b2 |
| `ej2_arch_depth` | `experiments/configs/ej2/arch_depth.json` | [64], [64, 64], [128, 64], [128, 64, 32], [256, 128] | E2-b2 |
| `ej2_opt` | `experiments/configs/ej2/opt.json` | {SGD, Momentum α=0.9, RMSProp, Adam, SGD + η adaptativo}, cada uno con su mejor η de un mini-barrido de 3 valores | E2-b3 |
| `ej2_extra` | `experiments/configs/ej2/extra_*.json` | batch ∈ {1, 16, 32, 128, full}; activación oculta ∈ {tanh, logística, ReLU}; salida ∈ {logística+MSE, softmax+CE}; init ∈ {uniforme 0.1, Xavier}; normalización ∈ {none, zscore}; L2 ∈ {0, 1e-4, 1e-3} | E2-b4 |
| `ej2_combo` | `experiments/configs/ej2/combo.json` | 2–3 valores de los 3 factores más influyentes | selección |
| `ej2_final` | `experiments/configs/ej2/final.json` | Mejor config (por accuracy de val media; empate → menos parámetros), reentrenada con todo `digits.csv` durante la mediana de `best_epoch`, `--final-eval` sobre `digits_test.csv` | E2-final |

## Parte 4 — Análisis y figuras (`analysis/ej2.py` → `figures/ej2/`)
1. Esquema del protocolo de evaluación **(E2-a)**
2. Loss de val vs época por η (SGD y Adam en paneles separados) + accuracy final de val vs η (log x) con barras de desvío **(E2-b1)**
3. Accuracy de val vs cantidad de parámetros (ancho y profundidad), con curvas train vs val para ver sobreajuste **(E2-b2)**
4. Curvas de val por optimizador vs **época** y vs **tiempo de pared** **(E2-b3)**
5. Tabla de extras: accuracy val (media ± desvío) por variante **(E2-b4)**
6. Matriz de confusión en test + recall por clase + pares más confundidos + grilla de ejemplos mal clasificados **(E2-final)**
7. Tabla "resumen del estudio": variante, mejor valor, accuracy val, tiempo/época.

## Plantilla de respuesta (`docs/resultados/ej2.md`)
- **(a) ¿Cómo evalúo el desempeño?** Protocolo + métricas y por qué.
- **(b) ¿Qué variantes realizo?** Una subsección por factor: qué se probó, qué se observó (con figura), conclusión.
- **Resultado en producción:** accuracy y macro-F1 en `digits_test.csv` (media ± desvío si se reentrena con varias semillas), matriz de confusión, errores típicos. Comparar con la accuracy de validación y explicar la diferencia (los 8 que el modelo nunca vio, el techo de ≈ 90 %).

## Criterios de aceptación
- [ ] `docs/datos/digits.md` completo, con el formato de imagen y el protocolo de validación decididos.
- [ ] Todos los barridos con `summary.csv`; ninguno leyó `digits_test.csv` (verificable en configs resueltas).
- [ ] `ej2_final/final_eval.json` generado una sola vez.
- [ ] Figuras 1–7 por script y `docs/resultados/ej2.md` completo.
- [ ] Mejor modelo guardado en `models/ej2_best/` (`model.npz` + `config.json`) para reusar en F13 y en los opcionales.

## Notas para el agente
- Si una corrida diverge (NaN/inf en la loss), `fit` la corta (`history.status == "diverged"`, F04), el runner la marca como `diverged` en `metrics.json` y el barrido continúa (es un resultado válido para E2-b1).
- Considerar `float32` y `--workers` si 784 features hacen lento el barrido (medir primero).
- No "arreglar" la falta de 8 en Ej2 (por ejemplo, sacando 8 del test): se reporta tal cual; el Ej3 es justamente donde aparecen datos nuevos..