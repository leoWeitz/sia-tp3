# F10 · Ej1 — Aprendizaje: perceptrón lineal vs no lineal

| | |
|---|---|
| **Objetivo** | Responder E1-A-a/b/c con evidencia: ¿underfitting?, ¿saturación de capacidades?, ¿cuál elegimos para generalizar? |
| **Requerimientos** | E1-A-a, E1-A-b, E1-A-c · R-01 (todas las muestras) · R-05 |
| **Depende de** | F07, F09 (y F08 en verde) |
| **Estimación** | 4 h de implementación de configs/análisis + tiempo de cómputo |

## Conceptos operacionalizados (para que "underfitting" y "saturación" sean medibles)

| Concepto | Cómo se mide en este TP |
|---|---|
| **Underfitting** | Error de entrenamiento alto: comparado contra (1) el predictor constante (predecir la media / la tasa de fraude) y (2) el otro perceptrón. Si el modelo apenas mejora al predictor constante con **todas** las muestras, subajusta. |
| **Saturación de la capacidad** | La curva de error de entrenamiento llega a una **meseta** que no baja con más épocas, otro η u otro optimizador. Para el lineal existe una cota exacta: la solución de mínimos cuadrados `np.linalg.lstsq` (solo como referencia, no como modelo): si el perceptrón lineal la alcanza, **agotó su capacidad**. Para el no lineal, la referencia es el mejor error alcanzado con entrenamiento largo (Adam, muchas épocas). |
| **Saturación de la neurona** (lectura complementaria) | Fracción de muestras con $\theta'(h)$ < 5 % de su máximo (logística con $|2\beta h|$ grande): la neurona deja de aprender. Se mide con y sin normalización y para varios β. |
| **Salida no probabilística** | Fracción de salidas del lineal fuera de [0, 1] (el enunciado pide una probabilidad). |

Criterio de meseta: mejora relativa del error < 0.1 % en el último 10 % de las épocas.

## Cómo se expresan los modelos en el motor
- Lineal: `"layers": ["auto", 1], "output_activation": "identity"` · No lineal: `"output_activation": "sigmoid"` (logística; con 2β si se aplicó F04-T0) · ambos con `"loss": "mse"` y `"split": {"kind": "none"}`.
- Los "η" de la tabla son del código (MSE promediada, `04-matematica.md` §2.4); ajustar el rango con `--smoke` si convergen demasiado lento o divergen.

## Experimentos (todos con `split: none` — R-01 — y ≥ 5 semillas)

| Nombre | Config | Barrido |
|---|---|---|
| `ej1_learning` | `experiments/configs/ej1/learning.json` | modelo ∈ {lineal, logística}; η (lineal: 1e-4, 1e-3, 1e-2 · logística: 1e-3, 1e-2, 1e-1); modo ∈ {online, batch}; épocas 500–1000 (ajustar por tiempo) |
| `ej1_beta` | `experiments/configs/ej1/beta.json` | logística con β ∈ {0.25, 0.5, 1, 2, 4}, mejor η de `ej1_learning` |
| `ej1_scaling` | `experiments/configs/ej1/scaling.json` | scaler ∈ {none, minmax, zscore} para ambos modelos (conecta con saturación de la neurona) |
| `ej1_long` | `experiments/configs/ej1/long.json` | ambos modelos con Adam, 3–5× más épocas → referencia de "mejor error alcanzable" |
| Referencias | `analysis/ej1_learning.py` | predictor constante y mínimos cuadrados (calculados en el análisis, no en el runner) |

Métrica de entrenamiento: MSE contra el target elegido en F09 (probabilidad de BigModel o etiqueta 0/1). Si existe la etiqueta real, loguear además accuracy/F1 con umbral 0.5 **solo como dato secundario** (el umbral se estudia en F11).

Para que la comparación sea justa: mismos features, mismo scaler, misma cantidad de épocas y mismas semillas de init en ambos modelos.

## Análisis y figuras (`analysis/ej1_learning.py` → `figures/ej1/learning/`)
1. MSE de entrenamiento vs época (escala log), lineal vs logística, media ± desvío entre semillas, mejor η de cada uno. Líneas horizontales: predictor constante y mínimos cuadrados. **(slide E1-A-a)**
2. Sensibilidad a η: MSE final vs η por modelo y modo.
3. Meseta: zoom en el último tramo + mejora relativa; `ej1_long` vs `ej1_learning`. **(slide E1-A-b)**
4. Histograma de salidas del lineal con [0, 1] marcado + fracción fuera de rango.
5. Fracción de neurona saturada vs β, con/sin normalización.
6. Tabla final: MSE train (media ± desvío), mejora vs constante, distancia a mínimos cuadrados, % salidas fuera de [0,1], tiempo por época. **(slide E1-A-c)**

## Plantilla de respuesta (a completar con los resultados, en `docs/resultados/ej1.md`)
- **a) ¿Underfitting?** Sí/No para cada modelo, con la cifra de MSE vs referencias.
- **b) ¿Saturación de capacidades?** Dónde se planchan las curvas, si el lineal alcanza mínimos cuadrados, si más épocas/otro optimizador cambian algo.
- **c) Elección:** qué perceptrón se lleva a generalización y por qué (potencial de aprendizaje + salida interpretable como probabilidad).

## Criterios de aceptación
- [x] Configs corren completas; `summary.csv` generado.
- [x] Figuras 1–6 generadas por script.
- [x] `docs/resultados/ej1.md` con las tres respuestas respaldadas por figura y número.
- [x] Elección E1-A-c (modelo y hiperparámetros de partida para F11) escrita en `docs/resultados/ej1.md`.

## Notas para el agente
- No correr los barridos completos sin OK (estimar tiempo primero con `--smoke` y avisar).
- No escribir conclusiones que no salgan de los números; si los resultados contradicen la intuición, reportarlo tal cual.