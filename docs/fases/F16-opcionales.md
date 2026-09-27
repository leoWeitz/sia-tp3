# F16 · Opcionales

> ⚠️ **Regla del enunciado (R-03):** no se arranca ninguno hasta que F15 tenga al menos un borrador completo con todos los obligatorios respondidos.

Priorizados por relación valor/costo (los prácticos reutilizan todo lo construido):

| Prioridad | Opcional | Tipo | Costo | Cómo |
|---|---|---|---|---|
| 1 | Ej1 · ReLU en el perceptrón no lineal | Práctico | Bajo (1 config) | Repetir `ej1_learning` con `"output_activation": "relu"`. Ojo: ReLU no está acotada a [0,1] → discutir si la salida sigue siendo una probabilidad (clip / no). ¿Cambian las conclusiones de underfitting, saturación y elección? |
| 2 | Ej2/3 · Robustez al ruido | Práctico | Bajo | Mejor modelo de F13 sobre `digits_test.csv` + ruido gaussiano σ ∈ {0, 0.05, 0.1, 0.2, 0.3, 0.5} (en la escala normalizada), 5 repeticiones de ruido. Curva accuracy vs σ; comparar modelo entrenado con y sin augmentation de ruido. Nota: evaluar sobre test con ruido **no** es tunear, pero no se usa para cambiar el modelo final. |
| 3 | Ej2/3 · Interpretabilidad | Práctico | Medio | (a) Pesos de la primera capa de cada neurona como imagen; (b) saliencia: $\partial O_c/\partial x$ con el backprop propio (extender `Network` con un método `input_gradient`); (c) opcional: gradiente × entrada o Integrated Gradients (promedio de gradientes en el camino de una imagen base negra a la imagen). Mostrar mapas por clase y para errores. |
| 4 | Ej1 · Calibración | Teórico (+ práctico chico) | Bajo | Explicar por qué importa que "0.8" signifique 80 % de fraude real (se usa el score como probabilidad y para fijar costos). Práctico opcional: diagrama de confiabilidad (reliability diagram) con 10 bins sobre las predicciones OOF + ECE. Mencionar Platt scaling / isotonic como ajustes posibles. |
| 5 | Ej1 · Feature engineering | Teórico | Bajo | A partir de `docs/datos/fraud_dataset.md`: qué features construir (p. ej. transformaciones log de montos, interacciones, agregados por usuario/tiempo si hubiera IDs/timestamps, indicadores) y cuáles descartar (IDs, fugas de información, redundantes muy correlacionadas). |

## Entregables
- `docs/resultados/opcionales.md` con cada punto: pregunta, qué se hizo, figura, conclusión en 2–3 líneas.
- Slides en el anexo de la presentación.