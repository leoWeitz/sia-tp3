# F15 · Presentación

| | |
|---|---|
| **Objetivo** | Una presentación que responda **cada pregunta del enunciado** con una conclusión clara y su evidencia, y un equipo preparado para las preguntas de los profesores |
| **Depende de** | F14 |
| **Estimación** | 6–8 h (armado + 2 ensayos) |

## Principios
- **Título = conclusión** ("El lineal se estanca en el mínimo de mínimos cuadrados: agotó su capacidad"), no tema ("Resultados lineal").
- **Una pregunta → una o dos slides → una figura protagonista.** El ID del requerimiento va chico en el pie (E1-A-b) para que el equipo se ubique.
- Números siempre con **media ± desvío** y `n` de semillas; el test set se menciona explícitamente como "producción".
- Nada de código en pantalla; sí el diagrama de arquitectura y fórmulas solo si aportan.
- Bloques claros por ejercicio y transiciones ensayadas.

## Guion propuesto

| # | Slide | Evidencia | Req. |
|---|---|---|---|
| 1 | Portada: TP3 Perceptrón Simple y Multicapa · SIA 2026 · Garrós (64375), Weitz (64365), Ruckauf (64356), Romanato (62072) | — | — |
| 2 | Qué resolvimos y cómo (agenda en una línea por ejercicio) | — | — |
| 3 | Implementación: módulos y flujo experimento → resultados → análisis | Diagrama de `03-arquitectura.md` | RNF |
| 4 | Validamos antes de experimentar: AND, lineal, tanh, XOR (escalón falla, MLP resuelve) + cálculo a mano coincide | `figures/validacion/` | V-01..05 |
| **Ej1** | | | |
| 5 | El problema: destilar BigModel en un TinyModel que devuelva una probabilidad | Esquema teacher → student | — |
| 6 | Qué hay en los datos (target, balance, escalas, limpieza) | EDA | R-04 |
| 7 | ¿Underfitting? | Curvas de error de entrenamiento + referencias | E1-A-a |
| 8 | ¿Saturación de capacidades? | Meseta, mínimos cuadrados, salidas fuera de [0,1], saturación de neurona | E1-A-b |
| 9 | Elegimos el no lineal / lineal porque… | Tabla comparativa | E1-A-c |
| 10 | Métricas elegidas y por qué la accuracy engaña | Tabla de métricas + ejemplo del predictor "nunca fraude" | E1-B-a |
| 11 | Cómo partimos los datos y por qué no se elige "el mejor fold" | Boxplot estrategias + mejor fold vs todo | E1-B-b |
| 12 | Mejor modelo en producción | Métricas de test + matriz de confusión | E1-B-c |
| 13 | **Recomendación a CompanyX:** umbral = X porque… (por cada 1000 transacciones: …) | Métricas vs umbral + umbral vs costo | E1-B-c |
| **Ej2** | | | |
| 14 | Datos y protocolo de evaluación (train/val/producción) | Esquema + balance | E2-a |
| 15 | Tasa de aprendizaje | Curvas por η | E2-b1 |
| 16 | Arquitectura | Accuracy vs parámetros | E2-b2 |
| 17 | Optimizadores (por época y por tiempo) | Curvas | E2-b3 |
| 18 | Otras variantes que movieron la aguja | Tabla | E2-b4 |
| 19 | Resultado en producción y errores típicos | Matriz 10×10 + ejemplos | E2-final |
| **Ej3** | | | |
| 20 | ¿Llegamos al 98 %? | Barras Ej2 vs Ej3 + línea 98 % | E3-a |
| 21 | Qué técnicas aportaron (ablación) | Tabla de ablación | E3-b |
| 22 | Más allá de nuestras técnicas: el efecto del dato | Curva vs cantidad de datos + distribuciones | E3-c |
| 23 | Conclusiones (3–4 bullets, uno por ejercicio + aprendizaje del proceso) | — | — |
| A1+ | Anexo: fórmulas y convenciones (2β, MSE promedio, bias), fixture a mano, reproducibilidad, opcionales | — | — |

## Preparación para preguntas (armar respuestas cortas en `docs/resultados/qa.md`)
- ¿Por qué no inicializar los pesos en 0? · ¿Qué hace β? ¿Por qué la logística tiene 2β?
- ¿Online vs batch: costo por época, memoria, ruido, convergencia? (preguntas de la Clase 11)
- ¿Por qué el perceptrón escalón no resuelve XOR y el MLP sí?
- ¿Qué pasa si no normalizan? ¿Con qué datos ajustaron la normalización?
- ¿Por qué estratificar? ¿Por qué k-fold? ¿Cómo evitaron usar el test para decidir?
- ¿Por qué ese umbral y no 0.5? ¿Qué cambia si fraude no detectado cuesta 10× más?
- ¿Por qué MSE y no entropía cruzada? (y qué pasó cuando probaron CE, si lo hicieron)
- ¿Cómo saben que el backprop está bien? (gradient check + fixture)
- Si E(w) = 0 en train, ¿es un clasificador perfecto? (pregunta de la clase de Métricas)
- ¿Qué es capacidad de un modelo y cómo se relaciona con under/overfitting? (Clase 13)

## Criterios de aceptación
- [ ] Toda pregunta del enunciado tiene su slide con conclusión en el título.
- [ ] Todas las figuras salen de `figures/` (sin capturas manuales).
- [ ] 2 ensayos completos cronometrados; duración dentro de lo pedido por la cátedra.
- [ ] `qa.md` con respuestas de ≤ 3 líneas.