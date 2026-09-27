# F08 · Ejercicios de validación — ✅ en tests · pendientes menores

| | |
|---|---|
| **Requerimientos** | V-01 a V-05 |

## Estado
`tests/test_validation.py` ya cubre AND, y = x, y = tanh(x), XOR `[2,2,1]` y `[2,3,2,1]` con 20 semillas, el escalón fallando en XOR, un paso de la regla del perceptrón a mano y la iteración de `docs/verificacion_manual.md`. **No hace falta volver a implementar nada de esto.**

## Pendientes
| ID | Tarea | Depende de | Para qué |
|---|---|---|---|
| F08-T3 | Rehacer en papel, con calculadora, la iteración de `docs/verificacion_manual.md` (2 integrantes) y dejar foto/escaneo en `docs/anexos/` | — | Lo pide el enunciado; sirve para la defensa oral |
| F08-T4 | (Opcional) fixture propio `[2,3,2,1]` a mano + test | — | El enunciado recomienda las dos arquitecturas |
| F08-T5 | Configs de validación en `experiments/configs/validacion/` (las crea F07) + `analysis/validacion.py` con figuras: recta de decisión de AND por época, ajuste de y = x y tanh, error vs época escalón vs MLP en XOR, regiones de decisión de XOR | F07 | Slide de respaldo "validamos antes de experimentar" |
| F08-T6 | Experimento de escala de inicialización en XOR: `uniform` con escala {0.1, 0.5, 1.0} y `xavier`, 20 semillas → % de semillas que resuelven | F07 | Slide de respaldo: con escala 0.1 la red queda en la meseta (salida ≈ 0); muestra por qué importa la inicialización |

## Criterios de aceptación
- [ ] Verificación en papel firmada por 2 integrantes.
- [ ] `python -m analysis.validacion` genera las figuras en `figures/validacion/`.