# F08 · Ejercicios de validación — ✅ hecho · falta T3 (papel)

| | |
|---|---|
| **Requerimientos** | V-01 a V-05 |

## Estado
`tests/test_validation.py` ya cubre AND, y = x, y = tanh(x), XOR `[2,2,1]` y `[2,3,2,1]` con 20 semillas, el escalón fallando en XOR, un paso de la regla del perceptrón a mano y las iteraciones de `docs/verificacion_manual.md` (`[2,2,1]`) y `docs/verificacion_manual_2321.md` (`[2,3,2,1]`).

Las figuras salen de las corridas de `experiments/configs/validacion/` (20 semillas cada una):

```bash
python -m experiments.runner experiments/configs/validacion/<config>.json   # las 7 configs, ~1,5 min en total
python -m analysis.validacion                                              # → figures/validacion/ + resumen.json
```

Los resultados (`results/val_*/`) no se versionan: se regeneran con esos dos comandos (`CLAUDE.md` §8). Las figuras de `figures/validacion/` sí se versionan. `figures/validacion/resumen.json` tiene los números de la slide (épocas hasta converger en AND, MSE finales, semillas que resuelven XOR por arquitectura e inicialización). Para la recta de AND por época, el runner tiene la opción `logging.save_weights_history` (pesos de cada época en `weights_history.npz`, `03-arquitectura.md` §4).

## Pendientes
| ID | Tarea | Depende de | Para qué |
|---|---|---|---|
| F08-T3 | Rehacer en papel, con calculadora, la iteración de `docs/verificacion_manual.md` (2 integrantes) y dejar foto/escaneo en `docs/anexos/` | — | Lo pide el enunciado; sirve para la defensa oral |
| F08-T4 | ✅ (Opcional) fixture propio `[2,3,2,1]` a mano + test: `docs/verificacion_manual_2321.md` y `test_verificacion_manual_2321`. Falta que el equipo lo verifique en papel | — | El enunciado recomienda las dos arquitecturas |
| F08-T5 | ✅ Configs de validación en `experiments/configs/validacion/` (+ `xor_step.json`) + `analysis/validacion.py` con figuras: recta de decisión de AND por época, ajuste de y = x y tanh, error vs época escalón vs MLP en XOR, regiones de decisión de XOR | F07 | Slide de respaldo "validamos antes de experimentar" |
| F08-T6 | ✅ Experimento de escala de inicialización en XOR: `uniform` con escala {0.1, 0.5, 1.0} (`xor_init_uniform.json`) y `xavier` (las corridas de `xor_221.json` y `xor_2321.json`, misma config), 20 semillas → % de semillas que resuelven | F07 | Slide de respaldo: con escala 0.1 la red queda en la meseta (salida ≈ 0); muestra por qué importa la inicialización |

## Criterios de aceptación
- [ ] Verificación en papel firmada por 2 integrantes.
- [x] `python -m analysis.validacion` genera las figuras en `figures/validacion/` (`tests/test_analysis_validacion.py`).
- [x] `logging.save_weights_history` guarda los pesos de cada época, también con trainer `perceptron` y al reanudar; por defecto no cambia nada (`tests/test_experiments.py`).
- [x] T6: `% de semillas que resuelven` por inicialización y arquitectura, y loss vs época con la meseta de U(±0.1) (`V-04_xor_init_*`).
- [x] T4: el motor reproduce cada número de `docs/verificacion_manual_2321.md` con tolerancia 1e-6.
