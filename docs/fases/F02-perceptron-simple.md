# F02 · Perceptrón simple (escalón, lineal, no lineal) — ✅ HECHO

Ya implementado (Etapas 2, 4 y 5 de `docs/PLAN_MOTOR.md`):

| Modelo | Cómo se expresa | Test |
|---|---|---|
| Escalón | `Network([n, 1], output_activation="step")` + `fit_perceptron(...)` (regla de Rosenblatt por lotes) | AND 20/20 semillas en 1–4 épocas; XOR nunca baja de 25 % de error; paso a mano |
| Lineal | `Network([n, 1], output_activation="identity")` + `fit` con MSE | y = x, 20/20, MSE < 1e-4 |
| No lineal | `Network([n, 1], output_activation="tanh"/"sigmoid")` + `fit` con MSE | y = tanh(x), 20/20, MSE < 1e-3 |
| Modos online / minibatch / batch | `batch_size=1 / k / None` | converge en los tres; batch ≡ paso manual |
| History + callbacks (PrintProgress, EarlyStopping) | `core/history.py`, `core/callbacks.py` | 28 tests en `test_training.py` |

Lo que falta para los ejercicios (varias métricas por época, columna `lr`, detección de divergencia, L2, augmentation) se agrega en **F04** sin romper estas interfaces.