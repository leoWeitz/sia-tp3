# F01 · Núcleo numérico — ✅ HECHO

Ya implementado (Etapa 1 de `docs/PLAN_MOTOR.md`): `core/activations.py`, `core/losses.py`, `core/initializers.py` con registros por nombre y tests (`test_activations.py`, `test_losses.py`, `test_initializers.py`).

| Spec original | Estado | Nota |
|---|---|---|
| step, identity, tanh(β), logística(β), relu, softmax | ✅ | La logística usa β **sin** el 2 de la cátedra → se alinea en **F04-T0** |
| Derivadas vs diferencias finitas | ✅ | ε = 1e-5, tol 1e-7 |
| Estabilidad numérica (sigmoid, softmax) | ✅ | |
| MSE, BCE, CCE + `softmax_delta` | ✅ | MSE promediada sin ½ (`04-matematica.md` §2.4) |
| uniform, xavier, he + reproducibilidad por semilla | ✅ | |

No hay nada que implementar en esta fase.