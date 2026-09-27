# F03 · Perceptrón multicapa + backprop — ✅ HECHO

Ya implementado (Etapas 2–3 de `docs/PLAN_MOTOR.md`): `core/layers.py` (`Dense`) y `core/network.py` (`Network.backward`).

| Criterio | Estado |
|---|---|
| Forward y backward matriciales, sin loops por muestra | ✅ |
| Gradient check (tanh/sigmoid/relu × MSE/CCE, perceptrón simple, varias ocultas) con error relativo < 1e-6 | ✅ (peor caso 6e-9) |
| El gradient check detecta 4 bugs típicos introducidos a propósito | ✅ |
| Forward a mano `[2,2,1]` | ✅ `test_forward.py` |
| Iteración completa a mano `[2,2,1]` | ✅ `docs/verificacion_manual.md` + test (falta rehacer en papel, F08-T3) |
| XOR `[2,3,2,1]` 20/20 y `[2,2,1]` 17/20 (mínimo local documentado) | ✅ |

## Extras opcionales (no bloquean nada)
- [ ] Test de simetría: con todos los pesos iniciales iguales, las neuronas de una capa oculta quedan idénticas tras entrenar (responde la pregunta de la Clase 11 "¿por qué no inicializar en 0?").
- [ ] Test de performance `@pytest.mark.slow`: forward+backward de `[784, 128, 64, 10]` con lote 128 en < 20 ms.