# 01 · Constitución del proyecto

Complementa `CLAUDE.md` §0 (git), §2 (reglas duras) y §5 (convenciones). Si una fase choca con una regla, se para y se discute con el equipo.

## C0 · Git (máxima prioridad)
- El agente **nunca** hace commit, push, merge, rebase, reset, tags ni PRs sin permiso explícito en el chat para esa acción concreta.
- Si se le da permiso: **nunca** agrega `Co-Authored-By`, "Generated with Claude Code" ni ninguna mención a Claude, Anthropic o IA en commits, PRs, código o comentarios.
- Al terminar una tarea, muestra `git status` / `git diff --stat` y propone el mensaje de commit.

## C1 · Stack
- Python ≥ 3.10, NumPy, pandas, matplotlib, pytest, ruff (lo que ya está en `requirements.txt`). Si hace falta `tqdm`, se agrega a `requirements.txt` en F07.
- Modelos, optimizadores, splits y métricas: **implementación propia** con NumPy. scikit-learn solo en `tests/` como oráculo.
- Todo cálculo del modelo es **matricial**. Loops permitidos: épocas, lotes, capas.

## C2 · Respetar el motor existente
- `core/` ya tiene interfaces probadas (154 tests en verde). Las fases nuevas **extienden** sin romper: toda firma pública existente se mantiene o se extiende con parámetros opcionales con default.
- Si una extensión obliga a cambiar un test existente, se explica en el PR y lo revisa otra persona del equipo.
- No se reestructura el repo (nada de `src/`, ni renombres).

## C3 · Fidelidad a la cátedra
- Fórmulas y equivalencias en `04-matematica.md`. Donde el código usa otra convención (MSE promediada sin ½, logística), la equivalencia con la teórica queda documentada y se menciona en la presentación.
- Variantes fuera de la teoría (softmax + CE, Xavier/He, ReLU) se reportan como variantes, comparadas contra la configuración "de la cátedra" (sigmoide/tanh + MSE).

## C4 · Reproducibilidad
- Toda aleatoriedad sale de un `np.random.Generator` sembrado. Nada de `np.random.seed`.
- ≥ 3 semillas por configuración; se reporta media ± desvío.
- Cada corrida guarda su config resuelta, commit de git y tiempos (`CLAUDE.md` §7).

## C5 · Disciplina de datos
- `datasets/*.csv` de solo lectura.
- **Test sagrado:** `digits_test.csv` y el holdout de test del Ej1 solo se leen con `--final-eval`, una vez por ejercicio, con la configuración ya elegida.
- Normalización ajustada solo con el train de cada partición/fold.
- Particiones de clasificación estratificadas por defecto.

## C6 · Separación experimento / análisis
- `experiments/` corre y guarda crudos; `analysis/` lee y grafica. Ninguna figura de la presentación se hace a mano.

## C7 · Calidad y flujo
- `pytest` y `ruff check . && ruff format --check .` en verde para cerrar una fase (el CI lo verifica en cada PR).
- Una rama por fase, un commit por tarea (`feat(F05): ...`), PR revisado por otra persona. Commits y pushes los decide el equipo (C0).
- Tests primero: cada criterio de aceptación de la spec es un test.

## C8 · Trabajo con el agente
- Una fase por sesión; lee `CLAUDE.md` y los docs antes de empezar.
- Ante ambigüedad pregunta; las decisiones menores las propone en el chat.
- No lanza barridos largos sin OK (primero `--smoke` + estimación de tiempo).
- Al cerrar, tilda los criterios de aceptación de la spec, actualiza la columna Estado de `docs/README.md` y propone los mensajes de commit, sin ejecutarlos (C0).

## C9 · Trabajo en paralelo
Cada fase toca un conjunto acotado de archivos (`03-arquitectura.md` §1). Si una fase necesita modificar un archivo de otra fase que está en curso, se avisa al equipo antes y se hace en un PR chico separado.