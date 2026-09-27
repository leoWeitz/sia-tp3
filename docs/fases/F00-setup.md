# F00 · Setup del repositorio — ✅ HECHO

Ya implementado (Etapa 0 de `docs/PLAN_MOTOR.md`).

| Criterio | Estado |
|---|---|
| Estructura `core/`, `data/`, `experiments/`, `analysis/`, `tests/`, `results/`, `docs/` | ✅ |
| `requirements.txt`, `pyproject.toml` (pytest + ruff) | ✅ |
| CI en GitHub Actions (`ruff check` + `pytest` en cada push/PR) | ✅ |
| `CLAUDE.md` en la raíz | ✅ (actualizado con las secciones 0 y 12) |

## Pendiente menor (lo hace quien integre los docs)
- [ ] Crear `datasets/` (está en `.gitignore`) y copiar los CSV del campus. Confirmar nombres reales: si difieren del enunciado, avisar al equipo y corregir `CLAUDE.md` §1.
- [ ] Agregar a `.gitignore` la excepción `!results/**/summary.csv` y `!results/**/final_eval.json`.
- [ ] Crear carpetas `notebooks/` y `figures/` (con `.gitkeep`).
- [ ] (Opcional) CI con la misma versión de Python que usa el equipo localmente; hoy es 3.10.