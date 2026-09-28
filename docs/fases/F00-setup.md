# F00 · Setup del repositorio — ✅ HECHO

Ya implementado (Etapa 0 de `docs/PLAN_MOTOR.md`), más los ajustes de integración de los docs.

| Criterio | Estado |
|---|---|
| Estructura `core/`, `data/`, `experiments/`, `analysis/`, `tests/`, `results/`, `docs/` | ✅ |
| `requirements.txt`, `pyproject.toml` (pytest + ruff) | ✅ |
| CI en GitHub Actions (`ruff check` + `pytest` en cada push/PR) | ✅ |
| `CLAUDE.md` en la raíz | ✅ (secciones 0 y 12) |
| `datasets/` con `.gitkeep`; los 4 CSV (`fraud_dataset.csv`, `digits.csv`, `digits_test.csv`, `more_digits.csv`) se copian localmente y **no se versionan** | ✅ |
| Nombres reales de los CSV = los del enunciado | ✅ |
| `docs/datos/fraud_dataset_documentation.pdf` (documentación oficial) | ✅ |
| `.gitignore`: `datasets/*` ignorado; en `results/` se versionan `config.json`, `metrics.json`, `summary.csv` y `final_eval.json` (la regla anterior no los versionaba por estar dentro de carpetas ignoradas) | ✅ |
| Carpetas `notebooks/`, `figures/` y `models/` con `.gitkeep` | ✅ |

## Pendiente (opcional)
- [ ] CI con la misma versión de Python que usa el equipo localmente; hoy es 3.10.