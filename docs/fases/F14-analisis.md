# F14 · Análisis consolidado y figuras finales

| | |
|---|---|
| **Objetivo** | Dejar todas las figuras y tablas de la presentación reproducibles con un solo comando, con estilo uniforme y trazables a los requerimientos |
| **Requerimientos** | RNF-05 · Constitución C4 y C6 |
| **Depende de** | F10–F13 (se va construyendo en paralelo) |
| **Estimación** | 3–4 h |

## Especificación
- `python -m analysis.figures` → corre `analysis/validacion.py`, `ej1_eda.py`, `ej1_learning.py`, `ej1_generalization.py`, `ej2_eda.py`, `ej2.py`, `ej3.py` y regenera `figures/**`.
- Cada figura:
  - nombre con el ID del requerimiento: `figures/ej2/E2-b1_lr_curves.png`
  - título, ejes con unidades, leyenda, **media ± desvío** (banda sombreada) cuando hay semillas, `n` de semillas en el pie
  - legible en proyector (fuente ≥ 14 pt en ejes), paleta consistente entre ejercicios (mismo color = mismo modelo/optimizador en todo el TP)
  - PNG 200 dpi + PDF
- `figures/INDEX.md` generado automáticamente: tabla ID → archivo → run(s) de origen → una línea de descripción.
- `docs/resultados/resumen.md`: los números clave de todo el TP en una página (lo que se dice en voz alta en la presentación).

## Chequeo de coherencia (checklist)
- [ ] Toda pregunta de `02-requerimientos.md` §3 tiene al menos una figura/tabla en `INDEX.md`.
- [ ] Todo número de `docs/resultados/*.md` sale de un `summary.csv` o `final_eval.json` (citar la ruta).
- [ ] Ninguna figura usa datos de test salvo las marcadas como "final".
- [ ] Los colores/estilos son consistentes entre Ej1, Ej2 y Ej3.
- [ ] `python -m analysis.figures` corre en limpio desde un clone nuevo + `results/` (sin reentrenar).

## Entregables
Scripts + `figures/` + `INDEX.md` + `resumen.md`, en una rama lista para PR. El agente propone los mensajes de commit (`feat(F14): ...`) y no commitea sin permiso (`CLAUDE.md` §0).