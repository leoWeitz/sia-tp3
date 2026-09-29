# F07 · Runner, configs, guardado/carga y agregación

| | |
|---|---|
| **Objetivo** | Lanzar barridos reproducibles desde JSON, ver el progreso, guardar todo lo crudo, reanudar entrenamientos y agregar semillas |
| **Requerimientos** | RNF-02 a RNF-06 · `CLAUDE.md` §6–§7 · `03-arquitectura.md` §4–§5 · Constitución C4–C6 |
| **Depende de** | F04 (optimizadores con `state_dict`, `fit(metrics=...)`). Usa F05/F06: **se puede empezar con los datasets sintéticos** y conectar `data/` y `core/metrics.py` cuando estén mergeados |
| **Estimación** | 1 día con el agente |

## Especificación

### `core/serialization.py`
```python
save_checkpoint(path, network, optimizer=None, extra: dict | None = None) -> None
load_checkpoint(path) -> tuple[Network, Optimizer | None, dict]
```
- `np.savez` con: pesos y bias por capa, `layer_sizes`, nombres y parámetros (β) de activaciones, inicializador, estado del optimizador (`state_dict`) y un JSON embebido con `extra` (época, estado de callbacks como `best`/contadores de η adaptativo, estado del `rng` vía `rng.bit_generator.state`).
- `load_checkpoint` reconstruye una red **entrenable** (no solo para predecir).

### `experiments/config.py`
- `load_config(path) -> dict`: JSON → defaults (`03-arquitectura.md` §4, más `dataset.format`: `"tabular"` | `"digits"`) → **validación** con errores legibles: clave desconocida, tipo inválido, `softmax` sin `categorical_crossentropy`, `test_path`/`holdout_test` evaluado sin `--final-eval`, `layers[-1]` incompatible con el target.
- `expand(config) -> list[dict]`: producto cartesiano de `sweep` (notación punto) × `seeds` [× folds si `split.kind == "kfold"`].
- `config_hash(config) -> str`: sha1 de la config resuelta **sin** semilla ni fold, 8 caracteres.
- `build(config, rng) -> (network, optimizer, loss, callbacks, metrics, augment)` usando los registros `get_activation`, `get_loss`, `get_optimizer`, `get_metric`, `get_augmentation`.

### `experiments/runner.py` (CLI)
```
python -m experiments.runner <config.json> [--smoke] [--resume <run_dir>] [--final-eval] [--force]
                                           [--only "clave=valor"] [--workers N]
```
Por cada corrida de `expand(config)`:
1. Si `results/<run_name>/<hash8>_s<seed>[_f<k>]/metrics.json` existe y no hay `--force` → saltear.
2. Cargar datos según `dataset.format` (clave nueva, default `"tabular"` → `load_csv`; `"digits"` → `load_digits_csv`; o `dataset.synthetic`), aplicar `holdout_test` (excluir esos índices), split → `prepare_fold`.
3. `rng = np.random.default_rng(seed)`; construir todo con `build`.
4. `fit` con `PrintProgress` (o `tqdm` si se agrega a `requirements.txt`), checkpoint cada `logging.checkpoint_every` épocas vía un callback `Checkpoint`.
5. Guardar `config.json`, `history.csv` (vía `History.to_csv`), `metrics.json` (métricas finales train/val con `classification_summary`, `best_epoch`, tiempo total, s/época, `n_params`, commit de git con `git rev-parse --short HEAD` o `"no-git"`, sha256 de los datos, `status`), `predictions.npz` (si `save_predictions`) y `model.npz` (si `save_model`).
- `--smoke`: primera config del sweep, 1 semilla, `epochs = min(epochs, 5)`, máx. 500 muestras, a `results/_smoke/`. Debe tardar < 60 s.
- `--resume <run_dir>`: carga `checkpoint.npz` y continúa hasta completar las épocas de la config.
- `--final-eval`: exige `dataset.test_path` (o `holdout_test`) y `selected_from`; entrena con todo el conjunto de desarrollo durante `best_epoch` (mediana de los runs seleccionados) o reusa el modelo, según config; evalúa **una vez** en test; escribe `results/<run_name>/final_eval.json`.
- `--workers N`: corridas en paralelo con `concurrent.futures.ProcessPoolExecutor` (cada una con su semilla).
- Log de barrido: `results/<run_name>/log.txt` con corrida i/N y tiempo estimado restante.

### `analysis/common.py` y `analysis/aggregate.py`
- `load_runs(results_dir) -> DataFrame`: config aplanada + métricas finales + meta, una fila por run.
- `load_histories(results_dir) -> DataFrame` en formato largo (`run`, `hash`, `seed`, `epoch`, `metric`, `value`).
- `aggregate(df, by) -> DataFrame` con media, desvío, min, max, n.
- `python -m analysis.aggregate results/<run_name>` → `summary.csv`.
- Estilo base de matplotlib compartido (fuente ≥ 14 pt, paleta fija, PNG 200 dpi + PDF).

### Configs de ejemplo
`experiments/configs/validacion/{and_step,linear_identity,nonlinear_tanh,xor_221,xor_2321}.json` usando `dataset.synthetic`. El escalón usa `"trainer": "perceptron"` para que el runner llame a `fit_perceptron`.

## Tests y criterios de aceptación (`tests/test_experiments.py`, `tests/test_serialization.py`, `tests/test_analysis.py`)
- [x] `expand` con 2 claves × 3 valores × 2 semillas → 18 configs; mismo hash para distinta semilla. (`test_expand_2_claves_x_3_valores_x_2_semillas`)
- [x] Validación rechaza: clave desconocida, softmax + mse, `test_path` sin `--final-eval`. (`test_validacion_rechaza_con_mensaje_que_nombra_la_clave`, 19 casos)
- [x] Corrida end-to-end de XOR desde JSON (en `tmp_path`) genera todos los archivos de `CLAUDE.md` §7. (`test_corrida_end_to_end_de_xor_genera_todos_los_archivos`)
- [x] **Resume idéntico:** 10 épocas de corrido ≡ 5 + cortar + `--resume` + 5 (pesos finales iguales, `atol=1e-12`), con GD y con Adam. (`test_resume_identico_a_no_cortar`: corte simulado en la época 6 de una corrida de 10 con checkpoint en la 5, con mini-batch, early stopping y η adaptativo; también coinciden el estado del optimizador y el historial)
- [x] `save_checkpoint` → `load_checkpoint` → mismas predicciones y el optimizador conserva su estado. (`tests/test_serialization.py`)
- [x] Relanzar un barrido no re-entrena runs existentes. (`test_relanzar_un_barrido_no_reentrena`)
- [x] `aggregate` calcula media/desvío correctos en un caso sintético. (`test_aggregate_media_desvio_min_max_y_n`)
- [x] Todas las configs de `validacion/` corren con `--smoke` en < 60 s. (`test_configs_de_validacion_corren_con_smoke_en_menos_de_60_s`: ~1 s en total)

Decisiones tomadas al implementar: ver `03-arquitectura.md` §3 (F07) y §4. Reanudar con augmentation no es idéntico bit a bit (el generador hijo de `fit` no se guarda); sin augmentation sí.

## Entregables
Rama `f07-runner` lista para PR + sección "Cómo correr un experimento" en el `README.md`. El agente propone los mensajes de commit (`feat(F07): ...`) y no commitea sin permiso (`CLAUDE.md` §0).