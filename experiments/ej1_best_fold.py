"""Ej1 · ¿el "mejor conjunto de entrenamiento" es el fold con mejor validación? (F11, E1-B-b).

    python -m experiments.ej1_best_fold experiments/configs/ej1/best_fold.json
                                        [--smoke] [--force] [--results-dir results]

La config es una config normal del runner con dataset.split kfold y
dataset.holdout_test (el TEST real del Ej1, que acá nunca se evalúa). Cada
semilla de `seeds` es una partición distinta:

1. Se excluye el TEST y, del resto (DESARROLLO), se separa un pseudo-test
   estratificado del 20 % (PSEUDO_TEST_RATIO) con su propio generador.
2. Sobre lo que queda se corre el k-fold de la config (train_run del runner,
   mismo contrato de results/): TEST ∪ pseudo-test se pasan como
   holdout_test.indices_path, así ningún fold los ve.
3. El "mejor fold" es el de menor val_loss. Además se entrena un modelo con
   todo el resto, sin validación, durante la mediana de best_epoch de los folds
   (el mismo criterio que --final-eval).
4. Cada modelo se evalúa en el pseudo-test. Si el fold elegido por su puntaje
   no generaliza mejor que el modelo con todo el resto, elegir la partición
   por su validación es un sesgo optimista.

Escribe en results/<run_name>/:
- <run_id>/ de cada fold y del modelo con todo el resto (como el runner);
- partition_s<seed>/: pseudo_idx.npy, excluded_idx.npy (TEST ∪ pseudo-test),
  pseudo_f<k>.npz y pseudo_rest.npz (idx, y_true, y_score en el pseudo-test)
  y partition.json (val_loss y métricas en el pseudo-test de cada modelo, el
  mejor fold y las épocas del modelo con todo el resto). partition.json se
  escribe último: relanzar saltea las particiones que lo tienen (salvo --force).
"""

import argparse
import copy
import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from data.loaders import Dataset
from data.splits import stratified_holdout
from experiments.config import ConfigError, expand, load_config, resolve_config, run_id
from experiments.runner import (
    SMOKE_DIR,
    RunnerError,
    _Log,
    _strata,
    evaluate_model,
    git_commit,
    holdout_test_indices,
    infer_task,
    load_dataset,
    train_run,
)

PSEUDO_TEST_RATIO = 0.2
# Stream propio del generador del pseudo-test (el runner usa 1, 2 y 3).
_PSEUDO_STREAM = 4


def pseudo_test_indices(y: np.ndarray, test_idx: np.ndarray, seed: int, task: str) -> np.ndarray:
    """Índices (ordenados) del pseudo-test: PSEUDO_TEST_RATIO de DESARROLLO, estratificado.

    y (p,) o (p, 1): target de todo el dataset. test_idx: filas del TEST, que
    quedan afuera. Estratifica igual que el runner (deciles si el target es continuo).
    """
    y = np.asarray(y)
    dev = np.setdiff1d(np.arange(len(y)), test_idx)
    rng = np.random.default_rng([seed, _PSEUDO_STREAM])
    _, pos = stratified_holdout(_strata(y[dev], task), 1.0 - PSEUDO_TEST_RATIO, rng)
    return np.sort(dev[pos])


def _check(config: Mapping[str, Any]) -> None:
    ds = config["dataset"]
    if ds["split"]["kind"] != "kfold":
        raise ConfigError("dataset.split: ej1_best_fold necesita kind 'kfold'")
    if ds["holdout_test"] is None or ds["holdout_test"]["indices_path"] is None:
        raise ConfigError(
            "dataset.holdout_test: ej1_best_fold necesita el TEST con indices_path, "
            "para excluirlo del pseudo-test"
        )


def _save_npy(path: Path, array: np.ndarray) -> None:
    with open(path, "wb") as f:
        np.save(f, array)


def _train(
    run: dict[str, Any],
    out: Path,
    *,
    dataset: Dataset,
    smoke: bool,
    force: bool,
    write: Callable[[str], None],
    commit: str,
) -> dict[str, Any]:
    """train_run de una corrida (o la saltea si ya tiene metrics.json) → su metrics.json."""
    run_dir = out / run_id(run)
    if force or not (run_dir / "metrics.json").exists():
        result = train_run(
            run,
            run_dir,
            smoke=smoke,
            write=lambda line: write("    " + line),
            dataset=dataset,
            commit=commit,
        )
        if result["status"] != "ok":
            raise RunnerError(f"{run_dir}: terminó con status {result['status']!r}")
    return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))


def _pseudo_eval(run_dir: Path, dataset: Dataset, pseudo: np.ndarray, path: Path) -> dict:
    """Métricas del model.npz de run_dir en el pseudo-test; guarda las predicciones en path."""
    y = np.asarray(dataset.y)
    scores, arrays = evaluate_model(run_dir / "model.npz", dataset.X[pseudo], y[pseudo])
    np.savez(path, idx=pseudo, **arrays)
    return scores


def run_partition(
    config: Mapping[str, Any],
    seed: int,
    out: Path,
    *,
    dataset: Dataset,
    smoke: bool = False,
    force: bool = False,
    write: Callable[[str], None] = print,
    commit: str = "",
) -> dict[str, Any]:
    """Una partición (semilla): pseudo-test, k-fold, modelo con todo el resto → partition.json."""
    start = time.perf_counter()
    ds = config["dataset"]
    y = np.asarray(dataset.y)
    task = infer_task(ds, y)
    test_idx = holdout_test_indices(dataset, ds, task)
    pseudo = pseudo_test_indices(y, test_idx, seed, task)

    part = out / f"partition_s{seed}"
    part.mkdir(parents=True, exist_ok=True)
    excluded = np.union1d(test_idx, pseudo)
    _save_npy(part / "pseudo_idx.npy", pseudo)
    _save_npy(part / "excluded_idx.npy", excluded)

    base = copy.deepcopy(dict(config))
    base["seeds"] = [seed]
    base["sweep"] = {}
    base["dataset"]["holdout_test"]["indices_path"] = str(part / "excluded_idx.npy")
    base = resolve_config(base)
    kwargs = {"dataset": dataset, "smoke": smoke, "force": force, "write": write, "commit": commit}

    folds = []
    for run in expand(base):
        write(f"  s{seed} fold {run['fold']}")
        m = _train(run, out, **kwargs)
        folds.append(
            {
                "fold": run["fold"],
                "run": run_id(run),
                "val_loss": m["val"]["loss"],
                "best_epoch": m["best_epoch"],
            }
        )
    best = min(folds, key=lambda f: f["val_loss"])

    rest_config = copy.deepcopy(base)
    epochs = max(1, int(np.round(np.median([f["best_epoch"] for f in folds]))))
    rest_config["training"]["epochs"] = epochs
    rest_config["training"]["early_stopping"] = None
    rest_config["dataset"]["split"] = {"kind": "none"}
    rest_config["logging"]["save_model"] = True
    [rest_run] = expand(resolve_config(rest_config))
    write(f"  s{seed} todo el resto ({epochs} épocas)")
    _train(rest_run, out, **kwargs)

    for f in folds:
        f["pseudo_test"] = _pseudo_eval(
            out / f["run"], dataset, pseudo, part / f"pseudo_f{f['fold']}.npz"
        )
    rest = {
        "run": run_id(rest_run),
        "epochs": epochs,
        "pseudo_test": _pseudo_eval(
            out / run_id(rest_run), dataset, pseudo, part / "pseudo_rest.npz"
        ),
    }
    summary = {
        "seed": seed,
        "pseudo_test_ratio": PSEUDO_TEST_RATIO,
        "n_test_excluded": len(test_idx),
        "n_pseudo_test": len(pseudo),
        "n_rest": len(y) - len(excluded),
        "pseudo_idx_path": str(part / "pseudo_idx.npy"),
        "excluded_idx_path": str(part / "excluded_idx.npy"),
        "folds": folds,
        "best_fold": best["fold"],
        "rest": rest,
        "smoke": smoke,
        "git_commit": commit,
        "seconds": time.perf_counter() - start,
    }
    (part / "partition.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def run_best_fold(
    config_path: str | PathLike,
    *,
    results_dir: str | PathLike = "results",
    smoke: bool = False,
    force: bool = False,
    write: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """Corre cada partición de la config (smoke: solo la primera, en results/_smoke/)."""
    config = load_config(config_path)
    _check(config)
    seeds = config["seeds"][:1] if smoke else config["seeds"]
    root = Path(results_dir)
    out = root / SMOKE_DIR / config["run_name"] if smoke else root / config["run_name"]
    force = force or smoke
    commit = git_commit()
    log = _Log(out / "log.txt", write)
    log(f"inicio: {len(seeds)} partición(es) de {config_path}, commit={commit}")
    dataset = load_dataset(config["dataset"])

    summaries = []
    for i, seed in enumerate(seeds, 1):
        path = out / f"partition_s{seed}" / "partition.json"
        if path.exists() and not force:
            log(f"[{i}/{len(seeds)}] partition_s{seed}: salteada (ya tiene partition.json)")
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
            continue
        summary = run_partition(
            config, seed, out, dataset=dataset, smoke=smoke, force=force, write=write, commit=commit
        )
        log(
            f"[{i}/{len(seeds)}] partition_s{seed}: mejor fold {summary['best_fold']}, "
            f"lectura del pseudo-test (n={summary['n_pseudo_test']}) en {summary['seconds']:.1f} s"
        )
        summaries.append(summary)
    log("fin")
    return summaries


def main(argv: Sequence[str] | None = None) -> int:
    """CLI. Devuelve 0 si terminó, 2 si la config es inválida o falla una corrida."""
    parser = argparse.ArgumentParser(
        prog="python -m experiments.ej1_best_fold",
        description="Mejor fold vs todo el resto, evaluados en un pseudo-test de DESARROLLO.",
    )
    parser.add_argument(
        "config", help="JSON del experimento (experiments/configs/ej1/best_fold.json)"
    )
    parser.add_argument("--smoke", action="store_true", help="solo la primera partición, 5 épocas")
    parser.add_argument("--force", action="store_true", help="rehace particiones y corridas")
    parser.add_argument("--results-dir", default="results", help="carpeta raíz de resultados")
    args = parser.parse_args(argv)
    try:
        run_best_fold(args.config, results_dir=args.results_dir, smoke=args.smoke, force=args.force)
    except (ConfigError, RunnerError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
