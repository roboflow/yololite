"""CSV resume, dataset discovery, and config-loading helpers."""

import os
from pathlib import Path

import pandas as pd


# ── CSV resume ───────────────────────────────────────────────────────────────

def load_variant_csv(csv_path: str) -> list[dict]:
    """Load a per-variant CSV into a list of row dicts.  Returns [] if missing."""
    if os.path.isfile(csv_path):
        return pd.read_csv(csv_path).to_dict("records")
    return []


def save_variant_csv(rows: list[dict], csv_path: str) -> None:
    """Save rows to a CSV file (overwrites)."""
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def build_completed_set(
    variant_rows: dict[str, list[dict]],
    key_fields: tuple[str, ...] = ("dataset", "variant"),
) -> set[tuple]:
    """Build a set of completed-run keys from the variant_rows mapping."""
    completed: set[tuple] = set()
    for rows in variant_rows.values():
        for r in rows:
            completed.add(tuple(r[k] for k in key_fields))
    return completed


# ── Dataset discovery ────────────────────────────────────────────────────────

def find_data_yaml(dataset_dir: str) -> str:
    """Return the path to data.yaml inside a dataset directory."""
    candidate = os.path.join(dataset_dir, "data.yaml")
    if os.path.isfile(candidate):
        return candidate
    for p in Path(dataset_dir).glob("*/data.yaml"):
        return str(p)
    raise FileNotFoundError(f"No data.yaml found in {dataset_dir}")


# ── Config loading ───────────────────────────────────────────────────────────

def load_training_config(
    variant_name: str,
    data_yaml: str,
    log_dir: str,
    *,
    epochs: int,
    batch_size: int,
    img_size: int,
    device: str,
    save_every: int | None = None,
    save_by: str = "AP",
    num_workers: int | None = None,
    resume: str | None = None,
) -> dict:
    """Build a fully configured training config dict for a variant.

    Resolves model YAML and train YAML from the package configs, calls
    ``load_configs()``, and applies the standard training overrides.
    """
    from importlib.resources import files as _pkg_files

    from yololite.scripts.args.build_args import load_configs

    from ._variants import YOLOLITE_VARIANTS

    subdir, yaml_file, _jobs_per_gpu = YOLOLITE_VARIANTS[variant_name]

    configs_root = _pkg_files("yololite").joinpath("configs")
    model_yaml = str(configs_root / subdir / yaml_file)
    train_yaml = str(configs_root / "train" / "standard_train.yaml")

    config = load_configs(
        model_yaml=model_yaml,
        train_yaml=train_yaml,
        data_yaml=data_yaml,
        log_dir=log_dir,
    )

    config["training"]["epochs"] = epochs
    config["training"]["batch_size"] = batch_size
    config["training"]["img_size"] = img_size
    config["training"]["device"] = device
    config["training"]["save_by"] = save_by
    if save_every is not None:
        config["training"]["save_every"] = save_every
    if num_workers is not None:
        config["training"]["num_workers"] = num_workers
    if resume is not None:
        config["training"]["resume"] = resume

    return config
