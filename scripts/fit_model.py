"""Re-fit every model parameter except the fixed lag vector."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from bt_fit_core import (
    BtBaseConfig,
    ProductionConfig,
    fit_bt_base,
    fit_production_branches,
)
from cofactor_fit_core import COFACTOR_SPECIES, fit_cofactors
from grower_fit_core import fit_grower_dm


PACKAGE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_DIR / "data"
PARAMETER_DIR = PACKAGE_DIR / "parameters"
NONGROWER_SPECIES = (
    "Af", "Ao", "As", "Bl.s", "Col", "Cs", "Et", "Eu.c",
    "Eu.l", "Im", "Ld", "Lsp", "Mi", "Pc", "Va", "Vp",
)


def _load_config() -> dict[str, object]:
    path = PACKAGE_DIR / "frozen_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("frozen_config.json must contain one JSON object.")
    return config


def _path_from_package(value: object) -> Path:
    path = (PACKAGE_DIR / str(value)).resolve()
    if PACKAGE_DIR.resolve() not in path.parents:
        raise ValueError(f"Configured path must stay inside the package: {value}")
    return path


def _fit_bt(config: dict[str, object], fixed: dict[str, object]):
    values = config["bt_resource_fit"]
    auc = config["resource_auc"]
    if not isinstance(values, dict) or not isinstance(auc, dict):
        raise TypeError("Bt fitting settings must be JSON objects.")
    fixed_values = values.get("fixed_resource_abundances", {})
    if not isinstance(fixed_values, dict):
        raise TypeError("fixed_resource_abundances must be a JSON object.")
    bt_config = BtBaseConfig(
        floor=float(values["floor"]),
        resource_reciprocal_abs_log2=float(values["reciprocal_abs_log2"]),
        resource_reciprocal_abs_difference=float(
            values["reciprocal_abs_difference"]
        ),
        resource_min_pair_growth=float(values["minimum_pair_growth"]),
        resource_max_nfev=100000,
        resource_lp_prediction_tolerance=1e-8,
        resource_pair_excluded_species=tuple(values["pair_excluded_species"]),
        fixed_resource_abundances=tuple(
            (str(name), float(amount))
            for name, amount in fixed_values.items()
        ),
        auc_initial_abundance=float(auc["initial_abundance"]),
        auc_cycles=int(auc["cycles"]),
        auc_hours_per_cycle=float(auc["hours_per_cycle"]),
        auc_dt=float(auc["dt"]),
        auc_dilution=float(auc["dilution"]),
        auc_mu_threshold=float(auc["mu_threshold"]),
    )
    return fit_bt_base(
        DATA_DIR,
        PARAMETER_DIR / "bt_base",
        bt_config,
        overwrite=True,
        binary_cr_path=_path_from_package(
            fixed["nongrower_binary_consumption_matrix"]
        ),
    )


def _fit_grower(config: dict[str, object], fixed: dict[str, object]):
    values = config["grower_dm_fit"]
    if not isinstance(values, dict):
        raise TypeError("grower_dm_fit must be a JSON object.")
    return fit_grower_dm(
        DATA_DIR,
        _path_from_package(fixed["grower_binary_consumption_matrix"]),
        PARAMETER_DIR / "grower",
        floor=float(values["floor"]),
        overwrite=True,
    )


def _fit_production(config: dict[str, object], bt_model) -> None:
    values = config["grower_production"]
    if not isinstance(values, dict):
        raise TypeError("grower_production must be a JSON object.")
    production_config = ProductionConfig(
        prior_weights=(float(values["regularization_weight"]),),
        clip_min=float(values["clip_min"]),
        profile_floor=float(values["profile_floor"]),
        multi_start=20,
        seed=2431,
        maxiter=5000,
        maxfun=200000,
    )
    output_root = PARAMETER_DIR / "production_fits"
    if output_root.exists():
        shutil.rmtree(output_root)
    fit_production_branches(
        DATA_DIR,
        bt_model,
        output_root,
        production_config,
        overwrite=True,
    )


def _fit_cofactor(
    config: dict[str, object], fixed: dict[str, object]
) -> pd.DataFrame:
    values = config["cofactor_fit"]
    if not isinstance(values, dict):
        raise TypeError("cofactor_fit must be a JSON object.")
    return fit_cofactors(
        DATA_DIR,
        PARAMETER_DIR / "bt_base",
        _path_from_package(fixed["nongrower_binary_consumption_matrix"]),
        PARAMETER_DIR / "cofactor_fit",
        floor=float(values["floor"]),
        F0=float(values["F0"]),
        C0=float(values["C0"]),
        lambda_F=float(values["lambda_F"]),
        lambda_C=float(values["lambda_C"]),
        lambda_ineq=float(values["lambda_ineq"]),
        active_margin_log2=float(values["active_margin_log2"]),
        F_bounds=tuple(float(value) for value in values["F_bounds"]),
        C_bounds=tuple(float(value) for value in values["C_bounds"]),
    )


def _compose_final_parameters(
    cofactor: pd.DataFrame,
    fixed_lag_path: Path,
) -> pd.DataFrame:
    lag = (
        pd.read_csv(fixed_lag_path)
        .set_index("species")["lag_h"]
        .apply(pd.to_numeric, errors="coerce")
        .reindex(NONGROWER_SPECIES)
    )
    fitted = cofactor.set_index("species").reindex(COFACTOR_SPECIES)
    if lag.isna().any() or fitted[["F", "C", "D"]].isna().any().any():
        raise ValueError("Fixed lag or fitted cofactor parameters are incomplete.")
    rows = []
    for species in NONGROWER_SPECIES:
        participates = species in COFACTOR_SPECIES
        rows.append(
            {
                "species": species,
                "participates_in_cofactor_fit": participates,
                "F": float(fitted.loc[species, "F"]) if participates else 0.0,
                "C": float(fitted.loc[species, "C"]) if participates else 1.0,
                "D": float(fitted.loc[species, "D"]) if participates else 0.0,
                "selected_lag_h": float(lag.loc[species]),
            }
        )
    result = pd.DataFrame(rows)
    output_dir = PARAMETER_DIR / "final_parameters"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "cofactor_and_lag.csv", index=False)
    return result


def fit_all() -> None:
    config = _load_config()
    fixed = config["fixed_inputs"]
    if not isinstance(fixed, dict):
        raise TypeError("fixed_inputs must be a JSON object.")
    bt_model = _fit_bt(config, fixed)
    _fit_grower(config, fixed)
    _fit_production(config, bt_model)
    cofactor = _fit_cofactor(config, fixed)
    final = _compose_final_parameters(
        cofactor, _path_from_package(fixed["lag_time"])
    )
    print(
        "Final parameter table complete: "
        f"{len(final)} nongrowers; fixed lag loaded from {fixed['lag_time']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit all continuous parameters using the fixed binary matrices."
    )
    return parser.parse_args()


def main() -> None:
    parse_args()
    fit_all()


if __name__ == "__main__":
    main()
