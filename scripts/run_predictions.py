"""Run every publication prediction from the single frozen final model."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from bt_fit_core import production_branch_name
from prediction_pipeline import (
    NONGROWER_SPECIES,
    PredictionBundle,
    PredictionConfig,
    load_grower_dm_model,
    predict_assemblies_in_spent_media,
    predict_full_communities_across_carbon_sources,
    predict_grower_nongrower_pairwise_in_dm,
    predict_nongrower_assemblies_in_bt,
    predict_random_assemblies_in_dm,
)


PACKAGE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_DIR / "data"
MODEL_DIR = PACKAGE_DIR / "parameters"
RESULTS_DIR = PACKAGE_DIR / "prediction_results"

PREDICTION_PARTS = {
    "bt": "nongrower_assemblies_in_bt_spent",
    "carbon": "full_communities_across_carbon_sources",
    "pairwise": "grower_nongrower_pairwise_in_dm",
    "grower_pairwise": "grower_grower_pairwise_in_dm",
    "dm": "random_core_full_assemblies_in_dm",
    "spent": "assemblies_in_bt_bd_btbd_spent",
}


def load_final_config() -> dict[str, object]:
    return json.loads((PACKAGE_DIR / "frozen_config.json").read_text())


def load_final_bundle() -> PredictionBundle:
    """Load the one retained Bt resource, R, F, D, and manual-lag solution."""

    rate_cr = pd.read_csv(
        MODEL_DIR / "bt_base" / "nongrower_R.csv",
        index_col="species",
    ).reindex(index=NONGROWER_SPECIES)
    resources = rate_cr.columns.tolist()

    y0 = (
        pd.read_csv(MODEL_DIR / "bt_base" / "bt_resource_Y0.csv")
        .set_index("resource_id")["bt_resource_y0"]
        .reindex(resources)
    )
    parameter_path = MODEL_DIR / "final_parameters" / "cofactor_and_lag.csv"
    parameters = (
        pd.read_csv(parameter_path)
        .set_index("species")
        .reindex(NONGROWER_SPECIES)
    )

    required = ["F", "D", "selected_lag_h"]
    if rate_cr.isna().any().any() or y0.isna().any():
        raise ValueError("The frozen Bt resource or nongrower R is incomplete.")
    if parameters[required].isna().any().any():
        raise ValueError("The frozen F, D, or manual lag vector is incomplete.")

    return PredictionBundle(
        rate_cr=rate_cr,
        bt_resource_y0=y0,
        F=parameters["F"],
        D=parameters["D"],
        lag=parameters["selected_lag_h"],
    )


def build_simulation_config(
    final_config: dict[str, object],
) -> PredictionConfig:
    values = final_config["simulation"]
    if not isinstance(values, dict):
        raise TypeError("final_config.json: simulation must be an object.")
    config = PredictionConfig(**values)
    config.validate()
    return config


def _predict_grower_pairwise(
    grower_model,
    config: PredictionConfig,
    output_dir: Path,
    overwrite: bool,
) -> pd.DataFrame:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    template = pd.read_csv(
        DATA_DIR / "grower_pairwise_coculture_relative_abundance_by_pair.csv"
    )
    species = grower_model.rate_cr.index.tolist()
    if template.columns[0] != "species" or template["species"].tolist() != species:
        raise ValueError("Grower-pairwise species order does not match the model.")
    communities = template.columns[1:].tolist()
    observed = template[communities].to_numpy(dtype=float)
    pair_mask = np.isfinite(observed)
    if not np.all(pair_mask.sum(axis=0) == 2):
        raise ValueError("Every grower-pairwise column must contain two species.")
    rate = grower_model.rate_cr.to_numpy(dtype=float)
    fresh_resources = grower_model.dm_resource_y0.to_numpy(dtype=float)

    def derivative(state: np.ndarray, current_rate: np.ndarray) -> np.ndarray:
        n_resources = current_rate.shape[1]
        biomass = state[: len(species)]
        resources = state[len(species):len(species) + n_resources]
        growth_driver = current_rate @ resources
        return np.concatenate(
            (
                biomass * growth_driver,
                -resources * (current_rate.T @ biomass),
            )
        )

    def one_cycle(initial_biomass: np.ndarray) -> np.ndarray:
        current_rate = rate.copy()
        state = np.concatenate((initial_biomass.copy(), fresh_resources.copy()))
        time = 0.0
        while time < config.hours_per_cycle - 1e-15:
            if current_rate.shape[1] == 0:
                break
            step = min(config.dt, config.hours_per_cycle - time)
            k1 = derivative(state, current_rate)
            k2 = derivative(state + 0.5 * step * k1, current_rate)
            k3 = derivative(state + 0.5 * step * k2, current_rate)
            k4 = derivative(state + step * k3, current_rate)
            state = np.maximum(
                state + (step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4),
                0.0,
            )
            n_resources = current_rate.shape[1]
            biomass = state[: len(species)]
            resources = state[len(species):len(species) + n_resources]
            keep = resources > 1e-9
            if not np.all(keep):
                current_rate = current_rate[:, keep]
                state = np.concatenate((biomass, resources[keep]))
            time += step
        return state[: len(species)]

    absolute = np.zeros_like(observed)
    relative = np.full_like(observed, np.nan)
    for community_index in range(len(communities)):
        members = pair_mask[:, community_index]
        initial = np.zeros(len(species))
        initial[members] = config.initial_abundance
        final = initial.copy()
        for _ in range(config.cycles):
            final = one_cycle(initial)
            initial = final / config.dilution
        absolute[:, community_index] = final
        total = float(final[members].sum())
        relative[members, community_index] = np.clip(
            final[members] / total,
            config.prediction_floor,
            None,
        )
    error = np.full_like(observed, np.nan)
    error[pair_mask] = np.log2(
        relative[pair_mask]
        / np.clip(observed[pair_mask], config.prediction_floor, None)
    )
    pair_error = pd.DataFrame(
        {
            "pair": communities,
            "mean_absolute_log2_error": np.nanmean(np.abs(error), axis=0),
        }
    )
    for filename, values in (
        ("predicted_absolute_abundance.csv", absolute),
        ("predicted_relative_abundance.csv", relative),
        ("log2_error.csv", error),
    ):
        frame = template.copy()
        frame.loc[:, communities] = pd.DataFrame(
            values, index=frame.index, columns=communities
        ).where(template[communities].notna())
        frame.to_csv(output_dir / filename, index=False)
    pair_error.to_csv(output_dir / "pair_error.csv", index=False)
    return pair_error


def run_predictions(
    parts: list[str],
    overwrite: bool = False,
) -> dict[str, pd.DataFrame]:
    """Run selected prediction families using no tunable scientific settings."""

    if "all" in parts:
        parts = list(PREDICTION_PARTS)
    unknown = sorted(set(parts) - set(PREDICTION_PARTS))
    if unknown:
        raise ValueError(f"Unknown prediction parts: {unknown}")

    final_config = load_final_config()
    config = build_simulation_config(final_config)
    bundle = load_final_bundle()
    grower_model = load_grower_dm_model(MODEL_DIR / "grower")
    production_settings = final_config["grower_production"]
    if not isinstance(production_settings, dict):
        raise TypeError("frozen_config.json: grower_production must be an object.")
    production_branches = (
        production_branch_name(
            float(production_settings["regularization_weight"])
        ),
    )
    dm_settings = final_config["dm_predictions"]
    if not isinstance(dm_settings, dict):
        raise TypeError("final_config.json: dm_predictions must be an object.")
    additional_lag = float(dm_settings["additional_nongrower_lag_hours"])
    universal_on = bool(dm_settings["nongrower_uses_universal_shared_niche"])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, pd.DataFrame] = {}

    if "bt" in parts:
        outputs["bt"] = predict_nongrower_assemblies_in_bt(
            DATA_DIR,
            bundle,
            RESULTS_DIR / PREDICTION_PARTS["bt"],
            config,
            overwrite=overwrite,
        )

    if "carbon" in parts:
        outputs["carbon"] = predict_full_communities_across_carbon_sources(
            DATA_DIR,
            MODEL_DIR,
            bundle,
            RESULTS_DIR / PREDICTION_PARTS["carbon"],
            config,
            production_branches=production_branches,
            overwrite=overwrite,
        )

    if "pairwise" in parts:
        outputs["pairwise"] = predict_grower_nongrower_pairwise_in_dm(
            DATA_DIR,
            MODEL_DIR,
            grower_model,
            bundle,
            RESULTS_DIR / PREDICTION_PARTS["pairwise"],
            config,
            production_branches=production_branches,
            additional_lags_h=(additional_lag,),
            include_universal_off=not universal_on,
            overwrite=overwrite,
        )

    if "grower_pairwise" in parts:
        outputs["grower_pairwise"] = _predict_grower_pairwise(
            grower_model,
            config,
            RESULTS_DIR / PREDICTION_PARTS["grower_pairwise"],
            overwrite,
        )

    if "dm" in parts:
        outputs["dm"] = predict_random_assemblies_in_dm(
            DATA_DIR,
            MODEL_DIR,
            grower_model,
            bundle,
            RESULTS_DIR / PREDICTION_PARTS["dm"],
            config,
            production_branches=production_branches,
            additional_lags_h=(additional_lag,),
            universal_shared_options=(universal_on,),
            overwrite=overwrite,
        )

    if "spent" in parts:
        outputs["spent"] = predict_assemblies_in_spent_media(
            DATA_DIR,
            MODEL_DIR,
            grower_model,
            bundle,
            RESULTS_DIR / PREDICTION_PARTS["spent"],
            config,
            production_branches=production_branches,
            overwrite=overwrite,
        )

    index_rows = []
    for part, frame in outputs.items():
        index_rows.append(
            {
                "prediction_family": part,
                "output_directory": PREDICTION_PARTS[part],
                "summary_rows": len(frame),
            }
        )
    pd.DataFrame(index_rows).to_csv(
        RESULTS_DIR / "prediction_index.csv",
        index=False,
    )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run predictions from the frozen publication model."
    )
    parser.add_argument(
        "--parts",
        nargs="+",
        default=["all"],
        choices=["all", *PREDICTION_PARTS],
        help="Prediction families to run (default: all).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace outputs for the selected prediction families.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_predictions(args.parts, overwrite=args.overwrite)
    print("Completed:", ", ".join(outputs))
    print("Saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
