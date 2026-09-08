#!/usr/bin/env python3
"""Scan Bt--nongrower initial-condition dependence in 90% DM + 10% Bt spent.

The implementation reuses the frozen production/cofactor equations from
the local batched dynamics kernel while adding cycle-level endpoint
extinction and complete trajectory summaries for the requested scan.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from dynamics import (
    _rk4_pairwise_production_batch,
)


ROOT = Path(__file__).resolve().parent.parent
FINALIZED_DIR = ROOT
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"

COFACTOR_SPECIES = (
    "Col",
    "Cs",
    "Et",
    "Eu.c",
    "Eu.l",
    "Im",
    "Ld",
    "Lsp",
    "Mi",
    "Va",
)
INITIAL_CONDITIONS = {
    "ng_low_bt_high": (1e-3, 1.0),
    "ng_high_bt_low": (1.0, 1e-3),
}
LAG_MODES = ("off", "on")
EXTINCTION_MODES = ("hard_extinction", "no_extinction_control")


@dataclass(frozen=True)
class ScanConfig:
    hours_per_cycle: float = 48.0
    max_cycles: int = 20
    dilution: float = 200.0
    endpoint_extinction_threshold: float = 1e-4
    max_step_h: float = 0.04
    adaptive_safety: float = 0.05
    minimum_step_h: float = 1e-8
    uptake_resource_sum_threshold: float = 1e-3
    convergence_tolerance_log10: float = 1e-4
    convergence_consecutive_cycles: int = 5
    convergence_log_floor: float = 1e-12
    dm_fraction: float = 0.9
    bt_spent_fraction: float = 0.1

    def validate(self) -> None:
        if self.hours_per_cycle <= 0 or self.max_cycles <= 0:
            raise ValueError("Cycle duration and count must be positive.")
        if self.dilution <= 0 or self.endpoint_extinction_threshold < 0:
            raise ValueError("Dilution must be positive and threshold nonnegative.")
        if not (0 < self.max_step_h and 0 < self.adaptive_safety < 1):
            raise ValueError("Invalid integration step settings.")
        if self.minimum_step_h <= 0 or self.minimum_step_h > self.max_step_h:
            raise ValueError("minimum_step_h must lie in (0, max_step_h].")
        if not np.isclose(self.dm_fraction + self.bt_spent_fraction, 1.0):
            raise ValueError("Medium fractions must sum to one.")


@dataclass
class FrozenParameters:
    nongrower_rate_cr: pd.DataFrame
    bt_dm_rate: pd.Series
    dm_y0: pd.Series
    bt_spent_y0: pd.Series
    bt_production: pd.Series
    cofactor: pd.DataFrame


def make_ratio_grid() -> np.ndarray:
    """Return 100 ratios: exact zero plus 99 log points including one."""

    positive = np.logspace(-2.0, 2.0, 99)
    if not np.isclose(positive[49], 1.0):
        raise AssertionError("The positive ratio grid must contain exact ratio 1.")
    return np.concatenate(([0.0], positive))


def _read_indexed_csv(path: Path, index_col: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if index_col not in frame.columns:
        raise ValueError(f"{path} is missing index column {index_col!r}.")
    return frame.set_index(index_col)


def load_frozen_parameters(finalized_dir: Path = FINALIZED_DIR, *, parameter_dir: Path | None = None) -> FrozenParameters:
    """Load and validate the exact frozen parameter files used by the scan."""

    model_dir = Path(parameter_dir) if parameter_dir is not None else Path(finalized_dir) / "parameters"
    ng_cr = _read_indexed_csv(
        model_dir / "bt_base" / "nongrower_R.csv", "species"
    ).loc[list(COFACTOR_SPECIES)]
    grower_cr = _read_indexed_csv(
        model_dir / "grower" / "model" / "grower_R.csv", "species"
    )
    bt_dm_rate = grower_cr.loc["Bt"].astype(float)
    dm_y0 = _read_indexed_csv(
        model_dir / "grower" / "model" / "dm_resource_y0.csv", "resource_id"
    )["dm_y0"].astype(float).reindex(bt_dm_rate.index)
    bt_y0 = _read_indexed_csv(
        model_dir / "bt_base" / "bt_resource_Y0.csv", "resource_id"
    )["bt_resource_y0"].astype(float).reindex(ng_cr.columns)
    production = _read_indexed_csv(
        model_dir
        / "production_fits"
        / "prod_prior-0p3"
        / "production_profiles.csv",
        "resource_id",
    )["Bt"].astype(float).reindex(ng_cr.columns)
    cofactor = _read_indexed_csv(
        model_dir / "final_parameters" / "cofactor_and_lag.csv", "species"
    ).loc[list(COFACTOR_SPECIES)]

    if ng_cr.isna().any().any() or dm_y0.isna().any() or bt_y0.isna().any():
        raise ValueError("Frozen resource tables are incomplete after reindexing.")
    if production.isna().any() or cofactor[["F", "D", "selected_lag_h"]].isna().any().any():
        raise ValueError("Frozen production/cofactor parameters are incomplete.")
    if np.any(ng_cr.to_numpy(dtype=float) < 0) or np.any(bt_dm_rate < 0):
        raise ValueError("Consumption rates must be nonnegative.")

    bt_support = bt_dm_rate.to_numpy(dtype=float) > 0
    if int(bt_support.sum()) != 6:
        raise ValueError(f"Expected Bt to consume 6 DM resources, got {bt_support.sum()}.")
    bt_positive = bt_dm_rate.to_numpy(dtype=float)[bt_support]
    if not np.allclose(bt_positive, bt_positive[0], rtol=1e-10, atol=1e-12):
        raise ValueError("Frozen Bt DM rates are not constant over Bt support.")

    for species, row in ng_cr.iterrows():
        positive = row.to_numpy(dtype=float)
        positive = positive[positive > 0]
        if positive.size == 0 or not np.allclose(
            positive, positive[0], rtol=1e-10, atol=1e-12
        ):
            raise ValueError(
                f"{species} lacks one common frozen Bt-product consumption rate."
            )

    return FrozenParameters(
        nongrower_rate_cr=ng_cr.astype(float),
        bt_dm_rate=bt_dm_rate,
        dm_y0=dm_y0,
        bt_spent_y0=bt_y0,
        bt_production=production,
        cofactor=cofactor,
    )


def common_positive_rate(row: pd.Series) -> float:
    positive = row.to_numpy(dtype=float)
    positive = positive[positive > 0]
    if positive.size == 0 or not np.allclose(positive, positive[0]):
        raise ValueError("Expected one common positive nongrower rate.")
    return float(positive[0])


def build_parameter_manifest(parameters: FrozenParameters) -> pd.DataFrame:
    bt_support = parameters.bt_dm_rate[parameters.bt_dm_rate > 0]
    rows = []
    for species in COFACTOR_SPECIES:
        row = parameters.nongrower_rate_cr.loc[species]
        rows.append(
            {
                "species": species,
                "r_ng_bt_production": common_positive_rate(row),
                "n_bt_product_resources_consumed": int((row > 0).sum()),
                "selected_lag_h": float(
                    parameters.cofactor.loc[species, "selected_lag_h"]
                ),
                "F": float(parameters.cofactor.loc[species, "F"]),
                "D": float(parameters.cofactor.loc[species, "D"]),
                "bt_dm_rate": float(bt_support.iloc[0]),
                "n_bt_dm_resources": len(bt_support),
                "bt_dm_resources": "|".join(bt_support.index),
            }
        )
    return pd.DataFrame(rows)


def build_run_table(
    parameters: FrozenParameters,
    ratios: Iterable[float],
    species: Iterable[str] = COFACTOR_SPECIES,
) -> pd.DataFrame:
    rows = []
    for name in species:
        base_rate = common_positive_rate(parameters.nongrower_rate_cr.loc[name])
        selected_lag = float(parameters.cofactor.loc[name, "selected_lag_h"])
        for ratio in ratios:
            for lag_mode in LAG_MODES:
                lag_h = 0.0 if lag_mode == "off" else selected_lag
                for initial_name, (ng0, bt0) in INITIAL_CONDITIONS.items():
                    for extinction_mode in EXTINCTION_MODES:
                        rows.append(
                            {
                                "run_id": len(rows),
                                "species": name,
                                "ratio": float(ratio),
                                "r_ng_bt_production": base_rate,
                                "r_ng_dm": float(ratio) * base_rate,
                                "lag_mode": lag_mode,
                                "lag_h": lag_h,
                                "initial_condition": initial_name,
                                "initial_ng": ng0,
                                "initial_bt": bt0,
                                "extinction_mode": extinction_mode,
                                "hard_extinction": extinction_mode == "hard_extinction",
                            }
                        )
    return pd.DataFrame(rows)


def assemble_batch_arrays(
    parameters: FrozenParameters,
    runs: pd.DataFrame,
    config: ScanConfig,
) -> dict[str, np.ndarray]:
    n_runs = len(runs)
    n_bt = parameters.nongrower_rate_cr.shape[1]
    n_resources = 1 + n_bt
    rate_cr = np.zeros((n_runs, 2, n_resources), dtype=float)
    production = np.zeros_like(rate_cr)
    D = np.zeros((n_runs, 2), dtype=float)
    F = np.zeros((n_runs, 2), dtype=float)
    lag = np.zeros((n_runs, 2), dtype=float)
    required = np.zeros((n_runs, 2), dtype=bool)

    species_index = {name: i for i, name in enumerate(COFACTOR_SPECIES)}
    ng_matrix = parameters.nongrower_rate_cr.loc[list(COFACTOR_SPECIES)].to_numpy(
        dtype=float
    )
    F_vector = parameters.cofactor.loc[list(COFACTOR_SPECIES), "F"].to_numpy(
        dtype=float
    )
    D_vector = parameters.cofactor.loc[list(COFACTOR_SPECIES), "D"].to_numpy(
        dtype=float
    )
    indices = np.array([species_index[name] for name in runs["species"]], dtype=int)
    bt_support = parameters.bt_dm_rate.to_numpy(dtype=float) > 0
    bt_dm_positive = parameters.bt_dm_rate.to_numpy(dtype=float)[bt_support]
    bt_dm_rate = float(bt_dm_positive[0])
    aggregated_dm_y0 = float(parameters.dm_y0.to_numpy(dtype=float)[bt_support].sum())

    rate_cr[:, 0, 1:] = ng_matrix[indices]
    rate_cr[:, 0, 0] = runs["r_ng_dm"].to_numpy(dtype=float)
    rate_cr[:, 1, 0] = bt_dm_rate
    production[:, 1, 1:] = parameters.bt_production.to_numpy(dtype=float)[None, :]
    D[:, 0] = D_vector[indices]
    F[:, 0] = F_vector[indices]
    lag[:, 0] = runs["lag_h"].to_numpy(dtype=float)
    required[:, 0] = True

    initial_resources = np.concatenate(
        (
            np.array([config.dm_fraction * aggregated_dm_y0]),
            config.bt_spent_fraction
            * parameters.bt_spent_y0.to_numpy(dtype=float),
        )
    )
    initial_biomass = runs[["initial_ng", "initial_bt"]].to_numpy(dtype=float)
    return {
        "rate_cr": rate_cr,
        "production": production,
        "D": D,
        "F": F,
        "lag": lag,
        "required": required,
        "initial_resources": initial_resources,
        "initial_biomass": initial_biomass,
    }


def apply_cycle_transfer(
    endpoint: np.ndarray,
    hard_extinction: np.ndarray,
    threshold: float,
    dilution: float,
) -> np.ndarray:
    endpoint = np.asarray(endpoint, dtype=float)
    hard_extinction = np.asarray(hard_extinction, dtype=bool)
    next_start = endpoint / dilution
    discard = hard_extinction[:, None] & (endpoint < threshold)
    next_start[discard] = 0.0
    return next_start


def _adaptive_step_bound(
    time_h: float,
    biomass: np.ndarray,
    resources: np.ndarray,
    environmental: np.ndarray,
    stored: np.ndarray,
    rate_cr: np.ndarray,
    D: np.ndarray,
    lag: np.ndarray,
    required: np.ndarray,
    config: ScanConfig,
) -> float:
    active = time_h >= lag
    cofactor_gate = np.where(required, stored > 0.0, True)
    gate = active & cofactor_gate
    usable = np.maximum(resources, 0.0)
    relative_growth = np.einsum("br,bsr->bs", usable, rate_cr) * gate
    # A nearly exhausted resource can retain an exponentially small positive
    # tail. It must not force tiny global steps after its absolute contribution
    # is already below the accepted nonnegativity tolerance.
    # Only the six scanned DM resources need an additional fast-turnover bound.
    # The 17 Bt-product resources retain their frozen rates and are handled at
    # the Finalized_Model maximum step, matching the publication simulations.
    scanned_dm_columns = np.any(rate_cr[:, 1, :] > 0.0, axis=0)
    materially_present = (usable > 1e-8) & scanned_dm_columns[None, :]
    resource_turnover = np.einsum(
        "bs,bsr->br",
        biomass * gate,
        rate_cr * materially_present[:, None, :],
    )
    accessible_resource_sum = np.einsum(
        "br,bsr->bs", usable, rate_cr > 0.0
    )
    uptake_gate = accessible_resource_sum > config.uptake_resource_sum_threshold
    uptake_turnover = (
        D
        * biomass
        * active
        * required
        * uptake_gate
        * (environmental[:, None] > 1e-8)
    )
    maximum = max(
        float(np.max(relative_growth, initial=0.0)),
        float(np.max(resource_turnover, initial=0.0)),
        float(np.max(uptake_turnover, initial=0.0)),
    )
    if maximum <= 0:
        bound = config.max_step_h
    else:
        bound = min(config.max_step_h, config.adaptive_safety / maximum)

    future_lags = lag[(lag > time_h + 1e-14) & (lag < config.hours_per_cycle)]
    if future_lags.size:
        bound = min(bound, float(np.min(future_lags) - time_h))
    return max(config.minimum_step_h, bound)


def simulate_scan_batch(
    arrays: dict[str, np.ndarray],
    runs: pd.DataFrame,
    config: ScanConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Run every condition and retain one row per run per serial cycle."""

    config.validate()
    rate_cr = arrays["rate_cr"]
    production = arrays["production"]
    D = arrays["D"]
    F = arrays["F"]
    lag = arrays["lag"]
    required = arrays["required"]
    initial_resources = arrays["initial_resources"]
    biomass = arrays["initial_biomass"].copy()
    hard_extinction = runs["hard_extinction"].to_numpy(dtype=bool)
    n_runs, n_species, n_resources = rate_cr.shape
    if biomass.shape != (n_runs, n_species):
        raise ValueError("Initial biomass does not match the assembled scan batch.")

    records: list[pd.DataFrame] = []
    residual_history = np.full((config.max_cycles, n_runs), np.nan, dtype=float)
    step_count = 0
    rejected_steps = 0
    smallest_step = config.max_step_h
    start_wall = time.perf_counter()

    for cycle in range(1, config.max_cycles + 1):
        cycle_start = biomass.copy()
        resources = np.broadcast_to(
            initial_resources, (n_runs, n_resources)
        ).copy()
        environmental = np.ones(n_runs, dtype=float)
        stored = np.zeros((n_runs, n_species), dtype=float)
        time_h = 0.0

        while time_h < config.hours_per_cycle - 1e-14:
            dt = min(
                _adaptive_step_bound(
                    time_h,
                    biomass,
                    resources,
                    environmental,
                    stored,
                    rate_cr,
                    D,
                    lag,
                    required,
                    config,
                ),
                config.hours_per_cycle - time_h,
            )
            while True:
                proposed = _rk4_pairwise_production_batch(
                    time_h,
                    biomass,
                    resources,
                    environmental,
                    stored,
                    dt,
                    rate_cr,
                    production,
                    D,
                    F,
                    lag,
                    required,
                    config.uptake_resource_sum_threshold,
                )
                finite = all(np.isfinite(item).all() for item in proposed)
                minimum = min(float(np.min(item, initial=0.0)) for item in proposed)
                if finite and minimum >= -1e-8:
                    break
                dt *= 0.5
                rejected_steps += 1
                if dt < config.minimum_step_h:
                    raise RuntimeError(
                        "Adaptive integration failed to find a finite nonnegative step."
                    )

            biomass, resources, environmental, stored = (
                np.maximum(item, 0.0) for item in proposed
            )
            time_h = min(config.hours_per_cycle, time_h + dt)
            step_count += 1
            smallest_step = min(smallest_step, dt)

        endpoint = biomass.copy()
        next_start = apply_cycle_transfer(
            endpoint,
            hard_extinction,
            config.endpoint_extinction_threshold,
            config.dilution,
        )
        residual = np.max(
            np.abs(
                np.log10(next_start + config.convergence_log_floor)
                - np.log10(cycle_start + config.convergence_log_floor)
            ),
            axis=1,
        )
        residual_history[cycle - 1] = residual
        records.append(
            pd.DataFrame(
                {
                    "run_id": runs["run_id"].to_numpy(dtype=int),
                    "cycle": cycle,
                    "cycle_start_ng": cycle_start[:, 0],
                    "cycle_start_bt": cycle_start[:, 1],
                    "cycle_end_ng": endpoint[:, 0],
                    "cycle_end_bt": endpoint[:, 1],
                    "next_start_ng": next_start[:, 0],
                    "next_start_bt": next_start[:, 1],
                    "cycle_map_residual_log10": residual,
                }
            )
        )
        biomass = next_start

    cycle_table = pd.concat(records, ignore_index=True)
    final = cycle_table[cycle_table["cycle"] == config.max_cycles].copy()
    final = runs.merge(final, on="run_id", how="left", validate="one_to_one")
    last_n = residual_history[-config.convergence_consecutive_cycles :]
    final["converged_last_n"] = np.all(
        last_n < config.convergence_tolerance_log10, axis=0
    )
    final["final_ng_above_threshold"] = (
        final["cycle_end_ng"] >= config.endpoint_extinction_threshold
    )
    final["final_bt_above_threshold"] = (
        final["cycle_end_bt"] >= config.endpoint_extinction_threshold
    )
    final["final_survival_class"] = np.select(
        [
            final["final_ng_above_threshold"] & final["final_bt_above_threshold"],
            final["final_ng_above_threshold"],
            final["final_bt_above_threshold"],
        ],
        ["coexistence", "ng_only", "bt_only"],
        default="washout",
    )
    diagnostics = {
        "elapsed_seconds": time.perf_counter() - start_wall,
        "accepted_integration_steps": int(step_count),
        "rejected_integration_steps": int(rejected_steps),
        "smallest_accepted_step_h": float(smallest_step),
    }
    return cycle_table, final, diagnostics


def compare_initial_conditions(final: pd.DataFrame, config: ScanConfig) -> pd.DataFrame:
    keys = ["species", "ratio", "lag_mode", "extinction_mode"]
    columns = [
        "cycle_end_ng",
        "cycle_end_bt",
        "final_survival_class",
        "converged_last_n",
        "cycle_map_residual_log10",
    ]
    wide = final.pivot(index=keys, columns="initial_condition", values=columns)
    wide.columns = [f"{value}__{initial}" for value, initial in wide.columns]
    wide = wide.reset_index()
    low = "ng_low_bt_high"
    high = "ng_high_bt_low"
    floor = config.convergence_log_floor
    for taxon in ("ng", "bt"):
        a = wide[f"cycle_end_{taxon}__{low}"].to_numpy(dtype=float)
        b = wide[f"cycle_end_{taxon}__{high}"].to_numpy(dtype=float)
        wide[f"endpoint_log10_difference_{taxon}"] = np.abs(
            np.log10(a + floor) - np.log10(b + floor)
        )
    wide["max_endpoint_log10_difference"] = wide[
        ["endpoint_log10_difference_ng", "endpoint_log10_difference_bt"]
    ].max(axis=1)
    low_total = (
        wide[f"cycle_end_ng__{low}"] + wide[f"cycle_end_bt__{low}"]
    )
    high_total = (
        wide[f"cycle_end_ng__{high}"] + wide[f"cycle_end_bt__{high}"]
    )
    low_ng_rel = np.divide(
        wide[f"cycle_end_ng__{low}"].to_numpy(dtype=float), low_total.to_numpy(dtype=float),
        out=np.zeros(len(wide)), where=low_total.to_numpy(dtype=float) > 0
    )
    high_ng_rel = np.divide(
        wide[f"cycle_end_ng__{high}"].to_numpy(dtype=float), high_total.to_numpy(dtype=float),
        out=np.zeros(len(wide)), where=high_total.to_numpy(dtype=float) > 0
    )
    wide["relative_composition_l1"] = 2.0 * np.abs(low_ng_rel - high_ng_rel)
    wide["survival_class_differs"] = (
        wide[f"final_survival_class__{low}"]
        != wide[f"final_survival_class__{high}"]
    )
    wide["both_initial_conditions_converged"] = (
        wide[f"converged_last_n__{low}"].astype(bool)
        & wide[f"converged_last_n__{high}"].astype(bool)
    )
    return wide

