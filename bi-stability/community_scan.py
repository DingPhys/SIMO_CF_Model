#!/usr/bin/env python3
"""Scan Bt with all ten cofactor-competing nongrowers in one community."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd



from dynamics import _rk4_pairwise_production_batch
from pairwise_scan import (
    COFACTOR_SPECIES,
    EXTINCTION_MODES,
    INITIAL_CONDITIONS,
    LAG_MODES,
    ScanConfig,
    apply_cycle_transfer,
    common_positive_rate,
)


LOCAL_SPECIES = COFACTOR_SPECIES + ("Bt",)
BT_INDEX = len(COFACTOR_SPECIES)


def build_run_table(ratios: np.ndarray) -> pd.DataFrame:
    """Build the shared-ratio community conditions."""
    rows: list[dict[str, object]] = []
    for ratio_index, ratio in enumerate(ratios):
        for lag_mode in LAG_MODES:
            for initial_name, (total_ng0, bt0) in INITIAL_CONDITIONS.items():
                for extinction_mode in EXTINCTION_MODES:
                    rows.append(
                        {
                            "run_id": len(rows),
                            "community": "Bt_plus_all_10_NG",
                            "ratio_index": ratio_index,
                            "ratio": float(ratio),
                            "lag_mode": lag_mode,
                            "initial_condition": initial_name,
                            "initial_ng_per_species": float(total_ng0)
                            / len(COFACTOR_SPECIES),
                            "initial_total_ng": float(total_ng0),
                            "initial_bt": float(bt0),
                            "extinction_mode": extinction_mode,
                            "hard_extinction": extinction_mode == "hard_extinction",
                        }
                    )
    return pd.DataFrame(rows)


def assemble_batch_arrays(
    parameters: object,
    runs: pd.DataFrame,
    config: ScanConfig,
) -> dict[str, np.ndarray]:
    """Assemble one 11-species consumer-resource system per scan run."""
    n_runs = len(runs)
    n_bt_products = parameters.nongrower_rate_cr.shape[1]
    n_resources = 1 + n_bt_products
    n_species = len(LOCAL_SPECIES)

    rate_cr = np.zeros((n_runs, n_species, n_resources), dtype=float)
    production = np.zeros_like(rate_cr)
    D = np.zeros((n_runs, n_species), dtype=float)
    F = np.zeros((n_runs, n_species), dtype=float)
    lag = np.zeros((n_runs, n_species), dtype=float)
    required = np.zeros((n_runs, n_species), dtype=bool)

    ng_matrix = parameters.nongrower_rate_cr.loc[list(COFACTOR_SPECIES)].to_numpy(
        dtype=float
    )
    base_rates = np.array(
        [
            common_positive_rate(parameters.nongrower_rate_cr.loc[name])
            for name in COFACTOR_SPECIES
        ],
        dtype=float,
    )
    ratios = runs["ratio"].to_numpy(dtype=float)
    rate_cr[:, :BT_INDEX, 0] = ratios[:, None] * base_rates[None, :]
    rate_cr[:, :BT_INDEX, 1:] = ng_matrix[None, :, :]

    bt_support = parameters.bt_dm_rate.to_numpy(dtype=float) > 0
    bt_dm_positive = parameters.bt_dm_rate.to_numpy(dtype=float)[bt_support]
    bt_dm_rate = float(bt_dm_positive[0])
    rate_cr[:, BT_INDEX, 0] = bt_dm_rate
    production[:, BT_INDEX, 1:] = parameters.bt_production.to_numpy(dtype=float)[
        None, :
    ]

    D[:, :BT_INDEX] = parameters.cofactor.loc[
        list(COFACTOR_SPECIES), "D"
    ].to_numpy(dtype=float)[None, :]
    F[:, :BT_INDEX] = parameters.cofactor.loc[
        list(COFACTOR_SPECIES), "F"
    ].to_numpy(dtype=float)[None, :]
    selected_lag = parameters.cofactor.loc[
        list(COFACTOR_SPECIES), "selected_lag_h"
    ].to_numpy(dtype=float)
    lag_on = runs["lag_mode"].eq("on").to_numpy(dtype=bool)
    lag[:, :BT_INDEX] = lag_on[:, None] * selected_lag[None, :]
    required[:, :BT_INDEX] = True

    aggregated_dm_y0 = float(
        parameters.dm_y0.to_numpy(dtype=float)[bt_support].sum()
    )
    initial_resources = np.concatenate(
        (
            np.array([config.dm_fraction * aggregated_dm_y0]),
            config.bt_spent_fraction
            * parameters.bt_spent_y0.to_numpy(dtype=float),
        )
    )
    initial_biomass = np.zeros((n_runs, n_species), dtype=float)
    initial_biomass[:, :BT_INDEX] = runs[
        "initial_ng_per_species"
    ].to_numpy(dtype=float)[:, None]
    initial_biomass[:, BT_INDEX] = runs["initial_bt"].to_numpy(dtype=float)

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


def adaptive_step_bound(
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
    """Return a batch-safe adaptive step for the 11-species system."""
    active = time_h >= lag
    cofactor_gate = np.where(required, stored > 0.0, True)
    gate = active & cofactor_gate
    usable = np.maximum(resources, 0.0)
    relative_growth = np.einsum("br,bsr->bs", usable, rate_cr) * gate

    scanned_dm_columns = np.any(rate_cr[:, BT_INDEX, :] > 0.0, axis=0)
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
    bound = (
        config.max_step_h
        if maximum <= 0
        else min(config.max_step_h, config.adaptive_safety / maximum)
    )
    future_lags = lag[(lag > time_h + 1e-14) & (lag < config.hours_per_cycle)]
    if future_lags.size:
        bound = min(bound, float(np.min(future_lags) - time_h))
    return max(config.minimum_step_h, bound)


def _cycle_frame(
    runs: pd.DataFrame,
    cycle: int,
    cycle_start: np.ndarray,
    endpoint: np.ndarray,
    next_start: np.ndarray,
    residual: np.ndarray,
) -> pd.DataFrame:
    data: dict[str, np.ndarray | int] = {
        "run_id": runs["run_id"].to_numpy(dtype=int),
        "cycle": cycle,
        "cycle_start_total_ng": cycle_start[:, :BT_INDEX].sum(axis=1),
        "cycle_start_bt": cycle_start[:, BT_INDEX],
        "cycle_end_total_ng": endpoint[:, :BT_INDEX].sum(axis=1),
        "cycle_end_bt": endpoint[:, BT_INDEX],
        "next_start_total_ng": next_start[:, :BT_INDEX].sum(axis=1),
        "next_start_bt": next_start[:, BT_INDEX],
        "cycle_map_residual_log10": residual,
    }
    for index, name in enumerate(LOCAL_SPECIES):
        data[f"cycle_start_{name}"] = cycle_start[:, index]
        data[f"cycle_end_{name}"] = endpoint[:, index]
        data[f"next_start_{name}"] = next_start[:, index]
    return pd.DataFrame(data)


def simulate_batch(
    parameters: object,
    runs: pd.DataFrame,
    config: ScanConfig,
    starting_biomass: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Simulate one ratio's eight lag/initial/extinction conditions."""
    config.validate()
    arrays = assemble_batch_arrays(parameters, runs, config)
    rate_cr = arrays["rate_cr"]
    production = arrays["production"]
    D = arrays["D"]
    F = arrays["F"]
    lag = arrays["lag"]
    required = arrays["required"]
    initial_resources = arrays["initial_resources"]
    biomass = arrays["initial_biomass"].copy() if starting_biomass is None else starting_biomass.copy()
    hard_extinction = runs["hard_extinction"].to_numpy(dtype=bool)

    n_runs = len(runs)
    records: list[pd.DataFrame] = []
    residual_history = np.full((config.max_cycles, n_runs), np.nan, dtype=float)
    step_count = 0
    rejected_steps = 0
    smallest_step = config.max_step_h
    start_wall = time.perf_counter()

    for cycle in range(1, config.max_cycles + 1):
        cycle_start = biomass.copy()
        resources = np.broadcast_to(
            initial_resources, (n_runs, len(initial_resources))
        ).copy()
        environmental = np.ones(n_runs, dtype=float)
        stored = np.zeros_like(biomass)
        time_h = 0.0

        while time_h < config.hours_per_cycle - 1e-14:
            dt = min(
                adaptive_step_bound(
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
            _cycle_frame(runs, cycle, cycle_start, endpoint, next_start, residual)
        )
        biomass = next_start

    cycle_table = pd.concat(records, ignore_index=True)
    final_cycle = cycle_table[cycle_table["cycle"] == config.max_cycles].copy()
    final = runs.merge(final_cycle, on="run_id", how="left", validate="one_to_one")
    last_n = residual_history[-config.convergence_consecutive_cycles :]
    final["converged_last_n"] = np.all(
        last_n < config.convergence_tolerance_log10, axis=0
    )
    ng_end_columns = [f"cycle_end_{name}" for name in COFACTOR_SPECIES]
    final["n_ng_above_threshold"] = (
        final[ng_end_columns] >= config.endpoint_extinction_threshold
    ).sum(axis=1)
    final["final_bt_above_threshold"] = (
        final["cycle_end_bt"] >= config.endpoint_extinction_threshold
    )
    final["surviving_ng_species"] = final[ng_end_columns].apply(
        lambda row: "|".join(
            name
            for name, value in zip(COFACTOR_SPECIES, row.to_numpy(dtype=float))
            if value >= config.endpoint_extinction_threshold
        ),
        axis=1,
    )
    final["final_survival_class"] = np.select(
        [
            final["final_bt_above_threshold"] & (final["n_ng_above_threshold"] > 0),
            final["n_ng_above_threshold"] > 0,
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


