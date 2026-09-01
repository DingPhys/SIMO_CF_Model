"""Simulation and error calculations for the publication predictions."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

NONGROWER_SPECIES = (
    "Af", "Ao", "As", "Bl.s", "Col", "Cs", "Et", "Eu.c",
    "Eu.l", "Im", "Ld", "Lsp", "Mi", "Pc", "Va", "Vp",
)
GROWER_SPECIES = (
    "Ai", "Ac", "Bfi", "Bf", "Bt", "Bu", "Bx", "Ba",
    "Csp", "Dl", "Ls", "Mf", "Pm", "Bd", "Bv", "Rg",
)


DEFAULT_PRODUCTION_BRANCHES = ("prod_prior-0p3",)
PRIMARY_CARBON_EXCLUSIONS = ("blank", "Fructose", "Glucuronic_acid")
PAIRWISE_ADDITIONAL_LAGS_H = (2.0, 4.0, 8.0)
DM_ASSEMBLY_ADDITIONAL_LAGS_H = (0.0, 2.0, 4.0, 6.0, 8.0)
UNIVERSAL_GROWER_RESOURCE = "F_All_Shared"
SPENT_MEDIUM_DATA_FILES = {
    "Bt_spent": "bt_spent_assemblies_mean_relative_abundance_matrix.csv",
    "Bd_spent": "bd_spent_assemblies_mean_relative_abundance_matrix.csv",
    "Bt_Bd_spent": "bt_bd_spent_assemblies_mean_relative_abundance_matrix.csv",
}
BT_BD_LEGACY_CONDITIONER_ABUNDANCES = {"Bt": 0.344120825, "Bd": 0.319379175}


def _portable_path(path: Path) -> str:
    """Store paths relative to this publication package when possible."""

    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(Path(__file__).resolve().parents[1]))
    except ValueError:
        return str(resolved)


@dataclass(frozen=True)
class PredictionConfig:
    """Shared serial-dilution protocol for both prediction families."""

    cycles: int = 10
    hours_per_cycle: float = 48.0
    dt: float = 0.04
    dilution: float = 200.0
    initial_abundance: float = 0.001
    prediction_floor: float = 1e-3
    uptake_resource_sum_threshold: float = 1e-3

    def validate(self) -> None:
        if self.cycles <= 0:
            raise ValueError("cycles must be positive.")
        if self.hours_per_cycle <= 0 or self.dt <= 0:
            raise ValueError("hours_per_cycle and dt must be positive.")
        if self.dilution <= 0 or self.initial_abundance <= 0:
            raise ValueError("dilution and initial_abundance must be positive.")
        if self.prediction_floor <= 0:
            raise ValueError("prediction_floor must be positive.")
        if self.uptake_resource_sum_threshold < 0:
            raise ValueError("uptake_resource_sum_threshold must be nonnegative.")


@dataclass
class PredictionBundle:
    """Frozen nongrower model used by all prediction scenarios here."""

    rate_cr: pd.DataFrame
    bt_resource_y0: pd.Series
    F: pd.Series
    D: pd.Series
    lag: pd.Series


@dataclass
class GrowerDMModel:
    """Frozen grower rate matrix and fresh-DM resource vector."""

    rate_cr: pd.DataFrame
    dm_resource_y0: pd.Series


def _prepare_output_directory(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {path}. "
                "Use --overwrite only when replacement is intended."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)


def _load_labeled_matrix(
    path: Path,
    row_label: str,
    expected_rows: Sequence[str],
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if row_label not in frame.columns:
        raise ValueError(f"Missing {row_label!r} column in {path}.")
    frame = frame.set_index(row_label)
    missing = [name for name in expected_rows if name not in frame.index]
    if missing:
        raise ValueError(f"Missing rows in {path}: {missing}.")
    frame = frame.loc[list(expected_rows)].apply(pd.to_numeric, errors="coerce")
    if frame.isna().any().any():
        raise ValueError(f"Non-numeric or missing model values in {path}.")
    return frame


def load_grower_dm_model(grower_results_root: Path) -> GrowerDMModel:
    """Load the frozen grower model without refitting it."""

    model_dir = grower_results_root.resolve() / "model"
    rate_cr = _load_labeled_matrix(
        model_dir / "grower_R.csv",
        "species",
        GROWER_SPECIES,
    )
    y0_path = model_dir / "dm_resource_y0.csv"
    y0_frame = pd.read_csv(y0_path)
    required = {"resource_id", "dm_y0"}
    if not required.issubset(y0_frame.columns):
        raise ValueError(f"DM resource file must contain {sorted(required)}: {y0_path}.")
    y0 = pd.to_numeric(
        y0_frame.set_index("resource_id")["dm_y0"], errors="coerce"
    ).reindex(rate_cr.columns)
    if y0.isna().any() or (y0 < 0).any():
        raise ValueError("DM resource vector is incomplete or negative.")
    if UNIVERSAL_GROWER_RESOURCE not in rate_cr.columns:
        raise ValueError(
            f"Missing universal grower resource {UNIVERSAL_GROWER_RESOURCE!r}."
        )
    return GrowerDMModel(
        rate_cr=rate_cr,
        dm_resource_y0=pd.Series(
            y0.to_numpy(dtype=float), index=rate_cr.columns, name="dm_y0"
        ),
    )


def _rhs_batch(
    time: float,
    biomass: np.ndarray,
    resources: np.ndarray,
    environmental_cofactor: np.ndarray,
    stored_cofactor: np.ndarray,
    rate_cr: np.ndarray,
    D: np.ndarray,
    F: np.ndarray,
    lag: np.ndarray,
    dt: float,
    cofactor_enabled: bool,
    uptake_resource_sum_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized derivative for many communities under one frozen model."""

    active = (time >= lag).astype(float)[None, :]
    usable_resources = np.maximum(resources, 0.0)
    accessible_resource_sum = usable_resources @ (rate_cr > 0.0).T
    growth_rate = usable_resources @ rate_cr.T

    if cofactor_enabled:
        cofactor_gate = (stored_cofactor > 0.0).astype(float)
        growth_gate = active * cofactor_gate
    else:
        growth_gate = np.broadcast_to(active, biomass.shape)

    raw_growth = biomass * growth_gate * growth_rate
    positive_growth = np.maximum(raw_growth, 0.0)
    growth_scale = np.ones_like(raw_growth)

    if cofactor_enabled:
        growth_limit = np.full_like(raw_growth, np.inf)
        positive_F = F > 0.0
        growth_limit[:, positive_F] = (
            np.maximum(stored_cofactor[:, positive_F], 0.0)
            / (F[positive_F][None, :] * dt)
        )
        limited_positive = np.minimum(positive_growth, growth_limit)
        biomass_change = np.where(raw_growth > 0.0, limited_positive, raw_growth)
        positive = positive_growth > 0.0
        growth_scale[positive] = limited_positive[positive] / positive_growth[positive]
    else:
        biomass_change = raw_growth

    species_activity = biomass * growth_gate * growth_scale
    resource_change = -usable_resources * (species_activity @ rate_cr)

    if cofactor_enabled:
        uptake_gate = active * (
            accessible_resource_sum > uptake_resource_sum_threshold
        ).astype(float)
        positive_environment = np.maximum(environmental_cofactor, 0.0)[:, None]
        uptake = D[None, :] * positive_environment * biomass * uptake_gate
        environmental_change = -np.sum(uptake, axis=1)
        stored_use = F[None, :] * np.maximum(biomass_change, 0.0)
        stored_change = uptake - stored_use
    else:
        environmental_change = np.zeros(biomass.shape[0], dtype=float)
        stored_change = np.zeros_like(stored_cofactor)

    return biomass_change, resource_change, environmental_change, stored_change


def _rk4_batch(
    time: float,
    biomass: np.ndarray,
    resources: np.ndarray,
    environmental_cofactor: np.ndarray,
    stored_cofactor: np.ndarray,
    dt: float,
    rate_cr: np.ndarray,
    D: np.ndarray,
    F: np.ndarray,
    lag: np.ndarray,
    cofactor_enabled: bool,
    uptake_resource_sum_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    args = (
        rate_cr,
        D,
        F,
        lag,
        dt,
        cofactor_enabled,
        uptake_resource_sum_threshold,
    )
    k1 = _rhs_batch(
        time,
        biomass,
        resources,
        environmental_cofactor,
        stored_cofactor,
        *args,
    )
    k2 = _rhs_batch(
        time + 0.5 * dt,
        biomass + 0.5 * dt * k1[0],
        resources + 0.5 * dt * k1[1],
        environmental_cofactor + 0.5 * dt * k1[2],
        stored_cofactor + 0.5 * dt * k1[3],
        *args,
    )
    k3 = _rhs_batch(
        time + 0.5 * dt,
        biomass + 0.5 * dt * k2[0],
        resources + 0.5 * dt * k2[1],
        environmental_cofactor + 0.5 * dt * k2[2],
        stored_cofactor + 0.5 * dt * k2[3],
        *args,
    )
    k4 = _rhs_batch(
        time + dt,
        biomass + dt * k3[0],
        resources + dt * k3[1],
        environmental_cofactor + dt * k3[2],
        stored_cofactor + dt * k3[3],
        *args,
    )
    return tuple(
        state
        + (dt / 6.0) * (s1 + 2.0 * s2 + 2.0 * s3 + s4)
        for state, s1, s2, s3, s4 in zip(
            (biomass, resources, environmental_cofactor, stored_cofactor),
            k1,
            k2,
            k3,
            k4,
        )
    )


def simulate_serial_dilution_batch(
    rate_cr: np.ndarray,
    initial_resources: np.ndarray,
    initial_biomass: np.ndarray,
    D: np.ndarray,
    F: np.ndarray,
    lag: np.ndarray,
    cofactor_enabled: bool,
    config: PredictionConfig,
) -> np.ndarray:
    """Simulate many communities together and return final absolute biomass."""

    config.validate()
    rate_cr = np.asarray(rate_cr, dtype=float)
    initial_resources = np.asarray(initial_resources, dtype=float)
    biomass = np.asarray(initial_biomass, dtype=float).copy()
    D = np.asarray(D, dtype=float)
    F = np.asarray(F, dtype=float)
    lag = np.asarray(lag, dtype=float)

    if biomass.ndim != 2 or initial_resources.ndim != 2:
        raise ValueError("initial_biomass and initial_resources must be 2D batches.")
    batch_size, n_species = biomass.shape
    if rate_cr.shape[0] != n_species:
        raise ValueError("CR species dimension does not match initial biomass.")
    if initial_resources.shape != (batch_size, rate_cr.shape[1]):
        raise ValueError("Resource batch shape does not match CR.")
    if D.shape != (n_species,) or F.shape != (n_species,) or lag.shape != (n_species,):
        raise ValueError("D, F, and lag must each contain one value per species.")
    if np.any(initial_resources < 0) or np.any(biomass < 0):
        raise ValueError("Initial resource and biomass values must be nonnegative.")

    final_biomass = biomass.copy()
    for _ in range(config.cycles):
        resources = initial_resources.copy()
        environmental_cofactor = np.ones(batch_size, dtype=float)
        stored_cofactor = np.zeros((batch_size, n_species), dtype=float)
        time = 0.0
        while time < config.hours_per_cycle - 1e-15:
            dt_step = min(config.dt, config.hours_per_cycle - time)
            biomass, resources, environmental_cofactor, stored_cofactor = _rk4_batch(
                time,
                biomass,
                resources,
                environmental_cofactor,
                stored_cofactor,
                dt_step,
                rate_cr,
                D,
                F,
                lag,
                cofactor_enabled,
                config.uptake_resource_sum_threshold,
            )
            biomass = np.maximum(biomass, 0.0)
            resources = np.maximum(resources, 0.0)
            environmental_cofactor = np.maximum(environmental_cofactor, 0.0)
            stored_cofactor = np.maximum(stored_cofactor, 0.0)
            next_time = time + dt_step
            time = (
                config.hours_per_cycle
                if abs(next_time - config.hours_per_cycle) <= 1e-15
                else next_time
            )
        final_biomass = biomass.copy()
        biomass = final_biomass / config.dilution
    return final_biomass


def _rhs_pairwise_production_batch(
    time: float,
    biomass: np.ndarray,
    resources: np.ndarray,
    environmental_cofactor: np.ndarray,
    stored_cofactor: np.ndarray,
    rate_cr: np.ndarray,
    resource_production: np.ndarray,
    D: np.ndarray,
    F: np.ndarray,
    lag: np.ndarray,
    cofactor_required: np.ndarray,
    dt: float,
    uptake_resource_sum_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Derivative for batches whose two local species differ by pair."""

    active = (time >= lag).astype(float)
    usable_resources = np.maximum(resources, 0.0)
    growth_rate = np.einsum("br,bsr->bs", usable_resources, rate_cr)
    accessible_resource_sum = np.einsum(
        "br,bsr->bs", usable_resources, rate_cr > 0.0
    )
    cofactor_gate = np.where(
        cofactor_required,
        stored_cofactor > 0.0,
        True,
    ).astype(float)
    growth_gate = active * cofactor_gate
    raw_growth = biomass * growth_gate * growth_rate
    positive_growth = np.maximum(raw_growth, 0.0)

    growth_limit = np.full_like(raw_growth, np.inf)
    limited = cofactor_required & (F > 0.0)
    growth_limit[limited] = (
        np.maximum(stored_cofactor[limited], 0.0) / (F[limited] * dt)
    )
    limited_positive = np.minimum(positive_growth, growth_limit)
    biomass_change = np.where(raw_growth > 0.0, limited_positive, raw_growth)
    growth_scale = np.ones_like(raw_growth)
    growing = positive_growth > 0.0
    growth_scale[growing] = limited_positive[growing] / positive_growth[growing]

    species_activity = biomass * growth_gate * growth_scale
    resource_sink = np.einsum("bs,bsr->br", species_activity, rate_cr)
    resource_source = np.einsum(
        "bs,bsr->br", biomass_change, resource_production
    )
    resource_change = -usable_resources * resource_sink + resource_source

    uptake_gate = (
        active
        * cofactor_required.astype(float)
        * (accessible_resource_sum > uptake_resource_sum_threshold).astype(float)
    )
    uptake = (
        D
        * np.maximum(environmental_cofactor, 0.0)[:, None]
        * biomass
        * uptake_gate
    )
    environmental_change = -np.sum(uptake, axis=1)
    stored_use = (
        F
        * cofactor_required.astype(float)
        * np.maximum(biomass_change, 0.0)
    )
    stored_change = uptake - stored_use
    return biomass_change, resource_change, environmental_change, stored_change


def _rk4_pairwise_production_batch(
    time: float,
    biomass: np.ndarray,
    resources: np.ndarray,
    environmental_cofactor: np.ndarray,
    stored_cofactor: np.ndarray,
    dt: float,
    rate_cr: np.ndarray,
    resource_production: np.ndarray,
    D: np.ndarray,
    F: np.ndarray,
    lag: np.ndarray,
    cofactor_required: np.ndarray,
    uptake_resource_sum_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    args = (
        rate_cr,
        resource_production,
        D,
        F,
        lag,
        cofactor_required,
        dt,
        uptake_resource_sum_threshold,
    )
    k1 = _rhs_pairwise_production_batch(
        time,
        biomass,
        resources,
        environmental_cofactor,
        stored_cofactor,
        *args,
    )
    k2 = _rhs_pairwise_production_batch(
        time + 0.5 * dt,
        biomass + 0.5 * dt * k1[0],
        resources + 0.5 * dt * k1[1],
        environmental_cofactor + 0.5 * dt * k1[2],
        stored_cofactor + 0.5 * dt * k1[3],
        *args,
    )
    k3 = _rhs_pairwise_production_batch(
        time + 0.5 * dt,
        biomass + 0.5 * dt * k2[0],
        resources + 0.5 * dt * k2[1],
        environmental_cofactor + 0.5 * dt * k2[2],
        stored_cofactor + 0.5 * dt * k2[3],
        *args,
    )
    k4 = _rhs_pairwise_production_batch(
        time + dt,
        biomass + dt * k3[0],
        resources + dt * k3[1],
        environmental_cofactor + dt * k3[2],
        stored_cofactor + dt * k3[3],
        *args,
    )
    return tuple(
        state + (dt / 6.0) * (s1 + 2.0 * s2 + 2.0 * s3 + s4)
        for state, s1, s2, s3, s4 in zip(
            (biomass, resources, environmental_cofactor, stored_cofactor),
            k1,
            k2,
            k3,
            k4,
        )
    )


def simulate_community_production_batch(
    rate_cr: np.ndarray,
    resource_production: np.ndarray,
    initial_resources: np.ndarray,
    initial_biomass: np.ndarray,
    D: np.ndarray,
    F: np.ndarray,
    lag: np.ndarray,
    cofactor_required: np.ndarray,
    config: PredictionConfig,
) -> np.ndarray:
    """Simulate batches of local communities with production and cofactor."""

    config.validate()
    rate_cr = np.asarray(rate_cr, dtype=float)
    resource_production = np.asarray(resource_production, dtype=float)
    initial_resources = np.asarray(initial_resources, dtype=float)
    biomass = np.asarray(initial_biomass, dtype=float).copy()
    D = np.asarray(D, dtype=float)
    F = np.asarray(F, dtype=float)
    lag = np.asarray(lag, dtype=float)
    cofactor_required = np.asarray(cofactor_required, dtype=bool)

    if rate_cr.ndim != 3:
        raise ValueError("rate_cr must have shape (batch, local_species, resources).")
    batch_size, local_species, n_resources = rate_cr.shape
    expected_species_shape = (batch_size, local_species)
    if resource_production.shape != rate_cr.shape:
        raise ValueError("resource_production must have the same shape as rate_cr.")
    if initial_resources.shape != (batch_size, n_resources):
        raise ValueError("initial_resources has an incompatible shape.")
    if biomass.shape != expected_species_shape:
        raise ValueError("initial_biomass has an incompatible shape.")
    for name, values in (
        ("D", D),
        ("F", F),
        ("lag", lag),
        ("cofactor_required", cofactor_required),
    ):
        if values.shape != expected_species_shape:
            raise ValueError(f"{name} has an incompatible shape.")
    if (
        np.any(rate_cr < 0)
        or np.any(resource_production < 0)
        or np.any(initial_resources < 0)
        or np.any(biomass < 0)
        or np.any(D < 0)
        or np.any(F < 0)
        or np.any(lag < 0)
    ):
        raise ValueError("Model parameters and initial states must be nonnegative.")

    final_biomass = biomass.copy()
    for _ in range(config.cycles):
        resources = initial_resources.copy()
        environmental_cofactor = np.ones(batch_size, dtype=float)
        stored_cofactor = np.zeros(expected_species_shape, dtype=float)
        time = 0.0
        while time < config.hours_per_cycle - 1e-15:
            dt_step = min(config.dt, config.hours_per_cycle - time)
            biomass, resources, environmental_cofactor, stored_cofactor = (
                _rk4_pairwise_production_batch(
                    time,
                    biomass,
                    resources,
                    environmental_cofactor,
                    stored_cofactor,
                    dt_step,
                    rate_cr,
                    resource_production,
                    D,
                    F,
                    lag,
                    cofactor_required,
                    config.uptake_resource_sum_threshold,
                )
            )
            biomass = np.maximum(biomass, 0.0)
            resources = np.maximum(resources, 0.0)
            environmental_cofactor = np.maximum(environmental_cofactor, 0.0)
            stored_cofactor = np.maximum(stored_cofactor, 0.0)
            next_time = time + dt_step
            time = (
                config.hours_per_cycle
                if abs(next_time - config.hours_per_cycle) <= 1e-15
                else next_time
            )
        final_biomass = biomass.copy()
        biomass = final_biomass / config.dilution
    return final_biomass


def simulate_pairwise_production_batch(
    rate_cr: np.ndarray,
    resource_production: np.ndarray,
    initial_resources: np.ndarray,
    initial_biomass: np.ndarray,
    D: np.ndarray,
    F: np.ndarray,
    lag: np.ndarray,
    cofactor_required: np.ndarray,
    config: PredictionConfig,
) -> np.ndarray:
    """Backward-compatible name for the generic local-community simulator."""

    return simulate_community_production_batch(
        rate_cr,
        resource_production,
        initial_resources,
        initial_biomass,
        D,
        F,
        lag,
        cofactor_required,
        config,
    )


def _matrix_with_species_rows(
    values: np.ndarray,
    species: Sequence[str],
    condition_names: Sequence[str],
    valid_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    values = np.asarray(values, dtype=float)
    if values.shape != (len(condition_names), len(species)):
        raise ValueError("Prediction matrix has unexpected shape.")
    frame = pd.DataFrame(values.T, index=species, columns=condition_names)
    if valid_mask is not None:
        mask = pd.DataFrame(
            np.asarray(valid_mask, dtype=bool).T,
            index=species,
            columns=condition_names,
        )
        frame = frame.where(mask)
    frame.index.name = "species"
    return frame.reset_index()


def _prepare_bt_assembly_scoring(
    values: np.ndarray,
    presence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the Eu readout merge while scoring every presented species."""

    result = np.where(presence, np.asarray(values, dtype=float), 0.0)
    score_mask = np.asarray(presence, dtype=bool).copy()
    euc = NONGROWER_SPECIES.index("Eu.c")
    eul = NONGROWER_SPECIES.index("Eu.l")
    both = presence[:, euc] & presence[:, eul]
    result[both, euc] += result[both, eul]
    result[both, eul] = 0.0
    return result, score_mask


def _normalize_within_presented(
    absolute: np.ndarray,
    presence: np.ndarray,
) -> np.ndarray:
    masked = np.where(presence, np.maximum(absolute, 0.0), 0.0)
    totals = masked.sum(axis=1, keepdims=True)
    return np.divide(masked, totals, out=np.zeros_like(masked), where=totals > 0)


def _assembly_type(name: str, n_presented: int) -> str:
    if name == "Full_community":
        return "full_community"
    if name.endswith("_dropout"):
        return "dropout"
    if n_presented == 2:
        return "pairwise"
    return "higher_order"


def predict_nongrower_assemblies_in_bt(
    data_dir: Path,
    bundle: PredictionBundle,
    output_dir: Path,
    config: PredictionConfig,
    overwrite: bool = False,
    max_communities: int | None = None,
) -> pd.DataFrame:
    """Predict all curated Bt-spent nongrower assemblies with matched ablation."""

    _prepare_output_directory(output_dir, overwrite)
    observed_path = data_dir / "Canonical_Relative_NG_Community_in_Bt_Spent_curated.csv"
    observed = pd.read_csv(observed_path)
    if "species" not in observed.columns:
        raise ValueError(f"Missing species column in {observed_path}.")
    observed = observed.set_index("species").reindex(list(NONGROWER_SPECIES))
    if observed.index.isna().any():
        raise ValueError("Bt assembly table is missing canonical nongrowers.")
    observed = observed.apply(pd.to_numeric, errors="coerce")
    if max_communities is not None:
        observed = observed.iloc[:, :max_communities]
    condition_names = observed.columns.tolist()
    observation_values = observed.T.to_numpy(dtype=float)
    presence = np.isfinite(observation_values)
    if np.any(presence.sum(axis=1) == 0):
        raise ValueError("At least one Bt assembly has no designed species.")

    batch_size = len(condition_names)
    initial_biomass = presence.astype(float) * config.initial_abundance
    initial_resources = np.broadcast_to(
        bundle.bt_resource_y0.to_numpy(dtype=float),
        (batch_size, bundle.rate_cr.shape[1]),
    ).copy()
    rate_cr = bundle.rate_cr.to_numpy(dtype=float)
    D = bundle.D.to_numpy(dtype=float)
    F = bundle.F.to_numpy(dtype=float)
    lag = bundle.lag.to_numpy(dtype=float)

    observed_mapped, observation_score_mask = _prepare_bt_assembly_scoring(
        np.where(presence, observation_values, 0.0), presence
    )
    observed_relative = _normalize_within_presented(observed_mapped, presence)
    summary_rows: list[dict[str, object]] = []

    for cofactor_enabled in (True, False):
        branch = "cofactor_on" if cofactor_enabled else "cofactor_off"
        branch_dir = output_dir / branch
        branch_dir.mkdir(parents=False, exist_ok=False)
        predicted_absolute_native = simulate_serial_dilution_batch(
            rate_cr,
            initial_resources,
            initial_biomass,
            D,
            F,
            lag,
            cofactor_enabled,
            config,
        )
        predicted_mapped, prediction_score_mask = _prepare_bt_assembly_scoring(
            predicted_absolute_native, presence
        )
        score_mask = observation_score_mask & prediction_score_mask
        predicted_relative = _normalize_within_presented(predicted_mapped, presence)
        signed_error = np.full_like(predicted_relative, np.nan)
        signed_error[score_mask] = np.log2(
            np.clip(predicted_relative[score_mask], config.prediction_floor, None)
            / np.clip(observed_relative[score_mask], config.prediction_floor, None)
        )

        _matrix_with_species_rows(
            predicted_absolute_native,
            NONGROWER_SPECIES,
            condition_names,
            presence,
        ).to_csv(branch_dir / "absolute_abundance_native.csv", index=False)
        _matrix_with_species_rows(
            predicted_relative,
            NONGROWER_SPECIES,
            condition_names,
            presence,
        ).to_csv(branch_dir / "relative_abundance_for_comparison.csv", index=False)
        _matrix_with_species_rows(
            signed_error,
            NONGROWER_SPECIES,
            condition_names,
            score_mask,
        ).to_csv(branch_dir / "signed_log2_error.csv", index=False)

        community_rows: list[dict[str, object]] = []
        for index, name in enumerate(condition_names):
            valid_error = signed_error[index, score_mask[index]]
            n_presented = int(presence[index].sum())
            row = {
                "community": name,
                "community_type": _assembly_type(name, n_presented),
                "n_presented": n_presented,
                "n_scored": int(score_mask[index].sum()),
                "cofactor_enabled": cofactor_enabled,
                "mean_abs_log2_error": float(np.mean(np.abs(valid_error))),
                "rmse_log2_error": float(np.sqrt(np.mean(valid_error**2))),
            }
            community_rows.append(row)
            summary_rows.append(row)
        pd.DataFrame(community_rows).to_csv(
            branch_dir / "community_metrics.csv", index=False
        )

        species_rows = []
        for species_index, species in enumerate(NONGROWER_SPECIES):
            valid = np.isfinite(signed_error[:, species_index])
            values = signed_error[valid, species_index]
            species_rows.append(
                {
                    "species": species,
                    "n_scored": int(valid.sum()),
                    "cofactor_enabled": cofactor_enabled,
                    "mean_abs_log2_error": (
                        float(np.mean(np.abs(values))) if values.size else np.nan
                    ),
                    "rmse_log2_error": (
                        float(np.sqrt(np.mean(values**2))) if values.size else np.nan
                    ),
                }
            )
        pd.DataFrame(species_rows).to_csv(
            branch_dir / "species_metrics.csv", index=False
        )

    _matrix_with_species_rows(
        observed_relative,
        NONGROWER_SPECIES,
        condition_names,
        presence,
    ).to_csv(output_dir / "observed_relative_abundance_for_comparison.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "model_comparison_by_community.csv", index=False)
    overall = (
        summary.groupby("cofactor_enabled", as_index=False)
        .agg(
            n_communities=("community", "size"),
            mean_community_abs_log2_error=("mean_abs_log2_error", "mean"),
            median_community_abs_log2_error=("mean_abs_log2_error", "median"),
        )
        .sort_values("cofactor_enabled", ascending=False)
    )
    overall.to_csv(output_dir / "model_comparison_summary.csv", index=False)
    return overall


def _load_production_profile(
    production_root: Path,
    branch: str,
    resource_names: Sequence[str],
) -> pd.DataFrame:
    path = production_root / branch / "production_profiles.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing production profile: {path}.")
    profile = pd.read_csv(path)
    if "resource_id" not in profile.columns:
        raise ValueError(f"Missing resource_id in {path}.")
    profile = profile.set_index("resource_id")
    missing_growers = [name for name in GROWER_SPECIES if name not in profile.columns]
    if missing_growers:
        raise ValueError(f"Missing growers in {path}: {missing_growers}.")
    profile = profile.loc[list(resource_names), list(GROWER_SPECIES)].apply(
        pd.to_numeric, errors="coerce"
    )
    if profile.isna().any().any() or (profile < 0).any().any():
        raise ValueError(f"Production profile is incomplete or negative: {path}.")
    return profile


def _replicate_output_frame(
    metadata: pd.DataFrame,
    species: Sequence[str],
    values: np.ndarray,
) -> pd.DataFrame:
    value_frame = pd.DataFrame(np.asarray(values, dtype=float), columns=species)
    return pd.concat([metadata.reset_index(drop=True), value_frame], axis=1)


def predict_full_communities_across_carbon_sources(
    data_dir: Path,
    results_root: Path,
    bundle: PredictionBundle,
    output_dir: Path,
    config: PredictionConfig,
    production_branches: Sequence[str] = DEFAULT_PRODUCTION_BRANCHES,
    primary_excluded_media: Sequence[str] = PRIMARY_CARBON_EXCLUSIONS,
    overwrite: bool = False,
    max_replicates: int | None = None,
) -> pd.DataFrame:
    """Predict canonical 32-species communities across carbon sources."""

    _prepare_output_directory(output_dir, overwrite)
    observed_path = (
        data_dir
        / "2024930_full_community_24_growers_16_canonical_nongrowers_"
        "absolute_abundance_by_replicate.csv"
    )
    observed = pd.read_csv(observed_path)
    required_columns = {
        "Sample ID",
        "Media",
        *GROWER_SPECIES,
        *NONGROWER_SPECIES,
    }
    missing = sorted(required_columns - set(observed.columns))
    if missing:
        raise ValueError(f"Carbon-source table is missing columns: {missing}.")
    if max_replicates is not None:
        observed = observed.iloc[:max_replicates].copy()

    metadata_columns = [
        "Sample ID",
        "Experiment ID",
        "growth_qc",
        "Growth Well ID",
        "16S Well ID",
        "Media",
        "Protocol",
        "Final OD",
    ]
    metadata_columns = [name for name in metadata_columns if name in observed.columns]
    canonical_species = list(GROWER_SPECIES) + list(NONGROWER_SPECIES)
    observed_absolute = observed[canonical_species].apply(
        pd.to_numeric, errors="coerce"
    )
    if observed_absolute.isna().any().any():
        raise ValueError("Canonical carbon-source abundance contains missing values.")
    observed_absolute_values = np.clip(
        observed_absolute.to_numpy(dtype=float), 0.0, None
    )
    observed_totals = observed_absolute_values.sum(axis=1, keepdims=True)
    scorable_observation = observed_totals[:, 0] > 0
    observed_relative = np.full_like(observed_absolute_values, np.nan)
    np.divide(
        observed_absolute_values,
        observed_totals,
        out=observed_relative,
        where=observed_totals > 0,
    )
    metadata = observed[metadata_columns].copy()
    metadata["observation_relative_abundance_defined"] = scorable_observation
    metadata["included_in_primary_score"] = (
        ~metadata["Media"].isin(list(primary_excluded_media))
        & scorable_observation
    )
    metadata["data_use_status"] = "explicit_user_selected_validation_only"
    metadata["contamination_status"] = "not_assessable"
    grower_absolute = observed_absolute_values[:, : len(GROWER_SPECIES)]
    batch_size = len(observed)
    initial_biomass = np.full(
        (batch_size, len(NONGROWER_SPECIES)),
        config.initial_abundance,
        dtype=float,
    )

    production_root = results_root / "production_fits"
    comparison_rows: list[dict[str, object]] = []
    for production_branch in production_branches:
        production = _load_production_profile(
            production_root,
            production_branch,
            bundle.rate_cr.columns,
        )
        produced_resources = grower_absolute @ production.to_numpy(dtype=float).T
        if np.any(produced_resources < 0):
            raise ValueError("Grower-produced resource matrix contains negative values.")

        for cofactor_enabled in (True, False):
            cofactor_name = "cofactor_on" if cofactor_enabled else "cofactor_off"
            branch_dir = output_dir / production_branch / cofactor_name
            branch_dir.mkdir(parents=True, exist_ok=False)
            predicted_nongrower_absolute = simulate_serial_dilution_batch(
                bundle.rate_cr.to_numpy(dtype=float),
                produced_resources,
                initial_biomass,
                bundle.D.to_numpy(dtype=float),
                bundle.F.to_numpy(dtype=float),
                bundle.lag.to_numpy(dtype=float),
                cofactor_enabled,
                config,
            )
            predicted_absolute = np.concatenate(
                [grower_absolute, predicted_nongrower_absolute], axis=1
            )
            predicted_totals = predicted_absolute.sum(axis=1, keepdims=True)
            if np.any(predicted_totals <= 0):
                raise ValueError("At least one predicted carbon community has zero total.")
            predicted_relative = predicted_absolute / predicted_totals
            signed_error = np.log2(
                np.clip(predicted_relative, config.prediction_floor, None)
                / np.clip(observed_relative, config.prediction_floor, None)
            )

            _replicate_output_frame(
                metadata, canonical_species, predicted_absolute
            ).to_csv(branch_dir / "predicted_absolute_by_replicate.csv", index=False)
            _replicate_output_frame(
                metadata, canonical_species, predicted_relative
            ).to_csv(branch_dir / "predicted_relative_by_replicate.csv", index=False)
            _replicate_output_frame(
                metadata, canonical_species, signed_error
            ).to_csv(branch_dir / "signed_log2_error_by_replicate.csv", index=False)

            replicate_metrics = metadata.copy()
            replicate_metrics["production_branch"] = production_branch
            replicate_metrics["cofactor_enabled"] = cofactor_enabled
            # Carbon-source prediction is evaluated only on the 16 canonical
            # nongrowers. Grower abundances are supplied by the experiment and
            # are used to construct produced resources, rather than predicted.
            scored_error = signed_error[:, len(GROWER_SPECIES) :]
            finite_error_count = np.isfinite(scored_error).sum(axis=1)
            replicate_mae = np.divide(
                np.nansum(np.abs(scored_error), axis=1),
                finite_error_count,
                out=np.full(batch_size, np.nan),
                where=finite_error_count > 0,
            )
            replicate_mse = np.divide(
                np.nansum(scored_error**2, axis=1),
                finite_error_count,
                out=np.full(batch_size, np.nan),
                where=finite_error_count > 0,
            )
            replicate_metrics["mean_abs_log2_error"] = replicate_mae
            replicate_metrics["rmse_log2_error"] = np.sqrt(replicate_mse)
            replicate_metrics.to_csv(
                branch_dir / "replicate_metrics.csv", index=False
            )

            predicted_relative_frame = pd.DataFrame(
                predicted_relative, columns=canonical_species
            )
            predicted_relative_frame.insert(
                0, "Media", observed["Media"].astype(str).to_numpy()
            )
            observed_relative_frame = pd.DataFrame(
                observed_relative, columns=canonical_species
            )
            observed_relative_frame.insert(
                0, "Media", observed["Media"].astype(str).to_numpy()
            )
            predicted_mean = predicted_relative_frame.groupby(
                "Media", sort=False
            )[canonical_species].mean()
            observed_mean = observed_relative_frame.groupby(
                "Media", sort=False
            )[canonical_species].mean()
            carbon_error_all = np.log2(
                predicted_mean.clip(lower=config.prediction_floor)
                / observed_mean.clip(lower=config.prediction_floor)
            )
            carbon_error = carbon_error_all[list(NONGROWER_SPECIES)]
            carbon_metrics = pd.DataFrame(
                {
                    "Media": carbon_error.index,
                    "production_branch": production_branch,
                    "cofactor_enabled": cofactor_enabled,
                    "included_in_primary_score": (
                        ~carbon_error.index.isin(list(primary_excluded_media))
                        & carbon_error.notna().any(axis=1).to_numpy()
                    ),
                    "mean_abs_log2_error": carbon_error.abs().mean(axis=1).to_numpy(),
                    "rmse_log2_error": np.sqrt(
                        (carbon_error**2).mean(axis=1)
                    ).to_numpy(),
                }
            )
            carbon_metrics.to_csv(
                branch_dir / "carbon_source_metrics.csv", index=False
            )
            predicted_mean.reset_index().to_csv(
                branch_dir / "predicted_mean_relative_by_carbon_source.csv",
                index=False,
            )
            carbon_error.reset_index().to_csv(
                branch_dir / "signed_log2_error_by_carbon_source.csv",
                index=False,
            )

            primary = carbon_metrics[carbon_metrics["included_in_primary_score"]]
            comparison_rows.append(
                {
                    "production_branch": production_branch,
                    "cofactor_enabled": cofactor_enabled,
                    "n_replicates": batch_size,
                    "n_carbon_sources_all": len(carbon_metrics),
                    "n_carbon_sources_primary": len(primary),
                    "n_scored_species": len(NONGROWER_SPECIES),
                    "primary_mean_carbon_source_abs_log2_error": float(
                        primary["mean_abs_log2_error"].mean()
                    ),
                    "primary_median_carbon_source_abs_log2_error": float(
                        primary["mean_abs_log2_error"].median()
                    ),
                    "all_mean_carbon_source_abs_log2_error": float(
                        carbon_metrics["mean_abs_log2_error"].mean()
                    ),
                }
            )

    _replicate_output_frame(
        metadata, canonical_species, observed_relative
    ).to_csv(output_dir / "observed_relative_canonical32_by_replicate.csv", index=False)
    comparison = pd.DataFrame(comparison_rows).sort_values(
        "primary_mean_carbon_source_abs_log2_error"
    )
    comparison.to_csv(output_dir / "model_comparison_summary.csv", index=False)
    return comparison


def _nongrower_universal_rates(rate_cr: pd.DataFrame) -> pd.Series:
    """Use each nongrower's common positive CR rate for the universal niche."""

    values = rate_cr.to_numpy(dtype=float)
    positive_count = np.sum(values > 0.0, axis=1)
    no_positive = positive_count == 0
    row_min = np.min(np.where(values > 0.0, values, np.inf), axis=1)
    row_max = np.max(np.where(values > 0.0, values, -np.inf), axis=1)
    inconsistent = ~np.isclose(row_min, row_max, rtol=1e-10, atol=1e-12)
    invalid = no_positive | inconsistent
    if np.any(invalid):
        bad = rate_cr.index[invalid].tolist()
        raise ValueError(
            "Universal-niche mapping requires one common positive CR rate within "
            f"each nongrower row; invalid rows: {bad}."
        )
    rates = row_max
    return pd.Series(rates, index=rate_cr.index, name="universal_niche_rate")


def _format_lag_directory(hours: float) -> str:
    if float(hours).is_integer():
        return f"extra_lag-{int(hours)}h"
    return f"extra_lag-{str(hours).replace('.', 'p')}h"


def predict_grower_nongrower_pairwise_in_dm(
    data_dir: Path,
    results_root: Path,
    grower_model: GrowerDMModel,
    bundle: PredictionBundle,
    output_dir: Path,
    config: PredictionConfig,
    production_branches: Sequence[str] = DEFAULT_PRODUCTION_BRANCHES,
    additional_lags_h: Sequence[float] = PAIRWISE_ADDITIONAL_LAGS_H,
    include_universal_off: bool = True,
    overwrite: bool = False,
    max_pairs: int | None = None,
) -> pd.DataFrame:
    """Predict grower-nongrower pairs across configurable production branches."""

    _prepare_output_directory(output_dir, overwrite)
    observed_path = data_dir / "grower_nongrower_pairwise_mean_relative_abundance_by_pair.csv"
    observed = pd.read_csv(observed_path)
    if "species" not in observed.columns:
        raise ValueError(f"Missing species column in {observed_path}.")
    canonical_species = list(GROWER_SPECIES) + list(NONGROWER_SPECIES)
    observed = observed.set_index("species")
    missing_species = [name for name in canonical_species if name not in observed.index]
    if missing_species:
        raise ValueError(f"Pairwise table is missing species: {missing_species}.")
    observed = observed.loc[canonical_species].apply(pd.to_numeric, errors="coerce")
    if max_pairs is not None:
        observed = observed.iloc[:, :max_pairs]
    pair_names = observed.columns.tolist()
    observed_values = observed.T.to_numpy(dtype=float)
    presence = np.isfinite(observed_values)
    if np.any(presence.sum(axis=1) != 2):
        bad = np.asarray(pair_names)[presence.sum(axis=1) != 2].tolist()
        raise ValueError(f"Every pair column must contain exactly two species: {bad}.")

    n_growers = len(GROWER_SPECIES)
    grower_presence = presence[:, :n_growers]
    nongrower_presence = presence[:, n_growers:]
    if np.any(grower_presence.sum(axis=1) != 1) or np.any(
        nongrower_presence.sum(axis=1) != 1
    ):
        raise ValueError("Every pair must contain one grower and one nongrower.")
    grower_index = np.argmax(grower_presence, axis=1)
    nongrower_index = np.argmax(nongrower_presence, axis=1)
    pair_count = len(pair_names)
    observed_local = np.column_stack(
        (
            observed_values[np.arange(pair_count), grower_index],
            observed_values[np.arange(pair_count), n_growers + nongrower_index],
        )
    )
    if not np.isfinite(observed_local).all() or np.any(observed_local < 0):
        raise ValueError("Observed pairwise relative abundances are invalid.")
    observed_totals = observed_local.sum(axis=1, keepdims=True)
    if np.any(observed_totals <= 0):
        raise ValueError("At least one observed pair has zero total abundance.")
    observed_local = observed_local / observed_totals

    grower_resource_names = grower_model.rate_cr.columns.tolist()
    nongrower_resource_names = bundle.rate_cr.columns.tolist()
    n_grower_resources = len(grower_resource_names)
    n_nongrower_resources = len(nongrower_resource_names)
    total_resources = n_grower_resources + n_nongrower_resources
    universal_index = grower_resource_names.index(UNIVERSAL_GROWER_RESOURCE)
    universal_rates = _nongrower_universal_rates(bundle.rate_cr)
    universal_rates.to_csv(output_dir / "nongrower_universal_niche_rates.csv")

    grower_rate_by_pair = grower_model.rate_cr.to_numpy(dtype=float)[grower_index]
    nongrower_rate_by_pair = bundle.rate_cr.to_numpy(dtype=float)[nongrower_index]
    D_by_pair = bundle.D.to_numpy(dtype=float)[nongrower_index]
    F_by_pair = bundle.F.to_numpy(dtype=float)[nongrower_index]
    baseline_lag_by_pair = bundle.lag.to_numpy(dtype=float)[nongrower_index]
    universal_rate_by_pair = universal_rates.to_numpy(dtype=float)[nongrower_index]
    initial_resource = np.zeros((pair_count, total_resources), dtype=float)
    initial_resource[:, :n_grower_resources] = (
        grower_model.dm_resource_y0.to_numpy(dtype=float)[None, :]
    )

    production_root = results_root / "production_fits"
    production_by_branch = {
        branch: _load_production_profile(
            production_root,
            branch,
            nongrower_resource_names,
        )
        for branch in production_branches
    }
    scenarios: list[dict[str, object]] = []
    if include_universal_off:
        for production_branch in production_branches:
            scenarios.append(
                {
                    "scenario": f"universal_off__{production_branch}",
                    "universal_shared": False,
                    "additional_nongrower_lag_h": 0.0,
                    "production_branch": production_branch,
                    "relative_directory": Path("universal_off") / production_branch,
                }
            )
    for additional_lag in additional_lags_h:
        if additional_lag < 0:
            raise ValueError("Additional nongrower lag must be nonnegative.")
        for production_branch in production_branches:
            scenarios.append(
                {
                    "scenario": (
                        f"universal_on__extra_lag-{additional_lag:g}h__"
                        f"{production_branch}"
                    ),
                    "universal_shared": True,
                    "additional_nongrower_lag_h": float(additional_lag),
                    "production_branch": production_branch,
                    "relative_directory": (
                        Path("universal_on")
                        / _format_lag_directory(float(additional_lag))
                        / production_branch
                    ),
                }
            )

    scenario_rate_cr: list[np.ndarray] = []
    scenario_production: list[np.ndarray] = []
    scenario_lag: list[np.ndarray] = []
    for scenario in scenarios:
        rate_cr = np.zeros((pair_count, 2, total_resources), dtype=float)
        rate_cr[:, 0, :n_grower_resources] = grower_rate_by_pair
        rate_cr[:, 1, n_grower_resources:] = nongrower_rate_by_pair
        if bool(scenario["universal_shared"]):
            rate_cr[:, 1, universal_index] = universal_rate_by_pair
        scenario_rate_cr.append(rate_cr)

        resource_production = np.zeros_like(rate_cr)
        production = production_by_branch[str(scenario["production_branch"])]
        resource_production[:, 0, n_grower_resources:] = (
            production.to_numpy(dtype=float).T[grower_index]
        )
        scenario_production.append(resource_production)
        scenario_lag.append(
            np.column_stack(
                (
                    np.zeros(pair_count, dtype=float),
                    baseline_lag_by_pair
                    + float(scenario["additional_nongrower_lag_h"]),
                )
            )
        )

    n_scenarios = len(scenarios)
    rate_cr_batch = np.concatenate(scenario_rate_cr, axis=0)
    production_batch = np.concatenate(scenario_production, axis=0)
    initial_resource_batch = np.tile(initial_resource, (n_scenarios, 1))
    initial_biomass_batch = np.full(
        (n_scenarios * pair_count, 2), config.initial_abundance, dtype=float
    )
    D_batch = np.tile(
        np.column_stack((np.zeros(pair_count), D_by_pair)), (n_scenarios, 1)
    )
    F_batch = np.tile(
        np.column_stack((np.zeros(pair_count), F_by_pair)), (n_scenarios, 1)
    )
    lag_batch = np.concatenate(scenario_lag, axis=0)
    cofactor_required = np.tile(
        np.array([[False, True]], dtype=bool), (n_scenarios * pair_count, 1)
    )

    predicted_batch = simulate_community_production_batch(
        rate_cr_batch,
        production_batch,
        initial_resource_batch,
        initial_biomass_batch,
        D_batch,
        F_batch,
        lag_batch,
        cofactor_required,
        config,
    ).reshape(n_scenarios, pair_count, 2)

    summary_rows: list[dict[str, object]] = []
    for scenario_index, scenario in enumerate(scenarios):
        branch_dir = output_dir / Path(str(scenario["relative_directory"]))
        branch_dir.mkdir(parents=True, exist_ok=False)
        predicted_local_absolute = predicted_batch[scenario_index]
        predicted_totals = predicted_local_absolute.sum(axis=1, keepdims=True)
        if np.any(predicted_totals <= 0):
            raise ValueError(f"Zero predicted total in {scenario['scenario']}.")
        predicted_local_relative = predicted_local_absolute / predicted_totals
        signed_error_local = np.log2(
            np.clip(predicted_local_relative, config.prediction_floor, None)
            / np.clip(observed_local, config.prediction_floor, None)
        )

        predicted_absolute = np.full((pair_count, len(canonical_species)), np.nan)
        predicted_relative = np.full_like(predicted_absolute, np.nan)
        signed_error = np.full_like(predicted_absolute, np.nan)
        row_index = np.arange(pair_count)
        predicted_absolute[row_index, grower_index] = predicted_local_absolute[:, 0]
        predicted_absolute[
            row_index, n_growers + nongrower_index
        ] = predicted_local_absolute[:, 1]
        predicted_relative[row_index, grower_index] = predicted_local_relative[:, 0]
        predicted_relative[
            row_index, n_growers + nongrower_index
        ] = predicted_local_relative[:, 1]
        signed_error[row_index, grower_index] = signed_error_local[:, 0]
        signed_error[row_index, n_growers + nongrower_index] = signed_error_local[:, 1]

        _matrix_with_species_rows(
            predicted_absolute,
            canonical_species,
            pair_names,
        ).to_csv(branch_dir / "predicted_absolute_abundance.csv", index=False)
        _matrix_with_species_rows(
            predicted_relative,
            canonical_species,
            pair_names,
        ).to_csv(branch_dir / "predicted_relative_abundance.csv", index=False)
        _matrix_with_species_rows(
            signed_error,
            canonical_species,
            pair_names,
        ).to_csv(branch_dir / "signed_log2_error.csv", index=False)

        pair_metrics = pd.DataFrame(
            {
                "pair": pair_names,
                "grower": np.asarray(GROWER_SPECIES)[grower_index],
                "nongrower": np.asarray(NONGROWER_SPECIES)[nongrower_index],
                "universal_shared": bool(scenario["universal_shared"]),
                "additional_nongrower_lag_h": float(
                    scenario["additional_nongrower_lag_h"]
                ),
                "production_branch": str(scenario["production_branch"]),
                "grower_signed_log2_error": signed_error_local[:, 0],
                "nongrower_signed_log2_error": signed_error_local[:, 1],
                "mean_abs_log2_error": np.mean(
                    np.abs(signed_error_local), axis=1
                ),
                "rmse_log2_error": np.sqrt(
                    np.mean(signed_error_local**2, axis=1)
                ),
            }
        )
        pair_metrics.to_csv(branch_dir / "pair_metrics.csv", index=False)

        species_rows: list[dict[str, object]] = []
        for role, names, indexes, errors in (
            ("grower", GROWER_SPECIES, grower_index, signed_error_local[:, 0]),
            (
                "nongrower",
                NONGROWER_SPECIES,
                nongrower_index,
                signed_error_local[:, 1],
            ),
        ):
            for species_index, species in enumerate(names):
                selected = indexes == species_index
                values = errors[selected]
                species_rows.append(
                    {
                        "role": role,
                        "species": species,
                        "n_pairs": int(selected.sum()),
                        "mean_abs_log2_error": (
                            float(np.mean(np.abs(values))) if values.size else np.nan
                        ),
                        "rmse_log2_error": (
                            float(np.sqrt(np.mean(values**2)))
                            if values.size
                            else np.nan
                        ),
                        "mean_signed_log2_error": (
                            float(np.mean(values)) if values.size else np.nan
                        ),
                    }
                )
        pd.DataFrame(species_rows).to_csv(
            branch_dir / "species_metrics.csv", index=False
        )

        summary_rows.append(
            {
                "scenario": str(scenario["scenario"]),
                "universal_shared": bool(scenario["universal_shared"]),
                "additional_nongrower_lag_h": float(
                    scenario["additional_nongrower_lag_h"]
                ),
                "production_branch": str(scenario["production_branch"]),
                "n_pairs": pair_count,
                "mean_abs_log2_error": float(np.mean(np.abs(signed_error_local))),
                "rmse_log2_error": float(np.sqrt(np.mean(signed_error_local**2))),
                "grower_mean_abs_log2_error": float(
                    np.mean(np.abs(signed_error_local[:, 0]))
                ),
                "nongrower_mean_abs_log2_error": float(
                    np.mean(np.abs(signed_error_local[:, 1]))
                ),
                "median_pair_mean_abs_log2_error": float(
                    pair_metrics["mean_abs_log2_error"].median()
                ),
            }
        )

    _matrix_with_species_rows(
        observed_values,
        canonical_species,
        pair_names,
    ).to_csv(output_dir / "observed_relative_abundance.csv", index=False)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["mean_abs_log2_error", "rmse_log2_error"]
    )
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    summary.to_csv(output_dir / "model_comparison_summary.csv", index=False)
    return summary


def _dm_community_type(name: str) -> str:
    if name == "Full_community_32_Glucose":
        return "full_community_32"
    if name in {
        "Bt_Bd_dropout",
        "Bd_dropout",
        "Bt_dropout",
        "Whole_community_13",
    }:
        return "core_or_dropout"
    return "random_assembly"


def _build_full_dm_scenario_model(
    grower_model: GrowerDMModel,
    bundle: PredictionBundle,
    production: pd.DataFrame,
    universal_shared: bool,
    additional_nongrower_lag_h: float,
    cofactor_enabled: bool,
    universal_rates: pd.Series,
) -> dict[str, np.ndarray]:
    """Assemble one frozen 32-species DM model for a prediction scenario."""

    n_growers = len(GROWER_SPECIES)
    n_nongrowers = len(NONGROWER_SPECIES)
    grower_resources = grower_model.rate_cr.columns.tolist()
    nongrower_resources = bundle.rate_cr.columns.tolist()
    n_grower_resources = len(grower_resources)
    n_nongrower_resources = len(nongrower_resources)
    total_resources = n_grower_resources + n_nongrower_resources

    rate_cr = np.zeros(
        (n_growers + n_nongrowers, total_resources), dtype=float
    )
    rate_cr[:n_growers, :n_grower_resources] = (
        grower_model.rate_cr.to_numpy(dtype=float)
    )
    rate_cr[n_growers:, n_grower_resources:] = bundle.rate_cr.to_numpy(
        dtype=float
    )
    if universal_shared:
        universal_index = grower_resources.index(UNIVERSAL_GROWER_RESOURCE)
        rate_cr[n_growers:, universal_index] = universal_rates.to_numpy(dtype=float)

    resource_production = np.zeros_like(rate_cr)
    resource_production[:n_growers, n_grower_resources:] = (
        production.to_numpy(dtype=float).T
    )
    initial_resources = np.zeros(total_resources, dtype=float)
    initial_resources[:n_grower_resources] = (
        grower_model.dm_resource_y0.to_numpy(dtype=float)
    )

    D = np.zeros(n_growers + n_nongrowers, dtype=float)
    F = np.zeros_like(D)
    cofactor_required = np.zeros_like(D, dtype=bool)
    if cofactor_enabled:
        D[n_growers:] = bundle.D.to_numpy(dtype=float)
        F[n_growers:] = bundle.F.to_numpy(dtype=float)
        cofactor_required[n_growers:] = True

    lag = np.zeros(n_growers + n_nongrowers, dtype=float)
    lag[n_growers:] = (
        bundle.lag.to_numpy(dtype=float) + additional_nongrower_lag_h
    )
    return {
        "rate_cr": rate_cr,
        "resource_production": resource_production,
        "initial_resources": initial_resources,
        "D": D,
        "F": F,
        "lag": lag,
        "cofactor_required": cofactor_required,
    }


def predict_random_assemblies_in_dm(
    data_dir: Path,
    results_root: Path,
    grower_model: GrowerDMModel,
    bundle: PredictionBundle,
    output_dir: Path,
    config: PredictionConfig,
    production_branches: Sequence[str] = DEFAULT_PRODUCTION_BRANCHES,
    additional_lags_h: Sequence[float] = DM_ASSEMBLY_ADDITIONAL_LAGS_H,
    universal_shared_options: Sequence[bool] = (False, True),
    overwrite: bool = False,
    max_communities: int | None = None,
) -> pd.DataFrame:
    """Run the DM assembly grid for configurable production branches."""

    _prepare_output_directory(output_dir, overwrite)
    observed_path = data_dir / "dm_assemblies_mean_relative_abundance_matrix.csv"
    observed = pd.read_csv(observed_path)
    if "species" not in observed.columns:
        raise ValueError(f"Missing species column in {observed_path}.")
    canonical_species = list(GROWER_SPECIES) + list(NONGROWER_SPECIES)
    observed = observed.set_index("species")
    missing_species = [name for name in canonical_species if name not in observed.index]
    if missing_species:
        raise ValueError(f"DM assembly matrix is missing species: {missing_species}.")
    observed = observed.loc[canonical_species].apply(pd.to_numeric, errors="coerce")
    if max_communities is not None:
        observed = observed.iloc[:, :max_communities]
    community_names = observed.columns.tolist()
    observed_values = observed.T.to_numpy(dtype=float)
    presence = np.isfinite(observed_values)
    richness = presence.sum(axis=1)
    if np.any(richness <= 0):
        bad = np.asarray(community_names)[richness <= 0].tolist()
        raise ValueError(f"DM communities without observed members: {bad}.")
    if np.any(np.where(presence, observed_values, 0.0) < 0):
        raise ValueError("DM assembly relative-abundance matrix contains negatives.")
    observed_totals = np.nansum(observed_values, axis=1, keepdims=True)
    if np.any(observed_totals <= 0):
        raise ValueError("At least one DM community has zero observed total.")
    observed_relative = np.divide(
        observed_values,
        observed_totals,
        out=np.full_like(observed_values, np.nan),
        where=presence,
    )
    community_types = np.asarray(
        [_dm_community_type(name) for name in community_names], dtype=object
    )

    grower_resource_names = grower_model.rate_cr.columns.tolist()
    nongrower_resource_names = bundle.rate_cr.columns.tolist()
    universal_rates = _nongrower_universal_rates(bundle.rate_cr)
    universal_rates.to_csv(output_dir / "nongrower_universal_niche_rates.csv")
    production_root = results_root / "production_fits"
    production_by_branch = {
        branch: _load_production_profile(
            production_root,
            branch,
            nongrower_resource_names,
        )
        for branch in production_branches
    }

    scenarios: list[dict[str, object]] = []
    scenario_models: list[dict[str, np.ndarray]] = []
    normalized_universal_options = tuple(
        bool(value) for value in universal_shared_options
    )
    if not normalized_universal_options:
        raise ValueError("universal_shared_options must not be empty.")
    if len(set(normalized_universal_options)) != len(
        normalized_universal_options
    ):
        raise ValueError("universal_shared_options must not contain duplicates.")
    for universal_shared in normalized_universal_options:
        universal_name = "universal_on" if universal_shared else "universal_off"
        for additional_lag in additional_lags_h:
            if additional_lag < 0:
                raise ValueError("Additional nongrower lag must be nonnegative.")
            lag_name = _format_lag_directory(float(additional_lag))
            for production_branch in production_branches:
                production = production_by_branch[production_branch]
                for cofactor_enabled in (True, False):
                    cofactor_name = (
                        "cofactor_on" if cofactor_enabled else "cofactor_off"
                    )
                    scenario = {
                        "scenario": (
                            f"{universal_name}__extra_lag-{additional_lag:g}h__"
                            f"{production_branch}__{cofactor_name}"
                        ),
                        "universal_shared": universal_shared,
                        "additional_nongrower_lag_h": float(additional_lag),
                        "production_branch": production_branch,
                        "cofactor_enabled": cofactor_enabled,
                        "relative_directory": (
                            Path(universal_name)
                            / lag_name
                            / production_branch
                            / cofactor_name
                        ),
                    }
                    scenarios.append(scenario)
                    scenario_models.append(
                        _build_full_dm_scenario_model(
                            grower_model,
                            bundle,
                            production,
                            universal_shared,
                            float(additional_lag),
                            cofactor_enabled,
                            universal_rates,
                        )
                    )
    expected_scenarios = (
        len(normalized_universal_options)
        * len(additional_lags_h)
        * len(production_branches)
        * 2
    )
    if len(scenarios) != expected_scenarios:
        raise ValueError(
            f"Expected {expected_scenarios} DM scenarios, constructed "
            f"{len(scenarios)} instead."
        )

    n_scenarios = len(scenarios)
    n_communities = len(community_names)
    n_species = len(canonical_species)
    predicted_absolute = np.full(
        (n_scenarios, n_communities, n_species), np.nan, dtype=float
    )

    # Group by data family so low-richness random assemblies are not padded to
    # the 32-species size of the added Glucose full community.
    for community_type in dict.fromkeys(community_types.tolist()):
        community_index = np.flatnonzero(community_types == community_type)
        max_richness = int(richness[community_index].max())
        member_index = np.zeros((len(community_index), max_richness), dtype=int)
        member_mask = np.zeros_like(member_index, dtype=bool)
        for local_community, global_community in enumerate(community_index):
            members = np.flatnonzero(presence[global_community])
            member_index[local_community, : len(members)] = members
            member_mask[local_community, : len(members)] = True

        rate_batches: list[np.ndarray] = []
        production_batches: list[np.ndarray] = []
        resource_batches: list[np.ndarray] = []
        D_batches: list[np.ndarray] = []
        F_batches: list[np.ndarray] = []
        lag_batches: list[np.ndarray] = []
        required_batches: list[np.ndarray] = []
        biomass_batches: list[np.ndarray] = []
        for model in scenario_models:
            local_rate = model["rate_cr"][member_index].copy()
            local_production = model["resource_production"][member_index].copy()
            local_D = model["D"][member_index].copy()
            local_F = model["F"][member_index].copy()
            local_lag = model["lag"][member_index].copy()
            local_required = model["cofactor_required"][member_index].copy()
            local_rate[~member_mask] = 0.0
            local_production[~member_mask] = 0.0
            local_D[~member_mask] = 0.0
            local_F[~member_mask] = 0.0
            local_lag[~member_mask] = 0.0
            local_required[~member_mask] = False
            rate_batches.append(local_rate)
            production_batches.append(local_production)
            resource_batches.append(
                np.broadcast_to(
                    model["initial_resources"],
                    (len(community_index), len(model["initial_resources"])),
                ).copy()
            )
            D_batches.append(local_D)
            F_batches.append(local_F)
            lag_batches.append(local_lag)
            required_batches.append(local_required)
            biomass_batches.append(
                member_mask.astype(float) * config.initial_abundance
            )

        local_prediction = simulate_community_production_batch(
            np.concatenate(rate_batches, axis=0),
            np.concatenate(production_batches, axis=0),
            np.concatenate(resource_batches, axis=0),
            np.concatenate(biomass_batches, axis=0),
            np.concatenate(D_batches, axis=0),
            np.concatenate(F_batches, axis=0),
            np.concatenate(lag_batches, axis=0),
            np.concatenate(required_batches, axis=0),
            config,
        ).reshape(n_scenarios, len(community_index), max_richness)
        for scenario_index in range(n_scenarios):
            for local_member in range(max_richness):
                valid = member_mask[:, local_member]
                predicted_absolute[
                    scenario_index,
                    community_index[valid],
                    member_index[valid, local_member],
                ] = local_prediction[scenario_index, valid, local_member]

    summary_rows: list[dict[str, object]] = []
    n_growers = len(GROWER_SPECIES)
    for scenario_index, scenario in enumerate(scenarios):
        branch_dir = output_dir / Path(str(scenario["relative_directory"]))
        branch_dir.mkdir(parents=True, exist_ok=False)
        absolute = predicted_absolute[scenario_index]
        totals = np.nansum(absolute, axis=1, keepdims=True)
        if np.any(totals <= 0):
            raise ValueError(f"Zero predicted total in {scenario['scenario']}.")
        relative = np.divide(
            absolute,
            totals,
            out=np.full_like(absolute, np.nan),
            where=presence,
        )
        signed_error = np.full_like(relative, np.nan)
        signed_error[presence] = np.log2(
            np.clip(relative[presence], config.prediction_floor, None)
            / np.clip(observed_relative[presence], config.prediction_floor, None)
        )

        _matrix_with_species_rows(
            absolute, canonical_species, community_names
        ).to_csv(branch_dir / "predicted_absolute_abundance.csv", index=False)
        _matrix_with_species_rows(
            relative, canonical_species, community_names
        ).to_csv(branch_dir / "predicted_relative_abundance.csv", index=False)
        _matrix_with_species_rows(
            signed_error, canonical_species, community_names
        ).to_csv(branch_dir / "signed_log2_error.csv", index=False)

        community_rows: list[dict[str, object]] = []
        for community_index, community in enumerate(community_names):
            values = signed_error[community_index, presence[community_index]]
            n_growers_present = int(
                presence[community_index, :n_growers].sum()
            )
            community_rows.append(
                {
                    "community": community,
                    "community_type": community_types[community_index],
                    "richness": int(richness[community_index]),
                    "n_growers": n_growers_present,
                    "n_nongrowers": int(richness[community_index])
                    - n_growers_present,
                    "universal_shared": bool(scenario["universal_shared"]),
                    "additional_nongrower_lag_h": float(
                        scenario["additional_nongrower_lag_h"]
                    ),
                    "production_branch": str(scenario["production_branch"]),
                    "cofactor_enabled": bool(scenario["cofactor_enabled"]),
                    "mean_abs_log2_error": float(np.mean(np.abs(values))),
                    "rmse_log2_error": float(np.sqrt(np.mean(values**2))),
                }
            )
        community_metrics = pd.DataFrame(community_rows)
        community_metrics.to_csv(
            branch_dir / "community_metrics.csv", index=False
        )

        species_rows: list[dict[str, object]] = []
        for species_index, species in enumerate(canonical_species):
            valid = presence[:, species_index]
            values = signed_error[valid, species_index]
            if len(values):
                species_mean_abs_error = float(np.mean(np.abs(values)))
                species_rmse = float(np.sqrt(np.mean(values**2)))
                species_mean_signed_error = float(np.mean(values))
            else:
                species_mean_abs_error = np.nan
                species_rmse = np.nan
                species_mean_signed_error = np.nan
            species_rows.append(
                {
                    "role": "grower" if species_index < n_growers else "nongrower",
                    "species": species,
                    "n_communities": int(valid.sum()),
                    "mean_abs_log2_error": species_mean_abs_error,
                    "rmse_log2_error": species_rmse,
                    "mean_signed_log2_error": species_mean_signed_error,
                }
            )
        pd.DataFrame(species_rows).to_csv(
            branch_dir / "species_metrics.csv", index=False
        )

        all_values = signed_error[presence]
        grower_values = signed_error[:, :n_growers][presence[:, :n_growers]]
        nongrower_values = signed_error[:, n_growers:][presence[:, n_growers:]]
        row: dict[str, object] = {
            "scenario": str(scenario["scenario"]),
            "universal_shared": bool(scenario["universal_shared"]),
            "additional_nongrower_lag_h": float(
                scenario["additional_nongrower_lag_h"]
            ),
            "production_branch": str(scenario["production_branch"]),
            "cofactor_enabled": bool(scenario["cofactor_enabled"]),
            "n_communities": n_communities,
            "n_scored_species_entries": int(presence.sum()),
            "mean_abs_log2_error": float(np.mean(np.abs(all_values))),
            "rmse_log2_error": float(np.sqrt(np.mean(all_values**2))),
            "mean_community_abs_log2_error": float(
                community_metrics["mean_abs_log2_error"].mean()
            ),
            "grower_mean_abs_log2_error": float(
                np.mean(np.abs(grower_values))
            ),
            "nongrower_mean_abs_log2_error": float(
                np.mean(np.abs(nongrower_values))
            ),
        }
        for community_type in (
            "random_assembly",
            "core_or_dropout",
            "full_community_32",
        ):
            selected = community_metrics[
                community_metrics["community_type"] == community_type
            ]
            row[f"{community_type}_n"] = len(selected)
            row[f"{community_type}_mean_community_abs_log2_error"] = (
                float(selected["mean_abs_log2_error"].mean())
                if len(selected)
                else np.nan
            )
        summary_rows.append(row)

    _matrix_with_species_rows(
        observed_relative, canonical_species, community_names
    ).to_csv(output_dir / "observed_relative_abundance.csv", index=False)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["mean_abs_log2_error", "rmse_log2_error"]
    )
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    summary.to_csv(output_dir / "model_comparison_summary.csv", index=False)
    return summary


def _load_spent_conditioner_abundances(data_dir: Path) -> pd.DataFrame:
    """Load single-conditioner yields and preserve legacy Bt+Bd coefficients."""

    monoculture_path = data_dir / "Canonical_16_grower_dm68_mean_growth.csv"
    monoculture = pd.read_csv(monoculture_path)
    required = {"species", "mean_growth"}
    if not required.issubset(monoculture.columns):
        raise ValueError(f"Missing {sorted(required)} in {monoculture_path}.")
    monoculture = monoculture.set_index("species")["mean_growth"]
    monoculture = pd.to_numeric(monoculture, errors="coerce")
    if monoculture.reindex(["Bt", "Bd"]).isna().any():
        raise ValueError("Bt/Bd monoculture yields are missing.")

    rows = [
        {
            "medium": "Bt_spent",
            "conditioner": "Bt",
            "abundance": float(monoculture.loc["Bt"]),
            "source": _portable_path(monoculture_path),
            "source_detail": "mean_growth",
        },
        {
            "medium": "Bd_spent",
            "conditioner": "Bd",
            "abundance": float(monoculture.loc["Bd"]),
            "source": _portable_path(monoculture_path),
            "source_detail": "mean_growth",
        },
    ]
    for conditioner, abundance in BT_BD_LEGACY_CONDITIONER_ABUNDANCES.items():
        rows.append(
            {
                "medium": "Bt_Bd_spent",
                "conditioner": conditioner,
                "abundance": float(abundance),
                "source": "legacy_spent_prediction_notebook",
                "source_detail": (
                    "Preserved data-derived constant from "
                    "Assemblies_in_Bt_Bd_BtBd_spent.ipynb; the exact upstream "
                    "absolute-abundance reconstruction is not recorded."
                ),
            }
        )
    return pd.DataFrame(rows)


def _construct_spent_initial_resources(
    grower_model: GrowerDMModel,
    production: pd.DataFrame,
    conditioner_abundances: pd.DataFrame,
) -> pd.DataFrame:
    """Construct completed spent media from residual DM plus prior products."""

    missing_growers = [
        name for name in GROWER_SPECIES if name not in production.columns
    ]
    if missing_growers:
        raise ValueError(f"Production profile is missing growers: {missing_growers}.")
    if list(production.columns) != list(GROWER_SPECIES):
        production = production.loc[:, list(GROWER_SPECIES)]

    grower_resources = grower_model.rate_cr.columns.tolist()
    byproduct_resources = production.index.tolist()
    dm_y0 = grower_model.dm_resource_y0.reindex(grower_resources)
    if dm_y0.isna().any():
        raise ValueError("Fresh-DM resource vector is incomplete.")

    profiles: dict[str, np.ndarray] = {}
    for medium in SPENT_MEDIUM_DATA_FILES:
        selected = conditioner_abundances[
            conditioner_abundances["medium"] == medium
        ]
        if selected.empty:
            raise ValueError(f"No conditioner abundance is defined for {medium}.")
        conditioners = selected["conditioner"].tolist()
        missing = [name for name in conditioners if name not in grower_model.rate_cr.index]
        if missing:
            raise ValueError(f"Conditioners are missing from grower CR: {missing}.")

        consumed = (
            grower_model.rate_cr.loc[conditioners].to_numpy(dtype=float) > 0
        ).any(axis=0)
        residual_dm = dm_y0.to_numpy(dtype=float) * (~consumed)
        preproduced = np.zeros(len(byproduct_resources), dtype=float)
        for row in selected.itertuples(index=False):
            preproduced += float(row.abundance) * production[row.conditioner].to_numpy(
                dtype=float
            )
        profiles[medium] = np.concatenate((residual_dm, preproduced))

    resource_ids = [f"DM::{name}" for name in grower_resources] + [
        f"Byproduct::{name}" for name in byproduct_resources
    ]
    result = pd.DataFrame(profiles, index=resource_ids)
    result.index.name = "resource_id"
    return result


def predict_assemblies_in_spent_media(
    data_dir: Path,
    results_root: Path,
    grower_model: GrowerDMModel,
    bundle: PredictionBundle,
    output_dir: Path,
    config: PredictionConfig,
    production_branches: Sequence[str] = DEFAULT_PRODUCTION_BRANCHES,
    overwrite: bool = False,
    max_communities_per_medium: int | None = None,
) -> pd.DataFrame:
    """Predict assemblies in completed Bt, Bd, and Bt+Bd spent media."""

    _prepare_output_directory(output_dir, overwrite)
    canonical_species = list(GROWER_SPECIES) + list(NONGROWER_SPECIES)
    family_dir = data_dir
    observed_frames: list[pd.DataFrame] = []
    community_names: list[str] = []
    community_labels: list[str] = []
    media: list[str] = []
    for medium, filename in SPENT_MEDIUM_DATA_FILES.items():
        path = family_dir / filename
        frame = pd.read_csv(path)
        if "species" not in frame.columns:
            raise ValueError(f"Missing species column in {path}.")
        frame = frame.set_index("species")
        missing = [name for name in canonical_species if name not in frame.index]
        if missing:
            raise ValueError(f"{path} is missing species: {missing}.")
        frame = frame.loc[canonical_species].apply(pd.to_numeric, errors="coerce")
        if max_communities_per_medium is not None:
            frame = frame.iloc[:, :max_communities_per_medium]
        labels = frame.columns.tolist()
        names = [f"{medium}::{label}" for label in labels]
        frame.columns = names
        observed_frames.append(frame)
        community_names.extend(names)
        community_labels.extend(labels)
        media.extend([medium] * len(labels))

    observed = pd.concat(observed_frames, axis=1)
    observed_values = observed.T.to_numpy(dtype=float)
    presence = np.isfinite(observed_values)
    richness = presence.sum(axis=1)
    if np.any(richness <= 0):
        bad = np.asarray(community_names)[richness <= 0].tolist()
        raise ValueError(f"Spent-medium communities without members: {bad}.")
    if np.any(np.where(presence, observed_values, 0.0) < 0):
        raise ValueError("Spent-medium observations contain negatives.")
    totals = np.nansum(observed_values, axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("At least one spent-medium observation has zero total.")
    observed_relative = np.divide(
        observed_values,
        totals,
        out=np.full_like(observed_values, np.nan),
        where=presence,
    )
    media_array = np.asarray(media, dtype=object)

    max_richness = int(richness.max())
    member_index = np.zeros((len(community_names), max_richness), dtype=int)
    member_mask = np.zeros_like(member_index, dtype=bool)
    for community_index in range(len(community_names)):
        members = np.flatnonzero(presence[community_index])
        member_index[community_index, : len(members)] = members
        member_mask[community_index, : len(members)] = True

    conditioner_abundances = _load_spent_conditioner_abundances(data_dir)
    conditioner_abundances.to_csv(
        output_dir / "conditioner_abundances.csv", index=False
    )
    production_root = results_root / "production_fits"
    nongrower_resource_names = bundle.rate_cr.columns.tolist()
    production_by_branch = {
        branch: _load_production_profile(
            production_root, branch, nongrower_resource_names
        )
        for branch in production_branches
    }

    scenarios: list[dict[str, object]] = []
    models: list[dict[str, np.ndarray]] = []
    spent_profiles: list[pd.DataFrame] = []
    unused_universal_rates = pd.Series(
        np.zeros(len(NONGROWER_SPECIES)), index=NONGROWER_SPECIES
    )
    for production_branch in production_branches:
        production = production_by_branch[production_branch]
        resources = _construct_spent_initial_resources(
            grower_model, production, conditioner_abundances
        )
        for cofactor_enabled in (True, False):
            cofactor_name = "cofactor_on" if cofactor_enabled else "cofactor_off"
            scenarios.append(
                {
                    "scenario": f"{production_branch}__{cofactor_name}",
                    "production_branch": production_branch,
                    "cofactor_enabled": cofactor_enabled,
                    "relative_directory": Path(production_branch) / cofactor_name,
                }
            )
            models.append(
                _build_full_dm_scenario_model(
                    grower_model,
                    bundle,
                    production,
                    universal_shared=False,
                    additional_nongrower_lag_h=0.0,
                    cofactor_enabled=cofactor_enabled,
                    universal_rates=unused_universal_rates,
                )
            )
            spent_profiles.append(resources)
    expected_scenarios = len(production_branches) * 2
    if len(scenarios) != expected_scenarios:
        raise ValueError(
            f"Expected {expected_scenarios} spent scenarios, "
            f"got {len(scenarios)}."
        )

    rate_batches: list[np.ndarray] = []
    production_batches: list[np.ndarray] = []
    resource_batches: list[np.ndarray] = []
    D_batches: list[np.ndarray] = []
    F_batches: list[np.ndarray] = []
    lag_batches: list[np.ndarray] = []
    required_batches: list[np.ndarray] = []
    biomass_batches: list[np.ndarray] = []
    for model, resources in zip(models, spent_profiles):
        local_rate = model["rate_cr"][member_index].copy()
        local_production = model["resource_production"][member_index].copy()
        local_D = model["D"][member_index].copy()
        local_F = model["F"][member_index].copy()
        local_lag = model["lag"][member_index].copy()
        local_required = model["cofactor_required"][member_index].copy()
        local_rate[~member_mask] = 0.0
        local_production[~member_mask] = 0.0
        local_D[~member_mask] = 0.0
        local_F[~member_mask] = 0.0
        local_lag[~member_mask] = 0.0
        local_required[~member_mask] = False
        rate_batches.append(local_rate)
        production_batches.append(local_production)
        resource_batches.append(
            np.vstack([resources[medium].to_numpy(dtype=float) for medium in media])
        )
        D_batches.append(local_D)
        F_batches.append(local_F)
        lag_batches.append(local_lag)
        required_batches.append(local_required)
        biomass_batches.append(member_mask.astype(float) * config.initial_abundance)

    local_prediction = simulate_community_production_batch(
        np.concatenate(rate_batches, axis=0),
        np.concatenate(production_batches, axis=0),
        np.concatenate(resource_batches, axis=0),
        np.concatenate(biomass_batches, axis=0),
        np.concatenate(D_batches, axis=0),
        np.concatenate(F_batches, axis=0),
        np.concatenate(lag_batches, axis=0),
        np.concatenate(required_batches, axis=0),
        config,
    ).reshape(len(scenarios), len(community_names), max_richness)

    predicted_absolute = np.full(
        (len(scenarios), len(community_names), len(canonical_species)),
        np.nan,
        dtype=float,
    )
    for scenario_index in range(len(scenarios)):
        for local_member in range(max_richness):
            valid = member_mask[:, local_member]
            predicted_absolute[
                scenario_index,
                np.flatnonzero(valid),
                member_index[valid, local_member],
            ] = local_prediction[scenario_index, valid, local_member]

    summary_rows: list[dict[str, object]] = []
    n_growers = len(GROWER_SPECIES)
    for scenario_index, scenario in enumerate(scenarios):
        branch_dir = output_dir / Path(str(scenario["relative_directory"]))
        branch_dir.mkdir(parents=True, exist_ok=False)
        resources = spent_profiles[scenario_index].reset_index()
        resources.insert(
            1,
            "resource_block",
            np.where(
                resources["resource_id"].str.startswith("DM::"),
                "residual_fresh_dm",
                "preproduced_byproduct",
            ),
        )
        resources.to_csv(branch_dir / "initial_spent_resources.csv", index=False)

        absolute = predicted_absolute[scenario_index]
        predicted_totals = np.nansum(absolute, axis=1, keepdims=True)
        if np.any(predicted_totals <= 0):
            raise ValueError(f"Zero predicted total in {scenario['scenario']}.")
        relative = np.divide(
            absolute,
            predicted_totals,
            out=np.full_like(absolute, np.nan),
            where=presence,
        )
        signed_error = np.full_like(relative, np.nan)
        signed_error[presence] = np.log2(
            np.clip(relative[presence], config.prediction_floor, None)
            / np.clip(observed_relative[presence], config.prediction_floor, None)
        )

        _matrix_with_species_rows(
            absolute, canonical_species, community_names
        ).to_csv(branch_dir / "predicted_absolute_abundance.csv", index=False)
        _matrix_with_species_rows(
            relative, canonical_species, community_names
        ).to_csv(branch_dir / "predicted_relative_abundance.csv", index=False)
        _matrix_with_species_rows(
            signed_error, canonical_species, community_names
        ).to_csv(branch_dir / "signed_log2_error.csv", index=False)

        community_rows: list[dict[str, object]] = []
        for community_index, community in enumerate(community_names):
            values = signed_error[community_index, presence[community_index]]
            n_growers_present = int(
                presence[community_index, :n_growers].sum()
            )
            community_rows.append(
                {
                    "community": community,
                    "community_label": community_labels[community_index],
                    "medium": media[community_index],
                    "richness": int(richness[community_index]),
                    "n_growers": n_growers_present,
                    "n_nongrowers": int(richness[community_index])
                    - n_growers_present,
                    "production_branch": str(scenario["production_branch"]),
                    "cofactor_enabled": bool(scenario["cofactor_enabled"]),
                    "mean_abs_log2_error": float(np.mean(np.abs(values))),
                    "rmse_log2_error": float(np.sqrt(np.mean(values**2))),
                }
            )
        community_metrics = pd.DataFrame(community_rows)
        community_metrics.to_csv(
            branch_dir / "community_metrics.csv", index=False
        )

        species_rows: list[dict[str, object]] = []
        for species_index, species in enumerate(canonical_species):
            valid = presence[:, species_index]
            values = signed_error[valid, species_index]
            species_rows.append(
                {
                    "role": "grower" if species_index < n_growers else "nongrower",
                    "species": species,
                    "n_communities": int(valid.sum()),
                    "mean_abs_log2_error": (
                        float(np.mean(np.abs(values))) if len(values) else np.nan
                    ),
                    "rmse_log2_error": (
                        float(np.sqrt(np.mean(values**2))) if len(values) else np.nan
                    ),
                    "mean_signed_log2_error": (
                        float(np.mean(values)) if len(values) else np.nan
                    ),
                }
            )
        pd.DataFrame(species_rows).to_csv(
            branch_dir / "species_metrics.csv", index=False
        )

        all_values = signed_error[presence]
        grower_values = signed_error[:, :n_growers][presence[:, :n_growers]]
        nongrower_values = signed_error[:, n_growers:][presence[:, n_growers:]]
        row: dict[str, object] = {
            "scenario": str(scenario["scenario"]),
            "production_branch": str(scenario["production_branch"]),
            "cofactor_enabled": bool(scenario["cofactor_enabled"]),
            "n_communities": len(community_names),
            "n_scored_species_entries": int(presence.sum()),
            "mean_abs_log2_error": float(np.mean(np.abs(all_values))),
            "rmse_log2_error": float(np.sqrt(np.mean(all_values**2))),
            "mean_community_abs_log2_error": float(
                community_metrics["mean_abs_log2_error"].mean()
            ),
            "grower_mean_abs_log2_error": float(np.mean(np.abs(grower_values))),
            "nongrower_mean_abs_log2_error": float(
                np.mean(np.abs(nongrower_values))
            ),
        }
        for medium in SPENT_MEDIUM_DATA_FILES:
            selected = community_metrics[community_metrics["medium"] == medium]
            row[f"{medium}_n"] = len(selected)
            row[f"{medium}_mean_community_abs_log2_error"] = float(
                selected["mean_abs_log2_error"].mean()
            )
        summary_rows.append(row)

    _matrix_with_species_rows(
        observed_relative, canonical_species, community_names
    ).to_csv(output_dir / "observed_relative_abundance.csv", index=False)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["mean_abs_log2_error", "rmse_log2_error"]
    )
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    summary.to_csv(output_dir / "model_comparison_summary.csv", index=False)
    return summary
