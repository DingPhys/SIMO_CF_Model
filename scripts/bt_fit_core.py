"""Fixed-structure Bt-resource and production fitting core.

This module contains only continuous-parameter fits using supplied fixed
consumption matrices. It contains no lag or resource-support fitting.
"""


from __future__ import annotations


import json


import math


import os


import shutil


from dataclasses import asdict, dataclass


from pathlib import Path


from typing import Iterable, Sequence


import numpy as np


import pandas as pd


from scipy.optimize import least_squares, linprog, nnls


NONGROWER_SPECIES = (
    "Af", "Ao", "As", "Bl.s", "Col", "Cs", "Et", "Eu.c",
    "Eu.l", "Im", "Ld", "Lsp", "Mi", "Pc", "Va", "Vp",
)


COFACTOR_SPECIES = (
    "Col", "Cs", "Et", "Eu.c", "Eu.l", "Im", "Ld", "Lsp", "Mi", "Va",
)


GROWER_SPECIES = (
    "Ai", "Ac", "Bfi", "Bf", "Bt", "Bu", "Bx", "Ba",
    "Csp", "Dl", "Ls", "Mf", "Pm", "Bd", "Bv", "Rg",
)


@dataclass(frozen=True)
class BtBaseConfig:
    """Numerical settings shared by every cofactor configuration."""

    floor: float = 1e-3
    resource_reciprocal_abs_log2: float = 0.8
    resource_reciprocal_abs_difference: float = 0.2
    resource_min_pair_growth: float = 0.01
    resource_max_nfev: int = 100000
    resource_lp_prediction_tolerance: float = 1e-8
    resource_pair_excluded_species: tuple[str, ...] = ("Cs",)
    fixed_universal_resource_abundance: float | None = None
    fixed_resource_abundances: tuple[tuple[str, float], ...] = ()

    auc_initial_abundance: float = 0.01
    auc_cycles: int = 10
    auc_hours_per_cycle: float = 48.0
    auc_dt: float = 0.01
    auc_dilution: float = 200.0
    auc_mu_threshold: float = 1e-3

    def validate(self) -> None:
        if self.floor <= 0:
            raise ValueError("floor must be positive.")
        if self.resource_reciprocal_abs_log2 <= 0:
            raise ValueError("resource_reciprocal_abs_log2 must be positive.")
        if self.resource_reciprocal_abs_difference <= 0:
            raise ValueError("resource_reciprocal_abs_difference must be positive.")
        if self.resource_min_pair_growth < 0:
            raise ValueError("resource_min_pair_growth must be nonnegative.")
        if self.resource_max_nfev <= 0:
            raise ValueError("resource_max_nfev must be positive.")
        if self.resource_lp_prediction_tolerance <= 0:
            raise ValueError("resource_lp_prediction_tolerance must be positive.")
        if len(set(self.resource_pair_excluded_species)) != len(
            self.resource_pair_excluded_species
        ):
            raise ValueError("resource_pair_excluded_species must be unique.")
        unknown_excluded_species = sorted(
            set(self.resource_pair_excluded_species) - set(COFACTOR_SPECIES)
        )
        if unknown_excluded_species:
            raise ValueError(
                "resource_pair_excluded_species contains species outside the "
                f"Bt resource fit: {unknown_excluded_species}"
            )
        if (
            self.fixed_universal_resource_abundance is not None
            and self.fixed_universal_resource_abundance < 0
        ):
            raise ValueError(
                "fixed_universal_resource_abundance must be nonnegative."
            )
        fixed_resource_names = [
            name for name, _ in self.fixed_resource_abundances
        ]
        if len(set(fixed_resource_names)) != len(fixed_resource_names):
            raise ValueError("fixed_resource_abundances resource names must be unique.")
        for name, value in self.fixed_resource_abundances:
            if not name:
                raise ValueError("Fixed resource names cannot be empty.")
            if not np.isfinite(value) or value < 0:
                raise ValueError(
                    f"Fixed resource abundance must be finite and nonnegative: "
                    f"{name}={value}"
                )
        if self.auc_initial_abundance <= 0 or self.auc_cycles <= 0:
            raise ValueError("AUC initial abundance and cycle count must be positive.")
        if self.auc_hours_per_cycle <= 0 or self.auc_dt <= 0:
            raise ValueError("AUC hours and dt must be positive.")
        if self.auc_dilution <= 0 or self.auc_mu_threshold <= 0:
            raise ValueError("AUC dilution and threshold must be positive.")


DEFAULT_PRODUCTION_PRIOR_WEIGHTS = (0.01, 0.1, 1.0)


@dataclass(frozen=True)
class ProductionConfig:
    """Settings shared by the fitted production branches."""

    prior_weights: tuple[float, ...] = DEFAULT_PRODUCTION_PRIOR_WEIGHTS
    clip_min: float = 1e-3
    profile_floor: float = 1e-12
    multi_start: int = 20
    seed: int = 2431
    maxiter: int = 5000
    maxfun: int = 200000

    def validate(self) -> None:
        if (
            not self.prior_weights
            or not np.isfinite(self.prior_weights).all()
            or min(self.prior_weights) < 0
        ):
            raise ValueError("Production prior weights must be finite and nonnegative.")
        if len(set(float(value) for value in self.prior_weights)) != len(
            self.prior_weights
        ):
            raise ValueError("Production prior weights must be unique.")
        if self.clip_min <= 0 or self.profile_floor <= 0:
            raise ValueError("Production floors must be positive.")
        if self.multi_start <= 0 or self.maxiter <= 0 or self.maxfun <= 0:
            raise ValueError("Production optimizer settings must be positive.")


@dataclass
class BtBaseModel:
    binary_cr: pd.DataFrame
    rate_cr: pd.DataFrame
    y0: pd.Series
    y0_before_tie_break: pd.Series
    auc: pd.DataFrame
    resource_fit_predictions: pd.DataFrame
    resource_fit_diagnostics: pd.DataFrame
    growth_rate_scaling: pd.DataFrame


@dataclass
class ProductionFit:
    profile: np.ndarray
    predicted: np.ndarray
    residual: np.ndarray
    objective: float
    data_loss: float
    prior_loss: float
    success: bool
    message: str
    iterations: int


def _format_number(value: float) -> str:
    text = format(float(value), ".12g")
    return text.replace("-", "m").replace(".", "p").replace("+", "")


def production_branch_name(prior_weight: float) -> str:
    """Return the stable result-folder name for one production prior weight."""

    value = float(prior_weight)
    if not np.isfinite(value) or value < 0:
        raise ValueError("Production prior weight must be finite and nonnegative.")
    return f"prod_prior-{_format_number(value)}"


def _json_ready(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Integrate with NumPy 1.x or 2.x."""

    function = getattr(np, "trapezoid", None)
    if function is None:
        function = np.trapz
    return float(function(y, x))


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _prepare_new_directory(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists and is not empty: {path}. "
                "Use --overwrite only when replacement is intentional."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _load_binary_cr(
    data_dir: Path,
    binary_cr_path: Path | None = None,
) -> pd.DataFrame:
    path = (
        binary_cr_path
        if binary_cr_path is not None
        else data_dir / "NonGrowers_CR_ZeroOne_nosixexclusive_R.csv"
    )
    frame = pd.read_csv(path)
    if frame.columns[0] != "species":
        raise ValueError(f"The first column of {path} must be 'species'.")
    frame = frame.set_index("species")
    missing = [name for name in NONGROWER_SPECIES if name not in frame.index]
    if missing:
        raise ValueError(f"Binary CR is missing species: {missing}")
    frame = frame.loc[list(NONGROWER_SPECIES)].apply(pd.to_numeric, errors="raise")
    if frame.shape[1] == 0:
        raise ValueError(f"No resource columns were found in {path}.")
    expected_resources = [
        f"Resource_{index}" for index in range(1, frame.shape[1] + 1)
    ]
    if frame.columns.tolist() != expected_resources:
        raise ValueError(
            "Binary CR resource columns must be a contiguous ordered sequence "
            "from Resource_1 through Resource_M."
        )
    values = frame.to_numpy(dtype=float)
    if not np.isin(values, [0.0, 1.0]).all():
        raise ValueError("Binary CR contains values other than 0 and 1.")
    frame.index.name = "species"
    return frame


def _load_double_spent(data_dir: Path, floor: float) -> pd.DataFrame:
    path = data_dir / "Nongrowers_in_Bt_double_spent.csv"
    frame = pd.read_csv(path)
    if frame.columns[0] != "Species":
        raise ValueError(f"The first column of {path} must be 'Species'.")
    frame = frame.set_index("Species")
    required_columns = list(NONGROWER_SPECIES) + ["Full_Bt", "Diluted_Bt"]
    missing_rows = [name for name in NONGROWER_SPECIES if name not in frame.index]
    missing_columns = [name for name in required_columns if name not in frame.columns]
    if missing_rows or missing_columns:
        raise ValueError(
            f"Incomplete double-spent table; rows={missing_rows}, columns={missing_columns}."
        )
    frame = frame.loc[list(NONGROWER_SPECIES), required_columns]
    frame = frame.apply(pd.to_numeric, errors="coerce").clip(lower=floor)
    for species in NONGROWER_SPECIES:
        frame.loc[species, species] = np.nan
    return frame


def _load_nongrower_growth_table(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "Non-growers_In_Bt_Spent_growth_curve_fit_summary.csv"
    frame = pd.read_csv(path).set_index("focal_species")
    required = {"lambda_h_mean_grow_only"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing growth-rate column in {path}.")
    missing = [name for name in NONGROWER_SPECIES if name not in frame.index]
    if missing:
        raise ValueError(f"Growth curve table is missing species: {missing}")
    return frame.loc[list(NONGROWER_SPECIES)]


def _fit_positive_resource_vector(
    design: np.ndarray,
    observed: np.ndarray,
    initial_theta: np.ndarray,
    floor: float,
    max_nfev: int,
    fixed_values: np.ndarray | None = None,
):
    observed = np.asarray(observed, dtype=float)
    design = np.asarray(design, dtype=float)
    valid = np.isfinite(observed)
    if not np.any(valid):
        raise ValueError("No finite observations are available for the resource fit.")
    design_valid = design[valid]
    observed_valid = observed[valid]

    if fixed_values is None:
        fixed_values = np.full(design.shape[1], np.nan, dtype=float)
    else:
        fixed_values = np.asarray(fixed_values, dtype=float)
    if fixed_values.shape != (design.shape[1],):
        raise ValueError("fixed_values must have one entry per resource.")
    fixed_mask = np.isfinite(fixed_values)
    if np.any(fixed_values[fixed_mask] < 0):
        raise ValueError("Fixed resource abundances must be nonnegative.")
    free_mask = ~fixed_mask
    if not np.any(free_mask):
        raise ValueError("At least one resource abundance must remain free.")

    initial_theta = np.asarray(initial_theta, dtype=float)
    if initial_theta.shape == (design.shape[1],):
        initial_theta = initial_theta[free_mask]
    elif initial_theta.shape != (int(free_mask.sum()),):
        raise ValueError("initial_theta does not match the free resource count.")

    def assemble_resource_vector(theta: np.ndarray) -> np.ndarray:
        resource = fixed_values.copy()
        resource[free_mask] = np.exp(theta)
        return resource

    def residual(theta: np.ndarray) -> np.ndarray:
        prediction = design_valid @ assemble_resource_vector(theta)
        return np.log2(np.clip(prediction, floor, None)) - np.log2(
            np.clip(observed_valid, floor, None)
        )

    result = least_squares(
        residual,
        x0=initial_theta,
        method="trf",
        max_nfev=max_nfev,
    )
    return assemble_resource_vector(result.x), result


def _resource_design(
    binary_sub: np.ndarray,
    pair_observed: np.ndarray,
    full_observed: np.ndarray,
    pair_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    rows: list[np.ndarray] = []
    observed: list[float] = []
    labels: list[dict[str, object]] = []
    for focal_index, focal in enumerate(COFACTOR_SPECIES):
        for donor_index, donor in enumerate(COFACTOR_SPECIES):
            if not pair_mask[focal_index, donor_index]:
                continue
            rows.append(binary_sub[focal_index] * (1.0 - binary_sub[donor_index]))
            observed.append(float(pair_observed[focal_index, donor_index]))
            labels.append(
                {
                    "observation_type": "double_spent",
                    "focal_species": focal,
                    "donor_species": donor,
                }
            )
        rows.append(binary_sub[focal_index].copy())
        observed.append(float(full_observed[focal_index]))
        labels.append(
            {
                "observation_type": "full_bt",
                "focal_species": focal,
                "donor_species": None,
            }
        )
    return np.vstack(rows), np.asarray(observed, dtype=float), labels


def _reciprocal_pair_screen(
    pair_observed: np.ndarray,
    full_observed: np.ndarray,
    config: BtBaseConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the fixed reciprocal-consistency screen used for the Bt fit."""

    # For focal i and donor j, compare the two reciprocal totals
    # FullBt_i + X(j in i-spent) and FullBt_j + X(i in j-spent).
    reciprocal_ij = full_observed[:, None] + pair_observed.T
    reciprocal_ji = full_observed[None, :] + pair_observed
    difference = reciprocal_ij - reciprocal_ji
    with np.errstate(divide="ignore", invalid="ignore"):
        log2_ratio = np.log2(reciprocal_ij / reciprocal_ji)

    pair_mask = (
        np.isfinite(pair_observed)
        & np.isfinite(log2_ratio)
        & (np.abs(log2_ratio) < config.resource_reciprocal_abs_log2)
        & (pair_observed > config.resource_min_pair_growth)
        & (np.abs(difference) < config.resource_reciprocal_abs_difference)
    )
    np.fill_diagonal(pair_mask, False)
    return pair_mask, log2_ratio, difference


def _exclude_species_from_pair_fit(
    pair_mask: np.ndarray,
    species: Sequence[str],
    excluded_species: Sequence[str],
) -> np.ndarray:
    """Remove every directed double-spent condition involving a species."""

    filtered = np.asarray(pair_mask, dtype=bool).copy()
    species_to_index = {name: index for index, name in enumerate(species)}
    missing = sorted(set(excluded_species) - set(species_to_index))
    if missing:
        raise ValueError(f"Cannot exclude unknown pair-fit species: {missing}")
    for name in excluded_species:
        index = species_to_index[name]
        filtered[index, :] = False
        filtered[:, index] = False
    return filtered


def _resource_log2_loss(
    design: np.ndarray,
    observed: np.ndarray,
    y0: np.ndarray,
    floor: float,
) -> float:
    prediction = np.clip(design @ y0, floor, None)
    observed_safe = np.clip(observed, floor, None)
    return float(np.sum(np.log2(prediction / observed_safe) ** 2))


def _configured_fixed_resource_values(
    binary_cr: pd.DataFrame,
    config: BtBaseConfig,
) -> pd.Series:
    """Build the complete resource-wise fixed-value vector for one fit."""

    fixed = pd.Series(np.nan, index=binary_cr.columns, dtype=float)
    universal_mask = binary_cr.gt(0).all(axis=0)
    if config.fixed_universal_resource_abundance is not None:
        if not universal_mask.any():
            raise ValueError(
                "No resource is used by every nongrower, so the universal "
                "resource abundance cannot be fixed."
            )
        fixed.loc[universal_mask] = config.fixed_universal_resource_abundance

    for resource_name, abundance in config.fixed_resource_abundances:
        if resource_name not in fixed.index:
            raise ValueError(f"Cannot fix unknown resource: {resource_name}")
        existing = fixed.loc[resource_name]
        if np.isfinite(existing) and not np.isclose(existing, abundance):
            raise ValueError(
                f"Conflicting fixed values for {resource_name}: "
                f"{existing} and {abundance}"
            )
        fixed.loc[resource_name] = abundance
    return fixed


def _minimize_shared_resource_mass(
    design: np.ndarray,
    y0_before_tie_break: np.ndarray,
    binary_sub: np.ndarray,
    tolerance: float,
    fixed_values: np.ndarray | None = None,
) -> tuple[np.ndarray, object, np.ndarray, float]:
    """Choose the minimum-shared-resource member of an equivalent fit."""

    prediction_before = design @ y0_before_tie_break
    n_consumers = np.sum(binary_sub > 0, axis=0)
    exclusive = n_consumers == 1
    penalty = (~exclusive).astype(float)
    if fixed_values is None:
        fixed_values = np.full(design.shape[1], np.nan, dtype=float)
    else:
        fixed_values = np.asarray(fixed_values, dtype=float)
    if fixed_values.shape != (design.shape[1],):
        raise ValueError("fixed_values must have one entry per resource.")
    bounds = [
        (float(value), float(value)) if np.isfinite(value) else (0.0, None)
        for value in fixed_values
    ]
    result = linprog(
        c=penalty,
        A_eq=design,
        b_eq=prediction_before,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Bt-resource LP tie-break failed: {result.message}")

    y0_after = np.asarray(result.x, dtype=float)
    maximum_prediction_change = float(
        np.max(np.abs(design @ y0_after - prediction_before))
    )
    scale = max(1.0, float(np.max(np.abs(prediction_before))))
    if maximum_prediction_change > tolerance * scale:
        raise RuntimeError(
            "Bt-resource LP changed fitted predictions by "
            f"{maximum_prediction_change:.3g}, above the allowed tolerance."
        )
    return y0_after, result, exclusive, maximum_prediction_change


def _fit_bt_resource(
    binary_cr: pd.DataFrame,
    double_spent: pd.DataFrame,
    config: BtBaseConfig,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    binary_sub = binary_cr.loc[list(COFACTOR_SPECIES)].to_numpy(dtype=float)
    pair_observed = double_spent.loc[
        list(COFACTOR_SPECIES), list(COFACTOR_SPECIES)
    ].to_numpy(dtype=float)
    full_observed = double_spent.loc[
        list(COFACTOR_SPECIES), "Full_Bt"
    ].to_numpy(dtype=float)

    pair_mask, reciprocal_log2_ratio, reciprocal_difference = (
        _reciprocal_pair_screen(pair_observed, full_observed, config)
    )
    retained_pairs_before_species_exclusion = int(pair_mask.sum())
    pair_mask = _exclude_species_from_pair_fit(
        pair_mask,
        COFACTOR_SPECIES,
        config.resource_pair_excluded_species,
    )
    design, observed, _ = _resource_design(
        binary_sub, pair_observed, full_observed, pair_mask
    )
    universal_mask = binary_cr.gt(0).all(axis=0).to_numpy(dtype=bool)
    fixed_resource_values = _configured_fixed_resource_values(binary_cr, config)
    fixed_values = fixed_resource_values.to_numpy(dtype=float)
    fixed_mask = np.isfinite(fixed_values)
    y0_before_tie_break, optimizer = _fit_positive_resource_vector(
        design,
        observed,
        initial_theta=np.zeros(binary_sub.shape[1]),
        floor=config.floor,
        max_nfev=config.resource_max_nfev,
        fixed_values=fixed_values,
    )
    (
        y0,
        lp_result,
        exclusive,
        maximum_prediction_change,
    ) = _minimize_shared_resource_mass(
        design,
        y0_before_tie_break,
        binary_sub,
        config.resource_lp_prediction_tolerance,
        fixed_values=fixed_values,
    )

    fit_rank = int(np.linalg.matrix_rank(design))
    loss_before = _resource_log2_loss(
        design, observed, y0_before_tie_break, config.floor
    )
    loss_after = _resource_log2_loss(design, observed, y0, config.floor)
    history = pd.DataFrame(
        [
            {
                "stage": "reciprocal_screen_log2_fit_lp_tie_break",
                "candidate_pairs": int(np.isfinite(pair_observed).sum()),
                "retained_pairs_before_species_exclusion": (
                    retained_pairs_before_species_exclusion
                ),
                "retained_pairs": int(pair_mask.sum()),
                "excluded_pair_species": ";".join(
                    config.resource_pair_excluded_species
                ),
                "pair_observations_removed_by_species_exclusion": (
                    retained_pairs_before_species_exclusion - int(pair_mask.sum())
                ),
                "full_bt_observations": int(np.isfinite(full_observed).sum()),
                "design_rows": int(design.shape[0]),
                "resource_count": int(design.shape[1]),
                "design_rank": fit_rank,
                "unidentifiable_dimensions": int(design.shape[1] - fit_rank),
                "universal_resource_count": int(universal_mask.sum()),
                "universal_resource_names": ";".join(
                    binary_cr.columns[universal_mask].tolist()
                ),
                "fixed_universal_resource_abundance": (
                    config.fixed_universal_resource_abundance
                ),
                "fixed_resource_count": int(fixed_mask.sum()),
                "fixed_resource_names": ";".join(
                    fixed_resource_values.index[fixed_mask].tolist()
                ),
                "fixed_resource_values": ";".join(
                    f"{name}={fixed_resource_values.loc[name]:.12g}"
                    for name in fixed_resource_values.index[fixed_mask]
                ),
                "optimizer_success": bool(optimizer.success),
                "optimizer_cost": float(optimizer.cost),
                "lp_success": bool(lp_result.success),
                "loss_before_tie_break": loss_before,
                "loss_after_tie_break": loss_after,
                "maximum_fitted_prediction_change": maximum_prediction_change,
                "shared_resource_sum_before": float(
                    y0_before_tie_break[~exclusive].sum()
                ),
                "shared_resource_sum_after": float(y0[~exclusive].sum()),
                "maximum_fixed_resource_change": (
                    float(
                        np.max(
                            np.abs(
                                y0[fixed_mask]
                                - y0_before_tie_break[fixed_mask]
                            )
                        )
                    )
                    if fixed_mask.any()
                    else 0.0
                ),
            }
        ]
    )

    resource_names = binary_cr.columns.tolist()
    y0_series = pd.Series(y0, index=resource_names, name="bt_resource_y0")
    y0_before_series = pd.Series(
        y0_before_tie_break,
        index=resource_names,
        name="bt_resource_y0_before_tie_break",
    )
    prediction_rows: list[dict[str, object]] = []
    for focal_index, focal in enumerate(COFACTOR_SPECIES):
        for donor_index, donor in enumerate(COFACTOR_SPECIES):
            observed_value = pair_observed[focal_index, donor_index]
            row = binary_sub[focal_index] * (1.0 - binary_sub[donor_index])
            predicted_before = float(row @ y0_before_tie_break)
            predicted_value = float(row @ y0)
            error = (
                float(
                    np.log2(
                        np.clip(predicted_value, config.floor, None)
                        / np.clip(observed_value, config.floor, None)
                    )
                )
                if np.isfinite(observed_value)
                else np.nan
            )
            prediction_rows.append(
                {
                    "observation_type": "double_spent",
                    "focal_species": focal,
                    "donor_species": donor,
                    "observed": observed_value,
                    "predicted_before_tie_break": predicted_before,
                    "predicted": predicted_value,
                    "tie_break_prediction_change": predicted_value - predicted_before,
                    "log2_error": error,
                    "reciprocal_log2_ratio": reciprocal_log2_ratio[
                        focal_index, donor_index
                    ],
                    "reciprocal_difference": reciprocal_difference[
                        focal_index, donor_index
                    ],
                    "used_in_final_fit": bool(pair_mask[focal_index, donor_index]),
                }
            )
        predicted_full_before = float(
            binary_sub[focal_index] @ y0_before_tie_break
        )
        predicted_full = float(binary_sub[focal_index] @ y0)
        prediction_rows.append(
            {
                "observation_type": "full_bt",
                "focal_species": focal,
                "donor_species": None,
                "observed": full_observed[focal_index],
                "predicted_before_tie_break": predicted_full_before,
                "predicted": predicted_full,
                "tie_break_prediction_change": (
                    predicted_full - predicted_full_before
                ),
                "log2_error": float(
                    np.log2(
                        np.clip(predicted_full, config.floor, None)
                        / np.clip(full_observed[focal_index], config.floor, None)
                    )
                ),
                "reciprocal_log2_ratio": np.nan,
                "reciprocal_difference": np.nan,
                "used_in_final_fit": True,
            }
        )
    return y0_series, y0_before_series, pd.DataFrame(prediction_rows), history


def _construct_rate_cr(
    binary_cr: pd.DataFrame,
    y0: pd.Series,
    growth_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rate = binary_cr.astype(float).copy()
    records: list[dict[str, object]] = []
    y0_values = y0.loc[binary_cr.columns].to_numpy(dtype=float)
    for species in COFACTOR_SPECIES:
        measured_rate = float(growth_table.loc[species, "lambda_h_mean_grow_only"])
        if not np.isfinite(measured_rate) or measured_rate <= 0:
            raise ValueError(f"Invalid fitted growth rate for {species}: {measured_rate}")
        support = binary_cr.loc[species].to_numpy(dtype=float)
        fresh_access = float(support @ y0_values)
        if fresh_access <= 0:
            raise ValueError(f"Cannot scale {species}; binary support has zero Bt access.")
        scale = measured_rate / fresh_access
        rate.loc[species] = support * scale
        records.append(
            {
                "species": species,
                "measured_growth_rate": measured_rate,
                "binary_fresh_access": fresh_access,
                "row_scale": scale,
                "achieved_growth_rate": float(rate.loc[species].to_numpy() @ y0_values),
            }
        )
    return rate, pd.DataFrame(records)


def _resource_rhs_fixed(
    state: np.ndarray,
    rate_cr: np.ndarray,
) -> np.ndarray:
    n_species = rate_cr.shape[0]
    biomass = state[:n_species]
    resources = state[n_species:]
    usable = np.where(resources > 0.0, resources, 0.0)
    biomass_change = biomass * (rate_cr @ usable)
    resource_change = -usable * (rate_cr.T @ biomass)
    return np.concatenate([biomass_change, resource_change])


def _rk4_resource(state: np.ndarray, dt: float, rate_cr: np.ndarray) -> np.ndarray:
    k1 = _resource_rhs_fixed(state, rate_cr)
    k2 = _resource_rhs_fixed(state + 0.5 * dt * k1, rate_cr)
    k3 = _resource_rhs_fixed(state + 0.5 * dt * k2, rate_cr)
    k4 = _resource_rhs_fixed(state + dt * k3, rate_cr)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _simulate_resource_cycle(
    rate_cr: np.ndarray,
    initial_biomass: np.ndarray,
    y0: np.ndarray,
    hours: float,
    dt: float,
    store: bool,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    state = np.concatenate([initial_biomass, y0]).astype(float)
    time = 0.0
    times = [0.0] if store else None
    states = [state.copy()] if store else None
    while time < hours - 1e-15:
        dt_step = min(dt, hours - time)
        state = _rk4_resource(state, dt_step, rate_cr)
        state = np.maximum(state, 0.0)
        next_time = time + dt_step
        time = hours if abs(next_time - hours) <= 1e-15 else next_time
        if store:
            assert times is not None and states is not None
            times.append(time)
            states.append(state.copy())
    return (
        state,
        np.asarray(times, dtype=float) if store else None,
        np.asarray(states, dtype=float) if store else None,
    )


def _auc_until_mu_threshold(
    times: np.ndarray,
    biomass: np.ndarray,
    resources: np.ndarray,
    binary_support: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    mu = resources @ binary_support
    below = np.flatnonzero(mu <= threshold)
    if below.size == 0:
        return {
            "auc": _trapezoid(biomass, times),
            "t_cut_h": float(times[-1]),
            "biomass_at_cut": float(biomass[-1]),
            "mu_initial": float(mu[0]),
            "mu_at_cut": float(mu[-1]),
            "threshold": threshold,
            "crossed": False,
        }
    index = int(below[0])
    if index == 0:
        return {
            "auc": 0.0,
            "t_cut_h": float(times[0]),
            "biomass_at_cut": float(biomass[0]),
            "mu_initial": float(mu[0]),
            "mu_at_cut": float(mu[0]),
            "threshold": threshold,
            "crossed": True,
        }
    t0, t1 = times[index - 1], times[index]
    m0, m1 = mu[index - 1], mu[index]
    x0, x1 = biomass[index - 1], biomass[index]
    alpha = (m0 - threshold) / (m0 - m1) if m0 != m1 else 1.0
    alpha = float(np.clip(alpha, 0.0, 1.0))
    t_cut = float(t0 + alpha * (t1 - t0))
    x_cut = float(x0 + alpha * (x1 - x0))
    auc = _trapezoid(
        np.r_[biomass[:index], x_cut],
        np.r_[times[:index], t_cut],
    )
    return {
        "auc": auc,
        "t_cut_h": t_cut,
        "biomass_at_cut": x_cut,
        "mu_initial": float(mu[0]),
        "mu_at_cut": threshold,
        "threshold": threshold,
        "crossed": True,
    }


def _compute_fixed_auc(
    binary_cr: pd.DataFrame,
    rate_cr: pd.DataFrame,
    y0: pd.Series,
    config: BtBaseConfig,
) -> pd.DataFrame:
    selected = list(COFACTOR_SPECIES)
    binary_sub = binary_cr.loc[selected].to_numpy(dtype=float)
    rate_sub = rate_cr.loc[selected].to_numpy(dtype=float)
    y0_values = y0.loc[binary_cr.columns].to_numpy(dtype=float)
    records: list[dict[str, object]] = []
    for focal_index, species in enumerate(selected):
        initial = np.zeros(len(selected), dtype=float)
        initial[focal_index] = config.auc_initial_abundance
        final_state = None
        final_times = None
        final_states = None
        for cycle in range(config.auc_cycles):
            final_state, times, states = _simulate_resource_cycle(
                rate_sub,
                initial,
                y0_values,
                config.auc_hours_per_cycle,
                config.auc_dt,
                store=(cycle == config.auc_cycles - 1),
            )
            initial = final_state[: len(selected)] / config.auc_dilution
            if times is not None:
                final_times, final_states = times, states
        assert final_state is not None and final_times is not None and final_states is not None
        biomass_curve = final_states[:, focal_index]
        resource_curve = final_states[:, len(selected):]
        result = _auc_until_mu_threshold(
            final_times,
            biomass_curve,
            resource_curve,
            binary_sub[focal_index],
            config.auc_mu_threshold,
        )
        result["species"] = species
        result["species_index"] = focal_index
        records.append(result)
    table = pd.DataFrame(records)
    if (table["auc"] <= 0).any() or not np.isfinite(table["auc"]).all():
        bad = table.loc[(table["auc"] <= 0) | ~np.isfinite(table["auc"]), "species"].tolist()
        raise ValueError(f"Cannot derive D because fixed AUC is invalid for: {bad}")
    return table


def _write_base_model(
    output_dir: Path,
    model: BtBaseModel,
    config: BtBaseConfig,
) -> None:
    del config
    output_dir.mkdir(parents=True, exist_ok=True)
    model.rate_cr.to_csv(output_dir / "nongrower_R.csv")
    model.y0.rename("bt_resource_y0").rename_axis(
        "resource_id"
    ).reset_index().to_csv(
        output_dir / "bt_resource_Y0.csv", index=False
    )
    model.auc.loc[:, ["species", "auc"]].to_csv(
        output_dir / "monoculture_AUC.csv", index=False
    )


def fit_bt_base(
    data_dir: Path,
    output_dir: Path,
    config: BtBaseConfig,
    overwrite: bool = False,
    binary_cr_path: Path | None = None,
) -> BtBaseModel:
    config.validate()
    _prepare_new_directory(output_dir, overwrite)
    binary_cr = _load_binary_cr(data_dir, binary_cr_path)
    double_spent = _load_double_spent(data_dir, config.floor)
    growth_table = _load_nongrower_growth_table(data_dir)
    y0, y0_before_tie_break, predictions, diagnostics = _fit_bt_resource(
        binary_cr, double_spent, config
    )
    rate_cr, scaling = _construct_rate_cr(binary_cr, y0, growth_table)
    auc = _compute_fixed_auc(binary_cr, rate_cr, y0, config)
    model = BtBaseModel(
        binary_cr=binary_cr,
        rate_cr=rate_cr,
        y0=y0,
        y0_before_tie_break=y0_before_tie_break,
        auc=auc,
        resource_fit_predictions=predictions,
        resource_fit_diagnostics=diagnostics,
        growth_rate_scaling=scaling,
    )
    _write_base_model(output_dir, model, config)
    print(f"Bt resource fit complete: {len(y0)} final resources.")
    return model


def _load_production_inputs(
    data_dir: Path,
    base: BtBaseModel,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, np.ndarray]:
    observed_path = data_dir / "Nongrowers_monoculture_in_grower_spent.csv"
    observed = pd.read_csv(observed_path).set_index("Species")
    missing_rows = [name for name in NONGROWER_SPECIES if name not in observed.index]
    missing_columns = [name for name in GROWER_SPECIES if name not in observed.columns]
    if missing_rows or missing_columns:
        raise ValueError(
            f"Incomplete production observations; rows={missing_rows}, columns={missing_columns}."
        )
    observed = observed.loc[list(NONGROWER_SPECIES), list(GROWER_SPECIES)].apply(
        pd.to_numeric, errors="coerce"
    )
    growth_path = data_dir / "Canonical_16_grower_dm68_mean_growth.csv"
    growth_table = pd.read_csv(growth_path).set_index("species")
    growth = pd.to_numeric(growth_table["mean_growth"], errors="coerce").loc[
        list(GROWER_SPECIES)
    ]
    if growth.isna().any() or (growth <= 0).any():
        raise ValueError("Grower monoculture growth contains invalid values.")
    bt_profile = base.y0.loc[base.binary_cr.columns].to_numpy(dtype=float) / float(
        growth.loc["Bt"]
    )
    return base.binary_cr, observed, growth, bt_profile


def _production_residual(
    predicted: np.ndarray,
    observed: np.ndarray,
    floor: float,
) -> np.ndarray:
    residual = np.full_like(np.asarray(observed, dtype=float), np.nan, dtype=float)
    valid = np.isfinite(observed) & np.isfinite(predicted)
    residual[valid] = np.log2(
        np.clip(predicted[valid], floor, None)
        / np.clip(observed[valid], floor, None)
    )
    return residual


def _production_upper_bound(
    observed: np.ndarray,
    grower_growth: float,
    reference: np.ndarray,
    floor: float,
) -> float:
    valid = np.isfinite(observed)
    observed_scale = (
        float(np.max(np.clip(observed[valid], floor, None))) / grower_growth
        if np.any(valid)
        else 0.0
    )
    reference_scale = float(np.max(np.clip(reference, 0.0, None)))
    return max(10.0 * max(observed_scale, reference_scale, floor), 1.0)


def _fit_one_production_profile(
    cr: np.ndarray,
    observed: np.ndarray,
    grower_growth: float,
    reference: np.ndarray,
    prior_weight: float,
    config: ProductionConfig,
    seed: int,
) -> ProductionFit:
    upper = _production_upper_bound(
        observed, grower_growth, reference, config.clip_min
    )
    # Resources absent from the Bt reference remain absent in this model
    # branch. Removing those coordinates also avoids the enormous numerical
    # gradient from log2((P + 1e-12) / 1e-12) at P=0.
    fixed_zero = reference <= 0.0
    fitted_resource = ~fixed_zero
    if not np.any(fitted_resource):
        raise ValueError("Bt production profile has no positive resources.")

    reference_fitted = reference[fitted_resource]
    lower_z = np.log2(config.profile_floor / reference_fitted)
    upper_z = np.log2(upper / reference_fitted)

    def profile_from_z(z: np.ndarray) -> np.ndarray:
        profile = np.zeros_like(reference, dtype=float)
        profile[fitted_resource] = reference_fitted * np.exp2(z)
        return profile

    def evaluate(z: np.ndarray) -> tuple[float, float, float, np.ndarray, np.ndarray]:
        profile = profile_from_z(z)
        predicted = grower_growth * (cr @ profile)
        residual = _production_residual(predicted, observed, config.clip_min)
        data_loss = float(np.sum(residual[np.isfinite(residual)] ** 2))
        # z is exactly log2(P / P_Bt) on fitted resources. Fixed-zero
        # resources contribute neither a parameter nor a prior term.
        prior_loss = float(np.sum(z**2))
        return (
            data_loss + prior_weight * prior_loss,
            data_loss,
            prior_loss,
            predicted,
            residual,
        )

    def residual_vector(z: np.ndarray) -> np.ndarray:
        _, _, _, _, data_residual = evaluate(z)
        pieces = [data_residual[np.isfinite(data_residual)]]
        if prior_weight > 0.0:
            pieces.append(np.sqrt(prior_weight) * z)
        return np.concatenate(pieces)

    starts = [np.zeros(reference_fitted.size, dtype=float)]
    valid = np.isfinite(observed)
    if np.any(valid):
        target = np.clip(observed[valid], config.clip_min, None) / grower_growth
        lsq_fitted, _ = nnls(cr[valid][:, fitted_resource], target)
        lsq_fitted = np.clip(lsq_fitted, config.profile_floor, upper)
        midpoint_fitted = np.clip(
            0.5 * (lsq_fitted + reference_fitted),
            config.profile_floor,
            upper,
        )
        starts.extend(
            [
                np.log2(lsq_fitted / reference_fitted),
                np.log2(midpoint_fitted / reference_fitted),
            ]
        )
    rng = np.random.default_rng(seed)
    base = starts[-1]
    while len(starts) < config.multi_start:
        starts.append(
            np.clip(
                base + rng.normal(0.0, 1.0, size=base.shape),
                lower_z,
                upper_z,
            )
        )

    best = None
    for start in starts[: config.multi_start]:
        result = least_squares(
            residual_vector,
            np.clip(start, lower_z, upper_z),
            bounds=(lower_z, upper_z),
            method="trf",
            max_nfev=config.maxfun,
        )
        result_objective = float(np.sum(result.fun**2))
        if best is None or result_objective < best[0]:
            best = (result_objective, result)
    if best is None:
        raise RuntimeError("No production optimizer run was attempted.")
    _, best_result = best
    objective, data_loss, prior_loss, predicted, residual = evaluate(best_result.x)
    return ProductionFit(
        profile=profile_from_z(best_result.x),
        predicted=predicted,
        residual=residual,
        objective=float(objective),
        data_loss=float(data_loss),
        prior_loss=float(prior_loss),
        success=bool(best_result.success),
        message=str(best_result.message),
        iterations=int(best_result.nfev),
    )


def _fixed_production_fit(
    profile: np.ndarray,
    cr: np.ndarray,
    observed: np.ndarray,
    growth: float,
    clip_min: float,
    message: str,
) -> ProductionFit:
    predicted = growth * (cr @ profile)
    residual = _production_residual(predicted, observed, clip_min)
    data_loss = float(np.sum(residual[np.isfinite(residual)] ** 2))
    return ProductionFit(
        profile=profile.copy(),
        predicted=predicted,
        residual=residual,
        objective=data_loss,
        data_loss=data_loss,
        prior_loss=0.0,
        success=True,
        message=message,
        iterations=0,
    )


def _write_production_branch(
    output_dir: Path,
    fits: dict[str, ProductionFit],
    resource_names: Sequence[str],
    nongrowers: Sequence[str],
    branch_config: dict[str, object],
) -> None:
    del nongrowers, branch_config
    profiles = pd.DataFrame(
        {grower: fits[grower].profile for grower in GROWER_SPECIES},
        index=resource_names,
    )
    profiles.index.name = "resource_id"
    profiles.to_csv(output_dir / "production_profiles.csv")


def fit_production_branches(
    data_dir: Path,
    base: BtBaseModel,
    output_root: Path,
    config: ProductionConfig,
    overwrite: bool = False,
) -> list[Path]:
    config.validate()
    binary_cr, observed, growth, bt_profile = _load_production_inputs(data_dir, base)
    cr_values = binary_cr.loc[list(NONGROWER_SPECIES)].to_numpy(dtype=float)
    resource_names = binary_cr.columns.tolist()
    written: list[Path] = []

    for prior_weight in config.prior_weights:
        branch_name = production_branch_name(prior_weight)
        branch_dir = output_root / branch_name
        _prepare_new_directory(branch_dir, overwrite)
        fits: dict[str, ProductionFit] = {}
        for grower_index, grower in enumerate(GROWER_SPECIES):
            observed_vector = observed[grower].to_numpy(dtype=float)
            if grower in {"Ai", "Bt"}:
                fits[grower] = _fixed_production_fit(
                    bt_profile,
                    cr_values,
                    observed_vector,
                    float(growth.loc[grower]),
                    config.clip_min,
                    "fixed to Bt production profile",
                )
            else:
                fits[grower] = _fit_one_production_profile(
                    cr_values,
                    observed_vector,
                    float(growth.loc[grower]),
                    bt_profile,
                    prior_weight,
                    config,
                    seed=config.seed + grower_index,
                )
        _write_production_branch(
            branch_dir,
            fits,
            resource_names,
            NONGROWER_SPECIES,
            {
                **asdict(config),
                "branch": "fitted",
                "prior_weight": prior_weight,
                "parameterization": "log2_ratio_to_bt_on_bt_positive_resources",
                "fixed_zero_resources": [
                    resource_names[index]
                    for index, value in enumerate(bt_profile)
                    if value <= 0.0
                ],
            },
        )
        written.append(branch_dir)
        failed = [
            grower
            for grower, fit in fits.items()
            if grower not in {"Ai", "Bt"} and not fit.success
        ]
        if failed:
            print(
                f"Production fit complete: {branch_name}; optimizer warning for "
                f"{', '.join(failed)} (see fit_diagnostics.csv)."
            )
        else:
            print(f"Production fit complete: {branch_name}; all fitted growers converged.")

    return written
