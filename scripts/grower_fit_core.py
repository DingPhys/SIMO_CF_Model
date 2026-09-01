"""Fit DM resource abundances with a fixed grower consumption matrix."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize, nnls


FIT_SPECIES = (
    "Ac", "Ba", "Bd", "Bf", "Bfi", "Bt", "Bu", "Bv", "Bx",
    "Csp", "Dl", "Ls", "Mf", "Pm", "Rg",
)

FINAL_SPECIES = (
    "Ai", "Ac", "Bfi", "Bf", "Bt", "Bu", "Bx", "Ba",
    "Csp", "Dl", "Ls", "Mf", "Pm", "Bd", "Bv", "Rg",
)


@dataclass
class GrowerModel:
    binary_support: pd.DataFrame
    dm_y0: pd.Series
    rate_matrix: pd.DataFrame
    mean_growth_rate: float


def _load_binary_support(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path).set_index("species")
    missing = [species for species in FINAL_SPECIES if species not in table.index]
    if missing:
        raise ValueError(f"Grower binary matrix is missing species: {missing}")
    table = table.loc[list(FINAL_SPECIES)].apply(pd.to_numeric, errors="raise")
    if not np.isin(table.to_numpy(), [0.0, 1.0]).all():
        raise ValueError("Grower binary matrix must contain only 0 and 1.")
    if "U_Ai" not in table.columns:
        raise ValueError("Grower binary matrix must contain U_Ai.")
    non_ai_access = table.loc[list(FIT_SPECIES)].drop(columns="U_Ai")
    if (non_ai_access.sum(axis=1) <= 0).any():
        bad = non_ai_access.index[non_ai_access.sum(axis=1) <= 0].tolist()
        raise ValueError(f"Growers without any fitted DM resource: {bad}")
    return table


def _load_observations(
    data_dir: Path,
    floor: float,
) -> tuple[pd.Series, pd.DataFrame]:
    mono_path = data_dir / "Canonical_16_grower_dm68_mean_growth.csv"
    monoculture = (
        pd.read_csv(mono_path)
        .set_index("species")["mean_growth"]
        .apply(pd.to_numeric, errors="coerce")
        .reindex(FINAL_SPECIES)
        .clip(lower=floor)
    )
    spent_path = data_dir / "grower_spent_final_delta_od_matrix.csv"
    spent = (
        pd.read_csv(spent_path)
        .set_index("focal_species")
        .reindex(index=FINAL_SPECIES, columns=FINAL_SPECIES)
        .apply(pd.to_numeric, errors="coerce")
        .clip(lower=floor)
    )
    if monoculture.isna().any():
        raise ValueError("Grower monoculture table is incomplete.")
    off_diagonal = ~np.eye(len(FINAL_SPECIES), dtype=bool)
    if np.isnan(spent.to_numpy()[off_diagonal]).any():
        raise ValueError("Grower spent-medium table is incomplete.")
    return monoculture, spent


def _build_design(
    support: pd.DataFrame,
    species: tuple[str, ...],
    monoculture: pd.Series,
    spent: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    values = support.loc[list(species)].to_numpy(dtype=float)
    rows: list[np.ndarray] = []
    observed: list[float] = []
    labels: list[dict[str, object]] = []
    for focal_index, focal in enumerate(species):
        rows.append(values[focal_index].copy())
        observed.append(float(monoculture.loc[focal]))
        labels.append(
            {
                "observation_type": "monoculture",
                "focal_species": focal,
                "conditioner_species": None,
            }
        )
    for focal_index, focal in enumerate(species):
        for conditioner_index, conditioner in enumerate(species):
            if focal_index == conditioner_index:
                continue
            rows.append(
                values[focal_index] * (1.0 - values[conditioner_index])
            )
            observed.append(float(spent.loc[focal, conditioner]))
            labels.append(
                {
                    "observation_type": "spent",
                    "focal_species": focal,
                    "conditioner_species": conditioner,
                }
            )
    return np.vstack(rows), np.asarray(observed), pd.DataFrame(labels)


def _log2_error(
    predicted: np.ndarray,
    observed: np.ndarray,
    floor: float,
) -> np.ndarray:
    return np.log2(
        np.clip(np.asarray(predicted), floor, None)
        / np.clip(np.asarray(observed), floor, None)
    )


def _fit_non_ai_resources(
    binary: pd.DataFrame,
    monoculture: pd.Series,
    spent: pd.DataFrame,
    floor: float,
) -> tuple[pd.Series, object]:
    resources = [name for name in binary.columns if name != "U_Ai"]
    design, observed, _ = _build_design(
        binary[resources], FIT_SPECIES, monoculture, spent
    )
    start, _ = nnls(design, np.clip(observed, floor, None))
    start = np.maximum(start, 1e-9)

    def objective_and_gradient(y0: np.ndarray) -> tuple[float, np.ndarray]:
        y0 = np.clip(y0, 0.0, None)
        predicted = design @ y0
        predicted_safe = np.clip(predicted, floor, None)
        observed_safe = np.clip(observed, floor, None)
        residual = np.log2(predicted_safe / observed_safe)
        value = 0.5 * float(np.dot(residual, residual))
        coefficient = np.zeros_like(predicted)
        active = predicted > floor
        coefficient[active] = residual[active] / (
            np.log(2.0) * predicted_safe[active]
        )
        return value, design.T @ coefficient

    result = minimize(
        lambda y0: objective_and_gradient(y0)[0],
        start,
        jac=lambda y0: objective_and_gradient(y0)[1],
        bounds=[(0.0, None)] * len(resources),
        method="L-BFGS-B",
        options={
            "maxiter": 10000,
            "maxfun": 200000,
            "maxls": 50,
            "ftol": 1e-15,
            "gtol": 1e-12,
        },
    )
    if not result.success:
        raise RuntimeError(f"Fixed grower-resource fit failed: {result.message}")
    return pd.Series(result.x, index=resources, name="dm_y0"), result


def _fit_ai_resource(
    monoculture: pd.Series,
    spent: pd.DataFrame,
    floor: float,
) -> tuple[float, object]:
    observed = np.asarray(
        [float(monoculture.loc["Ai"])]
        + [
            float(spent.loc["Ai", conditioner])
            for conditioner in FINAL_SPECIES
            if conditioner != "Ai"
        ]
    )
    result = least_squares(
        lambda theta: _log2_error(
            np.full(observed.shape, np.exp(theta[0])), observed, floor
        ),
        x0=np.zeros(1),
        method="trf",
        max_nfev=100000,
    )
    if not result.success:
        raise RuntimeError(f"Ai-exclusive resource fit failed: {result.message}")
    return float(np.exp(result.x[0])), result


def _growth_rate_scaling(
    data_dir: Path,
    binary: pd.DataFrame,
    y0: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    path = data_dir / "canonical_16_grower_dm68_atlas_scheme_fit_summary.csv"
    measured = (
        pd.read_csv(path)
        .set_index("focal_strain")["lambda_h_mean"]
        .apply(pd.to_numeric, errors="coerce")
        .reindex(FINAL_SPECIES)
    )
    if measured.isna().any():
        raise ValueError("Grower growth-rate table is incomplete.")
    target = float(measured.mean())
    binary_values = binary.to_numpy(dtype=float)
    y0_values = y0.reindex(binary.columns).to_numpy(dtype=float)
    fresh_access = binary_values @ y0_values
    if np.any(fresh_access <= 0):
        bad = np.asarray(FINAL_SPECIES)[fresh_access <= 0].tolist()
        raise ValueError(f"Growers with zero fresh-resource access: {bad}")
    row_scale = target / fresh_access
    rate = pd.DataFrame(
        binary_values * row_scale[:, None],
        index=binary.index,
        columns=binary.columns,
    )
    rate.index.name = "species"
    scaling = pd.DataFrame(
        {
            "species": FINAL_SPECIES,
            "measured_growth_rate": measured.to_numpy(),
            "target_growth_rate": rate.to_numpy() @ y0_values,
        }
    )
    return rate, scaling, target


def fit_grower_dm(
    data_dir: Path,
    binary_path: Path,
    output_dir: Path,
    floor: float,
    overwrite: bool = False,
) -> GrowerModel:
    """Fit continuous DM resource abundances for the supplied fixed matrix."""

    if floor <= 0:
        raise ValueError("floor must be positive.")
    model_dir = output_dir / "model"
    target_files = (
        model_dir / "dm_resource_y0.csv",
        model_dir / "grower_R.csv",
    )
    if not overwrite and any(path.exists() for path in target_files):
        raise FileExistsError(
            f"Grower fit outputs already exist in {model_dir}; use --overwrite."
        )
    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    binary = _load_binary_support(binary_path)
    monoculture, spent = _load_observations(data_dir, floor)
    non_ai_y0, optimizer = _fit_non_ai_resources(
        binary, monoculture, spent, floor
    )
    u_ai, _ = _fit_ai_resource(monoculture, spent, floor)
    y0 = pd.concat(
        [pd.Series({"U_Ai": u_ai}, name="dm_y0"), non_ai_y0]
    ).reindex(binary.columns)

    design, observed, labels = _build_design(
        binary, FINAL_SPECIES, monoculture, spent
    )
    prediction = design @ y0.to_numpy(dtype=float)
    labels["observed"] = observed
    labels["predicted"] = prediction
    labels["log2_error"] = _log2_error(prediction, observed, floor)
    labels["fit_role"] = np.where(
        labels["focal_species"].eq("Ai"),
        "fit_u_ai",
        np.where(
            labels["conditioner_species"].eq("Ai"),
            "validation_only_ai_conditioner",
            "fixed_support_fit",
        ),
    )
    rate, scaling, target = _growth_rate_scaling(data_dir, binary, y0)

    y0.rename_axis("resource_id").reset_index().to_csv(
        model_dir / "dm_resource_y0.csv", index=False
    )
    rate.to_csv(model_dir / "grower_R.csv")
    print(
        "Fixed grower-matrix fit complete: "
        f"{len(y0)} resources; objective={optimizer.fun:.10g}; "
        f"growth-rate target={target:.8f} h^-1"
    )
    return GrowerModel(binary, y0, rate, target)
