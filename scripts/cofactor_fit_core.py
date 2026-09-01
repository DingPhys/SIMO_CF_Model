"""Fit continuous cofactor parameters from Bt double-spent observations."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


COFACTOR_SPECIES = (
    "Col", "Cs", "Et", "Eu.c", "Eu.l",
    "Im", "Ld", "Lsp", "Mi", "Va",
)


def _load_inputs(
    data_dir: Path,
    bt_base_dir: Path,
    binary_cr_path: Path,
    floor: float,
) -> dict[str, object]:
    species = list(COFACTOR_SPECIES)
    binary_cr = (
        pd.read_csv(binary_cr_path)
        .set_index("species")
        .loc[species]
    )
    resource_y0 = (
        pd.read_csv(bt_base_dir / "bt_resource_Y0.csv")
        .set_index("resource_id")["bt_resource_y0"]
        .reindex(binary_cr.columns)
    )
    double_spent = (
        pd.read_csv(
            data_dir
            / "Nongrowers_in_Bt_double_spent.csv"
        )
        .set_index("Species")
        .loc[species, species + ["Full_Bt", "Diluted_Bt"]]
        .apply(pd.to_numeric, errors="coerce")
        .clip(lower=floor)
    )
    for name in species:
        double_spent.loc[name, name] = np.nan
    auc = (
        pd.read_csv(bt_base_dir / "monoculture_AUC.csv")
        .set_index("species")["auc"]
        .reindex(species)
    )
    if (
        binary_cr.isna().any().any()
        or resource_y0.isna().any()
        or auc.isna().any()
    ):
        raise ValueError("Cofactor fitting inputs are incomplete.")
    return {
        "species": species,
        "binary_cr": binary_cr,
        "resource_y0": resource_y0,
        "double_spent": double_spent,
        "auc": auc,
    }


def fit_cofactors(
    data_dir: Path,
    bt_base_dir: Path,
    binary_cr_path: Path,
    output_dir: Path,
    *,
    floor: float,
    F0: float,
    C0: float,
    lambda_F: float,
    lambda_C: float,
    lambda_ineq: float,
    active_margin_log2: float,
    F_bounds: tuple[float, float],
    C_bounds: tuple[float, float],
) -> pd.DataFrame:
    """Fit F and C jointly, then derive D from the fixed monoculture AUC."""

    if floor <= 0 or F0 <= 0 or C0 <= 0:
        raise ValueError("floor, F0, and C0 must be positive.")
    if min(lambda_F, lambda_C, lambda_ineq) < 0:
        raise ValueError("Cofactor regularization weights must be nonnegative.")
    if active_margin_log2 < 0:
        raise ValueError("active_margin_log2 must be nonnegative.")

    inputs = _load_inputs(data_dir, bt_base_dir, binary_cr_path, floor)
    species = inputs["species"]
    binary_cr = inputs["binary_cr"]
    resource_y0 = inputs["resource_y0"]
    double_spent = inputs["double_spent"]
    auc = inputs["auc"]

    binary = binary_cr.to_numpy(dtype=float)
    y0 = resource_y0.to_numpy(dtype=float)
    observed = double_spent.loc[species, species].to_numpy(dtype=float)
    baseline = np.einsum(
        "ijr,r->ij",
        binary[:, None, :] * (1.0 - binary[None, :, :]),
        y0,
    )
    valid = np.isfinite(observed)
    observed_log = np.full_like(observed, np.nan)
    baseline_log = np.full_like(baseline, np.nan)
    observed_log[valid] = np.log2(np.clip(observed[valid], floor, None))
    baseline_log[valid] = np.log2(np.clip(baseline[valid], floor, None))
    active = valid & (baseline_log > observed_log + active_margin_log2)
    inactive = valid & ~active
    if not active.any() or not inactive.any():
        raise ValueError("Cofactor fit requires both active and inactive pairs.")

    n_species = len(species)
    a0 = np.log2(F0)
    b0 = np.log2(C0)
    a_lower = np.full(n_species, np.log2(F_bounds[0]))
    a_upper = np.full(n_species, np.log2(F_bounds[1]))
    full_bt = double_spent["Full_Bt"].to_numpy(dtype=float)
    diluted_bt = double_spent["Diluted_Bt"].to_numpy(dtype=float)
    valid_full = np.isfinite(full_bt) & (full_bt > floor)
    valid_diluted = np.isfinite(diluted_bt) & (diluted_bt > floor)
    a_upper[valid_full] = np.minimum(
        a_upper[valid_full], np.log2(1.0 / full_bt[valid_full])
    )
    a_upper[valid_diluted] = np.minimum(
        a_upper[valid_diluted], np.log2(0.1 / diluted_bt[valid_diluted])
    )
    bounds = list(zip(a_lower, a_upper)) + [
        (np.log2(C_bounds[0]), np.log2(C_bounds[1]))
    ] * n_species

    def objective_and_gradient(theta: np.ndarray) -> tuple[float, np.ndarray]:
        a = theta[:n_species]
        b = theta[n_species:]
        cap_log = b[None, :] - a[:, None]
        gradient_a = np.zeros(n_species)
        gradient_b = np.zeros(n_species)

        active_residual = cap_log[active] - observed_log[active]
        loss = float(np.mean(active_residual**2))
        active_i, active_j = np.nonzero(active)
        active_scale = 2.0 / active.sum()
        np.add.at(gradient_a, active_i, -active_scale * active_residual)
        np.add.at(gradient_b, active_j, active_scale * active_residual)

        violation = baseline_log[inactive] - cap_log[inactive]
        positive_violation = np.maximum(0.0, violation)
        loss += lambda_ineq * float(np.mean(positive_violation**2))
        inactive_i, inactive_j = np.nonzero(inactive)
        inactive_scale = 2.0 * lambda_ineq / inactive.sum()
        np.add.at(gradient_a, inactive_i, inactive_scale * positive_violation)
        np.add.at(gradient_b, inactive_j, -inactive_scale * positive_violation)

        loss += lambda_F * float(np.mean((a - a0) ** 2))
        loss += lambda_C * float(np.mean((b - b0) ** 2))
        gradient_a += 2.0 * lambda_F * (a - a0) / n_species
        gradient_b += 2.0 * lambda_C * (b - b0) / n_species
        return loss, np.concatenate((gradient_a, gradient_b))

    start = np.concatenate(
        (
            np.clip(np.full(n_species, a0), a_lower, a_upper),
            np.clip(
                np.full(n_species, b0),
                np.log2(C_bounds[0]),
                np.log2(C_bounds[1]),
            ),
        )
    )
    result = minimize(
        lambda theta: objective_and_gradient(theta)[0],
        start,
        jac=lambda theta: objective_and_gradient(theta)[1],
        bounds=bounds,
        method="L-BFGS-B",
        options={
            "maxiter": 5000,
            "maxfun": 200000,
            "maxls": 50,
            "ftol": 1e-15,
            "gtol": 1e-12,
        },
    )
    if not result.success:
        raise RuntimeError(f"Cofactor fit failed: {result.message}")

    F = np.exp2(result.x[:n_species])
    C = np.exp2(result.x[n_species:])
    D = -np.log(C) / auc.to_numpy(dtype=float)
    prediction = np.minimum(baseline, C[None, :] / F[:, None])
    error = np.log2(
        np.clip(prediction[valid], floor, None)
        / np.clip(observed[valid], floor, None)
    )
    parameters = pd.DataFrame(
        {
            "species": species,
            "F": F,
            "C": C,
            "D": D,
        }
    )
    summary = pd.DataFrame(
        [
            {
                "objective": float(result.fun),
                "n_active_pairs": int(active.sum()),
                "n_inactive_pairs": int(inactive.sum()),
                "mean_abs_log2_error": float(np.mean(np.abs(error))),
                "rmse_log2_error": float(np.sqrt(np.mean(error**2))),
            }
        ]
    )
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parameters.to_csv(output_dir / "cofactor_parameters.csv", index=False)
    print(
        "Cofactor fit complete: "
        f"objective={result.fun:.10g}; "
        f"mean |log2 error|={summary.iloc[0]['mean_abs_log2_error']:.6f}"
    )
    return parameters
