"""Fit F and C from nongrower double-spent observations."""

import shutil

import numpy as np
import pandas as pd
from scipy.optimize import minimize


COFACTOR_SPECIES = (
    "Col", "Cs", "Et", "Eu.c", "Eu.l", "Im", "Ld", "Lsp", "Mi", "Va",
)


def fit_cofactors(
    data_dir,
    bt_base_dir,
    binary_cr_path,
    output_dir,
    *,
    floor,
    F0,
    C0,
    lambda_F,
    lambda_C,
    lambda_ineq,
    active_margin_log2,
    F_bounds,
    C_bounds,
):
    species = list(COFACTOR_SPECIES)
    binary = (
        pd.read_csv(binary_cr_path)
        .set_index("species")
        .loc[species]
    )
    resource = (
        pd.read_csv(bt_base_dir / "bt_resource_Y0.csv")
        .set_index("resource_id")["bt_resource_y0"]
        .reindex(binary.columns)
    )
    observed_table = (
        pd.read_csv(data_dir / "Nongrowers_in_Bt_double_spent.csv")
        .set_index("Species")
        .loc[species, species + ["Full_Bt", "Diluted_Bt"]]
        .clip(lower=floor)
    )
    for name in species:
        observed_table.loc[name, name] = np.nan

    B = binary.to_numpy(dtype=float)
    observed = observed_table[species].to_numpy(dtype=float)
    baseline = np.einsum(
        "ijr,r->ij",
        B[:, None, :] * (1.0 - B[None, :, :]),
        resource.to_numpy(dtype=float),
    )
    valid = np.isfinite(observed)
    observed_log = np.log2(np.clip(observed, floor, None))
    baseline_log = np.log2(np.clip(baseline, floor, None))
    active = valid & (baseline_log > observed_log + active_margin_log2)
    inactive = valid & ~active

    n = len(species)
    a0, b0 = np.log2(F0), np.log2(C0)
    a_min = np.full(n, np.log2(F_bounds[0]))
    a_max = np.full(n, np.log2(F_bounds[1]))
    full = observed_table["Full_Bt"].to_numpy(dtype=float)
    diluted = observed_table["Diluted_Bt"].to_numpy(dtype=float)
    a_max = np.minimum(a_max, np.log2(1.0 / full))
    a_max = np.minimum(a_max, np.log2(0.1 / diluted))
    bounds = list(zip(a_min, a_max)) + [
        (np.log2(C_bounds[0]), np.log2(C_bounds[1]))
    ] * n

    def loss(theta):
        a, b = theta[:n], theta[n:]
        cap_log = b[None, :] - a[:, None]
        active_residual = cap_log[active] - observed_log[active]
        violation = np.maximum(0.0, baseline_log[inactive] - cap_log[inactive])
        return (
            np.mean(active_residual**2)
            + lambda_ineq * np.mean(violation**2)
            + lambda_F * np.mean((a - a0) ** 2)
            + lambda_C * np.mean((b - b0) ** 2)
        )

    start = np.r_[
        np.clip(np.full(n, a0), a_min, a_max),
        np.clip(np.full(n, b0), np.log2(C_bounds[0]), np.log2(C_bounds[1])),
    ]
    result = minimize(
        loss,
        start,
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxiter": 5000},
    )
    F = np.exp2(result.x[:n])
    C = np.exp2(result.x[n:])
    auc = (
        pd.read_csv(bt_base_dir / "monoculture_AUC.csv")
        .set_index("species")["auc"]
        .reindex(species)
        .to_numpy()
    )
    parameters = pd.DataFrame(
        {"species": species, "F": F, "C": C, "D": -np.log(C) / auc}
    )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    parameters.to_csv(output_dir / "cofactor_parameters.csv", index=False)
    return parameters
