"""Fit DM resources and grower consumption rates."""

import shutil

import numpy as np
import pandas as pd

from resource_fit_core import fit_resources


GROWERS = (
    "Ai", "Ac", "Bfi", "Bf", "Bt", "Bu", "Bx", "Ba",
    "Csp", "Dl", "Ls", "Mf", "Pm", "Bd", "Bv", "Rg",
)
FIT_GROWERS = tuple(species for species in GROWERS if species != "Ai")
FIT_FLOOR = 1e-3


def build_design(binary, monoculture, spent, species):
    """Monoculture: B_i; spent medium: B_i * (1 - B_j)."""

    support = binary.loc[list(species)].to_numpy(dtype=float)
    rows, observed = [], []
    for i, focal in enumerate(species):
        rows.append(support[i])
        observed.append(monoculture.loc[focal])
        for j, conditioner in enumerate(species):
            if i != j:
                rows.append(support[i] * (1.0 - support[j]))
                observed.append(spent.loc[focal, conditioner])
    return np.vstack(rows), np.asarray(observed, dtype=float)


def fit_grower_dm(data_dir, binary_path, output_dir, overwrite=True):
    binary = pd.read_csv(binary_path).set_index("species").loc[list(GROWERS)]
    monoculture = (
        pd.read_csv(data_dir / "16_grower_dm68_mean_growth.csv")
        .set_index("species")["mean_growth"]
        .clip(lower=FIT_FLOOR)
    )
    spent = (
        pd.read_csv(data_dir / "grower_spent_mean_growth.csv")
        .set_index("focal_species")
        .clip(lower=FIT_FLOOR)
    )

    resource_names = [name for name in binary.columns if name != "U_Ai"]
    A, y = build_design(binary[resource_names], monoculture, spent, FIT_GROWERS)
    ordinary_resources, _ = fit_resources(A, y, resource_names, eps=FIT_FLOOR)

    ai_y = np.asarray(
        [monoculture.loc["Ai"]]
        + [spent.loc["Ai", species] for species in GROWERS if species != "Ai"]
    )
    ai_resource, _ = fit_resources(
        np.ones((len(ai_y), 1)), ai_y, ["U_Ai"], eps=FIT_FLOOR
    )
    resources = pd.concat([ai_resource, ordinary_resources]).reindex(binary.columns)
    resources.name = "dm_y0"

    growth = (
        pd.read_csv(data_dir / "16_grower_dm68_curve_fit.csv")
        .set_index("focal_strain")["growth_rate"]
        .reindex(GROWERS)
    )
    target_growth = growth.mean()
    access = binary.to_numpy() @ resources.to_numpy()
    rate = binary * (target_growth / access)[:, None]
    rate.index.name = "species"

    model_dir = output_dir / "model"
    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    resources.rename_axis("resource_id").reset_index().to_csv(
        model_dir / "dm_resource_y0.csv", index=False
    )
    rate.to_csv(model_dir / "grower_R.csv")
    return {"binary": binary, "y0": resources, "rate": rate}
