"""Regenerate the Figure4 source CSVs from the current data, parameters,
and prediction results.

Most files are plain copies; only ld_panel_plot_source.csv is computed
(Ld observed values plus resource-only / resource+cofactor predictions).

Run from the package root:  python scripts/prepare_figure4_data.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE_DIR = Path(__file__).resolve().parents[1]
FIGURE_DIR = PACKAGE_DIR / "figure_plots" / "Figure4"

COFACTOR_SPECIES = (
    "Col", "Cs", "Et", "Eu.c", "Eu.l", "Im", "Ld", "Lsp", "Mi", "Va",
)

# target file name -> source path, relative to the package root
COPIES = {
    "bt_resource_Y0.csv": "parameters/bt_base/bt_resource_Y0.csv",
    "cofactor_and_lag.csv": "parameters/final_parameters/cofactor_and_lag.csv",
    "nongrower_binary_CR.csv": "data/nongrower_binary_consumption_matrix.csv",
    "double_spent_growth.csv": "data/Nongrowers_in_Bt_double_spent.csv",
    "bt_cofactor_on_community_metrics.csv":
        "prediction_results/nongrowers_in_bt_spent/cofactor_on/community_metrics.csv",
    "bt_cofactor_off_community_metrics.csv":
        "prediction_results/nongrowers_in_bt_spent/cofactor_off/community_metrics.csv",
    "bt_observed_relative.csv":
        "prediction_results/nongrowers_in_bt_spent/observed_relative.csv",
    "bt_cofactor_on_relative.csv":
        "prediction_results/nongrowers_in_bt_spent/cofactor_on/predicted_relative.csv",
    "dm_cofactor_on_community_metrics.csv":
        "prediction_results/dm_communities/cofactor_on/community_metrics.csv",
    "dm_cofactor_off_community_metrics.csv":
        "prediction_results/dm_communities/cofactor_off/community_metrics.csv",
    "dm_observed_relative.csv":
        "prediction_results/dm_communities/observed_relative.csv",
    "dm_cofactor_on_relative.csv":
        "prediction_results/dm_communities/cofactor_on/predicted_relative.csv",
    "carbon_cofactor_on_metrics.csv":
        "prediction_results/carbon_sources/cofactor_on/carbon_source_metrics.csv",
    "carbon_cofactor_off_metrics.csv":
        "prediction_results/carbon_sources/cofactor_off/carbon_source_metrics.csv",
}


def _prepare_ld_panel(target: Path) -> None:
    """Ld in double-spent media (yield) and in assemblies containing Ld
    (relative abundance), observed and predicted by both model variants."""
    result_dir = PACKAGE_DIR / "prediction_results" / "nongrowers_in_bt_spent"
    metrics = pd.read_csv(result_dir / "cofactor_on" / "community_metrics.csv")
    observed = pd.read_csv(result_dir / "observed_relative.csv").set_index("species")
    pred_off = pd.read_csv(
        result_dir / "cofactor_off" / "predicted_relative.csv"
    ).set_index("species")
    pred_on = pd.read_csv(
        result_dir / "cofactor_on" / "predicted_relative.csv"
    ).set_index("species")
    binary = pd.read_csv(
        PACKAGE_DIR / "data" / "nongrower_binary_consumption_matrix.csv"
    ).set_index("species")
    y0 = pd.read_csv(
        PACKAGE_DIR / "parameters" / "bt_base" / "bt_resource_Y0.csv"
    ).set_index("resource_id")["bt_resource_y0"]
    parameters = pd.read_csv(
        PACKAGE_DIR / "parameters" / "final_parameters" / "cofactor_and_lag.csv"
    ).set_index("species")
    double_spent = pd.read_csv(
        PACKAGE_DIR / "data" / "Nongrowers_in_Bt_double_spent.csv"
    ).set_index("Species")

    ld_binary = binary.loc["Ld"].to_numpy(dtype=float)
    rows = []

    # Ld yield in each cofactor species' double-spent medium. Resource-only
    # prediction: resources Ld can use that the donor does not consume.
    # Cofactor prediction: capped by the donor's residual cofactor.
    for donor in (s for s in COFACTOR_SPECIES if s != "Ld"):
        donor_binary = binary.loc[donor].to_numpy(dtype=float)
        resource_prediction = max(
            float(np.sum(ld_binary * (1.0 - donor_binary) * y0.to_numpy())),
            1e-3,
        )
        rows.append({
            "data_group": "double_spent",
            "condition": f"Ld_in_{donor}_double_spent",
            "community_size": "",
            "metric": "yield",
            "actual_value": double_spent.at["Ld", donor],
            "resource_only_prediction": resource_prediction,
            "cofactor_prediction": min(
                resource_prediction,
                parameters.at[donor, "C"] / parameters.at["Ld", "F"],
            ),
        })

    # Ld relative abundance in every assembly that contains Ld.
    for _, metric in metrics.iterrows():
        community = str(metric["community"])
        if pd.isna(observed.at["Ld", community]):
            continue
        rows.append({
            "data_group": "pair" if metric["community_type"] == "pairwise"
                          else "larger",
            "condition": community,
            "community_size": metric["n_presented"],
            "metric": "relative_abundance",
            "actual_value": observed.at["Ld", community],
            "resource_only_prediction": pred_off.at["Ld", community],
            "cofactor_prediction": pred_on.at["Ld", community],
        })

    pd.DataFrame(rows).to_csv(target, index=False)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for name, source in COPIES.items():
        shutil.copyfile(PACKAGE_DIR / source, FIGURE_DIR / name)
    pd.DataFrame({"species": COFACTOR_SPECIES}).to_csv(
        FIGURE_DIR / "cofactor_species_order.csv", index=False
    )
    _prepare_ld_panel(FIGURE_DIR / "ld_panel_plot_source.csv")
    print("Figure4 source CSVs refreshed in", FIGURE_DIR)


if __name__ == "__main__":
    main()
