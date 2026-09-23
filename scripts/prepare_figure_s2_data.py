"""Regenerate the FigureS2 source CSVs from the current parameters and
prediction results.

Writes into figure_plots/FigureS2/:
  - full_32_CR_matrix.csv   consumption matrix, 32 species x 36 resources
  - full_32_PR_matrix.csv   production matrix, 32 species x 36 resources
  - nongrower_bt_spent_assembly_predictions.csv
        wide table: one column per assembly, 16 species rows holding the
        predicted relative abundance for presented species (empty cell =
        not presented), plus a final mean_abs_log2_error row
  - bt_resource_Y0.csv / dm_resource_y0.csv (copied unchanged)

Run from the package root:  python scripts/prepare_figure_s2_data.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE_DIR = Path(__file__).resolve().parents[1]
PARAMETER_DIR = PACKAGE_DIR / "parameters"
RESULTS_DIR = PACKAGE_DIR / "prediction_results"
FIGURE_DIR = PACKAGE_DIR / "figure_plots" / "FigureS2"

PRODUCTION_BRANCH = "prod_prior-0p3"

NONGROWERS = (
    "Af", "Ao", "As", "Bl.s", "Col", "Cs", "Et", "Eu.c",
    "Eu.l", "Im", "Ld", "Lsp", "Mi", "Pc", "Va", "Vp",
)
GROWERS = (
    "Ai", "Ac", "Bfi", "Bf", "Bt", "Bu", "Bx", "Ba",
    "Csp", "Dl", "Ls", "Mf", "Pm", "Bd", "Bv", "Rg",
)

BT_RESOURCES = [f"Resource_{i}" for i in range(1, 18)]


def _species_matrix(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).set_index("species")


def _format(value: float) -> str:
    """Compact, readable float formatting (10 significant digits)."""
    return f"{value:.10g}"


def _write_matrix(matrix: pd.DataFrame, target: Path) -> None:
    matrix.to_csv(target, float_format="%.10g")


def _full_matrix_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Concatenate nongrower and grower consumption, plus grower production.

    Columns are the 19 DM resources (grower_R order, ending with
    F_All_Shared) followed by the 17 Bt-produced resources. Nongrowers
    consume Bt resources; their F_All_Shared entry mirrors Resource_17,
    the shared resource every nongrower consumes. Growers consume DM
    resources and produce Bt resources.
    """
    nongrower_rate = _species_matrix(
        PARAMETER_DIR / "bt_base" / "nongrower_R.csv"
    ).reindex(NONGROWERS)
    grower_rate = _species_matrix(
        PARAMETER_DIR / "grower" / "model" / "grower_R.csv"
    ).reindex(GROWERS)
    production = pd.read_csv(
        PARAMETER_DIR
        / "production_fits"
        / PRODUCTION_BRANCH
        / "production_profiles.csv",
        index_col="resource_id",
    ).reindex(index=BT_RESOURCES, columns=GROWERS)

    dm_resources = grower_rate.columns.tolist()
    columns = dm_resources + BT_RESOURCES
    species = list(NONGROWERS) + list(GROWERS)

    consumption = pd.DataFrame(0.0, index=species, columns=columns)
    production_matrix = pd.DataFrame(0.0, index=species, columns=columns)
    consumption.loc[list(NONGROWERS), BT_RESOURCES] = nongrower_rate[
        BT_RESOURCES
    ]
    consumption.loc[list(NONGROWERS), "F_All_Shared"] = nongrower_rate[
        "Resource_17"
    ]
    consumption.loc[list(GROWERS), dm_resources] = grower_rate[dm_resources]
    production_matrix.loc[list(GROWERS), BT_RESOURCES] = (
        production.loc[BT_RESOURCES, list(GROWERS)].T
    )
    consumption.index.name = "species"
    production_matrix.index.name = "species"
    return consumption, production_matrix


def _prepare_s2_assembly_source(target: Path) -> None:
    """Wide per-assembly table for the nongrower prediction matrix figure."""
    result_dir = RESULTS_DIR / "nongrowers_in_bt_spent"
    metrics = pd.read_csv(result_dir / "cofactor_on" / "community_metrics.csv")
    predicted = _species_matrix(
        result_dir / "cofactor_on" / "predicted_relative.csv"
    ).reindex(NONGROWERS)
    observed = _species_matrix(
        result_dir / "observed_relative.csv"
    ).reindex(NONGROWERS)

    assemblies = metrics["community"].tolist()
    if not (
        (predicted.columns == assemblies).all()
        and (observed.columns == assemblies).all()
    ):
        raise ValueError("Assembly columns are not aligned with community_metrics.")

    presented = observed.notna()
    counts = presented.sum(axis=0).to_numpy()
    if not (counts == metrics["n_presented"].to_numpy()).all():
        raise ValueError("Presented-species counts do not match n_presented.")

    lines = [",".join(["species", *assemblies])]
    for species in NONGROWERS:
        cells = [
            _format(predicted.at[species, assembly])
            if presented.at[species, assembly]
            else ""
            for assembly in assemblies
        ]
        lines.append(",".join([species, *cells]))
    lines.append(
        ",".join(
            ["mean_abs_log2_error"]
            + [_format(v) for v in metrics["mean_abs_log2_error"]]
        )
    )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    consumption, production = _full_matrix_tables()
    _write_matrix(consumption, FIGURE_DIR / "full_32_CR_matrix.csv")
    _write_matrix(production, FIGURE_DIR / "full_32_PR_matrix.csv")
    shutil.copyfile(
        PARAMETER_DIR / "bt_base" / "bt_resource_Y0.csv",
        FIGURE_DIR / "bt_resource_Y0.csv",
    )
    shutil.copyfile(
        PARAMETER_DIR / "grower" / "model" / "dm_resource_y0.csv",
        FIGURE_DIR / "dm_resource_y0.csv",
    )
    _prepare_s2_assembly_source(
        FIGURE_DIR / "nongrower_bt_spent_assembly_predictions.csv"
    )
    print("FigureS2 source CSVs refreshed in", FIGURE_DIR)


if __name__ == "__main__":
    main()
