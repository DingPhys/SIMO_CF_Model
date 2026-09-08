"""Fit resources, rates, production, and cofactor parameters."""

import json
from pathlib import Path

import pandas as pd

from bt_fit_core import fit_bt_base, fit_production
from cofactor_fit_core import COFACTOR_SPECIES, fit_cofactors
from grower_fit_core import fit_grower_dm


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
DATA = ROOT / "data"
PARAMETERS = ROOT / "parameters"

NONGROWERS = (
    "Af", "Ao", "As", "Bl.s", "Col", "Cs", "Et", "Eu.c",
    "Eu.l", "Im", "Ld", "Lsp", "Mi", "Pc", "Va", "Vp",
)


def fit_model():
    settings = json.loads(
        (SCRIPT_DIR / "cofactor_config.json").read_text()
    )["cofactor_fit"]

    bt = fit_bt_base(
        DATA,
        DATA / "nongrower_binary_consumption_matrix.csv",
        PARAMETERS / "bt_base",
    )

    fit_grower_dm(
        DATA,
        DATA / "grower_binary_consumption_matrix.csv",
        PARAMETERS / "grower",
        overwrite=True,
    )

    fit_production(
        DATA,
        bt,
        PARAMETERS / "production_fits",
    )

    cofactor = fit_cofactors(
        DATA,
        PARAMETERS / "bt_base",
        DATA / "nongrower_binary_consumption_matrix.csv",
        PARAMETERS / "cofactor_fit",
        floor=settings["floor"],
        F0=settings["F0"],
        C0=settings["C0"],
        lambda_F=settings["lambda_F"],
        lambda_C=settings["lambda_C"],
        lambda_ineq=settings["lambda_ineq"],
        active_margin_log2=settings["active_margin_log2"],
        F_bounds=tuple(settings["F_bounds"]),
        C_bounds=tuple(settings["C_bounds"]),
    ).set_index("species")

    lag = pd.read_csv(DATA / "fixed_lag01.csv").set_index("species")["lag_h"]
    final = pd.DataFrame(index=NONGROWERS)
    final.index.name = "species"
    final["F"] = 0.0
    final["C"] = 1.0
    final["D"] = 0.0
    final.loc[list(COFACTOR_SPECIES), ["F", "C", "D"]] = cofactor[["F", "C", "D"]]
    final["selected_lag_h"] = lag.reindex(NONGROWERS)
    output = PARAMETERS / "final_parameters"
    output.mkdir(parents=True, exist_ok=True)
    final.reset_index().to_csv(output / "cofactor_and_lag.csv", index=False)


if __name__ == "__main__":
    fit_model()
