"""DM68: fit 16 growers from the raw Excel workbook and save mean growth rates and lags.

Run this script directly. Requires numpy, scipy, pandas, and openpyxl; no previous fit results are read.
Fit replicates from both batches separately, then average with equal weight; include only final OD - initial OD >= 0.01.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from growth_fit_common import fit_full_curve

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "Growth_data/Growth_curve_32_strains_monoculture_DM.xlsx"
OUTPUT = ROOT.parent / "data/16_grower_dm68_curve_fit.csv"
SPECIES = ["Ac", "Ai", "Ba", "Bd", "Bf", "Bfi", "Bt", "Bu", "Bv", "Bx", "Csp", "Dl", "Ls", "Mf", "Pm", "Rg"]


def read_curve(row):
    od = np.fromstring(row["Growth curve"], sep=" ")
    # Keep the existing time convention: distribute the actual points evenly between the recorded start and end.
    start, _, end = map(float, row["Growth curve time"].replace(" ", "").split(":"))
    time = np.linspace(start, end, len(od))
    assert len(od) >= 8 and np.isfinite(od).all() and end > start
    return time, od


def main():
    data = pd.read_excel(INPUT)
    data = data[(data["Media"] == "DM68") & data["Species"].isin(SPECIES)]
    assert len(data) == 96
    rows = []
    for species in SPECIES:
        rates, lags = [], []
        for _, row in data[data["Species"] == species].iterrows():
            if pd.notna(row["Note"]):
                continue
            time, od = read_curve(row)
            if od[-1] - od[0] < .01:
                continue
            rate, lag = fit_full_curve(time, od)
            rates.append(rate)
            lags.append(lag)
        rows.append({"focal_strain": species,
            "growth_rate": np.mean(rates) if rates else np.nan,
            "lag_time": np.mean(lags) if lags else np.nan})
        print(f"{species}: {len(rates)} 条曲线", flush=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT, index=False, float_format="%.8f", na_rep="")
    print(OUTPUT)


if __name__ == "__main__":
    main()
