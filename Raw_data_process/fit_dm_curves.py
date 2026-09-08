"""DM68: fit 16 growers from the raw Excel workbook and save mean growth rates and lags.

Run this script directly. Requires numpy, scipy, pandas, and openpyxl; no previous fit results are read.
Fit replicates from both batches separately, then average with equal weight; include only final OD - initial OD >= 0.01.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "Growth_data/Growth_curve_32_strains_monoculture_DM.xlsx"
OUTPUT = ROOT.parent / "data/16_grower_dm68_curve_fit.csv"
SPECIES = ["Ac", "Ai", "Ba", "Bd", "Bf", "Bfi", "Bt", "Bu", "Bv", "Bx", "Csp", "Dl", "Ls", "Mf", "Pm", "Rg"]
X0_MIN, X0_MAX = 0.001, 0.005
SEEDS = (17, 29, 41, 59, 71)


def read_curve(row):
    od = np.fromstring(row["Growth curve"], sep=" ")
    # Keep the existing time convention: distribute the actual points evenly between the recorded start and end.
    start, _, end = map(float, row["Growth curve time"].replace(" ", "").split(":"))
    time = np.linspace(start, end, len(od))
    assert len(od) >= 8 and np.isfinite(od).all() and end > start
    return time, od


def fit_full_curve(time, od):
    net = od - od.min()
    k = net[-1]  # Fix the asymptotic plateau; do not force the fitted curve through the 48 h endpoint.

    def loss(parameters):
        x0, rate = np.exp(parameters[:2])
        lag = parameters[2]
        predicted = k / (1 + (k / x0 - 1) * np.exp(-rate * np.maximum(time - lag, 0)))
        return np.mean(np.log2(np.maximum(predicted, .001) / np.maximum(net, .001)) ** 2)

    bounds = [(np.log(X0_MIN), np.log(X0_MAX)), (np.log(1e-5), np.log(50)), (0, time[-1])]
    candidates = []
    for seed in SEEDS:
        global_fit = differential_evolution(loss, bounds, seed=seed, popsize=25,
            maxiter=1500, tol=1e-9, atol=1e-12, polish=False)
        local_fit = minimize(loss, global_fit.x, method="Nelder-Mead", bounds=bounds,
            options={"maxiter": 3000, "xatol": 1e-10, "fatol": 1e-13})
        candidates.extend([global_fit, local_fit])
    parameters = min(candidates, key=lambda fit: fit.fun).x
    return float(np.exp(parameters[1])), float(parameters[2])


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
