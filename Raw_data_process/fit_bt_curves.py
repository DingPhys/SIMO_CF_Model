"""100% Bt spent medium: save the mean growth rate and lag for each nongrower.

Run this script directly. Requires numpy, scipy, pandas, and openpyxl; no previous fit results are read.
For species other than Im, average only replicates with final OD - initial OD >= 0.01.
For Im, use only 20250910 B3: fit lag from the full curve and rate from the tail starting at 20% of the final net OD.
"""
from pathlib import Path
from datetime import time as ExcelTime
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.optimize import differential_evolution, minimize
from scipy.special import expit

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "Growth_data/Non_growers_spent_medium_of_growers.xlsx"
OUTPUT = ROOT.parent / "data/Non-growers_In_Bt_Spent_growth_curve_fit_summary.csv"
SPECIES = ["Af", "Ao", "As", "Bl.s", "Col", "Cs", "Et", "Eu.c", "Eu.l", "Im", "Ld", "Lsp", "Mi", "Pc", "Va", "Vp"]
X0_MIN, X0_MAX = 0.001, 0.005
SEEDS = (17, 29, 41, 59, 71)


def read_curve(row):
    od = np.fromstring(row["Growth curve"], sep=" ")
    # Keep the existing time convention: distribute the actual points evenly between the recorded start and end.
    value = row["Growth curve time"]
    if isinstance(value, ExcelTime):
        # Excel converted the range text to a clock time; use the agreed 0-48 h protocol.
        start, end = 0.0, 48.0
        print(f"{row['Growth Well ID']}: Excel clock-time cell; using the 0-48 h protocol", flush=True)
    else:
        start, _, end = map(float, str(value).replace(" ", "").split(":"))
    time = np.linspace(start, end, len(od))
    assert len(od) >= 8 and np.isfinite(od).all() and end > start
    return time, od


def minimize_loss(loss, bounds):
    """Search the expanded bounds with five fixed random seeds, followed by local refinement."""
    candidates = []
    for seed in SEEDS:
        global_fit = differential_evolution(loss, bounds, seed=seed, popsize=25,
            maxiter=1500, tol=1e-9, atol=1e-12, polish=False)
        local_fit = minimize(loss, global_fit.x, method="Nelder-Mead", bounds=bounds,
            options={"maxiter": 3000, "xatol": 1e-10, "fatol": 1e-13})
        candidates.extend([global_fit, local_fit])
    return min(candidates, key=lambda fit: fit.fun).x


def log2_error(predicted, observed):
    return np.mean(np.log2(np.maximum(predicted, .001) / np.maximum(observed, .001)) ** 2)


def fit_full_curve(time, od):
    net = od - od.min()
    k = net[-1]  # Fix the asymptotic plateau; do not force the fitted curve through the 48 h endpoint.

    def loss(parameters):
        x0, rate = np.exp(parameters[:2])
        lag = parameters[2]
        predicted = k / (1 + (k / x0 - 1) * np.exp(-rate * np.maximum(time - lag, 0)))
        return log2_error(predicted, net)

    bounds = [(np.log(X0_MIN), np.log(X0_MAX)), (np.log(1e-5), np.log(50)), (0, time[-1])]
    parameters = minimize_loss(loss, bounds)
    return float(np.exp(parameters[1])), float(parameters[2])


def fit_im_tail_rate(time, od):
    net = od - od.min()
    k = net[-1]
    # Smooth only to select the start; fit the original, unsmoothed net OD.
    start = np.flatnonzero(median_filter(net, size=7, mode="nearest") >= .2 * k)[0]
    tail_time, tail_od = time[start:], net[start:]
    low_b = np.log(k / X0_MAX - 1)
    high_b = np.log(k / X0_MIN - 1)

    def unpack(parameters):
        rate = np.exp(parameters[0])
        # b = log(K/X0 - 1) + rate*lag; bounds allow a valid X0 and a lag no later than the tail start.
        b = low_b + parameters[1] * (high_b + rate * time[start] - low_b)
        return rate, b

    def loss(parameters):
        rate, b = unpack(parameters)
        return log2_error(k * expit(rate * tail_time - b), tail_od)

    parameters = minimize_loss(loss, [(np.log(1e-5), np.log(50)), (0, 1)])
    return float(unpack(parameters)[0])


def main():
    data = pd.read_excel(INPUT)
    rows = []
    for species in SPECIES:
        selected = data[data["Species"] == f"Bt_{species}_100%"]
        if species == "Bl.s":
            selected = selected.iloc[:0]  # Retain the output row but do not fit Bl.s.
        # Use 20250323 for Ld and 20250910 for all other species.
        batch = "20250323" if species == "Ld" else "20250910"
        selected = selected[selected["ExperimentID"].str.startswith(batch)]
        if species == "Im":
            selected = selected[selected["Growth Well ID"] == "Batch1_Plate4_B3"]
            assert len(selected) == 1
        rates, lags = [], []
        for _, row in selected.iterrows():
            if pd.notna(row["Note"]):
                continue
            time, od = read_curve(row)
            if species != "Im" and od[-1] - od[0] < .01:
                continue
            rate, lag = fit_full_curve(time, od)
            if species == "Im":
                rate = fit_im_tail_rate(time, od)
            rates.append(rate)
            lags.append(lag)
        rows.append({"focal_species": species,
            "growth_rate": np.mean(rates) if rates else np.nan,
            "lag_time_h": np.mean(lags) if lags else np.nan})
        print(f"{species}: {len(rates)} 条曲线", flush=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT, index=False, float_format="%.8f", na_rep="")
    print(OUTPUT)


if __name__ == "__main__":
    main()
