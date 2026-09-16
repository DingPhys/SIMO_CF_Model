"""100% Bt spent medium: save the mean growth rate and lag for each nongrower.

Run this script directly. Requires numpy, scipy, pandas, and openpyxl
"""
from pathlib import Path
from datetime import time as ExcelTime
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.special import expit

from growth_fit_common import fit_full_curve, minimize_grid, log2_error, X0_MIN, X0_MAX

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "Growth_data/Non_growers_spent_medium_of_growers.xlsx"
OUTPUT = ROOT.parent / "data/Non-growers_In_Bt_Spent_growth_curve_fit_summary.csv"
SPECIES = ["Af", "Ao", "As", "Bl.s", "Col", "Cs", "Et", "Eu.c", "Eu.l", "Im", "Ld", "Lsp", "Mi", "Pc", "Va", "Vp"]


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


def fit_im_tail_rate(time, od):
    net = od - od.min()
    k = net[-1]
    # Smooth only to select the start; fit the original, unsmoothed net OD.
    start = np.flatnonzero(median_filter(net, size=7, mode="nearest") >= .2 * k)[0]
    tail_time, tail_od = time[start:], net[start:]
    low_b = np.log(k / X0_MAX - 1)
    high_b = np.log(k / X0_MIN - 1)

    def loss(parameters):
        rate = np.exp(parameters[:, 0, None])
        fraction = parameters[:, 1, None]
        # b = log(K/X0 - 1) + rate*lag, with lag no later than the tail start.
        b = low_b + fraction * (high_b + rate * time[start] - low_b)
        predicted = k * expit(rate * tail_time[None, :] - b)
        return log2_error(predicted, tail_od)

    bounds = [(np.log(1e-5), np.log(50)), (0, 1)]
    parameters = minimize_grid(loss, bounds, [1201, 1201])
    return float(np.exp(parameters[0]))


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
        print(f"{species}: {len(rates)} curves", flush=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT, index=False, float_format="%.8f", na_rep="")
    print(OUTPUT)


if __name__ == "__main__":
    main()
