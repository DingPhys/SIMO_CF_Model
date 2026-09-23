"""Extract growth tables from the raw workbooks and fit the growth curves.

Tables written to ../data/:
    16_grower_dm68_mean_growth.csv      grower monoculture Delta OD in DM68
    grower_spent_mean_growth.csv        grower in grower spent medium
    Nongrowers_monoculture_in_grower_spent.csv
    Nongrowers_in_Bt_double_spent.csv   including the Full_Bt / Diluted_Bt columns
Curve fits (skipped with --skip-fits):
    16_grower_dm68_curve_fit.csv
    Non-growers_In_Bt_Spent_growth_curve_fit_summary.csv

Paths are relative to this script. Requires numpy, pandas, scipy and openpyxl.
"""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from datetime import time as ExcelTime
from pathlib import Path

import numpy as np
import pandas as pd

from growth_fit_common import fit_full_curve
from output_schema import GROWERS, NONGROWERS

RAW = Path(__file__).resolve().parent
OUT = RAW.parent / 'data'
OUT.mkdir(parents=True, exist_ok=True)

G = GROWERS
NG = NONGROWERS

# Historical spelling variants found in the workbooks.
ALIASES = {'Lsp50': 'Lsp', 'Afi': 'Af', 'Bls': 'Bl.s', 'Euc': 'Eu.c', 'Eul': 'Eu.l'}

GROWTH_THRESHOLD = 0.01


def text(value):
    """Stripped string; missing values become an empty string."""
    return '' if pd.isna(value) else str(value).strip()


def canonical(name):
    return ALIASES.get(text(name), text(name))


def exclusion_note(row):
    """Non-empty if the row is flagged as contaminated or not used."""
    note = text(row.get('Note')).lower().replace(' ', '_')
    if 'contamin' in note:
        return 'contamination'
    if 'not_used' in note:
        return 'not_used'
    return ''


def curve_values(row):
    """OD values of a growth curve; every point must be a finite number."""
    curve = row.get('Growth curve OD', row.get('Growth curve'))
    values = np.fromstring(str(curve), sep=' ')
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError(f'Invalid curve: {curve!r}')
    return values


def curve_values_ignore_na(curve):
    """OD values with unparseable (NA) points dropped."""
    values = [pd.to_numeric(value, errors='coerce') for value in str(curve).split()]
    values = [float(value) for value in values if pd.notna(value)]
    if len(values) < 2:
        raise ValueError(f'Invalid curve: {curve!r}')
    return values


def save_table(df, filename, index=True):
    df.to_csv(OUT / filename, index=index, lineterminator='\r\n')
    print(f'Saved: {OUT / filename}', flush=True)


# -----------------------------------------------------------------------------
# 1. Grower monoculture in DM68
# -----------------------------------------------------------------------------

def grower_dm_mean_growth():
    df = pd.read_excel(RAW / 'Growth_data/Growth_curve_32_strains_monoculture_DM.xlsx')
    deltas = {species: [] for species in G}
    for _, row in df.iterrows():
        species = canonical(row['Species'])
        if species not in G or text(row['Media']) != 'DM68':
            continue
        values = curve_values(row)
        if not exclusion_note(row):
            deltas[species].append(values[-1] - values[0])
    table = pd.DataFrame({
        'species': G,
        'mean_growth': [float(np.mean(deltas[s])) if deltas[s] else None for s in G],
    })
    save_table(table, '16_grower_dm68_mean_growth.csv', index=False)


# -----------------------------------------------------------------------------
# 2. Grower in grower spent medium
# -----------------------------------------------------------------------------

def grower_spent_mean_growth():
    df = pd.read_excel(RAW / 'Growth_data/Growers_pairwise_spent_medium.xlsx')
    deltas = {(species, conditioner): [] for species in G for conditioner in G}
    for _, row in df.iterrows():
        parts = text(row.get('Group')).split('_')
        if len(parts) != 2:
            continue
        conditioner, species = (canonical(part) for part in parts)
        if species not in G or conditioner not in G:
            continue
        values = curve_values(row)
        if not exclusion_note(row):
            deltas[species, conditioner].append(values[-1] - values[0])

    def cell(replicates):
        if not replicates:
            return None
        # A condition counts as growing if any replicate passes the threshold.
        if any(value > GROWTH_THRESHOLD + 1e-12 for value in replicates):
            return float(np.maximum(replicates, 0).mean())
        return 0.0

    table = pd.DataFrame(
        [[cell(deltas[species, conditioner]) for conditioner in G] for species in G],
        index=pd.Index(G, name='focal_species'), columns=G)
    save_table(table, 'grower_spent_mean_growth.csv')


# -----------------------------------------------------------------------------
# 3. Nongrower monoculture in grower spent medium (100%)
# -----------------------------------------------------------------------------

def nongrowers_in_grower_spent():
    df = pd.read_excel(RAW / 'Growth_data/Non_growers_spent_medium_of_growers_raw_data.xlsx')
    deltas = {(species, conditioner): [] for species in NG for conditioner in G}
    for _, row in df.iterrows():
        parts = text(row['Species']).split('_')
        if len(parts) != 3 or parts[2] != '100%':
            continue
        conditioner, species = canonical(parts[0]), canonical(parts[1])
        if species not in NG or conditioner not in G:
            continue
        if 'contamination' in text(row.get('Note')).lower():
            continue
        values = curve_values_ignore_na(row['Growth curve'])
        delta = values[-1] - values[0]
        # Average only growing replicates, across all batches.
        if delta >= GROWTH_THRESHOLD - 1e-12:
            deltas[species, conditioner].append(delta)
    table = pd.DataFrame(
        [[max(float(np.mean(deltas[s, c])), 0.) if deltas[s, c] else 0.0 for c in G] for s in NG],
        index=pd.Index(NG, name='Species'), columns=G)
    save_table(table, 'Nongrowers_monoculture_in_grower_spent.csv')


# -----------------------------------------------------------------------------
# 4. Nongrower in nongrower/Bt double spent medium
# -----------------------------------------------------------------------------

def nongrowers_in_double_spent():
    conditions = NG + ['Full_Bt', 'Diluted_Bt']
    deltas = {(species, c): [] for species in NG for c in conditions}

    # Nongrower-conditioned spent medium; growth is final minus minimum OD.
    df = pd.read_excel(RAW / 'Growth_data/Double_spent_medium.xlsx')
    for _, row in df.iterrows():
        parts = text(row['Species']).split('_')
        if len(parts) != 2:
            continue
        conditioner, species = (canonical(part) for part in parts)
        if species not in NG or conditioner not in NG:
            continue
        values = curve_values(row)
        if not exclusion_note(row):
            deltas[species, conditioner].append(values[-1] - values.min())

    # Bt-spent rows of the grower-spent workbook become the Full_Bt/Diluted_Bt columns.
    df = pd.read_excel(RAW / 'Growth_data/Non_growers_spent_medium_of_growers_raw_data.xlsx')
    for _, row in df.iterrows():
        parts = text(row['Species']).split('_')
        if len(parts) != 3 or parts[0] != 'Bt' or parts[2] not in ('100%', '10%'):
            continue
        species = canonical(parts[1])
        if species not in NG:
            continue
        conditioner = 'Full_Bt' if parts[2] == '100%' else 'Diluted_Bt'
        values = curve_values(row)
        delta = values[-1] - values[0]
        # Average only growing replicates (Delta OD >= 0.01).
        if not exclusion_note(row) and delta >= GROWTH_THRESHOLD - 1e-12:
            deltas[species, conditioner].append(delta)

    def cell(species, conditioner):
        replicates = deltas[species, conditioner]
        if replicates:
            return max(float(np.mean(replicates)), 0.)
        # The Bt conditions default to 0; untested nongrower pairs stay blank.
        return 0.0 if conditioner in ('Full_Bt', 'Diluted_Bt') else None

    table = pd.DataFrame(
        [[cell(species, c) for c in conditions] for species in NG],
        index=pd.Index(NG, name='Species'), columns=conditions)
    save_table(table, 'Nongrowers_in_Bt_double_spent.csv')


# -----------------------------------------------------------------------------
# Curve fitting (shared by the two fit jobs below)
# -----------------------------------------------------------------------------

def read_curve(row):
    """Parse 'Growth curve' and 'Growth curve time' into (time, od) arrays."""
    od = pd.to_numeric(pd.Series(str(row['Growth curve']).split()), errors='coerce').to_numpy()
    stamp = row['Growth curve time']
    if isinstance(stamp, ExcelTime):
        # The Ld range cells were autoformatted by Excel; the agreed protocol is 0-48 h.
        start, end = 0.0, 48.0
    else:
        start, _, end = map(float, str(stamp).replace(' ', '').split(':'))
    time = np.linspace(start, end, len(od))
    valid = np.isfinite(od)
    assert valid.sum() >= 8 and end > start
    # Keep timestamps of valid points; do not interpolate missing OD values.
    return time[valid], od[valid]


def fit_replicates(records):
    """Fit replicates in parallel and return one row per fitted replicate."""
    print(f'Fitting {len(records)} curves.', flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for result in pool.map(_fit_one, records):
            results.append(result)
            print(f"{len(results)}/{len(records)} {result['focal_species']}: "
                  f"{result['growth_rate']:.6g}/h", flush=True)
    return pd.DataFrame(results)


def _fit_one(record):
    time, od = read_curve(record['row'])
    fit_start = record.get('fit_start', 0.0)
    rate, lag = fit_full_curve(time, od, fit_start=fit_start)
    return {'focal_species': record['focal_species'], 'growth_rate': rate, 'lag_time_h': lag}


def save_fit_summary(frame, species_order, filename, index_name, lag_name, float_format,
                     crlf=False):
    grouped = frame.groupby('focal_species', sort=False).agg(
        growth_rate=('growth_rate', 'mean'), lag_time_h=('lag_time_h', 'mean'))
    summary = grouped.reindex(species_order).reset_index(names=index_name)
    summary = summary.rename(columns={'lag_time_h': lag_name})
    path = OUT / filename
    summary.to_csv(path, index=False, float_format=float_format, na_rep='',
                   lineterminator='\r\n' if crlf else '\n')
    if crlf:
        path.write_bytes(path.read_bytes().removesuffix(b'\r\n'))
    print(f'Saved: {path}', flush=True)
    print(summary.to_string(), flush=True)


# -----------------------------------------------------------------------------
# 5. Fit growers in DM68
# -----------------------------------------------------------------------------

def fit_dm_curves():
    data = pd.read_excel(RAW / 'Growth_data/Growth_curve_32_strains_monoculture_DM.xlsx')
    species_order = sorted(G)
    data = data[(data['Media'] == 'DM68') & data['Species'].isin(species_order)]
    assert len(data) == 96
    records = []
    for _, row in data.iterrows():
        if pd.notna(row['Note']):
            continue
        od = np.fromstring(str(row['Growth curve']), sep=' ')
        if len(od) < 2 or not np.isfinite(od).all() or od[-1] - od[0] < GROWTH_THRESHOLD:
            continue
        records.append({'focal_species': row['Species'], 'row': row})
    frame = fit_replicates(records)
    save_fit_summary(frame, species_order, '16_grower_dm68_curve_fit.csv',
                     index_name='focal_strain', lag_name='lag_time', float_format='%.8f')


# -----------------------------------------------------------------------------
# 6. Fit nongrowers in 100% Bt spent medium
# -----------------------------------------------------------------------------

def im_fit_window(species, experiment, well):
    """Im keeps only the B3 and B11 replicates (B12 excluded).

    B3 is fitted on the 24-48 h window (it does not grow early); B11 uses the
    full curve. Returns None for replicates that should not be fitted.
    """
    if species != 'Im':
        return 0.0
    if str(experiment).startswith('20250910') and well == 'Batch1_Plate4_B3':
        return 24.0
    if str(experiment).startswith('20250323') and well == 'Plate1_B11':
        return 0.0
    return None


def fit_bt_curves():
    data = pd.read_excel(RAW / 'Growth_data/Non_growers_spent_medium_of_growers_raw_data.xlsx')
    records = []
    for _, row in data.iterrows():
        parts = text(row['Species']).split('_')
        if len(parts) != 3 or parts[0] != 'Bt' or parts[2] != '100%':
            continue
        species = canonical(parts[1])
        if species not in NG or 'contamination' in text(row.get('Note')).lower():
            continue
        if species in ('Af', 'Bl.s'):
            continue  # Af and Bl.s are not fitted in the Bt-spent model.
        fit_start = im_fit_window(species, row['ExperimentID'], row['Growth_Well_ID'])
        if fit_start is None:
            continue
        od = pd.to_numeric(pd.Series(str(row['Growth curve']).split()), errors='coerce').to_numpy()
        valid = od[np.isfinite(od)]
        if len(valid) < 2 or valid[-1] - valid[0] < GROWTH_THRESHOLD - 1e-12:
            continue
        records.append({'focal_species': species, 'row': row, 'fit_start': fit_start})
    frame = fit_replicates(records)
    save_fit_summary(frame, NG, 'Non-growers_In_Bt_Spent_growth_curve_fit_summary.csv',
                     index_name='focal_species', lag_name='lag_time_h', float_format='%.10g',
                     crlf=True)


# -----------------------------------------------------------------------------

def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--skip-fits', action='store_true',
                        help='Rebuild only the tables; keep the existing fit results.')
    args = parser.parse_args()
    grower_dm_mean_growth()
    grower_spent_mean_growth()
    nongrowers_in_grower_spent()
    nongrowers_in_double_spent()
    if not args.skip_fits:
        fit_dm_curves()
        fit_bt_curves()


if __name__ == '__main__':
    main()
