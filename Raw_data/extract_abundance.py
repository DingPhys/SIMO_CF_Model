"""Extract community abundance tables from the raw 16S workbooks.

Tables written to ../data/:
    dm_assemblies_mean_relative_abundance_matrix.csv
    non-growers_in_Bt_spent_assemblies_mean_relative_abundance.csv
    full_community_different_carbon_sources_absolute_abundance.csv

Each replicate is normalized over its designed species, then replicates are
averaged with equal weight. Replicates flagged as contamination / not_used /
low_growth, without designed-species reads, or with more than 10% off-design
reads are excluded. Community columns follow the raw workbook, sorted by
community size (ties: nongrower order for the Bt-spent table, first
appearance for the DM table). Paths are relative to this script.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from output_schema import GROWERS, NONGROWERS

RAW = Path(__file__).resolve().parent
OUT = RAW.parent / 'data'
OUT.mkdir(parents=True, exist_ok=True)

G = GROWERS
NG = NONGROWERS
ALL = G + NG

# Taxa in the 16S tables beyond the 32 model species (used by the carbon table).
EXTRA_TAXA = ['Bb', 'Bbi', 'Bc', 'Bl', 'Bt47', 'Bt61', 'Pg', 'Pj']
FORTY = G + EXTRA_TAXA + NG

# 13-species community used in the DM assembly experiment.
CORE13 = ['Ai', 'Bt', 'Csp', 'Dl', 'Mf', 'Bd', 'Cs', 'Eu.c', 'Eu.l', 'Im', 'Mi', 'Va', 'Vp']

# Historical spelling variants found in the workbooks.
ALIASES = {'Lsp50': 'Lsp', 'Afi': 'Af', 'Bls': 'Bl.s', 'Euc': 'Eu.c', 'Eul': 'Eu.l'}

OFF_DESIGN_LIMIT = 0.10


def text(value):
    """Stripped string; missing values become an empty string."""
    return '' if pd.isna(value) else str(value).strip()


def number(value):
    """Finite float, or None for missing/unparseable values."""
    value = pd.to_numeric(value, errors='coerce')
    return float(value) if pd.notna(value) and np.isfinite(value) else None


def canonical(name):
    return ALIASES.get(text(name), text(name))


def read_workbook(relative_path):
    df = pd.read_excel(RAW / relative_path).dropna(how='all')
    df = df.rename(columns={column: ALIASES.get(column, column) for column in df.columns})
    return df


def sequenced_taxa(df):
    """All 16S count columns: the contiguous block from Ai to Vp in the header."""
    columns = list(df.columns)
    return columns[columns.index('Ai'):columns.index('Vp') + 1]


def designed_community(group, kind):
    """Return (member species in pool order, output label) for one assembly."""
    group = text(group)

    if kind == 'ng':
        pool = NG
        if group in ('Full_community', 'Bt'):
            return NG.copy(), 'Full_community'
        if group.endswith('_dropout'):
            dropped = canonical(group.removesuffix('_dropout'))
            return [species for species in NG if species != dropped], f'{dropped}_dropout'
    elif kind == 'dm':
        pool = ALL
        if group == 'Full_community':
            return ALL.copy(), 'Full_community_32_Glucose'
        if group == 'Whole_community':
            return CORE13.copy(), 'Whole_community_13'
        if group in ('Bt_dropout', 'Bd_dropout', 'Bt_Bd_dropout'):
            dropped = group.removesuffix('_dropout').split('_')
            return [species for species in CORE13 if species not in dropped], group
    else:
        raise ValueError("kind must be 'dm' or 'ng'")

    parts = [canonical(part) for part in group.split('_')]
    if not all(part in pool for part in parts):
        raise ValueError(f'Unknown community: {group}')
    members = sorted(set(parts), key=pool.index)
    # The Bt-spent workbook spells some pairs in both orders (Col_Af / Af_Col);
    # merge them under the nongrower-ordered name. DM labels are unique already.
    label = '_'.join(members) if kind == 'ng' else '_'.join(parts)
    return members, label


def community_matrix(source, kind, filename):
    df = read_workbook(source)
    taxa = sequenced_taxa(df)
    pool = NG if kind == 'ng' else ALL

    members = {}
    replicates = {}
    first_seen = []
    for _, row in df.iterrows():
        species, label = designed_community(row.get('Group'), kind)
        counts = {name: number(row.get(name)) or 0.0 for name in species}
        # When Eu.l is designed but Eu.c is not, recover Eu.c reads into Eu.l.
        if 'Eu.l' in species and 'Eu.c' not in species:
            counts['Eu.l'] += number(row.get('Eu.c')) or 0.0

        designed_reads = sum(counts.values())
        all_reads = sum(number(row.get(name)) or 0.0 for name in taxa)
        off_fraction = (all_reads - designed_reads) / all_reads if all_reads > 0 else None

        note = text(row.get('Note')).lower().replace(' ', '_')
        excluded = (
            'contamin' in note
            or 'not_used' in note
            or 'low_growth' in note
            or designed_reads <= 0
            or (off_fraction is not None and off_fraction > OFF_DESIGN_LIMIT)
        )
        if label not in members:
            members[label] = species
            replicates[label] = []
            first_seen.append(label)
        if not excluded:
            replicates[label].append(
                {name: counts[name] / designed_reads for name in species})

    # Keep communities with at least one accepted replicate, sorted by size.
    labels = [label for label in first_seen if replicates[label]]
    if kind == 'ng':
        # Same size: nongrower order of members; dropouts follow the dropped species.
        def order(label):
            if label.endswith('_dropout'):
                return (len(members[label]), [pool.index(label.removesuffix('_dropout'))])
            return (len(members[label]), [pool.index(s) for s in members[label]])
        labels.sort(key=order)
    else:
        labels.sort(key=lambda label: len(members[label]))

    table = pd.DataFrame(np.nan, index=pd.Index(pool, name='species'), columns=labels)
    for label in labels:
        species = members[label]
        for name in species:
            table.loc[name, label] = float(np.mean([r[name] for r in replicates[label]]))
        assert np.isclose(table.loc[species, label].sum(), 1.0)

    table.to_csv(OUT / filename, lineterminator='\r\n')
    print(f'Saved: {OUT / filename}', flush=True)


def carbon_absolute_abundance():
    df = read_workbook('Community_assembly/Full_community_assembly_DM_across_carbon_sources.xlsx')

    rows = []
    for _, row in df.iterrows():
        counts = {name: number(row.get(name)) or 0.0 for name in FORTY}
        total_reads = sum(counts.values())
        final_od = number(row.get('Final OD'))
        assert final_od is not None and total_reads > 0

        relative = {name: counts[name] / total_reads for name in FORTY}
        absolute = {name: final_od * relative[name] for name in FORTY}
        assert np.isclose(sum(absolute.values()), final_od)

        raw_qc = text(row.get('Note')).lower().replace('_', ' ')
        # Preserve the published label; this is not an independent contamination diagnosis.
        growth_qc = 'high other species' if raw_qc == 'not used' else raw_qc

        plate, well = text(row['Growth Well ID']).split('_', 1)
        batch = f'2024930_WZY_DM_assembly_{plate}'
        rows.append({
            'Sample ID': batch + well,
            'Experiment ID': batch,
            'growth_qc': growth_qc,
            'Growth Well ID': well,
            '16S Well ID': row.get('16S Well ID'),
            'Media': row.get('Media'),
            'Protocol': row.get('Protocol'),
            'Initial OD': '',
            'Final OD': final_od,
            'Growth curve time': row.get('Growth curve time'),
            'Growth curve OD': row.get('Growth curve OD'),
            'Notes': 'growth curve data NA, so 48 h label rather than 0: 0.05:48 was used',
            'Data location': (
                'Ho lab/7_Shared data/16s rRNA/Raw reads/2024930_DM_assembly/' + plate),
            **absolute,
        })

    table = pd.DataFrame(rows)
    table.to_csv(OUT / 'full_community_different_carbon_sources_absolute_abundance.csv',
                 index=False, lineterminator='\r\n')
    print(f"Saved: {OUT / 'full_community_different_carbon_sources_absolute_abundance.csv'}",
          flush=True)


def main():
    community_matrix('Community_assembly/DM_assembly.xlsx', 'dm',
                     'dm_assemblies_mean_relative_abundance_matrix.csv')
    community_matrix('Community_assembly/Non_grower_assembly_Bt_spent_medium.xlsx', 'ng',
                     'non-growers_in_Bt_spent_assemblies_mean_relative_abundance.csv')
    carbon_absolute_abundance()


if __name__ == '__main__':
    main()
