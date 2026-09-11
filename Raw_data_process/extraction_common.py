"""Extract observation tables from raw workbooks, preserving the published layout."""
from collections import defaultdict
from datetime import time as ExcelTime
from pathlib import Path
import csv

import numpy as np
import openpyxl

from output_schema import SCHEMAS, GROWERS, NONGROWERS

RAW = Path(__file__).resolve().parent
OUT = RAW.parent / 'data'
G = GROWERS
NG = NONGROWERS
ALL = G + NG
FORTY = G + ['Bb', 'Bbi', 'Bc', 'Bl', 'Bt47', 'Bt61', 'Pg', 'Pj'] + NG
CORE13 = ['Ai', 'Bt', 'Csp', 'Dl', 'Mf', 'Bd', 'Cs', 'Eu.c', 'Eu.l', 'Im', 'Mi', 'Va', 'Vp']
ALIASES = {'Lsp50': 'Lsp', 'Afi': 'Af', 'Bls': 'Bl.s', 'Euc': 'Eu.c', 'Eul': 'Eu.l'}


def text(value):
    return '' if value is None else str(value).strip()


def num(value):
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def read_excel(relative):
    path = RAW / relative
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = []
    try:
        for sheet in book:
            iterator = iter(sheet.values)
            headers = [text(value) for value in next(iterator)]
            species_headers = (
                headers[headers.index('Ai'):headers.index('Vp') + 1]
                if 'Ai' in headers and 'Vp' in headers else []
            )
            for row_number, values in enumerate(iterator, 2):
                if not any(value is not None for value in values):
                    continue
                row = {ALIASES.get(key, key): value for key, value in zip(headers, values) if key}
                row['source_key'] = f'{relative}|{sheet.title}|{row_number}'
                row['_species_headers'] = species_headers
                rows.append(row)
    finally:
        book.close()
    return rows


def od_metrics(row):
    od = np.fromstring(str(row.get('Growth curve OD', row.get('Growth curve'))), sep=' ')
    value = row.get('Growth curve time')
    if isinstance(value, ExcelTime):
        # The Ld range cells were autoformatted by Excel; the agreed protocol is 0–48 h.
        times = np.linspace(0.0, 48.0, len(od))
    else:
        parts = str(value).replace(' ', '').split(':')
        times = (np.linspace(float(parts[0]), float(parts[2]), len(od))
                 if len(parts) == 3 else np.fromstring(str(value), sep=' '))
    if len(od) < 2 or len(od) != len(times) or not np.isfinite(od).all() or not np.isfinite(times).all():
        raise ValueError(f'Invalid curve: {row["source_key"]}')
    return {'final_minus_initial': float(od[-1] - od[0]),
            'final_minus_minimum': float(od[-1] - od.min())}


def exclusion_note(row):
    note = text(row.get('Note')).lower().replace(' ', '_')
    if 'contamin' in note:
        return 'source_contamination'
    if 'not_used' in note:
        return 'source_not_used'
    return ''


def save_table(filename, rows):
    definition = SCHEMAS[filename]
    key = definition['row_key']
    by_key = {str(row[key]): row for row in rows}
    assert len(by_key) == len(rows), f'Duplicate row labels in {filename}'
    missing = set(definition['rows']) - set(by_key)
    assert not missing, f'Missing output rows in {filename}: {missing}'

    def cell(value):
        if value is None or isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return ''
        return value

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / filename).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=definition['columns'], extrasaction='ignore')
        writer.writeheader()
        writer.writerows(
            {column: cell(by_key[label].get(column)) for column in definition['columns']}
            for label in definition['rows']
        )
    print(f'{filename}: {len(definition["rows"])} rows saved', flush=True)


def matrix(row_species, columns, values, index='Species'):
    return [{index: species, **{column: values.get((species, column)) for column in columns}}
            for species in row_species]


def growth_matrix(filename, sources, row_species, columns, kind):
    groups = defaultdict(list)
    for source in sources:
        for row in read_excel(source):
            parts = text(row.get('Species', row.get('Group'))).split('_')
            if kind == 'double' and 'Non_growers_' in source:
                if len(parts) != 3 or parts[0] != 'Bt' or parts[2] not in ('100%', '10%'):
                    continue
                conditioner = 'Full_Bt' if parts[2] == '100%' else 'Diluted_Bt'
                species = parts[1]
                metric = 'final_minus_initial'
            elif kind == 'nongrower_spent':
                if len(parts) != 3 or parts[2] != '100%':
                    continue
                conditioner, species = parts[:2]
                metric = 'final_minus_initial'
            else:
                if len(parts) != 2:
                    continue
                conditioner, species = parts
                metric = 'final_minus_minimum' if kind == 'double' else 'final_minus_initial'
            species = ALIASES.get(species, species)
            conditioner = ALIASES.get(conditioner, conditioner)
            if species not in row_species or conditioner not in columns:
                continue
            growth = od_metrics(row)
            if not exclusion_note(row):
                groups[species, conditioner].append(growth[metric])

    values = {}
    for species in row_species:
        for conditioner in columns:
            replicates = groups.get((species, conditioner), [])
            value = None
            if replicates:
                if kind == 'grower_spent':
                    # Strict >0.01 trigger, with the original roundoff tolerance.
                    growing = any(value > .01 + 1e-12 for value in replicates)
                    value = float(np.maximum(replicates, 0).mean()) if growing else 0.
                else:
                    value = max(float(np.mean(replicates)), 0.)
            values[species, conditioner] = value
    index = 'focal_species' if kind == 'grower_spent' else 'Species'
    save_table(filename, matrix(row_species, columns, values, index))


def designed(group, kind):
    group = text(group)
    if kind == 'ng':
        if group in ('Full_community', 'Bt'):
            return NG.copy(), 'Full_community'
        if group.endswith('_dropout'):
            species = ALIASES.get(group[:-8], group[:-8])
            return [name for name in NG if name != species], species + '_dropout'
        pool = NG
    else:
        if group == 'Full_community':
            return ALL.copy(), 'Full_community_32_Glucose'
        if group == 'Whole_community':
            return CORE13.copy(), 'Whole_community_13'
        if group in ('Bt_dropout', 'Bd_dropout', 'Bt_Bd_dropout'):
            dropped = group.removesuffix('_dropout').split('_')
            return [name for name in CORE13 if name not in dropped], group
        pool = ALL
    species = [ALIASES.get(name, name) for name in group.split('_')]
    if not all(name in pool for name in species):
        raise ValueError(f'Unknown community {group}')
    return sorted(set(species), key=pool.index), None


def adjusted_counts(row, species):
    counts = {name: num(row.get(name)) or 0. for name in species}
    # Keep both channels when both are designed; recover Eu.c reads for Eu.l otherwise.
    if 'Eu.l' in species and 'Eu.c' not in species:
        counts['Eu.l'] += num(row.get('Eu.c')) or 0.
    return counts


def community_matrix(filename, source, kind, *, user_excluded_communities=(), user_excluded_samples=()):
    pool = NG if kind == 'ng' else ALL
    original_columns = SCHEMAS[filename]['columns'][1:]
    name_by_set = {}
    for column in original_columns:
        if 'community' not in column.lower() and 'dropout' not in column.lower():
            species = [ALIASES.get(name, name) for name in column.split('_')]
            if all(name in pool for name in species):
                name_by_set[frozenset(species)] = column

    accepted = defaultdict(list)
    members = {}
    for row in read_excel(source):
        species, label = designed(row.get('Group'), kind)
        name = label or name_by_set.get(frozenset(species), '_'.join(species))
        members[name] = species
        counts = adjusted_counts(row, species)
        designed_reads = sum(counts.values())
        all_reads = sum(num(row.get(name)) or 0 for name in row['_species_headers'])
        off_fraction = (all_reads - designed_reads) / all_reads if all_reads > 0 else None
        sample = (text(row.get('Experiment ID')), text(row.get('Growth Well ID')))
        reason = exclusion_note(row)
        if not reason and 'low_growth' in text(row.get('Note')).lower():
            reason = 'source_low_growth'
        if not reason and name in user_excluded_communities:
            reason = 'user_excluded_community'
        if not reason and sample in user_excluded_samples:
            reason = 'user_excluded_sample'
        if not reason and designed_reads <= 0:
            reason = 'no_designed_species_reads'
        if not reason and off_fraction is not None and off_fraction > .10:
            reason = 'off_design_fraction_gt_0.10'
        if not reason:
            accepted[name].append({name: counts[name] / designed_reads for name in species})

    columns = [name for name in original_columns if name in members] + sorted(set(members) - set(original_columns))
    values = {}
    for name in columns:
        replicates = accepted[name]
        for species in members[name]:
            values[species, name] = float(np.mean([row[species] for row in replicates])) if replicates else None
        if replicates:
            assert np.isclose(sum(values[species, name] for species in members[name]), 1.)
    save_table(filename, matrix(pool, columns, values, 'species'))


def grower_dm(filename, source):
    groups = defaultdict(list)
    for row in read_excel(source):
        species = ALIASES.get(text(row['Species']), text(row['Species']))
        if species not in G or text(row['Media']) != 'DM68':
            continue
        growth = od_metrics(row)
        if not exclusion_note(row):
            groups[species].append(growth['final_minus_initial'])
    rows = [dict(species=species, mean_growth=float(np.mean(groups[species])) if groups[species] else None)
            for species in G]
    save_table(filename, rows)


def carbon_absolute(filename, source):
    rows = []
    for row in read_excel(source):
        counts = {name: num(row.get(name)) or 0. for name in FORTY}
        total = sum(counts.values())
        final = num(row.get('Final OD'))
        assert final is not None and total > 0
        relative = {name: counts[name] / total for name in FORTY}
        absolute = {name: final * relative[name] for name in FORTY}
        raw_qc = text(row.get('Note')).lower().replace('_', ' ')
        # Preserve the published label; this is not an independent contamination diagnosis.
        qc = 'high other species' if raw_qc == 'not used' else raw_qc
        plate, well = text(row['Growth Well ID']).split('_', 1)
        batch = f'2024930_WZY_DM_assembly_{plate}'
        item = {
            'Sample ID': batch + well, 'Experiment ID': batch, 'growth_qc': qc,
            'Growth Well ID': well, '16S Well ID': row.get('16S Well ID'),
            'Media': row.get('Media'), 'Protocol': row.get('Protocol'), 'Initial OD': '',
            'Final OD': final, 'Growth curve time': row.get('Growth curve time'),
            'Growth curve OD': row.get('Growth curve OD'),
            'Notes': 'growth curve data NA, so 48 h label rather than 0: 0.05:48 was used',
            'Data location': f'Ho lab/7_Shared data/16s rRNA/Raw reads/2024930_DM_assembly/{plate}',
        }
        # Keep all replicate rows and their QC labels; scoring exclusions occur downstream.
        item.update(absolute)
        rows.append(item)
        assert np.isclose(sum(absolute.values()), final)
    save_table(filename, rows)
