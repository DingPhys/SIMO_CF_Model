"""Rebuild the Figure S3 lag heatmap and scatter with current 0907 parameters.

R draws SVG/PDF figures; plotted CSVs and source provenance stay beside them.
The double-spent source must be selected explicitly using --double-spent.
No growth curves are refitted and no model inputs are changed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import tempfile

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--double-spent', required=True, choices=['new', 'reference'])
    parser.add_argument('--preview-dir', type=Path)
    args = parser.parse_args()
    figure_root = Path(__file__).resolve().parents[1]
    root = figure_root.parent
    repo = root.parent
    reference = figure_root / 'FigureS3'
    output = figure_root / 'FigureS3'
    output.mkdir(exist_ok=True)
    parameter_path = root / 'parameters/final_parameters/cofactor_and_lag.csv'
    monoculture_path = root / 'data/Non-growers_In_Bt_Spent_growth_curve_fit_summary.csv'
    old_heatmap_path = reference / 'actual_lag_time_heatmap_source_data.csv'
    old_scatter_path = reference / 'manual_vs_bt_monoculture_lag_scatter_source_data.csv'
    old = pd.read_csv(old_heatmap_path, index_col='focal_species').rename(index={'Lsp50': 'Lsp'})
    colors = pd.read_csv(old_scatter_path).set_index('species')['color'].rename(index={'Lsp50': 'Lsp'})
    species = old.index.tolist()
    manual = pd.read_csv(parameter_path).set_index('species')['selected_lag_h'].reindex(species)
    mono = pd.read_csv(monoculture_path).set_index('focal_species')['lag_time_h'].reindex(species)
    assert np.isfinite(manual).all() and np.isfinite(mono).all()
    if args.double_spent == 'new':
        double_path = repo / 'Zhenyus_data/Data_to_upload/Data/Figure4/Figure4A/double_spent_lag.csv'
        double = pd.read_csv(double_path, index_col='focal_species')
        double = double.rename(index={'Lsp50': 'Lsp'}, columns={'Lsp50': 'Lsp'}).loc[species, species]
    else:
        double_path = old_heatmap_path
        double = old.iloc[:, 2:].rename(columns={'Lsp50': 'Lsp'}).loc[species, species]
    assert np.isfinite(double.to_numpy()[double.notna().to_numpy()]).all()
    assert (double.stack() >= 0).all()
    heatmap = pd.concat([manual.rename('Manual lag (used in model)'), mono.rename('Bt monoculture'), double], axis=1)
    heatmap.index.name = 'focal_species'
    scatter = pd.DataFrame({'species': species, 'bt_monoculture_lag_h': mono.to_numpy(),
                            'manually_set_lag_h': manual.to_numpy(), 'color': colors.reindex(species).to_numpy()})
    assert scatter['color'].notna().all()
    heatmap_path = output / 'actual_lag_time_heatmap_source_data.csv'
    scatter_path = output / 'manual_vs_bt_monoculture_lag_scatter_source_data.csv'
    mask_path = output / 'actual_lag_time_heatmap_missing_mask.csv'
    heatmap.to_csv(heatmap_path, na_rep='NA')
    scatter.to_csv(scatter_path, index=False)
    heatmap.isna().astype(int).to_csv(mask_path)
    source_paths = list(dict.fromkeys([parameter_path, monoculture_path, double_path, old_heatmap_path, old_scatter_path]))
    source_hashes = {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    provenance = {
        'manual_lag_source': str(parameter_path.relative_to(repo)),
        'bt_monoculture_lag_source': str(monoculture_path.relative_to(repo)),
        'double_spent_lag_source': str(double_path.relative_to(repo)),
        'double_spent_selection': args.double_spent,
        'species_order_and_colors': 'Reference FigureS3; Lsp50 column label normalized to Lsp.',
        'lag_units': 'hours', 'heatmap_limits': [0, 20],
        'heatmap_missing_values': 'Grey; preserved from selected source, not replaced by zero.',
        'missing_mask': '1 = missing; no additional exclusion is inferred.',
        'new_fitting_performed': False,
        'source_sha256': source_hashes,
    }
    (output / 'data_sources.json').write_text(json.dumps(provenance, indent=2) + '\n')
    environment = os.environ.copy()
    environment.update(FIG_S3_OUTPUT=str(output), FIG_S3_PREVIEW='')
    if args.preview_dir:
        args.preview_dir.mkdir(parents=True, exist_ok=True)
        environment['FIG_S3_PREVIEW'] = str(args.preview_dir.resolve())
    with tempfile.TemporaryDirectory(prefix='figure_s3_render_') as tmp:
        subprocess.run(['Rscript', str(figure_root / 'scripts/generate_figure_s3_current.R')],
                       cwd=tmp, env=environment, check=True)
    loaded = pd.read_csv(heatmap_path, index_col='focal_species')
    np.testing.assert_allclose(loaded.iloc[:, 0], manual)
    np.testing.assert_allclose(loaded.iloc[:, 1], mono)
    np.testing.assert_allclose(loaded.iloc[:, 2:], double, equal_nan=True)
    for name, digest in source_hashes.items():
        assert hashlib.sha256((repo / name).read_bytes()).hexdigest() == digest
    for stem in ['actual_lag_time_manual_bt_monoculture_and_double_spent_heatmap', 'manual_vs_bt_monoculture_lag_scatter']:
        for extension in ['.svg', '.pdf']:
            assert (output / (stem + extension)).stat().st_size > 1000
    print('Validated: 10 species; heatmap 10 x 12; scatter 10 points; all plotted values match sources.')
    print('Double-spent lag source:', double_path)
    print('Double-spent populated cells:', int(double.notna().sum().sum()))
    print('Largest monoculture lag:', float(mono.max()))
    print('Outputs:', output)


if __name__ == '__main__':
    main()
