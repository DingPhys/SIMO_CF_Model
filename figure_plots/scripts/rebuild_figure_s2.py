"""Rebuild Figure S2 from this package's current parameters, using the reference R style.

Run with jnb-legacy Python. Numerical preparation uses the package's simulator;
R handles all drawing. Two SVGs, two PDFs and their source CSV tables are written
to figure_plots/FigureS2.
Optional --preview-dir sends temporary R-rendered QA images elsewhere.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import os
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview-dir', type=Path)
    args = parser.parse_args()
    figure_root = Path(__file__).resolve().parents[1]
    root = figure_root.parent
    reference = figure_root / 'FigureS2'
    output = figure_root / 'FigureS2'
    output.mkdir(exist_ok=True)
    spec = importlib.util.spec_from_file_location('figure_s2_predictions', root / 'scripts/prediction_pipeline.py')
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)
    sources = sorted((root / 'parameters').rglob('*.csv')) + sorted((root / 'data').glob('*.csv'))
    original_hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}

    with tempfile.TemporaryDirectory(prefix='figure_s2_0907_') as temporary:
        temp = Path(temporary)
        model = temp / 'model'
        model.mkdir()
        (model / 'parameters').symlink_to(root / 'parameters', target_is_directory=True)
        (model / 'data').symlink_to(root / 'data', target_is_directory=True)
        stage = temp / 'plot_inputs'
        stage.mkdir()

        # Use the full DM model's exact universal-resource rule and fitted rates.
        grower, nongrower, _, _, _ = pipeline.load_model(root)
        consumption, production, *_ = pipeline.full_dm_model(root)
        resources = list(grower.columns) + list(nongrower.columns)
        old_consumption = pd.read_csv(reference / 'full_32_CR_matrix.csv', index_col='species')
        row_order = list(old_consumption.index)
        assert set(row_order) == set(pipeline.SPECIES)
        column_order = [c for c in old_consumption.columns if c in resources]
        column_order += [c for c in resources if c not in column_order]
        cr = pd.DataFrame(consumption, index=pipeline.SPECIES, columns=resources).loc[row_order, column_order]
        pr = pd.DataFrame(production, index=pipeline.SPECIES, columns=resources).loc[row_order, column_order]
        assert np.isfinite(cr).all().all() and np.isfinite(pr).all().all()
        assert not ((cr > 0) & (pr > 0)).any().any()
        cr.rename_axis('species').to_csv(stage / 'full_32_CR_matrix.csv')
        pr.rename_axis('species').to_csv(stage / 'full_32_PR_matrix.csv')
        shutil.copy2(root / 'parameters/grower/model/dm_resource_y0.csv', stage / 'dm_resource_y0.csv')
        shutil.copy2(root / 'parameters/bt_base/bt_resource_Y0.csv', stage / 'bt_resource_Y0.csv')

        # Always simulate from current parameters; historical prediction CSVs are not inputs.
        summary = pipeline.predict_nongrowers_in_bt(model)
        result = model / 'prediction_results/nongrowers_in_bt_spent'
        metrics = pd.read_csv(result / 'cofactor_on/community_metrics.csv').set_index('community')
        observed = pd.read_csv(result / 'observed_relative.csv', index_col=0)
        predicted = pd.read_csv(result / 'cofactor_on/predicted_relative.csv', index_col=0)
        error = pd.read_csv(result / 'cofactor_on/signed_log2_error.csv', index_col=0)
        reference_long = pd.read_csv(reference / 'nongrower_bt_spent_assembly_predictions.csv')
        old_assemblies = reference_long.drop_duplicates('assembly').sort_values('assembly_order')['assembly'].tolist()
        order = [a for a in old_assemblies if a in metrics.index]
        order += [a for a in metrics.index if a not in order]
        order = [a for a in order if np.isfinite(metrics.loc[a, 'mean_abs_log2_error'])]
        species = reference_long[['species_order', 'species']].drop_duplicates().sort_values('species_order')['species'].tolist()
        raw = pd.read_csv(root / 'data/non-growers_in_Bt_spent_assemblies_mean_relative_abundance.csv', index_col='species')
        rows = []
        for a_index, assembly in enumerate(order, 1):
            metric = metrics.loc[assembly]
            valid = raw[assembly].notna()
            assert int(valid.sum()) == int(metric.n_presented)
            for s_index, s in enumerate(species, 1):
                rows.append(dict(
                    assembly_order=a_index, assembly=assembly,
                    assembly_type=metric.community_type,
                    n_presented=int(metric.n_presented), n_scored=int(error[assembly].notna().sum()),
                    cofactor_enabled=True, assembly_mean_abs_log2_error=metric.mean_abs_log2_error,
                    assembly_rmse_log2_error=metric.rmse_log2_error,
                    species_order=s_index, species=s, presented=int(valid.loc[s]),
                    predicted_relative_abundance=predicted.loc[s, assembly],
                    observed_relative_abundance=observed.loc[s, assembly],
                    species_signed_log2_error=error.loc[s, assembly],
                    species_abs_log2_error=abs(error.loc[s, assembly]),
                ))
        long = pd.DataFrame(rows)
        assert len(long) == len(order) * len(species)
        assert np.allclose(long.groupby('assembly')['species_abs_log2_error'].mean().reindex(order),
                           metrics.loc[order, 'mean_abs_log2_error'])
        long.to_csv(stage / 'nongrower_bt_spent_assembly_predictions.csv', index=False)
        environment = os.environ.copy()
        environment.update(FIG_S2_INPUT=str(stage), FIG_S2_OUTPUT=str(output), FIG_S2_PREVIEW='')
        if args.preview_dir:
            args.preview_dir.mkdir(parents=True, exist_ok=True)
            environment['FIG_S2_PREVIEW'] = str(args.preview_dir.resolve())
        subprocess.run(['Rscript', str(figure_root / 'scripts/generate_figure_s2_current.R')],
                       env=environment, cwd=temp, check=True)
        # Preserve the exact input bytes read by R alongside the final figures.
        for source_csv in stage.glob('*.csv'):
            destination = output / source_csv.name
            shutil.copy2(source_csv, destination)
            assert destination.read_bytes() == source_csv.read_bytes()
        histogram = pd.read_csv(output / 'nongrower_bt_spent_assembly_prediction_matrix_histogram_source_data.csv')
        assert int(histogram['count'].sum()) == len(order)
        assert np.allclose(histogram['mean_error'], metrics.loc[order, 'mean_abs_log2_error'].mean())
        expected = {stem + extension for stem in (
            'full_32_consumption_production_matrix', 'nongrower_bt_spent_assembly_prediction_matrix'
        ) for extension in ('.svg', '.pdf')}
        assert expected.issubset({p.name for p in output.iterdir()})
        assert all((output / name).stat().st_size > 1000 for name in expected)
        for p, digest in original_hashes.items():
            assert hashlib.sha256(p.read_bytes()).hexdigest() == digest, f'Input changed: {p}'
        print(f'Validated: {len(order)} assemblies, {len(species)} nongrowers, {len(row_order)} matrix species.')
        print('Assembly mean error:', metrics.loc[order, 'mean_abs_log2_error'].mean())
        print('Reference assemblies excluded from current data:', sorted(set(old_assemblies) - set(order)))
        print('Outputs:', output)


if __name__ == '__main__':
    main()
