#!/usr/bin/env python3
"""Run the two Bt/NG scans.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import pairwise_scan as pair
import community_scan as community

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
def worker(kind, parameters, runs, config, starting_biomass):
    if kind == 'pairwise':
        arrays = pair.assemble_batch_arrays(parameters, runs, config)
        if starting_biomass is not None:
            arrays['initial_biomass'] = starting_biomass
        return pair.simulate_scan_batch(arrays, runs, config)
    return community.simulate_batch(parameters, runs, config, starting_biomass)


def save_summary(final, kind, output):
    output.mkdir(parents=True, exist_ok=True)
    final.sort_values('run_id').to_csv(output / f'run_summary_{kind}.csv', index=False)


def save_growth_axis(parameters, config, output):
    manifest = pair.build_parameter_manifest(parameters).set_index('species')
    support = parameters.bt_dm_rate[parameters.bt_dm_rate > 0].index
    pool = config.dm_fraction * parameters.dm_y0.loc[support].sum()
    coefficients = pd.DataFrame(index=manifest.index)
    coefficients['dm_growth_slope'] = pool * manifest.r_ng_bt_production
    coefficients['bt_spent_growth_intercept'] = config.bt_spent_fraction * (
        parameters.nongrower_rate_cr * parameters.bt_spent_y0).sum(axis=1)
    coefficients['bt_initial_growth_h_inv'] = pool * manifest.bt_dm_rate
    coefficients['ratio_at_equal_potential_growth'] = (
        coefficients.bt_initial_growth_h_inv - coefficients.bt_spent_growth_intercept
    ) / coefficients.dm_growth_slope
    coefficients.to_csv(output / 'growth_rate_axis_parameters.csv')


def run_block(kind, parameters, original_runs, previous, start_cycle, end_cycle, workers, config):
    if previous is None:
        runs = original_runs.copy()
    else:
        ids = previous.loc[~previous.converged_last_n, 'run_id']
        runs = original_runs[original_runs.run_id.isin(ids)].copy()
    if runs.empty:
        return previous
    print(f'{kind}, cycles {start_cycle + 1}-{end_cycle}: {len(runs)} trajectories', flush=True)
    block_config = replace(config, max_cycles=end_cycle-start_cycle)
    grouping = 'species' if kind == 'pairwise' else 'ratio_index'
    previous_indexed = None if previous is None else previous.set_index('run_id')
    next_columns = (['next_start_ng', 'next_start_bt'] if kind == 'pairwise'
                    else [f'next_start_{s}' for s in community.LOCAL_SPECIES])
    finals = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []
        for _, subset in runs.groupby(grouping, sort=True):
            start = (None if previous is None else
                     previous_indexed.loc[subset.run_id, next_columns].to_numpy(dtype=float))
            futures.append(executor.submit(worker, kind, parameters, subset, block_config, start))
        for i, future in enumerate(as_completed(futures), 1):
            final = future.result()
            final['cycle'] += start_cycle
            finals.append(final)
            if i % 20 == 0 or i == len(futures):
                print(f'  completed {i}/{len(futures)} groups', flush=True)
    new = pd.concat(finals, ignore_index=True)
    if previous is not None:
        new = pd.concat([previous[~previous.run_id.isin(new.run_id)], new], ignore_index=True)
    return new.sort_values('run_id').reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-cycles', type=int, default=100)
    parser.add_argument('--output-dir', type=Path, default=HERE / 'simulation_results')
    args = parser.parse_args()
    if args.max_cycles < 20 or args.max_cycles % 20:
        parser.error('--max-cycles must be a multiple of 20, at least 20')
    if args.workers < 1:
        parser.error('--workers must be positive')
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('Choose a new or empty --output-dir. Existing scan results are preserved.')
    parameter_dir = ROOT / 'parameters'
    config = pair.ScanConfig()
    parameters = pair.load_frozen_parameters(parameter_dir=parameter_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        # Resolve this path relative to the directory containing the manifest.
        parameter_source=os.path.relpath(parameter_dir, start=output),
        extinction_modes=["hard_extinction"],
        lag_modes=["on"],
        config=asdict(config), extension_cap=args.max_cycles,
        extension_rule='Continue unconverged runs in 20-cycle blocks; retain converged endpoints.',
        completed_cycles={}, status='running',
    )
    def save_manifest():
        (output / 'input_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    save_manifest()
    ratios = pair.make_ratio_grid()
    pd.DataFrame({'ratio_index': np.arange(len(ratios)), 'ratio': ratios}).to_csv(
        output / 'ratio_grid.csv', index=False)
    save_growth_axis(parameters, config, output)
    run_tables = {'pairwise': pair.build_run_table(parameters, ratios),
                  'community': community.build_run_table(ratios)}
    latest = {kind: None for kind in run_tables}
    start = time.perf_counter()
    for end_cycle in range(20, args.max_cycles + 1, 20):
        for kind, runs in run_tables.items():
            latest[kind] = run_block(kind, parameters, runs, latest[kind], end_cycle-20,
                                     end_cycle, args.workers, config)
            save_summary(latest[kind], kind, output)
            manifest['completed_cycles'][kind] = end_cycle
            save_manifest()
        if all(frame.converged_last_n.all() for frame in latest.values()):
            break
    manifest['status'] = 'complete'
    save_manifest()
    print(f'Finished in {time.perf_counter()-start:.1f} s. Results: {output}', flush=True)


if __name__ == '__main__':
    main()
