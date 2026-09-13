# Bi-stability

Scan nongrower consumption rates on DM resources and compare two initial NG:Bt ratios, with lag on. Only endpoint deletion is used: species below 1e-4 at the culture endpoint are removed before the next cycle.

## Run simulations

Parameters are read from `../parameters/`:

```bash
python -B run_all.py --output-dir simulation_results --workers 2 --max-cycles 100
Rscript plot_scan.R simulation_results
Rscript plot_growth_threshold_bars.R simulation_results
```

Unconverged trajectories continue in 20-cycle blocks up to the requested limit. Only the latest endpoints are saved.

## Saved results

`simulation_results/` contains:

- `run_summary_pairwise.csv`: Bt plus one nongrower, for each of ten nongrowers.
- `run_summary_community.csv`: Bt plus all ten nongrowers.
- `ratio_grid.csv`, `growth_rate_axis_parameters.csv`, and `input_manifest.json`: scan settings and plotting inputs. The parameter source is relative to the manifest directory.
- `figures/`: `bt_outcomes_lag_on` and `nongrowers_Bt_extinction`, each as SVG/PDF.


## Replot saved results

```bash
Rscript plot_scan.R
Rscript plot_growth_threshold_bars.R
```
