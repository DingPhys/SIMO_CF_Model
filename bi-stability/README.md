# Bi-stability

Scan nongrower consumption rates on DM resources and compare two initial NG:Bt ratios. Outputs describe Bt survival, endpoint differences and convergence; finite-time differences alone do not establish bistability.

## Run simulations
Activate `jnb-legacy`, then choose a new output directory. Simulations read directly from `../parameters/`:

```bash
python -B run_all.py --output-dir output_repeat --workers 4 --max-cycles 100
Rscript plot_scan.R output_repeat
Rscript plot_growth_threshold_bars.R output_repeat
```

## Replot saved results

From this folder:

```bash
Rscript plot_scan.R
Rscript plot_growth_threshold_bars.R
```
