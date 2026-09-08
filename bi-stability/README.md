# Bi-stability

In this folder we do scan of non-growers ultilizing DM resource growth rate and test if there are cases Bt can go extinction

## Run simulations
Activate `jnb-legacy`, then choose a new output directory. Simulations read directly from `../parameters/`:

```bash
python run_all.py --output-dir output_repeat --workers 4 --max-cycles 100
Rscript plot_scan.R output_repeat
Rscript plot_growth_threshold_bars.R output_repeat
```

## Replot saved results

From this folder:

```bash
Rscript plot_scan.R
Rscript plot_growth_threshold_bars.R
```
