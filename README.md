# Frozen consumer-resource model

This folder contains the processed inputs and minimum code needed to refit the
model and reproduce all predictions and errors. Lag times are the only fitted
quantities supplied as fixed values.

The two binary consumption matrices are processed model inputs. All continuous
resource abundances, consumption rates, production profiles, and cofactor
parameters are recalculated from the processed data.

## Run

Edit `frozen_config.json` to change model or simulation settings, then run:

```bash
python scripts/reproduce_all.py
```

Python dependencies are pinned in `requirements.txt`.

Generated outputs are written to `parameters/` and `prediction_results/`. The Bt
resource table contains only the final resource vector after the
linear-programming tie break.
