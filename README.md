# A consumer-resource model implementing cross-feeding and cofactor competition

This folder contains the processed inputs and code needed to refit the
model and reproduce all predictions and errors. Lag times are the only fitted
quantities supplied as fixed values.



## Run

Edit `frozen_config.json` to change model or simulation settings, then run:

```bash
python scripts/reproduce_all.py
```

Python dependencies are pinned in `requirements.txt`.

Generated outputs are written to `parameters/` and `prediction_results/`. 
