# Consumer-resource model with cofactor competetion implemented

This folder contains data and code for reproducing the model fitting and prediction results related to the paper 'Cross-feeding among gut bacteria involves a single-input multi-output structure and cofactor competition'.

Run the commands below from this folder.

 
## Install necessary packages

```bash
python -m pip install numpy pandas scipy openpyxl
```
 
## Extract data from raw data

```bash
python -B Raw_data/run_all.py
```


## Run all fittings and predictions


```bash
python scripts/reproduce_all.py
```


## Run part of the model
Fit only:

```bash
python scripts/fit_model.py
```

Nongrower lag times are read from `data/fixed_lag01.csv`, not optimized by this step. They may need manual tuning if the hyperparameters change.

Predict only, using the parameter CSV files already present:

```bash
python scripts/run_predictions.py
```

The simulations in `bi-stability/` use the fitted parameters in `parameters/` and are run separately.



## Code map

```text
scripts/resource_fit_core.py   one shared log2-ratio loss and least-squares fit
scripts/bt_fit_core.py         Bt-spent resource fit, consumption rates, AUC, grower production
scripts/grower_fit_core.py     grower resource fit and consumption rates
scripts/cofactor_fit_core.py   cofactor related parameters fit
scripts/fit_model.py           fitting entry point
scripts/prediction_pipeline.py shared simulation and all three prediction branches: nongrowers in Bt, DM assemblies, different carbon sources, error calculation
scripts/run_predictions.py     prediction entry point
scripts/reproduce_all.py       fit followed by prediction
```

## Figure make
to make figures, first run prepare_figure_s2_data.py and prepare_figure4_data.py, then run FigureS2.R and Figure4.R in figure_plots/scripts

