# Extraction data from raw data curves

`run_all.py` rebuilds every processed table in `../data/` from the raw
workbooks. Run `python -B run_all.py` from this directory; `--skip-fits`
rebuilds only the tables and keeps the existing curve-fit results.

Two scripts do the work:

- `extract_abundance.py` — 16S community tables:
  `dm_assemblies_mean_relative_abundance_matrix.csv`,
  `non-growers_in_Bt_spent_assemblies_mean_relative_abundance.csv`
  (each replicate normalized over its designed species, then averaged), and
  `full_community_different_carbon_sources_absolute_abundance.csv`
  (Final OD x relative abundance per replicate).
- `extract_growth.py` — growth tables from the OD curves
  (`16_grower_dm68_mean_growth.csv`, `grower_spent_mean_growth.csv`,
  `Nongrowers_monoculture_in_grower_spent.csv`,
  `Nongrowers_in_Bt_double_spent.csv`) and the growth-curve fits
  (`16_grower_dm68_curve_fit.csv`,
  `Non-growers_In_Bt_Spent_growth_curve_fit_summary.csv`).

