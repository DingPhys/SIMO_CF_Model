# Extraction data from raw data curves

run_all.py contains codes for extracting all data

extract_growth.py first extracts OD from the growth curves and save the OD matrix including grower in DM, grower in grower spent medium, nongrowers in Bt spent, nongrower in nongrower-Bt double spent, the results are saved in ../data/ folder, indcluding: 16_grower_dm68_mean_growth.csv, grower_spent_mean_growth.csv, Nongrowers_monoculture_in_grower_spent.csv, Nongrowers_in_Bt_double_spent.csv

extract_communities calculates the relative abundance in each experiment, and avaerge over replications that are used and save the final results dm_assemblies_mean_relative_abundance_matrix.csv and non-growers_in_Bt_spent_assemblies_mean_relative_abundance.csv in ../data/ folder

extract_carbon.py extract the absolute abundance for each replication by OD x relative abundance, and the absolute abundance for each replication is saved as full_community_different_carbon_sources_absolute_abundance.csv in ../data/ folder


fit_bt_curves.py fit the growth curves and extract OD and growth rate and lag time for nongrowers in Bt spent, results are saved in Non-growers_In_Bt_Spent_growth_curve_fit_summary.csv
fit_dm_curves.py do the same thing for growers in DM, saved as 16_grower_dm68_curve_fit.csv
The functions are in growth_fit_common.py and extractions_common.py


