"""Extract growth yields; uses relative paths anchored to this script directory."""
from extraction_common import G, NG, OUT, growth_matrix, grower_dm

if __name__ == "__main__":
    mono = "Growth_data/Non_growers_spent_medium_of_growers.xlsx"
    growth_matrix("Nongrowers_in_Bt_double_spent.csv",
        ["Growth_data/Double_spent_medium.xlsx", mono], NG, NG+["Full_Bt","Diluted_Bt"], "double")
    # The complete workbook supplies all 16 grower conditioners.
    growth_matrix("Nongrowers_monoculture_in_grower_spent.csv", [mono], NG, G, "nongrower_spent")
    grower_dm("16_grower_dm68_mean_growth.csv", "Growth_data/Growth_curve_32_strains_monoculture_DM.xlsx")
    growth_matrix("grower_spent_mean_growth.csv", ["Growth_data/Growers_pairwise_spent_medium.xlsx"], G, G, "grower_spent")
