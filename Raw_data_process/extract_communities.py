"""Normalize each included replicate over designed species, then average equally."""
from extraction_common import community_matrix

if __name__ == "__main__":
    community_matrix("dm_assemblies_mean_relative_abundance_matrix.csv", "Community_assembly/DM_assembly.xlsx", "dm")
    community_matrix("non-growers_in_Bt_spent_assemblies_mean_relative_abundance.csv", "Community_assembly/Non_grower_assembly_Bt_spent_medium.xlsx", "ng")
