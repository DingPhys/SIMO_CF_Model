"""Rebuild OD-weighted abundance by carbon-source replicate, preserving source QC labels."""
from extraction_common import carbon_absolute

if __name__ == "__main__":
    carbon_absolute("full_community_different_carbon_sources_absolute_abundance.csv", "Community_assembly/Full_community_assembly_DM_across_carbon_sources.xlsx")
