"""Process raw data files to all data used for the model, processed data will be put in ../data/ folder
Requires numpy, pandas, scipy and openpyxl. Paths are relative to this script.
"""
from pathlib import Path
import argparse
import csv
import subprocess
import sys
sys.dont_write_bytecode = True
from output_schema import SCHEMAS

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-fits", action="store_true", help="Keep existing rate/lag tables and rebuild yields/abundances only")
    args = parser.parse_args()
    steps = ["extract_growth.py", "extract_communities.py", "extract_carbon.py"]
    if not args.skip_fits:
        steps += ["fit_bt_curves.py", "fit_dm_curves.py"]
    for script in steps:
        subprocess.run([sys.executable, "-B", str(HERE / script)], check=True)
    for name, schema in SCHEMAS.items():
        path = HERE.parent / "data" / name
        if args.skip_fits and not path.exists():
            print(f"Not rebuilt: {name}; run without --skip-fits to generate it.")
            continue
        with path.open() as handle:
            reader = csv.DictReader(handle)
            assert reader.fieldnames == schema["columns"], f"Column mismatch: {name}"
            assert [row[schema["row_key"]] for row in reader] == schema["rows"], f"Row mismatch: {name}"
    print("Done. Tables saved to data/; raw workbooks and manual/binary inputs are unchanged.")


if __name__ == "__main__":
    main()
