"""Rebuild all processed tables in ../data/ from the raw workbooks.

Usage: python run_all.py [--skip-fits]
Extra arguments are passed to extract_growth.py (e.g. --skip-fits keeps the
existing curve-fit results and rebuilds the tables only).
Requires numpy, pandas, scipy and openpyxl. Paths are relative to this script.
"""
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main():
    for script in ('extract_abundance.py', 'extract_growth.py'):
        command = [sys.executable, '-B', str(HERE / script)]
        if script == 'extract_growth.py':
            command += sys.argv[1:]
        subprocess.run(command, check=True)
    print('Done. Tables saved to data/; raw workbooks are unchanged.')


if __name__ == '__main__':
    main()
