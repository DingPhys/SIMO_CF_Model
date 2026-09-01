"""Fit all non-lag parameters, then run every prediction and error calculation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    run([sys.executable, str(ROOT / "scripts" / "fit_model.py")])
    run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_predictions.py"),
            "--overwrite",
        ]
    )


if __name__ == "__main__":
    main()
