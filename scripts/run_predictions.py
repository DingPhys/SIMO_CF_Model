"""Run Bt-spent, DM, and carbon-source predictions."""

from pathlib import Path

from prediction_pipeline import run_predictions


ROOT = Path(__file__).resolve().parents[1]


def main():
    run_predictions(ROOT)


if __name__ == "__main__":
    main()
