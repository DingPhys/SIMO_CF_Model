"""Fit the model, then run DM and carbon-source predictions."""

from fit_model import fit_model
from run_predictions import main as predict


def main():
    fit_model()
    predict()


if __name__ == "__main__":
    main()
