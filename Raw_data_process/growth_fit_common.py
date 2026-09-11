"""Shared lagged-logistic fit with fixed bounds and five deterministic search starts."""
import numpy as np
from scipy.optimize import differential_evolution, minimize

X0_MIN, X0_MAX = 0.001, 0.005
SEEDS = (17, 29, 41, 59, 71)


def minimize_loss(loss, bounds):
    """Search the expanded bounds with five fixed random seeds, followed by local refinement."""
    candidates = []
    for seed in SEEDS:
        global_fit = differential_evolution(loss, bounds, seed=seed, popsize=25,
            maxiter=1500, tol=1e-9, atol=1e-12, polish=False)
        local_fit = minimize(loss, global_fit.x, method="Nelder-Mead", bounds=bounds,
            options={"maxiter": 3000, "xatol": 1e-10, "fatol": 1e-13})
        candidates.extend([global_fit, local_fit])
    return min(candidates, key=lambda fit: fit.fun).x


def log2_error(predicted, observed):
    return np.mean(np.log2(np.maximum(predicted, .001) / np.maximum(observed, .001)) ** 2)


def fit_full_curve(time, od):
    net = od - od.min()
    k = net[-1]  # Fix the asymptotic plateau; do not force the fitted curve through the 48 h endpoint.

    def loss(parameters):
        x0, rate = np.exp(parameters[:2])
        lag = parameters[2]
        predicted = k / (1 + (k / x0 - 1) * np.exp(-rate * np.maximum(time - lag, 0)))
        return log2_error(predicted, net)

    bounds = [(np.log(X0_MIN), np.log(X0_MAX)), (np.log(1e-5), np.log(50)), (0, time[-1])]
    parameters = minimize_loss(loss, bounds)
    return float(np.exp(parameters[1])), float(parameters[2])
