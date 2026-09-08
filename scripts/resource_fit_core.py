"""Shared resource-fitting functions for growers and nongrowers."""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, linprog


def residuals_log2_ratio(theta, A, y, eps=1e-3, mask_min=1e-3):
    R = np.exp(theta)
    pred = A @ R

    pred_safe = np.clip(pred, eps, None)
    y_safe = np.clip(y, eps, None)

    valid = (pred_safe >= mask_min) & (y_safe >= mask_min)
    if not np.any(valid):
        return np.zeros_like(pred_safe)

    return np.log2(pred_safe[valid]) - np.log2(y_safe[valid])


def fit_resources(A, y, resource_names, eps=1e-3):
    """Fit positive resources with the same loss for both model branches."""

    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float)
    result = least_squares(
        residuals_log2_ratio,
        x0=np.zeros(A.shape[1]),
        args=(A, y, eps, eps),
        method="trf",
        max_nfev=100000,
    )
    resources = pd.Series(
        np.exp(result.x),
        index=resource_names,
        name="resource_y0",
    )
    return resources, result


def minimum_shared_resource_solution(A, resources, binary_support):
    """Bt-only LP tie-break that preserves the least-squares predictions."""

    prediction = A @ resources.to_numpy(dtype=float)
    shared = (np.asarray(binary_support).sum(axis=0) != 1).astype(float)
    result = linprog(
        c=shared,
        A_eq=A,
        b_eq=prediction,
        bounds=[(0.0, None)] * len(resources),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(result.message)
    return pd.Series(result.x, index=resources.index, name=resources.name)
