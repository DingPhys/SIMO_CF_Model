"""Coarse-to-fine grid fitting of growth curves."""
import numpy as np

X0_MIN, X0_MAX = 0.001, 0.005


def grid_minimum(evaluate, axes):
    grids = np.meshgrid(*axes, indexing='ij')
    points = np.stack(grids, axis=-1).reshape(-1, len(axes))
    best = (np.inf, None)
    for offset in range(0, len(points), 1024):
        batch = points[offset:offset + 1024]
        values = evaluate(batch)
        i = np.argmin(values)
        if values[i] < best[0]:
            best = (float(values[i]), batch[i].copy())
    return best


def minimize_grid(loss, bounds, sizes):
    """Search a coarse grid, an 81-point fine grid, then refine locally."""
    coarse_axes = [np.linspace(low, high, n) for (low, high), n in zip(bounds, sizes)]
    best_loss, best = grid_minimum(loss, coarse_axes)

    # Search between the two coarse neighbors of each winning coordinate.
    fine_axes = []
    widths = []
    for value, axis in zip(best, coarse_axes):
        index = np.argmin(abs(axis - value))
        low = axis[max(0, index - 1)]
        high = axis[min(len(axis) - 1, index + 1)]
        fine_axes.append(np.linspace(low, high, 81))
        widths.append((high - low) / 2)
    best_loss, best = grid_minimum(loss, fine_axes)

    widths = np.asarray(widths)
    full_ranges = np.array([high - low for low, high in bounds])
    for _ in range(40):
        local_axes = []
        for center, width, (low, high) in zip(best, widths, bounds):
            left = max(low, center - width)
            right = min(high, center + width)
            axis = np.linspace(left, right, 21)
            # Retain the previous best so a new grid cannot discard it.
            local_axes.append(np.unique(np.append(axis, center)))

        new_loss, new_best = grid_minimum(loss, local_axes)
        near_edge = False
        for value, axis, (low, high) in zip(new_best, local_axes, bounds):
            near_left = value <= axis[1] and axis[0] > low
            near_right = value >= axis[-2] and axis[-1] < high
            if near_left or near_right:
                near_edge = True

        if new_loss < best_loss:
            best_loss, best = new_loss, new_best
        # At an interior window edge, move the center without shrinking.
        if not near_edge:
            widths *= 0.5
        if np.max(widths / full_ranges) < 1e-9:
            break
    return best


def log2_error(predicted, observed):
    ratio = np.maximum(predicted, .001) / np.maximum(observed, .001)
    return np.mean(np.log2(ratio) ** 2, axis=-1)


def fit_full_curve(time, od):
    net = od - od.min()
    k = net[-1]  # Fixed asymptotic plateau, not a forced endpoint.

    def loss(parameters):
        # Each row is one candidate: log(X0), log(rate), lag.
        x0 = np.exp(parameters[:, 0, None])
        rate = np.exp(parameters[:, 1, None])
        lag = parameters[:, 2, None]
        shifted_time = np.maximum(time[None, :] - lag, 0)
        predicted = k / (1 + (k / x0 - 1) * np.exp(-rate * shifted_time))
        return log2_error(predicted, net)

    bounds = [(np.log(X0_MIN), np.log(X0_MAX)), (np.log(1e-5), np.log(50)), (0, time[-1])]
    parameters = minimize_grid(loss, bounds, [21, 241, 193])
    return float(np.exp(parameters[1])), float(parameters[2])
