"""Bt resource fit, nongrower rates, monoculture AUC, and production profiles.

Only continuous quantities are fitted here; the binary consumption matrices
and all non-cofactor settings are fixed below.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, nnls

from resource_fit_core import fit_resources, minimum_shared_resource_solution

NONGROWER_SPECIES = (
    "Af", "Ao", "As", "Bl.s", "Col", "Cs", "Et", "Eu.c",
    "Eu.l", "Im", "Ld", "Lsp", "Mi", "Pc", "Va", "Vp",
)
COFACTOR_SPECIES = (
    "Col", "Cs", "Et", "Eu.c", "Eu.l", "Im", "Ld", "Lsp", "Mi", "Va",
)
GROWER_SPECIES = (
    "Ai", "Ac", "Bfi", "Bf", "Bt", "Bu", "Bx", "Ba",
    "Csp", "Dl", "Ls", "Mf", "Pm", "Bd", "Bv", "Rg",
)

# Fixed scientific and optimizer settings.
FIT_FLOOR = 1e-3
RECIPROCAL_ABS_LOG2 = 0.8
RECIPROCAL_ABS_DIFFERENCE = 0.2
MINIMUM_PAIR_GROWTH = 0.01
AUC_INITIAL_ABUNDANCE = 0.01
AUC_CYCLES = 10
AUC_HOURS = 48.0
AUC_DT = 0.01
AUC_DILUTION = 200.0
AUC_MU_THRESHOLD = 0.001
PRODUCTION_PRIOR = 0.3
PRODUCTION_CLIP_MIN = 0.001
PRODUCTION_PROFILE_FLOOR = 1e-12
PRODUCTION_DIRNAME = "prod_prior-0p3"
PRODUCTION_MULTI_START = 20
PRODUCTION_SEED = 2431
PRODUCTION_MAX_NFEV = 200000


def prepare_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def log2_fold_error(predicted, observed, floor: float):
    return np.log2(
        np.clip(predicted, floor, None) / np.clip(observed, floor, None)
    )


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Integrate with NumPy 1.x or 2.x."""

    function = getattr(np, "trapezoid", None) or np.trapz
    return float(function(y, x))


# ---------------------------------------------------------------------------
# Input tables
# ---------------------------------------------------------------------------

def load_binary_cr(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).set_index("species").loc[list(NONGROWER_SPECIES)]


def load_double_spent(data_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        data_dir / "Nongrowers_in_Bt_double_spent.csv"
    ).set_index("Species")
    frame = frame.loc[
        list(NONGROWER_SPECIES), list(NONGROWER_SPECIES) + ["Full_Bt", "Diluted_Bt"]
    ]
    frame = frame.apply(pd.to_numeric, errors="coerce").clip(lower=FIT_FLOOR)
    for species in NONGROWER_SPECIES:
        frame.loc[species, species] = np.nan
    return frame


def load_nongrower_growth_table(data_dir: Path) -> pd.DataFrame:
    return (
        pd.read_csv(data_dir / "Non-growers_In_Bt_Spent_growth_curve_fit_summary.csv")
        .set_index("focal_species")
        .loc[list(NONGROWER_SPECIES)]
    )


# ---------------------------------------------------------------------------
# Bt resource abundances (y0)
# ---------------------------------------------------------------------------

def reciprocal_screen(pair_observed: np.ndarray, full_observed: np.ndarray) -> np.ndarray:
    """Keep directed pairs whose two reciprocal totals are consistent.

    For focal i and donor j, compare FullBt_i + X(j in i-spent) against
    FullBt_j + X(i in j-spent); pairs that disagree too much are dropped.
    """

    reciprocal_ij = full_observed[:, None] + pair_observed.T
    reciprocal_ji = full_observed[None, :] + pair_observed
    with np.errstate(divide="ignore", invalid="ignore"):
        log2_ratio = np.log2(reciprocal_ij / reciprocal_ji)
    mask = (
        np.isfinite(pair_observed)
        & np.isfinite(log2_ratio)
        & (np.abs(log2_ratio) < RECIPROCAL_ABS_LOG2)
        & (pair_observed > MINIMUM_PAIR_GROWTH)
        & (np.abs(reciprocal_ij - reciprocal_ji) < RECIPROCAL_ABS_DIFFERENCE)
    )
    np.fill_diagonal(mask, False)
    return mask


def fit_bt_resources(binary_cr: pd.DataFrame, double_spent: pd.DataFrame) -> pd.Series:
    """Fit positive resource abundances, then LP tie-break toward exclusive mass."""

    sub = binary_cr.loc[list(COFACTOR_SPECIES)].to_numpy(dtype=float)
    pair = double_spent.loc[
        list(COFACTOR_SPECIES), list(COFACTOR_SPECIES)
    ].to_numpy(dtype=float)
    full = double_spent.loc[list(COFACTOR_SPECIES), "Full_Bt"].to_numpy(dtype=float)
    mask = reciprocal_screen(pair, full)

    # Design rows: pair rows explain i's growth in j-spent by the resources
    # i can use that j cannot; full rows use i's complete resource support.
    rows, observed = [], []
    for i in range(len(COFACTOR_SPECIES)):
        for j in range(len(COFACTOR_SPECIES)):
            if mask[i, j]:
                rows.append(sub[i] * (1.0 - sub[j]))
                observed.append(pair[i, j])
        rows.append(sub[i].copy())
        observed.append(full[i])
    design = np.vstack(rows)
    observed = np.asarray(observed, dtype=float)

    y0, _ = fit_resources(design, observed, binary_cr.columns, eps=FIT_FLOOR)
    y0.name = "bt_resource_y0"
    return minimum_shared_resource_solution(design, y0, sub)


def scale_growth_rates(
    binary_cr: pd.DataFrame, y0: pd.Series, growth_table: pd.DataFrame
) -> pd.DataFrame:
    """Scale each cofactor species' binary row to match its measured growth rate."""

    rate = binary_cr.astype(float).copy()
    y = y0.loc[binary_cr.columns].to_numpy(dtype=float)
    for species in COFACTOR_SPECIES:
        measured = float(growth_table.loc[species, "growth_rate"])
        if not np.isfinite(measured) or measured <= 0:
            raise ValueError(f"Invalid fitted growth rate for {species}: {measured}")
        access = float(binary_cr.loc[species].to_numpy(dtype=float) @ y)
        if access <= 0:
            raise ValueError(f"Cannot scale {species}; zero Bt resource access.")
        rate.loc[species] = binary_cr.loc[species] * (measured / access)
    rate.index.name = "species"
    return rate


# ---------------------------------------------------------------------------
# Monoculture AUC (used later to derive the cofactor uptake rate D)
# ---------------------------------------------------------------------------

def _resource_rhs(state: np.ndarray, rate_cr: np.ndarray) -> np.ndarray:
    n_species = rate_cr.shape[0]
    biomass, resources = state[:n_species], state[n_species:]
    usable = np.maximum(resources, 0.0)
    return np.concatenate(
        [biomass * (rate_cr @ usable), -usable * (rate_cr.T @ biomass)]
    )


def _rk4_step(state: np.ndarray, dt: float, rate_cr: np.ndarray) -> np.ndarray:
    k1 = _resource_rhs(state, rate_cr)
    k2 = _resource_rhs(state + 0.5 * dt * k1, rate_cr)
    k3 = _resource_rhs(state + 0.5 * dt * k2, rate_cr)
    k4 = _resource_rhs(state + dt * k3, rate_cr)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _simulate_one_cycle(
    rate_cr, initial_biomass, y0, hours, dt, store=False
):
    state = np.concatenate([initial_biomass, y0]).astype(float)
    time = 0.0
    times, states = [0.0], [state.copy()]
    while time < hours - 1e-15:
        step = min(dt, hours - time)
        state = np.maximum(_rk4_step(state, step, rate_cr), 0.0)
        time += step
        if store:
            times.append(time)
            states.append(state.copy())
    if store:
        return state, np.asarray(times), np.asarray(states)
    return state, None, None


def _auc_until_mu_threshold(times, biomass, resources, binary_support, threshold):
    """AUC of the biomass curve until accessible resources drop to threshold."""

    mu = resources @ binary_support
    below = np.flatnonzero(mu <= threshold)
    if below.size == 0 or below[0] == 0:
        cut = len(times) - 1 if below.size == 0 else 0
        return float(_trapezoid(biomass[: cut + 1], times[: cut + 1]))
    i = int(below[0])
    t0, t1, m0, m1 = times[i - 1], times[i], mu[i - 1], mu[i]
    alpha = float(np.clip((m0 - threshold) / (m0 - m1), 0.0, 1.0)) if m0 != m1 else 1.0
    t_cut = t0 + alpha * (t1 - t0)
    x_cut = biomass[i - 1] + alpha * (biomass[i] - biomass[i - 1])
    return float(_trapezoid(np.r_[biomass[:i], x_cut], np.r_[times[:i], t_cut]))


def monoculture_auc(
    binary_cr: pd.DataFrame, rate_cr: pd.DataFrame, y0: pd.Series
) -> pd.DataFrame:
    """Simulate each cofactor species alone and integrate its final-cycle AUC."""

    selected = list(COFACTOR_SPECIES)
    sub = binary_cr.loc[selected].to_numpy(dtype=float)
    rate = rate_cr.loc[selected].to_numpy(dtype=float)
    y = y0.loc[binary_cr.columns].to_numpy(dtype=float)
    records = []
    for i, species in enumerate(selected):
        initial = np.zeros(len(selected))
        initial[i] = AUC_INITIAL_ABUNDANCE
        for cycle in range(AUC_CYCLES):
            state, times, states = _simulate_one_cycle(
                rate, initial, y,
                AUC_HOURS, AUC_DT,
                store=(cycle == AUC_CYCLES - 1),
            )
            initial = state[: len(selected)] / AUC_DILUTION
        auc = _auc_until_mu_threshold(
            times,
            states[:, i],
            states[:, len(selected):],
            sub[i],
            AUC_MU_THRESHOLD,
        )
        if not np.isfinite(auc) or auc <= 0:
            raise ValueError(f"Fixed AUC is invalid for {species}; cannot derive D.")
        records.append({"species": species, "auc": auc})
    return pd.DataFrame(records)


def fit_bt_base(
    data_dir: Path,
    binary_cr_path: Path,
    output_dir: Path,
) -> dict:
    """Fit the Bt base model and write nongrower_R / bt_resource_Y0 / AUC."""

    prepare_output_dir(output_dir)
    binary_cr = load_binary_cr(binary_cr_path)
    double_spent = load_double_spent(data_dir)
    growth_table = load_nongrower_growth_table(data_dir)
    y0 = fit_bt_resources(binary_cr, double_spent)
    rate_cr = scale_growth_rates(binary_cr, y0, growth_table)
    auc = monoculture_auc(binary_cr, rate_cr, y0)

    rate_cr.to_csv(output_dir / "nongrower_R.csv")
    y0.rename_axis("resource_id").reset_index().to_csv(
        output_dir / "bt_resource_Y0.csv", index=False
    )
    auc.to_csv(output_dir / "monoculture_AUC.csv", index=False)
    return {"binary": binary_cr, "rate": rate_cr, "y0": y0, "auc": auc}


# ---------------------------------------------------------------------------
# Grower production profiles
# ---------------------------------------------------------------------------

def _production_upper_bound(observed, grower_growth, reference, clip_min):
    valid = np.isfinite(observed)
    observed_scale = (
        float(np.max(np.clip(observed[valid], clip_min, None))) / grower_growth
        if np.any(valid)
        else 0.0
    )
    reference_scale = float(np.max(np.clip(reference, 0.0, None)))
    return max(10.0 * max(observed_scale, reference_scale, clip_min), 1.0)


def _fit_one_production_profile(
    cr, observed, grower_growth, reference, prior_weight, clip_min, profile_floor, seed
):
    """Fit one grower's profile as log2 ratios to the Bt reference profile."""

    upper = _production_upper_bound(observed, grower_growth, reference, clip_min)
    # Resources absent from the Bt reference stay absent here. Removing those
    # coordinates also avoids the huge gradient of log2((P+floor)/floor) at P=0.
    fitted = reference > 0.0
    reference_fitted = reference[fitted]
    lower_z = np.log2(profile_floor / reference_fitted)
    upper_z = np.log2(upper / reference_fitted)

    def profile_from_z(z):
        profile = np.zeros_like(reference)
        profile[fitted] = reference_fitted * np.exp2(z)
        return profile

    def residual_vector(z):
        predicted = grower_growth * (cr @ profile_from_z(z))
        residual = log2_fold_error(predicted, observed, clip_min)
        pieces = [residual[np.isfinite(residual)]]
        if prior_weight > 0.0:
            pieces.append(np.sqrt(prior_weight) * z)
        return np.concatenate(pieces)

    starts = [np.zeros(reference_fitted.size)]
    valid = np.isfinite(observed)
    if np.any(valid):
        target = np.clip(observed[valid], clip_min, None) / grower_growth
        lsq, _ = nnls(cr[valid][:, fitted], target)
        lsq = np.clip(lsq, profile_floor, upper)
        midpoint = np.clip(0.5 * (lsq + reference_fitted), profile_floor, upper)
        starts.extend(
            [np.log2(lsq / reference_fitted), np.log2(midpoint / reference_fitted)]
        )
    rng = np.random.default_rng(seed)
    base = starts[-1]
    while len(starts) < PRODUCTION_MULTI_START:
        starts.append(
            np.clip(base + rng.normal(0.0, 1.0, size=base.shape), lower_z, upper_z)
        )

    best = None
    for start in starts[:PRODUCTION_MULTI_START]:
        result = least_squares(
            residual_vector,
            np.clip(start, lower_z, upper_z),
            bounds=(lower_z, upper_z),
            method="trf",
            max_nfev=PRODUCTION_MAX_NFEV,
        )
        objective = float(np.sum(result.fun**2))
        if best is None or objective < best[0]:
            best = (objective, result)
    return profile_from_z(best[1].x), bool(best[1].success)


def fit_production(data_dir: Path, bt: dict, output_root: Path) -> Path:
    """Fit one production profile per grower; Ai and Bt stay on the Bt profile."""

    observed = (
        pd.read_csv(data_dir / "Nongrowers_monoculture_in_grower_spent.csv")
        .set_index("Species")
        .loc[list(NONGROWER_SPECIES), list(GROWER_SPECIES)]
        .apply(pd.to_numeric, errors="coerce")
    )
    growth = pd.to_numeric(
        pd.read_csv(data_dir / "16_grower_dm68_mean_growth.csv")
        .set_index("species")["mean_growth"],
        errors="coerce",
    ).loc[list(GROWER_SPECIES)]
    if growth.isna().any() or (growth <= 0).any():
        raise ValueError("Grower monoculture growth contains invalid values.")
    bt_profile = (
        bt["y0"].loc[bt["binary"].columns].to_numpy(dtype=float) / float(growth.loc["Bt"])
    )
    cr = bt["binary"].loc[list(NONGROWER_SPECIES)].to_numpy(dtype=float)

    branch_dir = output_root / PRODUCTION_DIRNAME
    prepare_output_dir(branch_dir)
    profiles = {}
    for k, grower in enumerate(GROWER_SPECIES):
        if grower in ("Ai", "Bt"):
            profiles[grower] = bt_profile.copy()
            continue
        profile, _ = _fit_one_production_profile(
            cr,
            observed[grower].to_numpy(dtype=float),
            float(growth.loc[grower]),
            bt_profile,
            PRODUCTION_PRIOR,
            PRODUCTION_CLIP_MIN,
            PRODUCTION_PROFILE_FLOOR,
            seed=PRODUCTION_SEED + k,
        )
        profiles[grower] = profile

    frame = pd.DataFrame(profiles, index=bt["binary"].columns)
    frame.index.name = "resource_id"
    frame.to_csv(branch_dir / "production_profiles.csv")
    return branch_dir
