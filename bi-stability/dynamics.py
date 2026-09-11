"""Resource/production/stored-cofactor equations
"""
import numpy as np

def _rhs_pairwise_production_batch(
    time: float,
    biomass: np.ndarray,
    resources: np.ndarray,
    environmental_cofactor: np.ndarray,
    stored_cofactor: np.ndarray,
    rate_cr: np.ndarray,
    resource_production: np.ndarray,
    D: np.ndarray,
    F: np.ndarray,
    lag: np.ndarray,
    cofactor_required: np.ndarray,
    dt: float,
    uptake_resource_sum_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Derivative for batches whose two local species differ by pair."""

    active = (time >= lag).astype(float)
    usable_resources = np.maximum(resources, 0.0)
    growth_rate = np.einsum("br,bsr->bs", usable_resources, rate_cr)
    accessible_resource_sum = np.einsum(
        "br,bsr->bs", usable_resources, rate_cr > 0.0
    )
    cofactor_gate = np.where(
        cofactor_required,
        stored_cofactor > 0.0,
        True,
    ).astype(float)
    growth_gate = active * cofactor_gate
    raw_growth = biomass * growth_gate * growth_rate
    positive_growth = np.maximum(raw_growth, 0.0)

    growth_limit = np.full_like(raw_growth, np.inf)
    limited = cofactor_required & (F > 0.0)
    growth_limit[limited] = (
        np.maximum(stored_cofactor[limited], 0.0) / (F[limited] * dt)
    )
    limited_positive = np.minimum(positive_growth, growth_limit)
    biomass_change = np.where(raw_growth > 0.0, limited_positive, raw_growth)
    growth_scale = np.ones_like(raw_growth)
    growing = positive_growth > 0.0
    growth_scale[growing] = limited_positive[growing] / positive_growth[growing]

    species_activity = biomass * growth_gate * growth_scale
    resource_sink = np.einsum("bs,bsr->br", species_activity, rate_cr)
    resource_source = np.einsum(
        "bs,bsr->br", biomass_change, resource_production
    )
    resource_change = -usable_resources * resource_sink + resource_source

    uptake_gate = (
        active
        * cofactor_required.astype(float)
        * (accessible_resource_sum > uptake_resource_sum_threshold).astype(float)
    )
    uptake = (
        D
        * np.maximum(environmental_cofactor, 0.0)[:, None]
        * biomass
        * uptake_gate
    )
    environmental_change = -np.sum(uptake, axis=1)
    stored_use = (
        F
        * cofactor_required.astype(float)
        * np.maximum(biomass_change, 0.0)
    )
    stored_change = uptake - stored_use
    return biomass_change, resource_change, environmental_change, stored_change


def _rk4_pairwise_production_batch(
    time: float,
    biomass: np.ndarray,
    resources: np.ndarray,
    environmental_cofactor: np.ndarray,
    stored_cofactor: np.ndarray,
    dt: float,
    rate_cr: np.ndarray,
    resource_production: np.ndarray,
    D: np.ndarray,
    F: np.ndarray,
    lag: np.ndarray,
    cofactor_required: np.ndarray,
    uptake_resource_sum_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    args = (
        rate_cr,
        resource_production,
        D,
        F,
        lag,
        cofactor_required,
        dt,
        uptake_resource_sum_threshold,
    )
    k1 = _rhs_pairwise_production_batch(
        time,
        biomass,
        resources,
        environmental_cofactor,
        stored_cofactor,
        *args,
    )
    k2 = _rhs_pairwise_production_batch(
        time + 0.5 * dt,
        biomass + 0.5 * dt * k1[0],
        resources + 0.5 * dt * k1[1],
        environmental_cofactor + 0.5 * dt * k1[2],
        stored_cofactor + 0.5 * dt * k1[3],
        *args,
    )
    k3 = _rhs_pairwise_production_batch(
        time + 0.5 * dt,
        biomass + 0.5 * dt * k2[0],
        resources + 0.5 * dt * k2[1],
        environmental_cofactor + 0.5 * dt * k2[2],
        stored_cofactor + 0.5 * dt * k2[3],
        *args,
    )
    k4 = _rhs_pairwise_production_batch(
        time + dt,
        biomass + dt * k3[0],
        resources + dt * k3[1],
        environmental_cofactor + dt * k3[2],
        stored_cofactor + dt * k3[3],
        *args,
    )
    return tuple(
        state + (dt / 6.0) * (s1 + 2.0 * s2 + 2.0 * s3 + s4)
        for state, s1, s2, s3, s4 in zip(
            (biomass, resources, environmental_cofactor, stored_cofactor),
            k1,
            k2,
            k3,
            k4,
        )
    )

