"""Minimal predictions for Bt-spent, DM, and carbon-source communities."""

import shutil

import numpy as np
import pandas as pd


GROWERS = (
    "Ai", "Ac", "Bfi", "Bf", "Bt", "Bu", "Bx", "Ba",
    "Csp", "Dl", "Ls", "Mf", "Pm", "Bd", "Bv", "Rg",
)
NONGROWERS = (
    "Af", "Ao", "As", "Bl.s", "Col", "Cs", "Et", "Eu.c",
    "Eu.l", "Im", "Ld", "Lsp", "Mi", "Pc", "Va", "Vp",
)
SPECIES = GROWERS + NONGROWERS

# Fixed simulation and scoring settings.
CYCLES = 10
HOURS_PER_CYCLE = 48.0
DT = 0.04
DILUTION = 200.0
INITIAL_ABUNDANCE = 0.001
PREDICTION_FLOOR = 0.001
UPTAKE_RESOURCE_THRESHOLD = 0.001
ADDITIONAL_NONGROWER_LAG = 4.0
CARBON_QC_EXCLUSIONS = ("high other species", "low growth", "not used", "contamination")
RESULTS_DIRNAME = "prediction_results"


def rhs(time, X, Y, C, Q, R, P, D, F, lag, needs_cofactor, dt, threshold):
    """Consumer-resource model for a batch of communities."""

    active = (time >= lag)[None, :]
    usable_Y = np.maximum(Y, 0.0)
    growth_rate = usable_Y @ R.T
    accessible_Y = usable_Y @ (R > 0).T
    gate = active * np.where(needs_cofactor, Q > 0.0, True)

    raw_growth = X * gate * growth_rate
    positive_growth = np.maximum(raw_growth, 0.0)
    growth_limit = np.full_like(raw_growth, np.inf)
    limited = needs_cofactor & (F > 0.0)
    growth_limit[:, limited] = np.maximum(Q[:, limited], 0.0) / (F[limited] * dt)
    limited_growth = np.minimum(positive_growth, growth_limit)
    dX = np.where(raw_growth > 0.0, limited_growth, raw_growth)

    growth_scale = np.ones_like(raw_growth)
    growing = positive_growth > 0.0
    growth_scale[growing] = limited_growth[growing] / positive_growth[growing]
    activity = X * gate * growth_scale
    dY = -usable_Y * (activity @ R) + dX @ P

    uptake_gate = active * needs_cofactor * (accessible_Y > threshold)
    uptake = D[None, :] * np.maximum(C, 0.0)[:, None] * X * uptake_gate
    dC = -uptake.sum(axis=1)
    dQ = uptake - F[None, :] * needs_cofactor * np.maximum(dX, 0.0)
    return dX, dY, dC, dQ


def rk4(time, X, Y, C, Q, dt, R, P, D, F, lag, needs_cofactor, threshold):
    states = (X, Y, C, Q)
    args = (R, P, D, F, lag, needs_cofactor, dt, threshold)
    k1 = rhs(time, *states, *args)
    k2 = rhs(time + dt / 2, *(x + dt * k / 2 for x, k in zip(states, k1)), *args)
    k3 = rhs(time + dt / 2, *(x + dt * k / 2 for x, k in zip(states, k2)), *args)
    k4 = rhs(time + dt, *(x + dt * k for x, k in zip(states, k3)), *args)
    return tuple(
        x + dt * (a + 2 * b + 2 * c + d) / 6
        for x, a, b, c, d in zip(states, k1, k2, k3, k4)
    )


def simulate(R, P, fresh_Y, X0, D, F, lag, needs_cofactor):
    """Run repeated 48-hour growth and 200-fold dilution cycles."""

    X = np.asarray(X0, dtype=float).copy()
    fresh_Y = np.broadcast_to(fresh_Y, (len(X), R.shape[1])).copy()
    args = (R, P, D, F, lag, needs_cofactor, UPTAKE_RESOURCE_THRESHOLD)

    for _ in range(CYCLES):
        Y = fresh_Y.copy()
        C = np.ones(len(X))
        Q = np.zeros_like(X)
        time = 0.0
        while time < HOURS_PER_CYCLE - 1e-15:
            dt = min(DT, HOURS_PER_CYCLE - time)
            X, Y, C, Q = rk4(time, X, Y, C, Q, dt, *args)
            X, Y, C, Q = (np.maximum(value, 0.0) for value in (X, Y, C, Q))
            time += dt
        final_X = X.copy()
        X = final_X / DILUTION
    return final_X


def load_model(root):
    parameters = root / "parameters"
    grower_R = pd.read_csv(
        parameters / "grower" / "model" / "grower_R.csv", index_col="species"
    ).loc[list(GROWERS)]
    nongrower_R = pd.read_csv(
        parameters / "bt_base" / "nongrower_R.csv", index_col="species"
    ).loc[list(NONGROWERS)]
    dm_Y = (
        pd.read_csv(parameters / "grower" / "model" / "dm_resource_y0.csv")
        .set_index("resource_id")["dm_y0"]
        .reindex(grower_R.columns)
    )
    cofactor = (
        pd.read_csv(parameters / "final_parameters" / "cofactor_and_lag.csv")
        .set_index("species")
        .loc[list(NONGROWERS)]
    )
    production = (
        pd.read_csv(
            parameters / "production_fits" / "prod_prior-0p3" / "production_profiles.csv"
        )
        .set_index("resource_id")
        .loc[nongrower_R.columns, list(GROWERS)]
    )
    return grower_R, nongrower_R, dm_Y, cofactor, production


def full_dm_model(root):
    grower_R, nongrower_R, dm_Y, cofactor, production = load_model(root)
    ng_start = grower_R.shape[1]
    R = np.zeros((32, grower_R.shape[1] + nongrower_R.shape[1]))
    R[:16, :ng_start] = grower_R
    R[16:, ng_start:] = nongrower_R

    # The retained DM model lets every nongrower use the grower universal niche.
    positive = np.where(nongrower_R > 0, nongrower_R, np.nan)
    universal_rate = np.nanmax(positive, axis=1)
    R[16:, grower_R.columns.get_loc("F_All_Shared")] = universal_rate

    P = np.zeros_like(R)
    P[:16, ng_start:] = production.to_numpy().T
    fresh_Y = np.r_[dm_Y.to_numpy(), np.zeros(nongrower_R.shape[1])]
    D = np.r_[np.zeros(16), cofactor["D"].to_numpy()]
    F = np.r_[np.zeros(16), cofactor["F"].to_numpy()]
    lag = np.r_[
        np.zeros(16),
        cofactor["selected_lag_h"].to_numpy() + ADDITIONAL_NONGROWER_LAG,
    ]
    return R, P, fresh_Y, D, F, lag, production, nongrower_R


def relative_abundance(values, presence=None):
    values = np.asarray(values, dtype=float)
    if presence is not None:
        values = np.where(presence, values, 0.0)
    total = values.sum(axis=1, keepdims=True)
    relative = np.divide(values, total, out=np.zeros_like(values), where=total > 0)
    if presence is not None:
        relative = np.where(presence, relative, np.nan)
    return relative


def merge_eu(values, presence, species=SPECIES):
    values = np.asarray(values, dtype=float).copy()
    euc, eul = species.index("Eu.c"), species.index("Eu.l")
    both = presence[:, euc] & presence[:, eul]
    values[both, euc] += values[both, eul]
    values[both, eul] = 0.0
    return values


def log2_error(predicted, observed, floor):
    return np.log2(np.clip(predicted, floor, None) / np.clip(observed, floor, None))


def save_species_matrix(path, values, names, species=SPECIES):
    frame = pd.DataFrame(values.T, index=species, columns=names)
    frame.index.name = "species"
    frame.to_csv(path)


def reset_output(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def bt_assembly_type(name, richness):
    if name == "Full_community":
        return "full_community"
    if name.endswith("_dropout"):
        return "dropout"
    if richness == 2:
        return "pairwise"
    return "higher_order"


def predict_nongrowers_in_bt(root):
    """Predict curated nongrower assemblies in completed Bt spent medium."""

    output = root / RESULTS_DIRNAME / "nongrowers_in_bt_spent"
    reset_output(output)
    observed_frame = (
        pd.read_csv(
            root / "data" / "non-growers_in_Bt_spent_assemblies_mean_relative_abundance.csv"
        )
        .set_index("species")
        .loc[list(NONGROWERS)]
    )
    names = observed_frame.columns.tolist()
    observed_values = observed_frame.T.to_numpy(dtype=float)
    presence = np.isfinite(observed_values)
    observed_values = np.where(presence, observed_values, 0.0)
    observed_values = merge_eu(observed_values, presence, NONGROWERS)
    observed = relative_abundance(observed_values, presence)
    save_species_matrix(
        output / "observed_relative.csv", observed, names, NONGROWERS
    )

    _, nongrower_R, _, cofactor, _ = load_model(root)
    bt_y0 = (
        pd.read_csv(root / "parameters" / "bt_base" / "bt_resource_Y0.csv")
        .set_index("resource_id")["bt_resource_y0"]
        .reindex(nongrower_R.columns)
        .to_numpy(dtype=float)
    )
    R = nongrower_R.to_numpy(dtype=float)
    P = np.zeros_like(R)
    X0 = presence.astype(float) * INITIAL_ABUNDANCE
    rows = []
    for cofactor_on in (True, False):
        branch = output / ("cofactor_on" if cofactor_on else "cofactor_off")
        branch.mkdir()
        needs = np.full(len(NONGROWERS), cofactor_on)
        absolute = simulate(
            R,
            P,
            bt_y0,
            X0,
            cofactor["D"].to_numpy(),
            cofactor["F"].to_numpy(),
            cofactor["selected_lag_h"].to_numpy(),
            needs,
        )
        mapped_absolute = merge_eu(absolute, presence, NONGROWERS)
        relative = relative_abundance(mapped_absolute, presence)
        error = np.full_like(relative, np.nan)
        error[presence] = log2_error(
            relative[presence], observed[presence], PREDICTION_FLOOR
        )

        save_species_matrix(
            branch / "predicted_absolute_native.csv",
            np.where(presence, absolute, np.nan),
            names,
            NONGROWERS,
        )
        save_species_matrix(
            branch / "predicted_relative.csv", relative, names, NONGROWERS
        )
        save_species_matrix(
            branch / "signed_log2_error.csv", error, names, NONGROWERS
        )

        n_scored = np.sum(~np.isnan(error), axis=1)
        scored_communities = n_scored > 0
        # Retain missing communities as NaN without averaging empty rows.
        community_mae = np.divide(
            np.nansum(np.abs(error), axis=1), n_scored,
            out=np.full(len(names), np.nan), where=scored_communities,
        )
        community_rmse = np.sqrt(np.divide(
            np.nansum(error**2, axis=1), n_scored,
            out=np.full(len(names), np.nan), where=scored_communities,
        ))
        richness = presence.sum(axis=1)
        pd.DataFrame(
            {
                "community": names,
                "community_type": [
                    bt_assembly_type(name, count)
                    for name, count in zip(names, richness)
                ],
                "n_presented": richness,
                "mean_abs_log2_error": community_mae,
                "rmse_log2_error": community_rmse,
            }
        ).to_csv(branch / "community_metrics.csv", index=False)

        species_rows = []
        for index, species in enumerate(NONGROWERS):
            values = error[:, index]
            values = values[np.isfinite(values)]
            species_rows.append(
                {
                    "species": species,
                    "n_scored": len(values),
                    "mean_abs_log2_error": np.mean(np.abs(values)),
                    "rmse_log2_error": np.sqrt(np.mean(values**2)),
                }
            )
        pd.DataFrame(species_rows).to_csv(
            branch / "species_metrics.csv", index=False
        )

        complex_assemblies = (richness > 2) & scored_communities
        rows.extend(
            [
                {
                    "cofactor_enabled": cofactor_on,
                    "community_subset": "all assemblies",
                    "n_communities": int(scored_communities.sum()),
                    "mean_community_abs_log2_error": (
                        np.mean(community_mae[scored_communities])
                        if scored_communities.any() else np.nan
                    ),
                },
                {
                    "cofactor_enabled": cofactor_on,
                    "community_subset": "complex assemblies",
                    "n_communities": int(complex_assemblies.sum()),
                    "mean_community_abs_log2_error": (
                        np.mean(community_mae[complex_assemblies])
                        if complex_assemblies.any() else np.nan
                    ),
                },
            ]
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(output / "model_comparison_summary.csv", index=False)
    return summary


def predict_dm(root):
    """Predict random, core/dropout, and full DM communities."""

    output = root / RESULTS_DIRNAME / "dm_communities"
    reset_output(output)
    observed_frame = (
        pd.read_csv(root / "data" / "dm_assemblies_mean_relative_abundance_matrix.csv")
        .set_index("species")
        .loc[list(SPECIES)]
    )
    names = observed_frame.columns.tolist()
    observed_values = observed_frame.T.to_numpy(dtype=float)
    presence = np.isfinite(observed_values)
    observed = merge_eu(relative_abundance(observed_values, presence), presence)
    save_species_matrix(output / "observed_relative.csv", observed, names)

    R, P, fresh_Y, D, F, lag, _, _ = full_dm_model(root)
    X0 = presence.astype(float) * INITIAL_ABUNDANCE
    rows = []
    for cofactor_on in (True, False):
        branch = output / ("cofactor_on" if cofactor_on else "cofactor_off")
        branch.mkdir()
        needs = np.r_[np.zeros(16, dtype=bool), np.full(16, cofactor_on)]
        absolute = simulate(R, P, fresh_Y, X0, D, F, lag, needs)
        relative = merge_eu(relative_abundance(absolute, presence), presence)
        error = np.full_like(relative, np.nan)
        error[presence] = log2_error(
            relative[presence], observed[presence], PREDICTION_FLOOR
        )
        save_species_matrix(branch / "predicted_absolute.csv", np.where(presence, absolute, np.nan), names)
        save_species_matrix(branch / "predicted_relative.csv", relative, names)
        save_species_matrix(branch / "signed_log2_error.csv", error, names)

        community_mae = np.nanmean(np.abs(error), axis=1)
        community_rmse = np.sqrt(np.nanmean(error**2, axis=1))
        pd.DataFrame(
            {"community": names, "mean_abs_log2_error": community_mae, "rmse_log2_error": community_rmse}
        ).to_csv(branch / "community_metrics.csv", index=False)
        rows.append(
            {
                "cofactor_enabled": cofactor_on,
                "mean_community_abs_log2_error": np.mean(community_mae),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "model_comparison_summary.csv", index=False)
    return summary


def predict_carbon_sources(root):
    """Predict the 16 nongrowers in full communities across carbon sources."""

    output = root / RESULTS_DIRNAME / "carbon_sources"
    reset_output(output)
    data = pd.read_csv(
        root / "data"
        / "full_community_different_carbon_sources_absolute_abundance.csv"
    )
    observed_absolute = np.clip(
        data[list(SPECIES)].to_numpy(dtype=float),
        0.0,
        None,
    )
    observed_total = observed_absolute.sum(axis=1, keepdims=True)
    observed = np.divide(
        observed_absolute,
        observed_total,
        out=np.full_like(observed_absolute, np.nan),
        where=observed_total > 0,
    )
    observed = merge_eu(observed, np.isfinite(observed))
    full_presence = np.ones_like(observed, dtype=bool)

    _, nongrower_R, _, cofactor, production = load_model(root)
    produced_Y = observed_absolute[:, :16] @ production.to_numpy().T
    X0 = np.full((len(data), 16), INITIAL_ABUNDANCE)
    zero_production = np.zeros_like(nongrower_R.to_numpy())
    metadata = data[[column for column in data.columns if column not in SPECIES]].copy()
    metadata["contamination_status"] = "not_assessable"
    metadata["modeling_action_default"] = "review_required"
    growth_qc = metadata["growth_qc"].fillna("").str.strip().str.lower()
    unknown_qc = sorted(set(growth_qc) - {"", *CARBON_QC_EXCLUSIONS})
    if unknown_qc:
        raise ValueError(f"Unknown growth_qc labels: {unknown_qc}")
    score_replicate = ~growth_qc.isin(CARBON_QC_EXCLUSIONS)
    metadata["included_in_carbon_score"] = score_replicate
    metadata["carbon_score_exclusion_reason"] = growth_qc.replace("", np.nan)

    selected_media = data.loc[score_replicate, "Media"]
    by_media_obs = (
        pd.DataFrame(observed[score_replicate], columns=SPECIES)
        .assign(Media=selected_media.to_numpy())
        .groupby("Media", sort=False)[list(SPECIES)]
        .mean()
    )
    media_order = data["Media"].drop_duplicates().tolist()
    qc_table = pd.DataFrame(
        {
            "Media": data["Media"],
            "used": score_replicate,
            "high_other_species": growth_qc.eq("high other species"),
            "low_growth": growth_qc.eq("low growth"),
        }
    )
    qc_counts = qc_table.groupby("Media", sort=False).agg(
        n_replicates_total=("used", "size"),
        n_replicates_used=("used", "sum"),
        n_excluded_high_other_species=("high_other_species", "sum"),
        n_excluded_low_growth=("low_growth", "sum"),
    ).reindex(media_order)

    rows = []
    for cofactor_on in (True, False):
        branch = output / ("cofactor_on" if cofactor_on else "cofactor_off")
        branch.mkdir()
        needs = np.full(16, cofactor_on)
        predicted_ng = simulate(
            nongrower_R.to_numpy(),
            zero_production,
            produced_Y,
            X0,
            cofactor["D"].to_numpy(),
            cofactor["F"].to_numpy(),
            cofactor["selected_lag_h"].to_numpy(),
            needs,
        )
        predicted_absolute = np.c_[observed_absolute[:, :16], predicted_ng]
        predicted = merge_eu(relative_abundance(predicted_absolute), full_presence)
        error = log2_error(predicted, observed, PREDICTION_FLOOR)

        for filename, values in (
            ("predicted_absolute_by_replicate.csv", predicted_absolute),
            ("predicted_relative_by_replicate.csv", predicted),
            ("signed_log2_error_by_replicate.csv", error),
        ):
            pd.concat([metadata, pd.DataFrame(values, columns=SPECIES)], axis=1).to_csv(
                branch / filename, index=False
            )

        by_media_pred = (
            pd.DataFrame(predicted[score_replicate], columns=SPECIES)
            .assign(Media=selected_media.to_numpy())
            .groupby("Media", sort=False)[list(SPECIES)]
            .mean()
        )
        media_error = log2_error(
            by_media_pred, by_media_obs, PREDICTION_FLOOR
        )[list(NONGROWERS)]
        media_metrics = qc_counts.copy()
        media_metrics["included_in_score"] = media_metrics["n_replicates_used"] > 0
        media_metrics["mean_abs_log2_error"] = media_error.abs().mean(axis=1)
        media_metrics["rmse_log2_error"] = np.sqrt((media_error**2).mean(axis=1))
        media_metrics = media_metrics.reset_index()
        media_metrics.to_csv(branch / "carbon_source_metrics.csv", index=False)
        by_media_pred.reset_index().to_csv(branch / "predicted_mean_relative_by_carbon_source.csv", index=False)
        media_error.reset_index().to_csv(branch / "signed_log2_error_by_carbon_source.csv", index=False)
        rows.append(
            {
                "cofactor_enabled": cofactor_on,
                "mean_carbon_source_abs_log2_error": media_metrics.loc[
                    media_metrics["included_in_score"], "mean_abs_log2_error"
                ].mean(),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "model_comparison_summary.csv", index=False)
    return summary


def run_predictions(root):
    output = root / RESULTS_DIRNAME
    output.mkdir(exist_ok=True)
    bt = predict_nongrowers_in_bt(root)
    dm = predict_dm(root)
    carbon = predict_carbon_sources(root)
    summary = pd.concat(
        [
            bt.assign(dataset="nongrowers in Bt spent"),
            dm.assign(dataset="DM communities"),
            carbon.assign(dataset="carbon sources"),
        ],
        ignore_index=True,
    )
    summary.to_csv(output / "summary.csv", index=False)
    return summary
