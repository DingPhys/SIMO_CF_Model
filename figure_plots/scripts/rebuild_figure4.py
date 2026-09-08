"""Rebuild four Figure4 panels from the current 0907 data and parameters.

Run with jnb-legacy Python and Rscript on PATH. All drawing is done by the
embedded R code; this single script exports SVG/PDF and exact source CSVs.
Optional --preview-dir writes R-rendered inspection images outside Figure4.
The official data, parameters and prediction_results directories are unchanged.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview-dir', type=Path)
    args = parser.parse_args()
    figure_root = Path(__file__).resolve().parents[1]
    root = figure_root.parent
    output = figure_root / 'Figure4'
    output.mkdir(exist_ok=True)
    sources = sorted((root / 'parameters').rglob('*.csv')) + sorted((root / 'data').glob('*.csv'))
    sources += [root / 'scripts/prediction_pipeline.py']
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    spec = importlib.util.spec_from_file_location('figure4_current_predictions', sources[-1])
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)
    input_paths = {
        'nongrower_binary_CR.csv': 'data/nongrower_binary_consumption_matrix.csv',
        'bt_resource_Y0.csv': 'parameters/bt_base/bt_resource_Y0.csv',
        'cofactor_and_lag.csv': 'parameters/final_parameters/cofactor_and_lag.csv',
        'double_spent_growth.csv': 'data/Nongrowers_in_Bt_double_spent.csv',
    }
    for name, path in input_paths.items():
        shutil.copy2(root / path, output / name)
    cf_species = ['Col', 'Cs', 'Et', 'Eu.c', 'Eu.l', 'Im', 'Ld', 'Lsp', 'Mi', 'Va']
    pd.DataFrame({'species': cf_species}).to_csv(output / 'cofactor_species_order.csv', index=False)

    with tempfile.TemporaryDirectory(prefix='figure4_0907_') as temporary:
        temp = Path(temporary)
        model = temp / 'model'
        model.mkdir()
        (model / 'data').symlink_to(root / 'data', target_is_directory=True)
        (model / 'parameters').symlink_to(root / 'parameters', target_is_directory=True)
        print('Recomputing Bt-spent, DM and carbon-source predictions from current parameters.', flush=True)
        summary = pipeline.run_predictions(model)
        summary.to_csv(output / 'prediction_summary.csv', index=False)
        results = model / 'prediction_results'
        for short, directory in [('bt', 'nongrowers_in_bt_spent'), ('dm', 'dm_communities')]:
            base = results / directory
            shutil.copy2(base / 'observed_relative.csv', output / f'{short}_observed_relative.csv')
            for branch in ['cofactor_on', 'cofactor_off']:
                shutil.copy2(base / branch / 'community_metrics.csv',
                             output / f'{short}_{branch}_community_metrics.csv')
                shutil.copy2(base / branch / 'predicted_relative.csv',
                             output / f'{short}_{branch}_relative.csv')
        for branch in ['cofactor_on', 'cofactor_off']:
            shutil.copy2(results / 'carbon_sources' / branch / 'carbon_source_metrics.csv',
                         output / f'carbon_{branch}_metrics.csv')

        binary = pd.read_csv(output / 'nongrower_binary_CR.csv', index_col='species')
        y0 = pd.read_csv(output / 'bt_resource_Y0.csv', index_col='resource_id')['bt_resource_y0'].reindex(binary.columns)
        parameters = pd.read_csv(output / 'cofactor_and_lag.csv', index_col='species')
        double = pd.read_csv(output / 'double_spent_growth.csv', index_col='Species')
        assert np.isfinite(y0).all()
        ld_rows = []
        for donor in cf_species:
            if donor == 'Ld':
                continue
            resource = max(float((binary.loc['Ld'] * (1 - binary.loc[donor]) * y0).sum()), 1e-3)
            ld_rows.append(dict(data_group='double_spent', condition=f'Ld_in_{donor}_double_spent',
                                community_size=np.nan, metric='yield', actual_value=double.loc['Ld', donor],
                                resource_only_prediction=resource,
                                cofactor_prediction=min(resource, parameters.loc[donor, 'C'] / parameters.loc['Ld', 'F'])))
        observed = pd.read_csv(output / 'bt_observed_relative.csv', index_col='species')
        cofactor = pd.read_csv(output / 'bt_cofactor_on_relative.csv', index_col='species')
        resource = pd.read_csv(output / 'bt_cofactor_off_relative.csv', index_col='species')
        membership = pd.read_csv(root / 'data/non-growers_in_Bt_spent_assemblies_mean_relative_abundance.csv', index_col='species')
        # Use current input order so a clean rebuild never needs an earlier figure output.
        assert set(observed.columns).issubset(membership.columns)
        order = [name for name in membership.columns if name in observed.columns]
        for name in order:
            n_species = int(membership[name].notna().sum())
            if n_species < 2 or pd.isna(membership.loc['Ld', name]):
                continue
            ld_rows.append(dict(data_group='pair' if n_species == 2 else 'larger', condition=name,
                                community_size=n_species, metric='relative_abundance',
                                actual_value=observed.loc['Ld', name],
                                resource_only_prediction=resource.loc['Ld', name],
                                cofactor_prediction=cofactor.loc['Ld', name]))
        ld_frame = pd.DataFrame(ld_rows)
        assert np.isfinite(ld_frame[['actual_value', 'resource_only_prediction', 'cofactor_prediction']]).all().all()
        ld_frame.to_csv(output / 'ld_panel_plot_source.csv', index=False)

        environment = os.environ.copy()
        environment.update(FIG4_DIR=str(output), FIG4_PREVIEW='')
        if args.preview_dir:
            preview = args.preview_dir.resolve()
            if preview == output.resolve() or output.resolve() in preview.parents:
                raise ValueError('Preview images must be outside the final Figure4 folder.')
            preview.mkdir(parents=True, exist_ok=True)
            environment['FIG4_PREVIEW'] = str(preview)
        r_script = temp / 'render_figure4.R'
        r_script.write_text(R_SOURCE)
        print('Rendering the four panels in R.', flush=True)
        subprocess.run(['Rscript', str(r_script)], env=environment, cwd=temp, check=True)

    stems = ['ld_yield_relative_resource_vs_cofactor', 'double_spent_monoculture_resource_vs_cofactor_fit',
             'relative_abundance_prediction_grid_3x4', 'prediction_error_four_groups_two_rows_compact']
    for stem in stems:
        for extension in ['.svg', '.pdf']:
            assert (output / (stem + extension)).stat().st_size > 1000
    assert not list(output.glob('*.png')) and not list(output.glob('*.tiff'))
    for path, digest in hashes.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, f'Input changed: {path}'
    manifest = {
        'model_root': str(root),
        'rendering': 'Embedded R code adapted from figure_plots/figures/Figure4/generate_figure4.R',
        'input_sha256': {str(p.relative_to(root)): digest for p, digest in hashes.items()},
        'ld_condition_order_source': 'data/non-growers_in_Bt_spent_assemblies_mean_relative_abundance.csv',
        'notes': [
            'Predictions freshly simulated; existing prediction_results are not used or modified.',
            'Double-spent predictions use binary access and current resource amounts, capped by C_donor/F_focal.',
            'Double-spent self-conditioning diagonal excluded; Full Bt monocultures included.',
            'Spearman rho uses plotted yields floored at 0.001; raw-yield rho is also exported.',
            'Ld panel includes double-spent yield and community relative abundance; only communities with >2 species have black outlines.',
            'Ld floor-level displacement is display-only, with original values retained in the source CSV.',
            'The twelve reference grid conditions are retained without imposing a target prediction error.',
            'Error groups use current pipeline QC, Eu.c/Eu.l scoring convention, and observed coexistence >0.001 for pairs.',
        ],
    }
    (output / 'data_sources.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Ld quantities:', ld_frame.groupby('data_group').size().to_dict())
    print('Outputs:', output)


# The complete R renderer is embedded below so that only this script is needed.
R_SOURCE = r'''
#!/usr/bin/env Rscript

# Render the four Figure4 panels from freshly generated current-model CSVs.

# ---- double-spent monoculture fit ----
local({
# Reproduce the existing double-spent + Full-Bt monoculture fit panel in R.
# The visual contract intentionally matches the established paper figure.

suppressPackageStartupMessages({
  library(ggplot2)
  library(grid)
})

script_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_argument) != 1L) stop("Run this file with Rscript.")
script_path <- normalizePath(sub("^--file=", "", script_argument), mustWork = TRUE)
bundle_dir <- normalizePath(Sys.getenv("FIG4_DIR"), mustWork = TRUE)
figure_dir <- bundle_dir

floor_value <- 1e-3
paper_colours <- c(
  resource_only = "#A7A9AC",
  cofactor = "#ED174C",
  text = "#111111",
  guide = "#777777"
)

point_size_main_mm <- sqrt(62) * 25.4 / 72
axis_line_mm <- 0.5 * 25.4 / 72
guide_line_mm <- 0.8 * 25.4 / 72

cofactor_species <- read.csv(
  file.path(bundle_dir, "cofactor_species_order.csv"),
  check.names = FALSE,
  stringsAsFactors = FALSE
)$species

binary_frame <- read.csv(
  file.path(bundle_dir, "nongrower_binary_CR.csv"),
  check.names = FALSE,
  stringsAsFactors = FALSE
)
rownames(binary_frame) <- binary_frame$species
binary_frame$species <- NULL

resource_table <- read.csv(
  file.path(bundle_dir, "bt_resource_Y0.csv"),
  check.names = FALSE,
  stringsAsFactors = FALSE
)
resource_values <- setNames(
  resource_table$bt_resource_y0,
  resource_table$resource_id
)
resources <- colnames(binary_frame)
resource_values <- resource_values[resources]

parameter_frame <- read.csv(
  file.path(bundle_dir, "cofactor_and_lag.csv"),
  check.names = FALSE,
  stringsAsFactors = FALSE
)
rownames(parameter_frame) <- parameter_frame$species

observed_frame <- read.csv(
  file.path(bundle_dir, "double_spent_growth.csv"),
  check.names = FALSE,
  stringsAsFactors = FALSE
)
rownames(observed_frame) <- observed_frame$Species

binary_selected <- as.matrix(binary_frame[cofactor_species, resources, drop = FALSE])
storage.mode(binary_selected) <- "double"
bt_y0 <- as.numeric(resource_values)

n_species <- length(cofactor_species)
resource_only_double <- matrix(NA_real_, n_species, n_species)
for (focal_index in seq_len(n_species)) {
  for (donor_index in seq_len(n_species)) {
    resource_only_double[focal_index, donor_index] <- sum(
      binary_selected[focal_index, ] *
        (1 - binary_selected[donor_index, ]) *
        bt_y0
    )
  }
}
resource_only_double <- pmax(resource_only_double, floor_value)

resource_only_monoculture <- pmax(
  as.numeric(binary_selected %*% bt_y0),
  floor_value
)

F_selected <- parameter_frame[cofactor_species, "F"]
C_selected <- parameter_frame[cofactor_species, "C"]
cofactor_capacity_double <- outer(1 / F_selected, C_selected, "*")
cofactor_double <- pmin(resource_only_double, cofactor_capacity_double)
cofactor_monoculture <- pmin(resource_only_monoculture, 1 / F_selected)

observed_double <- as.matrix(
  observed_frame[cofactor_species, cofactor_species, drop = FALSE]
)
storage.mode(observed_double) <- "double"
observed_monoculture <- observed_frame[cofactor_species, "Full_Bt"]

rows <- vector("list", n_species * (n_species + 1))
row_index <- 0L
for (focal_index in seq_len(n_species)) {
  for (donor_index in seq_len(n_species)) {
    row_index <- row_index + 1L
    rows[[row_index]] <- data.frame(
      condition = "double_spent",
      focal_species = cofactor_species[focal_index],
      donor_species = cofactor_species[donor_index],
      observed_yield = observed_double[focal_index, donor_index],
      resource_only_prediction = resource_only_double[focal_index, donor_index],
      cofactor_prediction = cofactor_double[focal_index, donor_index],
      stringsAsFactors = FALSE
    )
  }
  row_index <- row_index + 1L
  rows[[row_index]] <- data.frame(
    condition = "monoculture_full_bt",
    focal_species = cofactor_species[focal_index],
    donor_species = NA_character_,
    observed_yield = observed_monoculture[focal_index],
    resource_only_prediction = resource_only_monoculture[focal_index],
    cofactor_prediction = cofactor_monoculture[focal_index],
    stringsAsFactors = FALSE
  )
}

comparison <- do.call(rbind, rows)
diagonal <- comparison$condition == "double_spent" &
  comparison$focal_species == comparison$donor_species
comparison <- comparison[!diagonal, , drop = FALSE]
valid <- is.finite(comparison$observed_yield) &
  is.finite(comparison$resource_only_prediction) &
  is.finite(comparison$cofactor_prediction)
comparison <- comparison[valid, , drop = FALSE]
comparison$is_ld <- comparison$focal_species == "Ld"
comparison$observed_plot <- pmax(comparison$observed_yield, floor_value)
comparison$resource_only_plot <- pmax(
  comparison$resource_only_prediction,
  floor_value
)
comparison$cofactor_plot <- pmax(comparison$cofactor_prediction, floor_value)

write.csv(comparison, file.path(figure_dir,
  "double_spent_monoculture_resource_vs_cofactor_fit_source_data.csv"), row.names = FALSE)
statistics <- data.frame(
  model = c("With cofactor", "Without cofactor"),
  n = nrow(comparison),
  spearman_rho_plotted = c(
    cor(comparison$cofactor_plot, comparison$observed_plot, method = "spearman"),
    cor(comparison$resource_only_plot, comparison$observed_plot, method = "spearman")),
  spearman_rho_raw = c(
    cor(comparison$cofactor_prediction, comparison$observed_yield, method = "spearman"),
    cor(comparison$resource_only_prediction, comparison$observed_yield, method = "spearman"))
)
write.csv(statistics, file.path(figure_dir,
  "double_spent_monoculture_resource_vs_cofactor_fit_statistics.csv"), row.names = FALSE)

log_tick_labels <- c(
  expression(10^"-3"),
  expression(10^"-2"),
  expression(10^"-1"),
  expression(10^"0")
)

figure <- ggplot() +
  geom_segment(
    aes(x = floor_value, y = floor_value, xend = 1, yend = 1),
    colour = paper_colours[["guide"]],
    linewidth = guide_line_mm,
    linetype = "22"
  ) +
  geom_point(
    data = comparison,
    aes(x = resource_only_plot, y = observed_plot),
    colour = paper_colours[["resource_only"]],
    size = point_size_main_mm,
    alpha = 0.80,
    shape = 16
  ) +
  geom_point(
    data = comparison,
    aes(x = cofactor_plot, y = observed_plot),
    colour = paper_colours[["cofactor"]],
    size = point_size_main_mm,
    alpha = 0.88,
    shape = 16
  ) +
  annotate("text", x = 1.2e-3, y = c(1.35, 0.28),
    label = c("With cofactor", "Without cofactor"),
    colour = c(paper_colours[["cofactor"]], paper_colours[["resource_only"]]),
    hjust = 0, family = "Arial", size = 8.5 / ggplot2::.pt) +
  scale_x_log10(
    breaks = c(1e-3, 1e-2, 1e-1, 1),
    labels = log_tick_labels,
    expand = c(0, 0)
  ) +
  scale_y_log10(
    breaks = c(1e-3, 1e-2, 1e-1, 1),
    labels = log_tick_labels,
    expand = c(0, 0)
  ) +
  coord_fixed(
    ratio = 1,
    xlim = c(floor_value, 1),
    ylim = c(floor_value, 1),
    clip = "off"
  ) +
  labs(
    x = "Predicted yield (OD)",
    y = "Actual yield (OD)"
  ) +
  theme_classic(base_size = 10, base_family = "Arial") +
  theme(
    text = element_text(colour = paper_colours[["text"]]),
    axis.title = element_text(size = 10, colour = paper_colours[["text"]]),
    axis.title.x = element_text(margin = margin(t = 7, unit = "pt")),
    axis.title.y = element_text(margin = margin(r = 7, unit = "pt")),
    axis.text = element_text(size = 8, colour = paper_colours[["text"]]),
    axis.line = element_line(colour = paper_colours[["text"]], linewidth = axis_line_mm),
    axis.ticks = element_line(colour = paper_colours[["text"]], linewidth = axis_line_mm),
    axis.ticks.length = unit(2.5, "pt"),
    legend.position = "none",
    plot.margin = margin(t = 16, r = 15, b = 4, l = 4, unit = "pt"),
    panel.grid = element_blank(),
    plot.background = element_rect(fill = "white", colour = NA),
    panel.background = element_rect(fill = "white", colour = NA)
  )

base_path <- file.path(
  figure_dir,
  "double_spent_monoculture_resource_vs_cofactor_fit"
)

svglite::svglite(
  paste0(base_path, ".svg"),
  width = 208.363625 / 72,
  height = 207.414158 / 72,
  bg = "white"
)
print(figure)
dev.off()

grDevices::cairo_pdf(
  paste0(base_path, ".pdf"),
  width = 208.979 / 72,
  height = 208.333 / 72,
  family = "Arial",
  bg = "white"
)
print(figure)
dev.off()

if (nzchar(Sys.getenv("FIG4_PREVIEW"))) {
ragg::agg_png(
  file.path(Sys.getenv("FIG4_PREVIEW"), paste0(basename(base_path), ".png")),
  width = 1729,
  height = 1728,
  units = "px",
  res = 600,
  background = "white"
)
print(figure)
dev.off()
}

message("Number of fitting data points: ", nrow(comparison))
message("Number of Ld data points: ", sum(comparison$is_ld))
message("Saved: ", base_path, ".[svg|pdf]")
})

# ---- Ld panel ----
local({
# Observed-versus-predicted Ld quantities under resource-only and
# resource-plus-cofactor competition. Point fill encodes model formulation;
# black point outlines identify >2-species communities; point shape encodes quantity type
# (yield vs relative abundance); arrows connect paired predictions. Crowded
# floor-level points use deterministic display-only displacement while their
# raw values remain available in the exported source-data table.

required_packages <- c("ggplot2", "svglite", "ragg")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages) > 0L) {
  stop(
    "Missing R package(s): ",
    paste(missing_packages, collapse = ", "),
    call. = FALSE
  )
}

command_args <- commandArgs(trailingOnly = FALSE)
script_arg <- grep("^--file=", command_args, value = TRUE)
if (length(script_arg) != 1L) {
  stop("Run this file with Rscript.", call. = FALSE)
}

script_path <- normalizePath(sub("^--file=", "", script_arg))
bundle_dir <- normalizePath(Sys.getenv("FIG4_DIR"), mustWork = TRUE)
figure_dir <- bundle_dir

input_path <- file.path(bundle_dir, "ld_panel_plot_source.csv")
plot_data <- read.csv(
  input_path,
  check.names = FALSE,
  stringsAsFactors = FALSE
)

required_columns <- c(
  "data_group",
  "condition",
  "metric",
  "actual_value",
  "resource_only_prediction",
  "cofactor_prediction"
)
missing_columns <- setdiff(required_columns, names(plot_data))
if (length(missing_columns) > 0L) {
  stop(
    "Missing input column(s): ",
    paste(missing_columns, collapse = ", "),
    call. = FALSE
  )
}
if (anyNA(plot_data[required_columns])) {
  stop("The plotting columns contain missing values.", call. = FALSE)
}

# Keep yield only for double-spent media. Pairwise and larger communities
# contribute relative-abundance observations only.
plot_data <- plot_data[
  plot_data$data_group == "double_spent" |
    plot_data$metric == "relative_abundance",
  ,
  drop = FALSE
]

plot_floor <- 1e-3
plot_data$pair_id <- seq_len(nrow(plot_data))
plot_data$actual_plot <- pmax(plot_data$actual_value, plot_floor)
plot_data$resource_plot <- pmax(
  plot_data$resource_only_prediction,
  plot_floor
)
plot_data$cofactor_plot <- pmax(
  plot_data$cofactor_prediction,
  plot_floor
)

# Match the by-metric figure's handling of the dense larger-community cluster.
# Raw values remain unchanged; only the plotted positions are separated.
crowded_x_limits <- c(0.05, 0.20)
crowded_pair <-
  plot_data$data_group == "larger" &
  plot_data$metric == "relative_abundance" &
  plot_data$actual_plot <= plot_floor &
  plot_data$resource_plot >= crowded_x_limits[[1]] &
  plot_data$resource_plot <= crowded_x_limits[[2]]

plot_data$actual_display <- plot_data$actual_plot
plot_data$resource_display <- plot_data$resource_plot
plot_data$cofactor_display <- plot_data$cofactor_plot
plot_data$display_adjusted <- crowded_pair

crowded_index <- which(crowded_pair)
if (length(crowded_index) > 0L) {
  set.seed(20260831)
  y_levels <- 10^seq(
    log10(1.00e-3),
    log10(1.20e-3),
    length.out = length(crowded_index)
  )
  plot_data$actual_display[crowded_index] <-
    y_levels[order(stats::runif(length(crowded_index)))]

  resource_multipliers <- 10^seq(
    -0.03,
    0.03,
    length.out = length(crowded_index)
  )
  plot_data$resource_display[crowded_index] <-
    plot_data$resource_plot[crowded_index] *
      resource_multipliers[order(stats::runif(length(crowded_index)))]

  cofactor_at_floor <-
    plot_data$cofactor_plot[crowded_index] <= plot_floor
  floor_index <- crowded_index[cofactor_at_floor]
  if (length(floor_index) > 0L) {
    floor_x_levels <- 10^seq(
      log10(1.00e-3),
      log10(1.30e-3),
      length.out = length(floor_index)
    )
    plot_data$cofactor_display[floor_index] <-
      floor_x_levels[order(stats::runif(length(floor_index)))]
  }
  nonfloor_index <- crowded_index[!cofactor_at_floor]
  if (length(nonfloor_index) > 0L) {
    nonfloor_multipliers <- 10^seq(
      -0.03,
      0.03,
      length.out = length(nonfloor_index)
    )
    plot_data$cofactor_display[nonfloor_index] <-
      plot_data$cofactor_plot[nonfloor_index] *
        nonfloor_multipliers[order(stats::runif(length(nonfloor_index)))]
  }
}

experiment_levels <- c("double_spent", "pair", "larger")
experiment_labels <- c(
  "Double-spent media",
  "Pairwise community",
  "Larger community"
)
if (!setequal(unique(plot_data$data_group), experiment_levels)) {
  stop("Unexpected experiment types in the Ld source data.", call. = FALSE)
}
plot_data$experiment_type <- factor(
  plot_data$data_group,
  levels = experiment_levels,
  labels = experiment_labels
)

quantity_levels <- c("yield", "relative_abundance")
quantity_labels <- c("Yield", "Relative abundance")
plot_data$quantity <- factor(
  plot_data$metric,
  levels = quantity_levels,
  labels = quantity_labels
)

# Shorten long arrows in log space so their heads remain visible at the edge
# of the destination point. Very short arrows retain their true endpoints.
resource_log <- log10(plot_data$resource_display)
cofactor_log <- log10(plot_data$cofactor_display)
arrow_delta <- cofactor_log - resource_log
arrow_direction <- sign(arrow_delta)
long_arrow <- abs(arrow_delta) > 0.16
arrow_start_log <- resource_log
arrow_end_log <- cofactor_log
arrow_start_log[long_arrow] <- resource_log[long_arrow] +
  arrow_direction[long_arrow] * 0.055
arrow_end_log[long_arrow] <- cofactor_log[long_arrow] -
  arrow_direction[long_arrow] * 0.075

arrow_data <- data.frame(
  pair_id = plot_data$pair_id,
  actual_plot = plot_data$actual_display,
  arrow_start = 10^arrow_start_log,
  arrow_end = 10^arrow_end_log,
  stringsAsFactors = FALSE
)

resource_points <- data.frame(
  pair_id = plot_data$pair_id,
  data_group = plot_data$data_group,
  experiment_type = plot_data$experiment_type,
  condition = plot_data$condition,
  metric = plot_data$metric,
  quantity = plot_data$quantity,
  method = "Resource competition",
  actual = plot_data$actual_value,
  prediction = plot_data$resource_only_prediction,
  actual_plot_raw = plot_data$actual_plot,
  prediction_plot_raw = plot_data$resource_plot,
  actual_plot = plot_data$actual_display,
  prediction_plot = plot_data$resource_display,
  display_adjusted = plot_data$display_adjusted,
  stringsAsFactors = FALSE
)
cofactor_points <- data.frame(
  pair_id = plot_data$pair_id,
  data_group = plot_data$data_group,
  experiment_type = plot_data$experiment_type,
  condition = plot_data$condition,
  metric = plot_data$metric,
  quantity = plot_data$quantity,
  method = "+ Cofactor competition",
  actual = plot_data$actual_value,
  prediction = plot_data$cofactor_prediction,
  actual_plot_raw = plot_data$actual_plot,
  prediction_plot_raw = plot_data$cofactor_plot,
  actual_plot = plot_data$actual_display,
  prediction_plot = plot_data$cofactor_display,
  display_adjusted = plot_data$display_adjusted,
  stringsAsFactors = FALSE
)
point_data <- rbind(resource_points, cofactor_points)
point_data$method <- factor(
  point_data$method,
  levels = c("Resource competition", "+ Cofactor competition")
)

source_data_path <- file.path(
  figure_dir,
  "ld_yield_relative_resource_vs_cofactor_source_data.csv"
)
write.csv(point_data, source_data_path, row.names = FALSE)

method_colors <- c(
  "Resource competition" = grDevices::adjustcolor(
    "#A7A9AC",
    alpha.f = 0.80
  ),
  "+ Cofactor competition" = grDevices::adjustcolor(
    "#ED174C",
    alpha.f = 0.88
  )
)

pt_to_mm <- function(points) points * 25.4 / 72
axis_linewidth <- pt_to_mm(0.5)
guide_linewidth <- pt_to_mm(0.8)
arrow_linewidth <- pt_to_mm(0.45)
# Only larger-community diamonds receive a black edge.
point_stroke <- pt_to_mm(0.9)
ld_point_size_mm <- sqrt(62) * 25.4 / 72
# Use shape 16 for yield circles, exactly as in the double-spent panel.
# Filled diamonds retain the default stroke allowance with a transparent edge.
ld_invisible_stroke <- 0.5

log10_labels <- function(values) {
  parse(text = sprintf('10^"%d"', round(log10(values))))
}

figure <- ggplot2::ggplot() +
  ggplot2::geom_abline(
    intercept = 0,
    slope = 1,
    colour = "#777777",
    linewidth = guide_linewidth,
    linetype = "22",
    lineend = "butt"
  ) +
  ggplot2::geom_segment(
    data = arrow_data,
    ggplot2::aes(
      x = arrow_start,
      xend = arrow_end,
      y = actual_plot,
      yend = actual_plot
    ),
    colour = "#929292",
    linewidth = arrow_linewidth,
    alpha = 0.62,
    lineend = "round",
    arrow = grid::arrow(
      length = grid::unit(1.15, "mm"),
      angle = 24,
      type = "open"
    )
  ) +
  ggplot2::geom_point(
    data = point_data[point_data$metric == "yield", ],
    ggplot2::aes(x = prediction_plot, y = actual_plot, colour = method),
    shape = 16, size = ld_point_size_mm
  ) +
  ggplot2::geom_point(
    data = point_data[point_data$data_group != "larger" & point_data$metric != "yield", ],
    ggplot2::aes(x = prediction_plot, y = actual_plot, fill = method, shape = quantity),
    colour = "transparent", size = ld_point_size_mm, stroke = ld_invisible_stroke
  ) +
  ggplot2::geom_point(
    data = point_data[point_data$data_group == "larger", ],
    ggplot2::aes(x = prediction_plot, y = actual_plot, fill = method, shape = quantity),
    colour = "black", size = ld_point_size_mm, stroke = point_stroke
  ) +
  ggplot2::annotate("text", x = 2.0e-3, y = c(1.4, 0.82),
    label = c("With cofactor", "Without cofactor"),
    colour = c("#ED174C", "#A7A9AC"), hjust = 0,
    family = "Arial", size = 8.5 / ggplot2::.pt) +
  ggplot2::annotate("point", x = 1.38e-3, y = 0.49,
    shape = 16, colour = "#B8B8B8", size = ld_point_size_mm) +
  ggplot2::annotate("point", x = 1.38e-3, y = 0.29,
    shape = 23, fill = "#B8B8B8", colour = "transparent", stroke = ld_invisible_stroke, size = ld_point_size_mm) +
  ggplot2::annotate("point", x = 1.38e-3, y = 0.17,
    shape = 23, fill = "#B8B8B8", colour = "black", stroke = point_stroke, size = ld_point_size_mm) +
  ggplot2::annotate("text", x = 2.0e-3, y = c(0.49, 0.29, 0.17),
    label = c("Yield in double-spent", "Relative abundance", ">2-species"),
    colour = "black", hjust = 0, family = "Arial", size = 8.5 / ggplot2::.pt) +
  ggplot2::scale_x_log10(
    breaks = c(1e-3, 1e-2, 1e-1, 1),
    labels = log10_labels,
    expand = c(0, 0)
  ) +
  ggplot2::scale_y_log10(
    breaks = c(1e-3, 1e-2, 1e-1, 1),
    labels = log10_labels,
    expand = c(0, 0)
  ) +
  ggplot2::scale_colour_manual(values = method_colors, drop = FALSE) +
  ggplot2::scale_fill_manual(
    values = method_colors,
    drop = FALSE
  ) +
  ggplot2::scale_shape_manual(
    name = NULL,
    values = c("Yield" = 21, "Relative abundance" = 23),
    drop = FALSE
  ) +
  ggplot2::coord_fixed(
    ratio = 1,
    xlim = c(plot_floor, 1),
    ylim = c(plot_floor, 1),
    clip = "off"
  ) +
  ggplot2::labs(
    x = "Predicted yield or relative abundance",
    y = "Actual yield or relative abundance",
    fill = NULL
  ) +
  ggplot2::theme_classic(
    base_size = 10,
    base_family = "Arial"
  ) +
  ggplot2::theme(
    axis.line = ggplot2::element_line(
      colour = "#111111",
      linewidth = axis_linewidth
    ),
    axis.ticks = ggplot2::element_line(
      colour = "#111111",
      linewidth = axis_linewidth
    ),
    axis.ticks.length = grid::unit(2.5, "pt"),
    axis.title = ggplot2::element_text(
      colour = "#111111",
      size = 10
    ),
    axis.title.x = ggplot2::element_text(
      margin = ggplot2::margin(t = 7, unit = "pt")
    ),
    axis.title.y = ggplot2::element_text(
      margin = ggplot2::margin(r = 7, unit = "pt")
    ),
    axis.text = ggplot2::element_text(
      colour = "#111111",
      size = 8
    ),
    legend.position = "none",
    legend.justification = c(0, 1),
    legend.background = ggplot2::element_blank(),
    legend.box.background = ggplot2::element_blank(),
    legend.margin = ggplot2::margin(0, 0, 0, 0, unit = "pt"),
    legend.box.margin = ggplot2::margin(0, 0, 0, 0, unit = "pt"),
    legend.spacing.y = grid::unit(0.5, "pt"),
    legend.key.height = grid::unit(8.2, "pt"),
    legend.key.width = grid::unit(8.5, "pt"),
    legend.title = ggplot2::element_text(
      size = 8,
      face = "plain",
      margin = ggplot2::margin(t = 2, b = 0, unit = "pt")
    ),
    legend.text = ggplot2::element_text(size = 8),
    plot.margin = ggplot2::margin(16, 15, 4, 4, unit = "pt")
  )

output_base <- file.path(
  figure_dir,
  "ld_yield_relative_resource_vs_cofactor"
)

svglite::svglite(
  paste0(output_base, ".svg"),
  width = 208.363625 / 72,
  height = 207.414158 / 72,
  bg = "white"
)
print(figure)
grDevices::dev.off()

grDevices::cairo_pdf(
  paste0(output_base, ".pdf"),
  width = 208.979 / 72,
  height = 208.333 / 72,
  family = "Arial",
  bg = "white"
)
print(figure)
grDevices::dev.off()

if (nzchar(Sys.getenv("FIG4_PREVIEW"))) {
ragg::agg_png(
  file.path(Sys.getenv("FIG4_PREVIEW"), paste0(basename(output_base), ".png")),
  width = 1729,
  height = 1728,
  units = "px",
  res = 600,
  background = "white"
)
print(figure)
grDevices::dev.off()
}


cat("Experimental quantities: ", nrow(plot_data), "\n", sep = "")
cat("Displayed points: ", nrow(point_data), "\n", sep = "")
cat("Display-adjusted crowded quantities: ", sum(crowded_pair), "\n", sep = "")
cat("Experiment types: ", paste(experiment_labels, collapse = ", "), "\n", sep = "")
cat("Saved: ", paste0(output_base, ".svg"), "\n", sep = "")
cat("Saved: ", paste0(output_base, ".pdf"), "\n", sep = "")
cat("Saved: ", source_data_path, "\n", sep = "")
})

# ---- prediction errors ----
local({
# Reproduce the established compact two-row model-error figure in R.
# Only the requested group order differs from the previous rendering.

suppressPackageStartupMessages({
  library(grid)
})

find_project_root <- function(start = getwd()) {
  candidate <- normalizePath(start, mustWork = TRUE)
  repeat {
    marker <- file.path(candidate, "Finalized_Model", "final_config.json")
    if (file.exists(marker)) {
      return(candidate)
    }
    parent <- dirname(candidate)
    if (identical(parent, candidate)) {
      stop("Could not find Finalized_Model/final_config.json.")
    }
    candidate <- parent
  }
}

read_table <- function(path) {
  read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
}

get_matched_errors <- function(
  cofactor_table,
  resource_table,
  identifiers,
  id_column,
  error_column = "mean_abs_log2_error"
) {
  cofactor_table <- cofactor_table[!duplicated(cofactor_table[[id_column]]), ]
  resource_table <- resource_table[!duplicated(resource_table[[id_column]]), ]
  cofactor_index <- match(identifiers, cofactor_table[[id_column]])
  resource_index <- match(identifiers, resource_table[[id_column]])
  matched <- !is.na(cofactor_index) & !is.na(resource_index)
  identifiers <- identifiers[matched]
  cofactor_index <- cofactor_index[matched]
  resource_index <- resource_index[matched]
  cofactor_values <- as.numeric(cofactor_table[[error_column]][cofactor_index])
  resource_values <- as.numeric(resource_table[[error_column]][resource_index])
  valid <- is.finite(cofactor_values) & is.finite(resource_values)
  list(
    ids = identifiers[valid],
    cofactor = cofactor_values[valid],
    resource = resource_values[valid]
  )
}

script_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_argument) != 1L) stop("Run this file with Rscript.")
script_path <- normalizePath(sub("^--file=", "", script_argument), mustWork = TRUE)
bundle_dir <- normalizePath(Sys.getenv("FIG4_DIR"), mustWork = TRUE)
figure_dir <- bundle_dir
results_dir <- bundle_dir

error_clip <- 6
pair_coexistence_floor <- 1e-3
cofactor_colour <- "#ED174C"
resource_colour <- "#A7A9AC"
text_colour <- "#111111"

# 1. Nongrower assemblies and coexisting pairs in Bt-spent medium.
bt_root <- file.path(results_dir, "nongrower_assemblies_in_bt_spent")
bt_cofactor_metrics <- read_table(
  file.path(bundle_dir, "bt_cofactor_on_community_metrics.csv")
)
bt_resource_metrics <- read_table(
  file.path(bundle_dir, "bt_cofactor_off_community_metrics.csv")
)

complex_names <- bt_cofactor_metrics$community[
  bt_cofactor_metrics$n_presented > 2
]
complex_group <- get_matched_errors(
  bt_cofactor_metrics,
  bt_resource_metrics,
  complex_names,
  "community"
)

bt_observed <- read_table(
  file.path(bundle_dir, "bt_observed_relative.csv")
)
rownames(bt_observed) <- bt_observed$species
bt_observed$species <- NULL
pair_names <- bt_cofactor_metrics$community[
  bt_cofactor_metrics$n_presented == 2
]
coexisting_pair_names <- pair_names[vapply(
  pair_names,
  function(community) {
    if (!community %in% colnames(bt_observed)) {
      return(FALSE)
    }
    observed_values <- suppressWarnings(as.numeric(bt_observed[[community]]))
    observed_values <- observed_values[is.finite(observed_values)]
    length(observed_values) == 2 &&
      all(observed_values > pair_coexistence_floor)
  },
  logical(1)
)]
pair_group <- get_matched_errors(
  bt_cofactor_metrics,
  bt_resource_metrics,
  coexisting_pair_names,
  "community"
)

# 2. Assemblies in DM.
dm_root <- file.path(
  results_dir,
  "random_core_full_assemblies_in_dm",
  "universal_on",
  "extra_lag-4h",
  "prod_prior-0p3"
)
dm_cofactor_metrics <- read_table(
  file.path(bundle_dir, "dm_cofactor_on_community_metrics.csv")
)
dm_resource_metrics <- read_table(
  file.path(bundle_dir, "dm_cofactor_off_community_metrics.csv")
)
dm_group <- get_matched_errors(
  dm_cofactor_metrics,
  dm_resource_metrics,
  dm_cofactor_metrics$community,
  "community"
)

# 3. Full communities across primary carbon sources.
carbon_root <- file.path(
  results_dir,
  "full_communities_across_carbon_sources",
  "prod_prior-0p3"
)
carbon_cofactor_metrics <- read_table(
  file.path(bundle_dir, "carbon_cofactor_on_metrics.csv")
)
carbon_resource_metrics <- read_table(
  file.path(bundle_dir, "carbon_cofactor_off_metrics.csv")
)
primary_mask <- tolower(as.character(
  carbon_cofactor_metrics$included_in_score
)) %in% c("true", "1")
carbon_group <- get_matched_errors(
  carbon_cofactor_metrics,
  carbon_resource_metrics,
  carbon_cofactor_metrics$Media[primary_mask],
  "Media"
)

plot_groups <- list(
  complex = c(complex_group, list(label = "Complex\nassemblies")),
  carbon = c(carbon_group, list(label = "Carbon\nsources")),
  dm = c(dm_group, list(label = "DM\nassemblies")),
  pairs = c(pair_group, list(label = "Coexisting\npairs"))
)

error_source <- do.call(rbind, lapply(names(plot_groups), function(group) {
  values <- plot_groups[[group]]
  data.frame(group = group, condition = values$ids,
    cofactor_error = values$cofactor, resource_only_error = values$resource,
    cofactor_error_plot = pmin(values$cofactor, error_clip),
    resource_only_error_plot = pmin(values$resource, error_clip))
}))
write.csv(error_source, file.path(figure_dir,
  "prediction_error_four_groups_two_rows_compact_source_data.csv"), row.names = FALSE)

# Preserve a fixed jitter for every group/condition independently of panel order.
set.seed(1234)
jitter_width <- 0.045
for (group_name in names(plot_groups)) {
  plot_groups[[group_name]]$cofactor_jitter <- runif(
    length(plot_groups[[group_name]]$cofactor),
    -jitter_width,
    jitter_width
  )
  plot_groups[[group_name]]$resource_jitter <- runif(
    length(plot_groups[[group_name]]$resource),
    -jitter_width,
    jitter_width
  )
}

box_offset <- 0.15
box_width <- 0.24
cap_width <- 0.12

draw_box_and_points <- function(values, jitter, center, colour) {
  values <- as.numeric(values)
  values <- values[is.finite(values)]
  displayed <- pmin(values, error_clip)
  stats <- boxplot.stats(displayed, coef = 1.5, do.conf = FALSE)$stats

  grid.lines(
    x = unit(c(center, center), "native"),
    y = unit(c(stats[1], stats[2]), "native"),
    gp = gpar(col = colour, lwd = 0.45)
  )
  grid.lines(
    x = unit(c(center, center), "native"),
    y = unit(c(stats[4], stats[5]), "native"),
    gp = gpar(col = colour, lwd = 0.45)
  )
  grid.lines(
    x = unit(c(center - cap_width / 2, center + cap_width / 2), "native"),
    y = unit(c(stats[1], stats[1]), "native"),
    gp = gpar(col = colour, lwd = 0.45, lineend = "round")
  )
  grid.lines(
    x = unit(c(center - cap_width / 2, center + cap_width / 2), "native"),
    y = unit(c(stats[5], stats[5]), "native"),
    gp = gpar(col = colour, lwd = 0.45, lineend = "round")
  )
  grid.rect(
    x = unit(center, "native"),
    y = unit((stats[2] + stats[4]) / 2, "native"),
    width = unit(box_width, "native"),
    height = unit(stats[4] - stats[2], "native"),
    gp = gpar(fill = "white", col = colour, lwd = 0.45)
  )
  grid.lines(
    x = unit(c(center - box_width / 2, center + box_width / 2), "native"),
    y = unit(c(stats[3], stats[3]), "native"),
    gp = gpar(col = colour, lwd = 0.5)
  )
  grid.points(
    x = unit(center + jitter[seq_along(displayed)], "native"),
    y = unit(displayed, "native"),
    pch = 16,
    size = unit(3.2, "pt"),
    gp = gpar(col = adjustcolor(colour, alpha.f = 0.58))
  )
}

draw_panel <- function(groups, x_labels) {
  pushViewport(viewport(xscale = c(-0.48, 1.48), yscale = c(0, 6), clip = "off"))

  for (group_index in seq_along(groups)) {
    group <- groups[[group_index]]
    center <- group_index - 1
    draw_box_and_points(
      group$cofactor,
      group$cofactor_jitter,
      center - box_offset,
      cofactor_colour
    )
    draw_box_and_points(
      group$resource,
      group$resource_jitter,
      center + box_offset,
      resource_colour
    )
  }

  grid.lines(
    x = unit(c(-0.48, 1.48), "native"),
    y = unit(c(0, 0), "native"),
    gp = gpar(col = text_colour, lwd = 0.5)
  )
  grid.lines(
    x = unit(c(-0.48, -0.48), "native"),
    y = unit(c(0, 6), "native"),
    gp = gpar(col = text_colour, lwd = 0.5)
  )

  for (tick in c(0, 2, 4, 6)) {
    grid.lines(
      x = unit.c(unit(-0.48, "native"), unit(-2.8, "pt")),
      y = unit(c(tick, tick), "native"),
      gp = gpar(col = text_colour, lwd = 0.5)
    )
    grid.text(
      as.character(tick),
      x = unit(-5.0, "pt"),
      y = unit(tick, "native"),
      just = "right",
      gp = gpar(fontfamily = "Arial", fontsize = 8, col = text_colour)
    )
  }

  for (tick_index in seq_along(x_labels)) {
    x_value <- tick_index - 1
    grid.lines(
      x = unit(c(x_value, x_value), "native"),
      y = unit.c(unit(0, "native"), unit(-2.8, "pt")),
      gp = gpar(col = text_colour, lwd = 0.5)
    )
    label_lines <- strsplit(x_labels[tick_index], "\n", fixed = TRUE)[[1]]
    for (line_index in seq_along(label_lines)) {
      label <- label_lines[[line_index]]
      y_label <- unit(-6 - (line_index - 1) * 10, "pt")
      if (identical(label, "in Bt-spent")) {
        pieces <- c("in ", "Bt", "-spent")
        faces <- c("plain", "italic", "plain")
        widths <- lapply(seq_along(pieces), function(i) grobWidth(textGrob(
          pieces[[i]], gp = gpar(fontfamily = "Arial", fontsize = 8, fontface = faces[[i]]))))
        total <- Reduce(`+`, widths)
        offset <- unit(x_value, "native") - total / 2
        for (i in seq_along(pieces)) {
          grid.text(pieces[[i]], x = offset, y = y_label, just = c("left", "top"),
            gp = gpar(fontfamily = "Arial", fontsize = 8, fontface = faces[[i]], col = text_colour))
          offset <- offset + widths[[i]]
        }
      } else {
        grid.text(label, x = unit(x_value, "native"), y = y_label, just = "top",
          gp = gpar(fontfamily = "Arial", fontsize = 8, col = text_colour))
      }
    }
  }

  popViewport()
}

draw_illustrator_safe_y_label <- function() {
  # Two lines with an explicitly positioned 8 pt subscript; no scaled plotmath text.
  grid.text("Model error", x = unit(5.0, "mm"), y = unit(43.5, "mm"), rot = 90,
    gp = gpar(fontfamily = "Arial", fontsize = 10, col = text_colour))
  prefix <- "(log"
  subscript <- "2"
  suffix <- "(fold change) per species)"
  regular_gp <- gpar(fontfamily = "Arial", fontsize = 10, col = text_colour)
  subscript_gp <- gpar(fontfamily = "Arial", fontsize = 8, col = text_colour)
  prefix_width <- grobWidth(textGrob(prefix, gp = regular_gp))
  subscript_width <- grobWidth(textGrob(subscript, gp = subscript_gp))
  suffix_width <- grobWidth(textGrob(suffix, gp = regular_gp))
  total_width <- prefix_width + subscript_width + suffix_width
  pushViewport(viewport(x = unit(10.0, "mm"), y = unit(43.5, "mm"),
    width = total_width, height = unit(5, "mm"), angle = 90))
  grid.text(prefix, x = unit(0, "npc"), y = unit(2.5, "mm"), just = "left", gp = regular_gp)
  grid.text(subscript, x = prefix_width, y = unit(1.75, "mm"), just = "left", gp = subscript_gp)
  grid.text(suffix, x = prefix_width + subscript_width, y = unit(2.5, "mm"), just = "left", gp = regular_gp)
  popViewport()
}

draw_figure <- function() {
  grid.newpage()
  panel_width_mm <- 150 / 72 * 25.4
  panel_center_x_mm <- 19.6 + panel_width_mm / 2
  top_panel <- viewport(x = unit(panel_center_x_mm, "mm"), y = unit(63.2, "mm"),
    width = unit(panel_width_mm, "mm"), height = unit(22.6, "mm"))
  bottom_panel <- viewport(x = unit(panel_center_x_mm, "mm"), y = unit(23.7, "mm"),
    width = unit(panel_width_mm, "mm"), height = unit(22.7, "mm"))
  pushViewport(top_panel)
  draw_panel(list(plot_groups$complex, plot_groups$carbon),
    c(">2-species\nin Bt-spent", "Full community in DM\nwith glucose replaced"))
  popViewport()
  pushViewport(bottom_panel)
  draw_panel(list(plot_groups$dm, plot_groups$pairs),
    c("Assemblies\nin DM", "Coexisting pairs\nin Bt-spent"))
  popViewport()
  draw_illustrator_safe_y_label()
  grid.text("With cofactor", x = unit(25.2, "mm"), y = unit(78.5, "mm"), just = "left",
    gp = gpar(fontfamily = "Arial", fontsize = 10, col = cofactor_colour))
  grid.text("Without cofactor", x = unit(25.2, "mm"), y = unit(74.2, "mm"), just = "left",
    gp = gpar(fontfamily = "Arial", fontsize = 10, col = resource_colour))
}

base_path <- file.path(
  figure_dir,
  "prediction_error_four_groups_two_rows_compact"
)

svglite::svglite(
  paste0(base_path, ".svg"),
  width = 237.310021 / 72,
  height = 239.757316 / 72,
  bg = "white"
)
draw_figure()
dev.off()

grDevices::cairo_pdf(
  paste0(base_path, ".pdf"),
  width = 237.66874 / 72,
  height = 239.748 / 72,
  family = "Arial",
  bg = "white"
)
draw_figure()
dev.off()

if (nzchar(Sys.getenv("FIG4_PREVIEW"))) {
ragg::agg_png(
  file.path(Sys.getenv("FIG4_PREVIEW"), paste0(basename(base_path), ".png")),
  width = 1979,
  height = 1995,
  units = "px",
  res = 600,
  background = "white"
)
draw_figure()
dev.off()
}

message(
  "Group counts: Complex=", length(plot_groups$complex$cofactor),
  ", Carbon=", length(plot_groups$carbon$cofactor),
  ", DM=", length(plot_groups$dm$cofactor),
  ", Pairs=", length(plot_groups$pairs$cofactor)
)
message("Saved: ", base_path, ".[svg|pdf]")
})

# ---- relative-abundance grid ----
local({
# Rebuild the 3 x 4 observed-versus-predicted relative-abundance grid in R.
# The geometry follows the existing manuscript figure. Nongrowers use the
# supplied R palette, while growers are shown in neutral grey.

required_packages <- c("svglite", "ragg")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages) > 0L) {
  stop(
    "Missing R package(s): ",
    paste(missing_packages, collapse = ", "),
    call. = FALSE
  )
}

command_args <- commandArgs(trailingOnly = FALSE)
script_arg <- grep("^--file=", command_args, value = TRUE)
if (length(script_arg) != 1L) {
  stop("Run this file with Rscript.", call. = FALSE)
}

script_path <- normalizePath(sub("^--file=", "", script_arg))
bundle_dir <- normalizePath(Sys.getenv("FIG4_DIR"), mustWork = TRUE)
figure_dir <- bundle_dir

read_species_matrix <- function(path) {
  frame <- read.csv(
    path,
    row.names = 1,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
  as.matrix(data.frame(lapply(frame, as.numeric), check.names = FALSE)) |>
    `rownames<-`(rownames(frame)) |>
    `colnames<-`(colnames(frame))
}

bt_observed <- read_species_matrix(file.path(bundle_dir, "bt_observed_relative.csv"))
bt_predicted <- read_species_matrix(file.path(bundle_dir, "bt_cofactor_on_relative.csv"))
dm_observed <- read_species_matrix(file.path(bundle_dir, "dm_observed_relative.csv"))
dm_predicted <- read_species_matrix(file.path(bundle_dir, "dm_cofactor_on_relative.csv"))

growers <- c(
  "Ai", "Ac", "Bfi", "Bf", "Bt", "Bu", "Bx", "Ba",
  "Csp", "Dl", "Ls", "Mf", "Pm", "Bd", "Bv", "Rg"
)
nongrowers <- c(
  "Af", "Ao", "As", "Bl.s", "Col", "Cs", "Et", "Eu.c",
  "Eu.l", "Im", "Ld", "Lsp", "Mi", "Pc", "Va", "Vp"
)
all_species <- c(growers, nongrowers)

species_colors <- c(
  "Ai" = "#1b9e77", "Af" = "#e31a1c",
  "Ao" = "#e7298a", "As" = "#fb9a99",
  "Ac" = "#f4a4a5", "Bfi" = "#cc6600",
  "Bf" = "#b35900", "Bt" = "#ff7f00",
  "Bu" = "#ffe6cc", "Bx" = "#ff9933",
  "Ba" = "#1f78b4", "Bl.s" = "#e8deed",
  "Csp" = "#986dc5", "Cs" = "#6a3d9a",
  "Col" = "#f2edf8", "Dl" = "#b2df8a",
  "Et" = "#8585ad", "Eu.c" = "#62a02c",
  "Eu.l" = "#4a7821", "Im" = "#52527a",
  "Lsp" = "#8dde87", "Ld" = "#33a02c",
  "Ls" = "#d9f4d7", "Mf" = "#fca636",
  "Mi" = "#fdbf6f", "Pm" = "#cab2d6",
  "Bd" = "#ffb366", "Bv" = "#ffcc99",
  "Pc" = "#cccc00", "Rg" = "#ffff99",
  "Va" = "#ecc2ac", "Vp" = "#b15928"
)

if (!setequal(names(species_colors), all_species)) {
  stop("The species palette does not match the canonical 32 species.", call. = FALSE)
}

panel_specs <- data.frame(
  panel = seq_len(12L),
  source = c(
    rep("bt", 10L),
    rep("dm", 2L)
  ),
  community = c(
    "Eu.c_Lsp",
    "Col_Va",
    "Cs_Va",
    "Col_Lsp_Va",
    "As_Cs_Eu.c_Vp",
    "Ao_Eu.l_Im_Mi_Pc",
    "Va_dropout",
    "Cs_dropout",
    "Mi_dropout",
    "Full_community",
    "Whole_community_13",
    "Full_community_32_Glucose"
  ),
  stringsAsFactors = FALSE
)

floor_abundance <- 1e-3

get_panel_data <- function(source, community, panel_index) {
  if (identical(source, "bt")) {
    observed_matrix <- bt_observed
    predicted_matrix <- bt_predicted
  } else {
    observed_matrix <- dm_observed
    predicted_matrix <- dm_predicted
  }

  if (!community %in% colnames(observed_matrix)) {
    stop("Community missing from observed matrix: ", community, call. = FALSE)
  }
  if (!community %in% colnames(predicted_matrix)) {
    stop("Community missing from predicted matrix: ", community, call. = FALSE)
  }

  species_here <- all_species[
    all_species %in% rownames(observed_matrix) &
      all_species %in% rownames(predicted_matrix)
  ]
  observed <- observed_matrix[species_here, community]
  predicted <- predicted_matrix[species_here, community]
  keep <- is.finite(observed) & is.finite(predicted)

  values <- data.frame(
    panel = panel_index,
    source = source,
    community = community,
    species = species_here[keep],
    observed = as.numeric(observed[keep]),
    predicted = as.numeric(predicted[keep]),
    stringsAsFactors = FALSE
  )
  values$observed_plot <- pmax(values$observed, floor_abundance)
  values$predicted_plot <- pmax(values$predicted, floor_abundance)
  values$abs_log2_error <- abs(
    log2(values$predicted_plot / values$observed_plot)
  )
  values
}

panel_data <- Map(
  get_panel_data,
  panel_specs$source,
  panel_specs$community,
  panel_specs$panel
)
names(panel_data) <- panel_specs$community

panel_summary <- do.call(
  rbind,
  lapply(panel_data, function(values) {
    data.frame(
      panel = values$panel[[1]],
      source = values$source[[1]],
      community = values$community[[1]],
      n_species = nrow(values),
      mean_abs_log2_error = mean(values$abs_log2_error),
      stringsAsFactors = FALSE
    )
  })
)
rownames(panel_summary) <- NULL

selected_pair <- panel_summary[
  panel_summary$community == "Cs_Va",
  ,
  drop = FALSE
]
selected_pair_values <- panel_data[["Cs_Va"]]
selected_higher <- panel_summary[
  panel_summary$community == "Col_Lsp_Va",
  ,
  drop = FALSE
]
# Retain the reference community; do not impose an old error target on new data.
if (nrow(selected_pair) != 1L || selected_pair$n_species != 2L) {
  stop("The reference Cs_Va pair is missing or has unexpected membership.")
}
if (
  nrow(selected_higher) != 1L ||
    selected_higher$n_species <= 2L ||
    selected_higher$n_species >= 10L
) {
  stop("The selected higher-order community is no longer between 2 and 10 species.", call. = FALSE)
}

source_data_path <- file.path(
  figure_dir,
  "relative_abundance_prediction_grid_3x4_source_data.csv"
)
summary_path <- file.path(
  figure_dir,
  "relative_abundance_prediction_grid_3x4_panel_summary.csv"
)
write.csv(do.call(rbind, panel_data), source_data_path, row.names = FALSE)
write.csv(panel_summary, summary_path, row.names = FALSE)

# Reference annotations repeat the full-community Eu.c coordinate; they are
# separate from the observed/predicted species records and error calculations.
euc_reference <- panel_data[["Full_community"]]
euc_reference <- euc_reference[euc_reference$species == "Eu.c", , drop = FALSE]
stopifnot(nrow(euc_reference) == 1L)
reference_panels <- c(7L, 8L, 9L, 10L)
reference_source <- data.frame(
  panel = reference_panels,
  community = panel_specs$community[reference_panels],
  reference_community = "Full_community",
  reference_species = "Eu.c",
  observed = euc_reference$observed,
  predicted = euc_reference$predicted,
  observed_plot = euc_reference$observed_plot,
  predicted_plot = euc_reference$predicted_plot,
  diameter_multiplier = 0.7,
  marker = "black outline, transparent fill"
)
write.csv(reference_source, file.path(figure_dir,
  "relative_abundance_prediction_grid_3x4_reference_markers.csv"), row.names = FALSE)

figure_width_in <- 4.8
figure_height_in <- 3.45
figure_width_mm <- figure_width_in * 25.4
figure_height_mm <- figure_height_in * 25.4

# Match the existing Matplotlib geometry after equal-aspect adjustment.
panel_left_fraction <- 0.095
panel_right_fraction <- 0.855
grid_bottom_fraction <- 0.125
grid_top_fraction <- 0.975
horizontal_space_ratio <- 0.18
vertical_space_ratio <- 0.04

panel_width_mm <- (
  (panel_right_fraction - panel_left_fraction) * figure_width_mm /
    (4 + 3 * horizontal_space_ratio)
)
horizontal_gap_mm <- horizontal_space_ratio * panel_width_mm
grid_row_height_mm <- (
  (grid_top_fraction - grid_bottom_fraction) * figure_height_mm /
    (3 + 2 * vertical_space_ratio)
)
vertical_grid_gap_mm <- vertical_space_ratio * grid_row_height_mm
panel_vertical_inset_mm <- (grid_row_height_mm - panel_width_mm) / 2

panel_x_mm <- panel_left_fraction * figure_width_mm +
  (0:3) * (panel_width_mm + horizontal_gap_mm)
panel_y_bottom_mm <- grid_bottom_fraction * figure_height_mm +
  panel_vertical_inset_mm +
  (0:2) * (grid_row_height_mm + vertical_grid_gap_mm)
panel_y_mm <- rev(panel_y_bottom_mm)

axis_color <- "#111111"
guide_color <- "#777777"
grower_color <- "#B8B8B8"

# R grid lwd = 1 is 0.75 pt in svglite output.
pt_to_grid_lwd <- function(points) points / 0.75
axis_lwd <- pt_to_grid_lwd(0.5)
tick_lwd <- pt_to_grid_lwd(0.5)
guide_lwd <- pt_to_grid_lwd(0.8)
point_edge_lwd <- pt_to_grid_lwd(0.25)
tick_font_size_pt <- 8

draw_power_label <- function(
  exponent,
  x_mm,
  y_mm,
  just = c("centre", "centre"),
  rotation = 0
) {
  label <- bquote(10^.(as.character(exponent)))
  grid::grid.text(
    label,
    x = grid::unit(x_mm, "mm"),
    y = grid::unit(y_mm, "mm"),
    just = just,
    rot = rotation,
    gp = grid::gpar(
      fontfamily = "Arial",
      fontsize = tick_font_size_pt,
      col = axis_color
    )
  )
}

draw_panel <- function(panel_index, values) {
  row_index <- (panel_index - 1L) %/% 4L + 1L
  column_index <- (panel_index - 1L) %% 4L + 1L
  x_mm <- panel_x_mm[[column_index]]
  y_mm <- panel_y_mm[[row_index]]

  grid::pushViewport(
    grid::viewport(
      x = grid::unit(x_mm, "mm"),
      y = grid::unit(y_mm, "mm"),
      width = grid::unit(panel_width_mm, "mm"),
      height = grid::unit(panel_width_mm, "mm"),
      just = c("left", "bottom"),
      xscale = c(-3, 0),
      yscale = c(-3, 0),
      clip = "off"
    )
  )

  grid::grid.lines(
    x = grid::unit(c(-3, 0), "native"),
    y = grid::unit(c(-3, 0), "native"),
    gp = grid::gpar(
      col = guide_color,
      lwd = guide_lwd,
      lty = c(2, 2),
      lineend = "butt"
    )
  )
  grid::grid.rect(
    gp = grid::gpar(
      fill = NA,
      col = axis_color,
      lwd = axis_lwd
    )
  )

  for (minor_value in c(-2, -1)) {
    grid::grid.lines(
      x = grid::unit(c(minor_value, minor_value), "native"),
      y = grid::unit(c(0, -2.2), c("npc", "points")),
      gp = grid::gpar(col = axis_color, lwd = tick_lwd)
    )
    grid::grid.lines(
      x = grid::unit(c(0, -2.2), c("npc", "points")),
      y = grid::unit(c(minor_value, minor_value), "native"),
      gp = grid::gpar(col = axis_color, lwd = tick_lwd)
    )
  }
  for (major_value in c(-3, 0)) {
    grid::grid.lines(
      x = grid::unit(c(major_value, major_value), "native"),
      y = grid::unit(c(0, -3.5), c("npc", "points")),
      gp = grid::gpar(col = axis_color, lwd = tick_lwd)
    )
    grid::grid.lines(
      x = grid::unit(c(0, -3.5), c("npc", "points")),
      y = grid::unit(c(major_value, major_value), "native"),
      gp = grid::gpar(col = axis_color, lwd = tick_lwd)
    )
  }

  x_native <- log10(values$predicted_plot)
  y_native <- log10(values$observed_plot)
  for (point_index in seq_len(nrow(values))) {
    species_name <- values$species[[point_index]]
    is_grower <- species_name %in% growers
    alpha_value <- if (species_name %in% growers) 0.88 else 0.96
    point_color <- if (species_name %in% growers) {
      grower_color
    } else {
      species_colors[[species_name]]
    }
    grid::grid.circle(
      x = grid::unit(x_native[[point_index]], "native"),
      y = grid::unit(y_native[[point_index]], "native"),
      r = grid::unit(sqrt(38) / 2 * if (is_grower) 0.8 else 1, "points"),
      gp = grid::gpar(
        fill = grDevices::adjustcolor(
          point_color,
          alpha.f = alpha_value
        ),
        col = if (is_grower) NA else axis_color,
        lwd = if (is_grower) 0 else point_edge_lwd
      )
    )
  }

  if (panel_index %in% reference_panels) {
    grid::grid.circle(
      x = grid::unit(log10(euc_reference$predicted_plot), "native"),
      y = grid::unit(log10(euc_reference$observed_plot), "native"),
      r = grid::unit(sqrt(38) / 2 * 0.7, "points"),
      gp = grid::gpar(fill = NA, col = "black", lwd = point_edge_lwd)
    )
  }

  grid::popViewport()

  if (row_index == 3L) {
    draw_power_label(
      -3,
      x_mm + 0.35,
      y_mm - 4.0,
      just = c("centre", "centre")
    )
    draw_power_label(
      0,
      x_mm + panel_width_mm - 0.35,
      y_mm - 4.0,
      just = c("centre", "centre")
    )
  }
  if (column_index == 1L) {
    draw_power_label(
      -3,
      x_mm - 3.1,
      y_mm,
      just = c("right", "centre")
    )
    draw_power_label(
      0,
      x_mm - 3.1,
      y_mm + panel_width_mm,
      just = c("right", "centre")
    )
  }
}

draw_figure <- function() {
  grid::grid.newpage()
  grid::pushViewport(
    grid::viewport(
      x = grid::unit(0, "mm"),
      y = grid::unit(0, "mm"),
      width = grid::unit(figure_width_mm, "mm"),
      height = grid::unit(figure_height_mm, "mm"),
      just = c("left", "bottom"),
      clip = "off"
    )
  )

  for (panel_index in seq_len(nrow(panel_specs))) {
    draw_panel(panel_index, panel_data[[panel_index]])
  }

  grid::grid.text(
    "Predicted relative abundance",
    x = grid::unit(0.475 * figure_width_mm, "mm"),
    y = grid::unit(3.1, "mm"),
    gp = grid::gpar(
      fontfamily = "Arial",
      fontsize = 10,
      col = axis_color
    )
  )
  grid::grid.text(
    "Actual relative abundance",
    x = grid::unit(3.0, "mm"),
    y = grid::unit(0.54 * figure_height_mm, "mm"),
    rot = 90,
    gp = grid::gpar(
      fontfamily = "Arial",
      fontsize = 10,
      col = axis_color
    )
  )

  legend_x_mm <- 0.875 * figure_width_mm
  legend_bottom_mm <- 0.13 * figure_height_mm
  legend_height_mm <- 0.80 * figure_height_mm
  legend_y_mm <- legend_bottom_mm + rev(seq(0.02, 0.98, length.out = 16)) *
    legend_height_mm
  for (legend_index in seq_along(nongrowers)) {
    species_name <- nongrowers[[legend_index]]
    grid::grid.text(
      species_name,
      x = grid::unit(legend_x_mm, "mm"),
      y = grid::unit(legend_y_mm[[legend_index]], "mm"),
      just = c("left", "centre"),
      gp = grid::gpar(
        fontfamily = "Arial",
        fontface = "italic",
        fontsize = 10,
        col = species_colors[[species_name]]
      )
    )
  }

  grid::popViewport()
  invisible(NULL)
}

output_base <- file.path(
  figure_dir,
  "relative_abundance_prediction_grid_3x4"
)

svglite::svglite(
  paste0(output_base, ".svg"),
  width = figure_width_in,
  height = figure_height_in,
  bg = "white"
)
draw_figure()
grDevices::dev.off()

grDevices::cairo_pdf(
  paste0(output_base, ".pdf"),
  width = figure_width_in,
  height = figure_height_in,
  family = "Arial",
  bg = "white"
)
draw_figure()
grDevices::dev.off()

if (nzchar(Sys.getenv("FIG4_PREVIEW"))) {
ragg::agg_png(
  file.path(Sys.getenv("FIG4_PREVIEW"), paste0(basename(output_base), ".png")),
  width = figure_width_in,
  height = figure_height_in,
  units = "in",
  res = 600,
  background = "white"
)
draw_figure()
grDevices::dev.off()
}


cat("Selected coexisting pair: Cs_Va; mean abs log2 error = ",
    sprintf("%.3f", selected_pair$mean_abs_log2_error), "\n", sep = "")
cat("Selected higher-order community: Col_Lsp_Va; n = ",
    selected_higher$n_species, "; mean abs log2 error = ",
    sprintf("%.3f", selected_higher$mean_abs_log2_error), "\n", sep = "")
cat("Saved: ", paste0(output_base, ".svg"), "\n", sep = "")
cat("Saved: ", paste0(output_base, ".pdf"), "\n", sep = "")
cat("Saved: ", source_data_path, "\n", sep = "")
cat("Saved: ", summary_path, "\n", sep = "")
})
'''


if __name__ == '__main__':
    main()
