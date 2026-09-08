#!/usr/bin/env Rscript

# Draw Figure S3 from the current source CSVs prepared by rebuild_figure_s3.py.
# Preserve the original R style; final figure exports are SVG and PDF only.
output_dir <- Sys.getenv("FIG_S3_OUTPUT")
preview_dir <- Sys.getenv("FIG_S3_PREVIEW")
if (!dir.exists(output_dir)) stop("Missing FIG_S3_OUTPUT directory.")

# ---- lag heatmap ----
local({
suppressPackageStartupMessages({
  library(ComplexHeatmap)
  library(circlize)
  library(grid)
})
script_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_argument) != 1L) stop("Run this file with Rscript.")
script_path <- normalizePath(sub("^--file=", "", script_argument), mustWork = TRUE)
bundle_dir <- output_dir
figure_dir <- bundle_dir
output_stem <- file.path(bundle_dir, "actual_lag_time_manual_bt_monoculture_and_double_spent_heatmap")
table <- read.csv(file.path(bundle_dir, "actual_lag_time_heatmap_source_data.csv"), check.names = FALSE, stringsAsFactors = FALSE)
if (!"focal_species" %in% names(table)) stop("Missing focal_species in heatmap source data.")
rownames(table) <- table$focal_species
table$focal_species <- NULL
lag_time <- as.matrix(table)
storage.mode(lag_time) <- "double"
colnames(lag_time)[colnames(lag_time) == "Manual lag (used in model)"] <- "Manual lag"

font_family <- "Arial"
label_size <- 10
tick_size <- 8
cell_size_mm <- 5
axis_line_lwd <- 2 / 3
cell_line_lwd <- 0.45
lag_time_col_fun <- colorRamp2(
  c(0, 20 / 3, 40 / 3, 20),
  c("#FFFFFF", "#9CCAE3", "#6BAED6", "#08519C")
)

heatmap <- Heatmap(
  lag_time,
  name = "lag time",
  col = lag_time_col_fun,
  na_col = "#808080",
  cluster_rows = FALSE,
  cluster_columns = FALSE,
  show_heatmap_legend = FALSE,
  show_row_names = TRUE,
  row_names_side = "left",
  row_names_gp = gpar(fontsize = label_size, fontfamily = font_family, col = "#111111"),
  show_column_names = TRUE,
  column_names_side = "bottom",
  column_names_rot = 90,
  column_names_gp = gpar(fontsize = label_size, fontfamily = font_family, col = "#111111"),
  column_names_max_height = unit(25, "mm"),
  row_title = "Species",
  row_title_side = "left",
  row_title_rot = 90,
  row_title_gp = gpar(fontsize = label_size, fontfamily = font_family, col = "#111111"),
  column_title = "Spent medium of",
  column_title_side = "bottom",
  column_title_gp = gpar(fontsize = label_size, fontfamily = font_family, col = "#111111"),
  border = TRUE,
  border_gp = gpar(col = "black", lwd = axis_line_lwd),
  rect_gp = gpar(col = "white", lwd = cell_line_lwd),
  width = unit(ncol(lag_time) * cell_size_mm, "mm"),
  height = unit(nrow(lag_time) * cell_size_mm, "mm")
)

legend <- Legend(
  col_fun = lag_time_col_fun,
  at = c(0, 5, 10, 15, 20),
  labels = c("0", "5", "10", "15", "20"),
  title = "Lag time (h)",
  title_position = "leftcenter-rot",
  title_gp = gpar(fontsize = label_size, fontfamily = font_family, col = "#111111"),
  labels_gp = gpar(fontsize = tick_size, fontfamily = font_family, col = "#111111"),
  legend_height = unit(35, "mm"),
  grid_width = unit(4, "mm"),
  tick_length = unit(0, "mm"),
  border = NA
)

draw_heatmap <- function() {
  draw(
    heatmap,
    heatmap_legend_list = list(legend),
    heatmap_legend_side = "right",
    padding = unit(c(4, 4, 9, 4), "mm")
  )
  for (separator_index in c(1, 2)) {
    decorate_heatmap_body("lag time", {
      separator_x <- unit(separator_index / ncol(lag_time), "npc")
      grid.lines(
        x = unit.c(separator_x, separator_x),
        y = unit(c(0, 1), "npc"),
        gp = gpar(col = "black", lwd = axis_line_lwd)
      )
    })
  }
}

width_mm <- 100
height_mm <- 90
width_in <- width_mm / 25.4
height_in <- height_mm / 25.4

svglite::svglite(paste0(output_stem, ".svg"), width = width_in, height = height_in, bg = "white")
draw_heatmap()
dev.off()

if (nzchar(preview_dir)) {
ragg::agg_png(
  file.path(preview_dir, paste0(basename(output_stem), ".png")),
  width = width_in,
  height = height_in,
  units = "in",
  res = 160,
  background = "white"
)
draw_heatmap()
dev.off()}


grDevices::cairo_pdf(
  paste0(output_stem, ".pdf"),
  width = width_in,
  height = height_in,
  family = font_family
)
draw_heatmap()
dev.off()

message(sprintf("Wrote R-generated raw lag-time figure: %s.[svg|pdf]", output_stem))
message("Missing/unavailable lag values are encoded as NA in the saved source CSV.")
})

# ---- lag scatter ----
local({
script_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_argument) != 1L) stop("Run this file with Rscript.")
script_path <- normalizePath(sub("^--file=", "", script_argument), mustWork = TRUE)
bundle_dir <- output_dir
output_stem <- file.path(bundle_dir, "manual_vs_bt_monoculture_lag_scatter")
scatter_data <- read.csv(file.path(bundle_dir, "manual_vs_bt_monoculture_lag_scatter_source_data.csv"), check.names = FALSE, stringsAsFactors = FALSE)
required_columns <- c("species", "bt_monoculture_lag_h", "manually_set_lag_h", "color")
if (!all(required_columns %in% names(scatter_data))) stop("Scatter source data are incomplete.")

font_family <- "Arial"
label_size <- 10
tick_size <- 8
axis_line_lwd <- 2 / 3
point_size <- 2.2
# Expand only as needed to show every current lag; retain 5-hour tick spacing.
x_max <- max(15, ceiling(max(scatter_data$bt_monoculture_lag_h, na.rm = TRUE) / 5) * 5)
y_max <- max(20, ceiling(max(scatter_data$manually_set_lag_h, na.rm = TRUE) / 5) * 5)

draw_scatter <- function() {
  old_parameters <- par(no.readonly = TRUE)
  on.exit(par(old_parameters), add = TRUE)
  par(
    family = font_family,
    mar = c(4.5, 4.6, 0.6, 8.6),
    cex.axis = tick_size / 12,
    cex.lab = label_size / 12,
    lwd = axis_line_lwd,
    las = 1,
    xpd = NA
  )
  plot(
    scatter_data$bt_monoculture_lag_h,
    scatter_data$manually_set_lag_h,
    type = "n",
    xlim = c(0, x_max),
    ylim = c(0, y_max),
    xaxs = "i",
    yaxs = "i",
    xlab = "Lag time in monoculture (h)",
    ylab = "Manually set lag time (h)",
    axes = FALSE
  )
  # Reference line: equal manually set and monoculture lag times.
  segments(0, 0, min(x_max, y_max), min(x_max, y_max), lty = 2, lwd = axis_line_lwd, col = "#111111")
  axis(1, at = seq(0, x_max, by = 5), lwd = axis_line_lwd, lwd.ticks = axis_line_lwd)
  axis(2, at = seq(0, y_max, by = 5), lwd = axis_line_lwd, lwd.ticks = axis_line_lwd)
  points(
    scatter_data$bt_monoculture_lag_h,
    scatter_data$manually_set_lag_h,
    pch = 16,
    col = scatter_data$color,
    cex = point_size
  )
  legend_y <- c(18.7, 17.0, 15.3, 13.6, 11.9)
  left_indices <- 1:5
  right_indices <- 6:10
  text(
    x = x_max * (16.3 / 15),
    y = legend_y,
    labels = scatter_data$species[left_indices],
    col = scatter_data$color[left_indices],
    font = 3,
    adj = c(0, 0.5),
    cex = 0.88
  )
  text(
    x = x_max * (19.2 / 15),
    y = legend_y,
    labels = scatter_data$species[right_indices],
    col = scatter_data$color[right_indices],
    font = 3,
    adj = c(0, 0.5),
    cex = 0.88
  )
}

width_mm <- 124
height_mm <- 88
width_in <- width_mm / 25.4
height_in <- height_mm / 25.4

svglite::svglite(paste0(output_stem, ".svg"), width = width_in, height = height_in, bg = "white")
draw_scatter()
dev.off()

if (nzchar(preview_dir)) {
ragg::agg_png(
  file.path(preview_dir, paste0(basename(output_stem), ".png")),
  width = width_in,
  height = height_in,
  units = "in",
  res = 160,
  background = "white"
)
draw_scatter()
dev.off()}


grDevices::cairo_pdf(
  paste0(output_stem, ".pdf"),
  width = width_in,
  height = height_in,
  family = font_family
)
draw_scatter()
dev.off()

message(sprintf("Wrote R-generated lag scatter: %s.[svg|pdf]", output_stem))
})
