#!/usr/bin/env Rscript
# FigureS3: read the existing CSVs in ../FigureS3 and save SVG/PDF figures there.
# Run: Rscript figure_plots/scripts/FigureS3.R (from the package root).
# These scripts only draw figures; they do not rerun model predictions.

suppressPackageStartupMessages({
  library(grid)
  library(ComplexHeatmap)
  library(circlize)
})

script_file <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))
figure_dir <- normalizePath(file.path(dirname(script_file), "..", "FigureS3"))

# SVG and PDF use the original dimensions (in inches).
save_figure <- function(stem, draw, width, height,
                        pdf_width = width, pdf_height = height) {
  svglite::svglite(paste0(stem, ".svg"), width = width, height = height, bg = "white")
  draw()
  dev.off()
  grDevices::cairo_pdf(paste0(stem, ".pdf"), width = pdf_width,
                      height = pdf_height, family = "Arial", bg = "white")
  draw()
  invisible(dev.off())
}

# ---- lag heatmap ----
local({
  output_stem <- file.path(figure_dir, "actual_lag_time_manual_bt_monoculture_and_double_spent_heatmap")
  table <- read.csv(file.path(figure_dir, "actual_lag_time_heatmap_source_data.csv"), check.names = FALSE, stringsAsFactors = FALSE)
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

  save_figure(output_stem, draw_heatmap, width_in, height_in)

})
