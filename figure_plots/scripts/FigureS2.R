#!/usr/bin/env Rscript
# FigureS2: read the existing CSVs in ../FigureS2 and save SVG/PDF figures there.
# Run: Rscript figure_plots/scripts/FigureS2.R (from the package root).
# These scripts only draw figures; they do not rerun model predictions.

suppressPackageStartupMessages({
  library(grid)
  library(ComplexHeatmap)
  library(circlize)
})

script_file <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))
figure_dir <- normalizePath(file.path(dirname(script_file), "..", "FigureS2"))

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

# ---- full 32-species CR/PR matrix ----
local({
  # Figure S2A: combined consumption + production matrix for the full
  # 32-species panel. Consumption cells use the parameter-matrix blue ramp,
  # production cells the red ramp; the two never overlap in a cell, so both
  # fit in a single heatmap. Rows keep the file order (nongrowers then
  # growers) with no row split. Colorbars follow
  # plot_parameter_matrix_bt_resource_consumption_production.R.

  # Measure text on the same Cairo/Arial device used for the final PDF.
  grDevices::cairo_pdf(tempfile("figure_s2_layout_", fileext = ".pdf"),
                      width = 165 / 25.4, height = 150 / 25.4, family = "Arial")

  read_species_matrix <- function(path) {
    frame <- read.csv(
      path,
      check.names = FALSE,
      stringsAsFactors = FALSE,
      fileEncoding = "UTF-8-BOM"
    )
    if (!"species" %in% names(frame)) {
      stop("Missing 'species' column in ", path, call. = FALSE)
    }
    rownames(frame) <- frame$species
    frame$species <- NULL
    matrix <- as.matrix(frame)
    storage.mode(matrix) <- "double"
    matrix
  }

  consumption_matrix <- read_species_matrix(
    file.path(figure_dir, "full_32_CR_matrix.csv")
  )
  production_matrix <- read_species_matrix(
    file.path(figure_dir, "full_32_PR_matrix.csv")
  )

  if (
    !identical(rownames(consumption_matrix), rownames(production_matrix)) ||
      !identical(colnames(consumption_matrix), colnames(production_matrix))
  ) {
    stop("Consumption and production matrices are not aligned.", call. = FALSE)
  }
  if (any(consumption_matrix > 0 & production_matrix > 0)) {
    stop("A cell carries both consumption and production.", call. = FALSE)
  }

  # Resource abundances for the top bar. The matrix columns are not in the
  # abundance-table order, so align explicitly by resource_id.
  dm_resource_table <- read.csv(
    file.path(figure_dir, "dm_resource_y0.csv"),
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
  if (!all(c("resource_id", "dm_y0") %in% names(dm_resource_table))) {
    stop("The DM resource table is missing required columns.", call. = FALSE)
  }
  if (!all(dm_resource_table$resource_id %in% colnames(consumption_matrix))) {
    stop("A DM resource is missing from the matrix columns.", call. = FALSE)
  }
  dm_resource_values <- setNames(
    as.numeric(dm_resource_table$dm_y0),
    dm_resource_table$resource_id
  )

  # Keep only resources whose total abundance is at least 1.5e-3. Abundance
  # comes from the DM table for DM resources and from the Bt table for the
  # Bt-only Resource_* columns. This drops the near-zero U_Bfi / U_Bf, the
  # five ~1e-3 U_* resources, and the zero-abundance Resource_11-14,
  # leaving 12 DM resources and 13 Bt-produced resources.
  bt_resource_table <- read.csv(
    file.path(figure_dir, "bt_resource_Y0.csv"),
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
  bt_resource_values <- setNames(
    as.numeric(bt_resource_table$bt_resource_y0),
    bt_resource_table$resource_id
  )
  abundance_floor <- 1.5e-3
  all_columns <- colnames(consumption_matrix)
  column_abundance <- ifelse(
    all_columns %in% names(dm_resource_values),
    unname(dm_resource_values[all_columns]),
    unname(bt_resource_values[all_columns])
  )
  if (anyNA(column_abundance)) {
    stop("A matrix column has no abundance in either resource table.", call. = FALSE)
  }
  keep_columns <- column_abundance >= abundance_floor
  dropped_resources <- all_columns[!keep_columns]
  consumption_matrix <- consumption_matrix[, keep_columns, drop = FALSE]
  production_matrix <- production_matrix[, keep_columns, drop = FALSE]

  # Plot retained DM resources first and retained Bt-produced resources second.
  # This changes only the display order; no source values are altered.
  retained_columns <- colnames(consumption_matrix)
  dm_columns <- retained_columns[retained_columns %in% names(dm_resource_values)]
  bt_columns <- retained_columns[retained_columns %in% names(bt_resource_values)]
  if (length(dm_columns) + length(bt_columns) != length(retained_columns)) {
    stop("A retained resource could not be classified as DM or Bt-produced.", call. = FALSE)
  }
  plot_columns <- c(dm_columns, bt_columns)
  consumption_matrix <- consumption_matrix[, plot_columns, drop = FALSE]
  production_matrix <- production_matrix[, plot_columns, drop = FALSE]

  # Linear resource-abundance bar. DM resources are blue and Bt-produced
  # resources are red, matching their requested grouping in the matrix.
  resource_bar_values <- unname(column_abundance[match(
    plot_columns,
    all_columns
  )])
  resource_bar_colours <- c(
    rep("#08519C", length(dm_columns)),
    rep("#D73027", length(bt_columns))
  )

  # Ramps identical to the parameter-matrix figure.
  consumption_col_fun <- colorRamp2(
    c(0, 8 / 3, 16 / 3, 8),
    c("#FFFFFF", "#9CCAE3", "#6BAED6", "#08519C")
  )
  production_col_fun <- colorRamp2(
    c(0, 0.6),
    c("#FFFFFF", "#D73027")
  )

  # R grid's lwd = 1 is exported by svglite as 0.75 pt.
  axis_line_lwd <- 2 / 3
  cell_size_mm <- 3.65

  n_rows <- nrow(consumption_matrix)
  n_cols <- ncol(consumption_matrix)
  resource_bar_ylim_max <- 0.4
  if (max(resource_bar_values) > resource_bar_ylim_max) {
    resource_bar_ylim_max <- ceiling(max(resource_bar_values) * 10) / 10
  }

  top_annotation <- HeatmapAnnotation(
    `Resource abundance` = anno_barplot(
      resource_bar_values,
      gp = gpar(fill = resource_bar_colours, col = NA),
      border = FALSE,
      baseline = 0,
      ylim = c(0, resource_bar_ylim_max),
      axis = TRUE,
      axis_param = list(
        side = "left",
        at = seq(0, resource_bar_ylim_max, by = 0.1),
        labels = format(
          seq(0, resource_bar_ylim_max, by = 0.1),
          nsmall = 1,
          trim = TRUE
        ),
        gp = gpar(fontsize = 8, fontfamily = "Arial", lwd = axis_line_lwd),
        facing = "outside"
      ),
      height = unit(15, "mm")
    ),
    # Reserve the annotation-name gutter, then replace this transparent
    # placeholder with an Illustrator-safe vector/text label after draw().
    annotation_label = "YmuOD",
    annotation_name_side = "left",
    annotation_name_rot = 90,
    annotation_name_gp = gpar(
      fontsize = 10,
      fontfamily = "Arial",
      col = "transparent"
    ),
    gap = unit(0, "mm")
  )

  legend_annotation <- rowAnnotation(
    scales = anno_empty(
      border = FALSE,
      width = unit(24, "mm")
    )
  )

  heatmap <- Heatmap(
    consumption_matrix,
    name = "consumption",
    col = consumption_col_fun,
    cluster_rows = FALSE,
    cluster_columns = FALSE,
    show_row_names = TRUE,
    row_names_side = "left",
    row_names_gp = gpar(fontsize = 10, fontfamily = "Arial", col = "#111111"),
    show_column_names = FALSE,
    column_title = "Coarse-grained resources",
    column_title_side = "bottom",
    column_title_gp = gpar(
      fontsize = 10,
      fontfamily = "Arial",
      col = "#111111"
    ),
    show_heatmap_legend = FALSE,
    border = TRUE,
    border_gp = gpar(col = "black", lwd = axis_line_lwd),
    rect_gp = gpar(type = "none"),
    width = unit(n_cols * cell_size_mm, "mm"),
    height = unit(n_rows * cell_size_mm, "mm"),
    top_annotation = top_annotation,
    right_annotation = legend_annotation,
    cell_fun = function(j, i, x, y, width, height, fill) {
      consumption_value <- consumption_matrix[i, j]
      production_value <- production_matrix[i, j]
      cell_fill <- if (consumption_value > 0) {
        consumption_col_fun(consumption_value)
      } else if (production_value > 0) {
        production_col_fun(production_value)
      } else {
        "#FFFFFF"
      }
      grid.rect(
        x = x,
        y = y,
        width = width,
        height = height,
        gp = gpar(fill = cell_fill, col = "white", lwd = 0.45)
      )
    }
  )

  # Illustrator can distort an isolated, rotated one-letter SVG text object.
  # Draw only the capital Y as vector geometry; keep mu and OD as editable
  # Arial text so the full mathematical label remains easy to edit.
  draw_vector_y <- function(
    left_mm = 1.5,
    center_y_mm = 4.0,
    width_mm = 2.5,
    height_mm = 3.5,
    colour = "#111111"
  ) {
    x_norm <- c(0, 0.14, 0.50, 0.86, 1, 0.58, 0.58, 0.42, 0.42)
    y_norm <- c(1, 1, 0.58, 1, 1, 0.47, 0, 0, 0.47)
    grid.polygon(
      x = unit(left_mm + width_mm * x_norm, "mm"),
      y = unit(center_y_mm - height_mm / 2 + height_mm * y_norm, "mm"),
      gp = gpar(fill = colour, col = NA)
    )
  }

  draw_illustrator_safe_y_label <- function() {
    pushViewport(
      viewport(
        x = unit(-9.5, "mm"),
        y = unit(0.5, "npc"),
        width = unit(24, "mm"),
        height = unit(7.5, "mm"),
        angle = 90
      )
    )
    draw_vector_y()
    grid.text(
      "μ",
      x = unit(5.0, "mm"),
      y = unit(2.3, "mm"),
      just = "left",
      gp = gpar(fontsize = 7, fontfamily = "Arial", col = "#111111")
    )
    grid.text(
      "(OD)",
      x = unit(7.2, "mm"),
      y = unit(4.0, "mm"),
      just = "left",
      gp = gpar(fontsize = 10, fontfamily = "Arial", col = "#111111")
    )
    popViewport()
  }

  # Match the narrow, borderless gradient bars used elsewhere in the manuscript:
  # compact vertical ramp, short ticks on the right, and right-aligned labels.
  draw_custom_colorbar <- function(
    col_fun,
    limits,
    at,
    labels,
    center_y,
    title
  ) {
    bar_left <- unit(0.3, "mm")
    bar_width_mm <- 2.2
    bar_height_mm <- 10.5
    bar_bottom <- unit(center_y, "npc") - unit(bar_height_mm / 2, "mm")
    n_strips <- 128
    strip_height_mm <- bar_height_mm / n_strips
    strip_values <- seq(limits[1], limits[2], length.out = n_strips)
    strip_colours <- col_fun(strip_values)

    for (strip_index in seq_len(n_strips)) {
      grid.rect(
        x = bar_left + unit(bar_width_mm / 2, "mm"),
        y = bar_bottom + unit((strip_index - 0.5) * strip_height_mm, "mm"),
        width = unit(bar_width_mm, "mm"),
        height = unit(strip_height_mm * 1.03, "mm"),
        gp = gpar(fill = strip_colours[strip_index], col = NA)
      )
    }

    bar_right <- bar_left + unit(bar_width_mm, "mm")
    for (tick_index in seq_along(at)) {
      tick_fraction <- (at[tick_index] - limits[1]) / diff(limits)
      tick_y <- bar_bottom + unit(tick_fraction * bar_height_mm, "mm")
      grid.lines(
        x = unit.c(bar_right, bar_right + unit(0.65, "mm")),
        y = unit.c(tick_y, tick_y),
        gp = gpar(col = "black", lwd = axis_line_lwd)
      )
      grid.text(
        labels[tick_index],
        x = bar_right + unit(1.0, "mm"),
        y = tick_y,
        just = "left",
        gp = gpar(fontsize = 7, fontfamily = "Arial", col = "#111111")
      )
    }

    grid.text(
      title,
      x = bar_left,
      y = bar_bottom + unit(bar_height_mm + 2.0, "mm"),
      just = c("left", "bottom"),
      gp = gpar(fontsize = 10, fontfamily = "Arial", col = "#111111")
    )
  }

  draw_figure <- function() {
    ht_opt$COLUMN_ANNO_PADDING <- unit(0, "mm")
    ht_opt$ROW_ANNO_PADDING <- unit(0, "mm")
    draw(
      heatmap,
      padding = unit(c(2.5, 2.5, 2.5, 2.5), "mm")
    )
    decorate_annotation("Resource abundance", {
      draw_illustrator_safe_y_label()
      if (length(dm_columns) > 0L && length(bt_columns) > 0L) {
        group_boundary <- length(dm_columns) / n_cols
        grid.lines(
          x = unit.c(unit(group_boundary, "npc"), unit(group_boundary, "npc")),
          y = unit.c(unit(0, "npc"), unit(1, "npc")),
          gp = gpar(col = "black", lwd = axis_line_lwd)
        )
      }
    })
    decorate_heatmap_body("consumption", {
      if (length(dm_columns) > 0L && length(bt_columns) > 0L) {
        group_boundary <- length(dm_columns) / n_cols
        grid.lines(
          x = unit.c(unit(group_boundary, "npc"), unit(group_boundary, "npc")),
          y = unit.c(unit(0, "npc"), unit(1, "npc")),
          gp = gpar(col = "black", lwd = axis_line_lwd)
        )
      }
    })
    decorate_annotation("scales", {
      draw_custom_colorbar(
        consumption_col_fun,
        limits = c(0, 8),
        at = c(0, 2, 4, 6, 8),
        labels = c("0", "2", "4", "6", "8"),
        center_y = 0.70,
        title = "Consumption"
      )
      draw_custom_colorbar(
        production_col_fun,
        limits = c(0, 0.6),
        at = c(0, 0.2, 0.4, 0.6),
        labels = c("0", "0.2", "0.4", "0.6"),
        center_y = 0.47,
        title = "Production"
      )
    })
  }

  grDevices::dev.off()  # Close the temporary layout-measurement device.

  output_base <- file.path(
    figure_dir,
    "full_32_consumption_production_matrix"
  )
  width_mm <- 165
  height_mm <- 150
  width_in <- width_mm / 25.4
  height_in <- height_mm / 25.4

  save_figure(output_base, draw_figure, width_in, height_in)

})

# ---- nongrower assembly prediction matrix ----
local({
  # Plot the finalized nongrower assemblies in Bt-spent medium as an assembly
  # membership matrix with the mean absolute log2 prediction error above each
  # column and a matching error histogram at right. The script reads only the
  # accompanying wide-form source-data CSV: the header row holds the assembly
  # names, the next 16 rows hold one row per species (a non-empty cell means
  # the species is presented in that assembly; the cell value itself is the
  # predicted relative abundance and is not used here), and a final
  # "mean_abs_log2_error" row holds the per-assembly prediction error.

  data_path <- file.path(
    figure_dir,
    "nongrower_bt_spent_assembly_predictions.csv"
  )
  output_stem <- file.path(
    figure_dir,
    "nongrower_bt_spent_assembly_prediction_matrix"
  )

  raw_lines <- read.csv(
    data_path,
    check.names = FALSE,
    stringsAsFactors = FALSE,
    colClasses = "character"
  )

  # Drop the fully empty padding rows below the error row.
  non_empty_rows <- apply(
    raw_lines,
    1L,
    function(row) any(!is.na(row) & nzchar(trimws(row)))
  )
  raw_lines <- raw_lines[non_empty_rows, , drop = FALSE]

  if (names(raw_lines)[1] != "species") {
    stop("The first column must be 'species'.", call. = FALSE)
  }
  error_row_index <- which(raw_lines[[1]] == "mean_abs_log2_error")
  if (length(error_row_index) != 1L) {
    stop("Expected exactly one 'mean_abs_log2_error' row.", call. = FALSE)
  }

  species_names <- raw_lines[[1]][seq_len(error_row_index - 1L)]
  n_species <- length(species_names)
  if (n_species != 16L) {
    stop("Expected 16 species rows above the error row.", call. = FALSE)
  }

  assembly_names <- names(raw_lines)[-1]
  n_assemblies <- length(assembly_names)

  membership <- matrix(
    0L,
    nrow = n_species,
    ncol = n_assemblies,
    dimnames = list(species_names, assembly_names)
  )
  species_cells <- as.matrix(raw_lines[seq_len(error_row_index - 1L), -1, drop = FALSE])
  membership[!is.na(species_cells) & nzchar(trimws(species_cells))] <- 1L

  errors <- as.numeric(as.character(raw_lines[error_row_index, -1]))
  if (anyNA(errors)) {
    stop("The error row contains a non-numeric value.", call. = FALSE)
  }

  # The wide file carries no assembly_type column; recover it from the
  # membership counts (2 = pairwise, 3-5 = higher_order, 15 = dropout,
  # 16 = full_community). Only the group boundaries are used downstream.
  membership_counts <- colSums(membership)
  assembly_types <- ifelse(
    membership_counts == 2L,
    "pairwise",
    ifelse(
      membership_counts == 16L,
      "full_community",
      ifelse(membership_counts == 15L, "dropout", "higher_order")
    )
  )

  assembly_table <- data.frame(
    assembly = assembly_names,
    assembly_type = assembly_types,
    assembly_mean_abs_log2_error = errors,
    stringsAsFactors = FALSE
  )
  species_table <- data.frame(
    species = species_names,
    stringsAsFactors = FALSE
  )

  # Match the established deep blue used by the other finalized R figures.
  deep_blue <- "#08519C"
  histogram_blue <- "#9ECAE1"
  separator_grey <- "#D9D9D9"
  axis_line_lwd <- 2 / 3  # 0.5 pt in the exported svglite output
  base_family <- "Arial"

  errors <- assembly_table$assembly_mean_abs_log2_error
  y_axis_max <- 6
  display_errors <- pmin(errors, y_axis_max)
  y_ticks <- c(2, 4, 6)
  mean_error <- mean(errors)
  histogram_breaks <- seq(0, y_axis_max, by = 0.3)
  error_histogram <- hist(
    display_errors,
    breaks = histogram_breaks,
    plot = FALSE,
    right = FALSE,
    include.lowest = TRUE
  )
  histogram_x_max <- max(error_histogram$counts)
  if (!is.finite(histogram_x_max) || histogram_x_max <= 0) {
    stop("The prediction-error histogram has no finite counts.")
  }

  group_ends <- which(
    assembly_table$assembly_type[-n_assemblies] !=
      assembly_table$assembly_type[-1]
  )

  draw_figure <- function() {
    layout(
      matrix(c(1, 2, 3, 4), nrow = 2, byrow = TRUE),
      widths = c(8.0, 0.75),
      heights = c(0.95, 2.15)
    )

    # Error bars. The zero line is also the top border of the assembly matrix.
    par(
      mar = c(0, 4.9, 0.35, 0.10),
      mgp = c(2.25, 0.55, 0),
      tcl = -0.23,
      xaxs = "i",
      yaxs = "i",
      family = base_family,
      ps = 10
    )
    plot.new()
    plot.window(
      xlim = c(0.5, n_assemblies + 0.5),
      ylim = c(0, y_axis_max)
    )
    rect(
      xleft = seq_len(n_assemblies) - 0.36,
      ybottom = 0,
      xright = seq_len(n_assemblies) + 0.36,
      ytop = display_errors,
      col = deep_blue,
      border = NA
    )
    axis(
      side = 2,
      at = y_ticks,
      labels = y_ticks,
      las = 1,
      cex.axis = 0.8,
      lwd = axis_line_lwd,
      lwd.ticks = axis_line_lwd
    )
    segments(
      x0 = 0.5,
      y0 = 0,
      x1 = n_assemblies + 0.5,
      y1 = 0,
      lwd = axis_line_lwd,
      col = "black"
    )
    segments(
      x0 = 0.5,
      y0 = 0,
      x1 = 0.5,
      y1 = y_axis_max,
      lwd = axis_line_lwd,
      col = "black"
    )
    mtext(
      expression(atop(Error~"("*log[2]~fc, per~species*")")),
      side = 2,
      line = 2.35,
      cex = 1.0,
      family = base_family
    )

    # Horizontal histogram aligned to the same prediction-error scale.
    par(
      mar = c(0, 0.10, 0.35, 0.30),
      xaxs = "i",
      yaxs = "i",
      family = base_family,
      ps = 10
    )
    plot.new()
    plot.window(
      xlim = c(0, histogram_x_max * 1.08),
      ylim = c(0, y_axis_max)
    )
    rect(
      xleft = 0,
      ybottom = error_histogram$breaks[-length(error_histogram$breaks)],
      xright = error_histogram$counts,
      ytop = error_histogram$breaks[-1],
      col = histogram_blue,
      border = "white",
      lwd = 0.35
    )
    segments(
      x0 = 0,
      y0 = mean_error,
      x1 = histogram_x_max * 1.08,
      y1 = mean_error,
      col = deep_blue,
      lwd = 1.35
    )
    text(
      x = histogram_x_max * 1.04,
      y = mean_error + 0.12,
      labels = sprintf("Mean\n%.2f", mean_error),
      adj = c(1, 0),
      cex = 0.68,
      family = base_family
    )
    segments(
      x0 = 0,
      y0 = 0,
      x1 = histogram_x_max * 1.08,
      y1 = 0,
      lwd = axis_line_lwd,
      col = "black"
    )
    segments(
      x0 = 0,
      y0 = 0,
      x1 = 0,
      y1 = y_axis_max,
      lwd = axis_line_lwd,
      col = "black"
    )

    # Binary assembly membership matrix. Row 1 is drawn at the top.
    par(
      mar = c(2.25, 4.9, 0, 0.10),
      mgp = c(1.45, 0.45, 0),
      tcl = 0,
      xaxs = "i",
      yaxs = "i",
      family = base_family,
      ps = 10
    )
    plot.new()
    plot.window(
      xlim = c(0.5, n_assemblies + 0.5),
      ylim = c(0.5, n_species + 0.5)
    )

    presented_cells <- which(membership == 1L, arr.ind = TRUE)
    cell_x <- presented_cells[, "col"]
    cell_y <- n_species + 1 - presented_cells[, "row"]
    rect(
      xleft = cell_x - 0.5,
      ybottom = cell_y - 0.5,
      xright = cell_x + 0.5,
      ytop = cell_y + 0.5,
      col = deep_blue,
      border = NA
    )

    if (length(group_ends) > 0L) {
      abline(
        v = group_ends + 0.5,
        col = separator_grey,
        lwd = axis_line_lwd
      )
    }
    rect(
      xleft = 0.5,
      ybottom = 0.5,
      xright = n_assemblies + 0.5,
      ytop = n_species + 0.5,
      border = "black",
      lwd = axis_line_lwd
    )
    axis(
      side = 2,
      at = n_species + 1 - seq_len(n_species),
      labels = species_table$species,
      las = 1,
      tick = FALSE,
      cex.axis = 1.0,
      line = -0.15
    )
    mtext("Species", side = 2, line = 3.0, cex = 1.0, family = base_family)
    mtext("Assemblies", side = 1, line = 1.05, cex = 1.0, family = base_family)

    # Keep the lower-right region empty except for the histogram-axis label.
    par(
      mar = c(2.25, 0.10, 0, 0.30),
      family = base_family,
      ps = 10
    )
    plot.new()
    text(
      x = 0.5,
      y = 0.985,
      labels = "Counts",
      adj = c(0.5, 1),
      cex = 0.75,
      family = base_family
    )
  }

  width_mm <- 183
  height_mm <- 90
  width_in <- width_mm / 25.4
  height_in <- height_mm / 25.4

  save_figure(output_stem, draw_figure, width_in, height_in)

})
