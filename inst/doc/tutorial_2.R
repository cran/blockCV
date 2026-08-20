## ----echo=FALSE---------------------------------------------------------------
options(scipen = 10)

## ----message=FALSE, warning=FALSE---------------------------------------------
library(blockCV)
library(sf)
library(terra)

points <- read.csv(system.file("extdata/", "species.csv", package = "blockCV"))
pa_data <- sf::st_as_sf(points, coords = c("x", "y"), crs = 7845)

rasters <- terra::rast(
  list.files(system.file("extdata/au/", package = "blockCV"), full.names = TRUE)
)

## ----warning=FALSE, message=FALSE---------------------------------------------
sb <- cv_spatial(x = pa_data,
                 column = "occ",
                 r = rasters,
                 k = 5,
                 size = 350000,
                 selection = "random",
                 iteration = 50,
                 progress = FALSE,
                 report = FALSE,
                 plot = FALSE)

set.seed(6)
bscv <- cv_cluster(x = pa_data,
                   column = "occ",
                   k = 5,
                   balance = TRUE,
                   k_multiplier = 3,
                   report = FALSE)

set.seed(6)
ecv <- cv_cluster(x = pa_data,
                  column = "occ",
                  r = rasters,
                  k = 5,
                  scale = TRUE)

bloo <- cv_buffer(x = pa_data,
                  column = "occ",
                  size = 350000,
                  progress = FALSE,
                  report = FALSE)

knn_blocks <- cv_knndm(x = pa_data,
                       column = "occ",
                       r = rasters,
                       k = 5,
                       clustering = "blocks",
                       keep_blocks = TRUE,
                       num_sample = 3000,
                       nk_len = 40,
                       seed = 6,
                       report = FALSE,
                       plot = FALSE)

## -----------------------------------------------------------------------------
summary_basic <- cv_summary(sb)
summary_basic

## -----------------------------------------------------------------------------
summary_full <- cv_summary(cv = sb,
                           x = pa_data,
                           r = rasters,
                           num_sample = 3000,
                           seed = 6,
                           progress = FALSE)

summary_full$distances
summary_full$novelty
summary_full$warnings

## ----fig.height=4, fig.width=6------------------------------------------------
sim <- cv_similarity(cv = ecv,
                     x = pa_data,
                     r = rasters,
                     method = "MESS",
                     plot = FALSE,
                     progress = FALSE)

sim$plot
sim$extrapolation
sim$overall

## ----fig.height=4.5, fig.width=6----------------------------------------------
cv_similarity(cv = ecv,
              x = pa_data,
              r = rasters,
              method = "MESS",
              type = "map",
              plot = TRUE,
              progress = FALSE)


## ----fig.height=4, fig.width=6------------------------------------------------
cv_similarity(cv = ecv,
              x = pa_data,
              r = rasters,
              method = "L2",
              num_plots = 1:3,
              num_sample = 3000,
              seed = 6,
              plot = TRUE,
              progress = FALSE)


## ----warning=FALSE, message=FALSE, fig.height=7, fig.width=8------------------
distance_theme <- ggplot2::theme(
  legend.position = "bottom",
  plot.subtitle = ggplot2::element_text(size = 7)
)

dist_sb <- cv_distance(cv = sb,
                       x = pa_data,
                       r = rasters,
                       num_sample = 3000,
                       num_random = 5,
                       seed = 6,
                       plot = FALSE)

dist_bscv <- cv_distance(cv = bscv,
                         x = pa_data,
                         r = rasters,
                         num_sample = 3000,
                         num_random = 5,
                         seed = 6,
                         plot = FALSE)

dist_bloo <- cv_distance(cv = bloo,
                         x = pa_data,
                         r = rasters,
                         add_random = FALSE,
                         num_sample = 3000,
                         seed = 6,
                         plot = FALSE)

dist_knn <- cv_distance(cv = knn_blocks,
                        x = pa_data,
                        r = rasters,
                        num_sample = 3000,
                        num_random = 5,
                        seed = 6,
                        plot = FALSE)

cowplot::plot_grid(
  dist_sb$plot +
    ggplot2::labs(title = "Spatial blocks") +
    distance_theme,
  dist_bscv$plot +
    ggplot2::labs(title = "Spatial clustering (balanced)") +
    distance_theme,
  dist_bloo$plot +
    ggplot2::labs(title = "Buffering LOO") +
    distance_theme,
  dist_knn$plot +
    ggplot2::labs(title = "kNNDM blocks") +
    distance_theme,
  ncol = 2
)

## -----------------------------------------------------------------------------
dist_sb$W
dist_sb$distances

## ----fig.height=4, fig.width=6------------------------------------------------
dist_feature <- cv_distance(cv = sb,
                            x = pa_data,
                            r = rasters,
                            space = "feature",
                            num_sample = 3000,
                            seed = 6,
                            plot = FALSE)

dist_feature$plot

## ----warning=FALSE, message=FALSE, fig.height=5, fig.width=7.2----------------
sac_raster <- cv_spatial_autocor(r = rasters,
                                 num_sample = 3000,
                                 progress = FALSE,
                                 plot = FALSE)

plot(sac_raster)

## -----------------------------------------------------------------------------
sac_raster$range
summary(sac_raster)

## ----warning=FALSE, message=FALSE, fig.height=5, fig.width=7.2----------------
sac_points <- cv_spatial_autocor(x = pa_data,
                                 column = "occ",
                                 plot = FALSE)

plot(sac_points)

## ----eval=FALSE---------------------------------------------------------------
# cv_block_size(x = pa_data,
#               column = "occ",
#               r = rasters,
#               min_size = 2e5,
#               max_size = 9e5)

