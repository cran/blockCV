## ----eval=FALSE---------------------------------------------------------------
# # install stable version from CRAN
# install.packages("blockCV", dependencies = TRUE)
# 
# # install latest update from GitHub
# remotes::install_github("rvalavi/blockCV", dependencies = TRUE)
# 

## ----message=TRUE, warning=TRUE-----------------------------------------------
# loading the package
library(blockCV)


## ----fig.height=5, fig.width=7.2, warning=FALSE, message=FALSE----------------
library(sf) # working with spatial vector data
library(terra) # working with spatial raster data
library(tmap) # plotting maps

# load raster data
rasters <- terra::rast(
  list.files(system.file("extdata/au/", package = "blockCV"), full.names = TRUE)
)


## ----fig.height=4.5, fig.width=7.1--------------------------------------------
# load species presence-absence data and convert to sf
points <- read.csv(system.file("extdata/", "species.csv", package = "blockCV"))
head(points)


## ----fig.height=4.5, fig.width=7.1--------------------------------------------
pa_data <- sf::st_as_sf(points, coords = c("x", "y"), crs = 7845)

## ----fig.height=4.5, fig.width=7.1--------------------------------------------
tm_shape(rasters[[1]]) +
  tm_raster(
    col.scale = tm_scale_continuous(values = gray.colors(10)),
    col.legend = tm_legend_hide()
  ) +
  tm_shape(pa_data) +
  tm_dots(
    fill = "occ",
    fill.scale = tm_scale_categorical(),
    size = 0.5,
    fill_alpha = 0.5
  )


## ----results='hide', fig.keep='all', warning=FALSE, message=FALSE, fig.height=5, fig.width=7----
sb1 <- cv_spatial(x = pa_data,
                  column = "occ", # the response column (binary or multi-class)
                  k = 5, # number of folds
                  size = 350000, # size of the blocks in metres
                  selection = "random", # random blocks-to-fold
                  iteration = 50, # find evenly dispersed folds
                  biomod2 = TRUE) # also create folds for biomod2


## ----warning=FALSE, message=FALSE, fig.height=5, fig.width=7------------------
sb2 <- cv_spatial(x = pa_data,
                  column = "occ",
                  r = rasters, # optionally add a raster layer
                  k = 5, 
                  size = 350000, 
                  hexagon = TRUE, # use hexagonal blocks
                  selection = "random",
                  progress = FALSE, # turn off progress bar for vignette
                  iteration = 50, 
                  biomod2 = TRUE, 
                  report = TRUE,
                  plot = TRUE)


## ----warning=FALSE, message=FALSE, fig.height=5, fig.width=7------------------
# systematic fold assignment 
# and also use row/column for creating blocks instead of size
sb3 <- cv_spatial(x = pa_data,
                  column = "occ",
                  rows_cols = c(12, 10),
                  hexagon = FALSE,
                  selection = "systematic", 
                  report = TRUE,
                  plot = TRUE)


## ----warning=FALSE, message=FALSE, fig.height=5, fig.width=7------------------
# checkerboard block to CV fold assignment
sb4 <- cv_spatial(x = pa_data,
                  column = "occ",
                  size = 350000,
                  hexagon = FALSE,
                  selection = "checkerboard", 
                  report = TRUE, 
                  plot = TRUE)


## ----warning=FALSE, message=FALSE, fig.height=5, fig.width=7------------------
tm_shape(sb4$blocks) +
  tm_fill(
    fill = "folds",
    fill.scale = tm_scale_categorical()
  )


## -----------------------------------------------------------------------------
# spatial clustering
set.seed(6)
scv <- cv_cluster(x = pa_data,
                  column = "occ", # optional: counting number of train/test records
                  k = 5,
                  report = TRUE)

## -----------------------------------------------------------------------------
# balanced spatial clustering
set.seed(6)
bscv <- cv_cluster(x = pa_data,
                   column = "occ",
                   k = 5,
                   balance = TRUE,
                   k_multiplier = 3,
                   report = TRUE)

## ----warning=FALSE, message=FALSE, fig.height=4, fig.width=8------------------
scv_plot <- cv_plot(scv, pa_data, combine_folds = TRUE) +
    ggplot2::labs(title = "Default clustering")

bscv_plot <- cv_plot(bscv, pa_data, combine_folds = TRUE) +
    ggplot2::labs(title = "Balanced clustering")

cowplot::plot_grid(scv_plot, bscv_plot, nrow = 1)


## -----------------------------------------------------------------------------
# load species presence-background data and convert to sf
points_pb <- read.csv(system.file("extdata/", "species_pb.csv", package = "blockCV"))
pb_data <- sf::st_as_sf(points_pb, coords = c("x", "y"), crs = 7845)

# balance folds on presences only (presence-background data)
set.seed(6)
pbcv <- cv_cluster(x = pb_data,
                   column = "occ",
                   k = 5,
                   balance = TRUE,
                   presence_bg = TRUE, # balance on presences (1s) only
                   report = TRUE)

## ----warning=FALSE, message=FALSE---------------------------------------------
# environmental clustering
set.seed(6)
ecv <- cv_cluster(x = pa_data,
                  column = "occ",
                  r = rasters,
                  k = 5, 
                  scale = TRUE)


## ----warning=FALSE, message=FALSE---------------------------------------------
# spatially constrained environmental clustering
set.seed(6)
secv <- cv_cluster(x = pa_data,
                   column = "occ",
                   r = rasters,
                   k = 5,
                   scale = TRUE,
                   spatial_weight = 0.4) # blend geography into the environmental clusters


## ----warning=FALSE, message=FALSE, fig.height=4, fig.width=8------------------
ecv_plot <- cv_plot(ecv, pa_data, combine_folds = TRUE) +
    ggplot2::labs(title = "Environmental clustering")

secv_plot <- cv_plot(secv, pa_data, combine_folds = TRUE) +
    ggplot2::labs(title = "Spatially constrained (spatial_weight = 0.4)")

cowplot::plot_grid(ecv_plot, secv_plot, nrow = 1)


## -----------------------------------------------------------------------------
set.seed(6)
coords <- sf::st_coordinates(pa_data)
pa_data$site <- paste0("site_", stats::kmeans(coords, centers = 10)$cluster)

## -----------------------------------------------------------------------------
site_lgo <- cv_group(x = pa_data,
                     group_col = "site",
                     column = "occ",
                     report = TRUE)

## -----------------------------------------------------------------------------
site_cv <- cv_group(x = pa_data,
                    group_col = "site",
                    column = "occ",
                    k = 5,
                    balance = TRUE,
                    iteration = 50,
                    seed = 6,
                    report = TRUE)

## ----warning=FALSE, message=FALSE, fig.height=4.5, fig.width=6----------------
cv_plot(site_cv, pa_data, combine_folds = TRUE) +
    ggplot2::labs(title = "Grouped folds")


## ----results='hide', fig.keep='all'-------------------------------------------
bloo <- cv_buffer(x = pa_data,
                  column = "occ",
                  size = 350000)


## ----fig.height=4, fig.width=7------------------------------------------------
nncv <- cv_nndm(x = pa_data,
                column = "occ",
                r = rasters,
                size = 350000,
                num_sample = 5000, 
                sampling = "random",
                min_train = 0.1,
                plot = TRUE)


## ----warning=FALSE, message=FALSE, fig.height=4, fig.width=7------------------
knn_hier <- cv_knndm(x = pa_data,
                     column = "occ",
                     r = rasters,
                     k = 5,
                     clustering = "hierarchical",
                     num_sample = 5000,
                     nk_len = 50,
                     seed = 6,
                     report = TRUE,
                     plot = TRUE)


## ----warning=FALSE, message=FALSE, fig.height=4, fig.width=7------------------
knn_blocks <- cv_knndm(x = pa_data,
                       column = "occ",
                       r = rasters,
                       k = 5,
                       clustering = "blocks", 
                       keep_blocks = TRUE,
                       num_sample = 5000,
                       nk_len = 50,
                       seed = 6,
                       report = TRUE,
                       plot = TRUE)


## ----warning=FALSE, message=FALSE, fig.height=4.5, fig.width=7----------------
cv_plot(knn_blocks, pa_data, combine_folds = TRUE)


## ----warning=FALSE, message=FALSE, fig.height=6, fig.width=8------------------
cv_plot(cv = scv, x = pa_data)


## ----warning=FALSE, message=FALSE, fig.height=4.5, fig.width=6----------------
cv_plot(cv = scv, x = pa_data, combine_folds = TRUE) +
    ggplot2::labs(
        title = "Spatial clustering folds",
        x = "Longitude",
        y = "Latitude"
    )


## ----warning=FALSE, message=FALSE, fig.height=4.5, fig.width=6----------------
cv_plot(cv = pbcv, x = pb_data, combine_folds = TRUE, bg_alpha = 0.05) +
    ggplot2::labs(title = "Presence-background folds (background faded)")


## ----warning=FALSE, message=FALSE, fig.height=5, fig.width=8------------------
cv_plot(cv = bloo,
        x = pa_data,
        num_plots = c(1, 50, 100)) # only show folds 1, 50 and 100


## ----warning=FALSE, message=FALSE, fig.height=5, fig.width=7------------------
cv_plot(cv = sb1,
        r = rasters,
        raster_colors = terrain.colors(10, alpha = 0.5),
        label_size = 4) 


