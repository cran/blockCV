## ----echo=FALSE---------------------------------------------------------------
options(scipen = 10)

## ----message=FALSE, warning=FALSE---------------------------------------------
library(blockCV)
library(sf)
library(terra)

# import presence-absence species data
points <- read.csv(system.file("extdata/", "species.csv", package = "blockCV"))

# make an sf object from the data.frame
pa_data <- sf::st_as_sf(points, coords = c("x", "y"), crs = 7845)

# load raster covariates
covars <- terra::rast(
  list.files(system.file("extdata/au/", package = "blockCV"), full.names = TRUE)
)

## -----------------------------------------------------------------------------
training <- terra::extract(covars, pa_data, ID = FALSE)
training$occ <- as.factor(pa_data$occ)

head(training)

## ----fig.height=5, fig.width=7, message=FALSE, warning=FALSE------------------
set.seed(123)

sb1 <- cv_spatial(
  x = pa_data,
  column = "occ",
  r = covars,
  size = 450000,
  k = 5,
  selection = "random",
  iteration = 50,
  progress = FALSE,
  report = TRUE,
  plot = TRUE
)

## ----eval=FALSE---------------------------------------------------------------
# library(caret)
# 
# train_index <- lapply(sb1$folds_list, function(fold) fold[[1]])
# test_index <- lapply(sb1$folds_list, function(fold) fold[[2]])
# 
# names(train_index) <- paste0("Fold", seq_along(train_index))
# names(test_index) <- names(train_index)
# 
# control <- trainControl(
#   method = "cv",
#   number = length(train_index),
#   index = train_index,
#   indexOut = test_index,
#   search = "random"
# )
# 
# set.seed(123)
# 
# rf_model <- train(
#   occ ~ .,
#   data = training,
#   method = "rf",
#   trControl = control,
#   tuneLength = 4,
#   ntree = 500
# )
# 
# rf_model
# rf_model$resample
# plot(rf_model)

