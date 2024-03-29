% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/cv_nndm.R
\name{cv_nndm}
\alias{cv_nndm}
\title{Use the Nearest Neighbour Distance Matching (NNDM) to separate train and test folds}
\usage{
cv_nndm(
  x,
  column = NULL,
  r,
  size,
  num_sample = 10000,
  sampling = "random",
  min_train = 0.05,
  presence_bg = FALSE,
  add_bg = FALSE,
  plot = TRUE,
  report = TRUE
)
}
\arguments{
\item{x}{a simple features (sf) or SpatialPoints object of spatial sample data (e.g., species
data or ground truth sample for image classification).}

\item{column}{character; indicating the name of the column in which response variable (e.g. species data as a binary
response i.e. 0s and 1s) is stored. This is required when \code{presence_bg = TRUE}, otherwise optional.}

\item{r}{a terra SpatRaster object of a predictor variable. This defines the area that model is going to predict.}

\item{size}{numeric value of the range of spatial autocorrelation (the \code{phi} parameter).
This distance should be in \strong{metres}. The range could be explored by \code{\link{cv_spatial_autocor}}.}

\item{num_sample}{integer; the number of sample points from predictor (\code{r}) to be used for calculating
the G function of prediction points.}

\item{sampling}{either \code{"random"} or \code{"regular"} for sampling prediction points.
When  \code{sampling = "regular"}, the actual number of samples might be less than \code{num_sample}
for non-rectangular rasters (points falling on no-value areas are removed).}

\item{min_train}{numeric; between 0 and 1. A constraint on the minimum proportion of train points in each fold.}

\item{presence_bg}{logical; whether to treat data as species presence-background data. For all other data
types (presence-absence, continuous, count or multi-class responses), this option should be \code{FALSE}.}

\item{add_bg}{logical; add background points to the test set when \code{presence_bg = TRUE}. We do not
recommend this according to Radosavljevic & Anderson (2014). Keep it \code{FALSE}, unless you mean to add
the background pints to testing points.}

\item{plot}{logical; whether to plot the G functions.}

\item{report}{logical; whether to generate print summary of records in each fold; for very big
datasets, set to \code{FALSE} for slightly faster calculation.}
}
\value{
An object of class S3. A list of objects including:
    \itemize{
    \item{folds_list - a list containing the folds. Each fold has two vectors with the training (first) and testing (second) indices}
    \item{k - number of the folds}
    \item{size - the distance band to separated trainig and testing folds)}
    \item{column - the name of the column if provided}
    \item{presence_bg - whether this was treated as presence-background data}
    \item{records - a table with the number of points in each category of training and testing}
    }
}
\description{
A fast implementation of the Nearest Neighbour Distance Matching (NNDM) algorithm (Milà et al., 2022) in C++. Similar
to \code{\link{cv_buffer}}, this is a variation of leave-one-out (LOO) cross-validation. It tries to match the
nearest neighbour distance distribution function between the test and training data to the nearest neighbour
distance distribution function between the target prediction and training points (Milà et al., 2022).
}
\details{
When working with presence-background (presence and pseudo-absence) species distribution
data (should be specified by \code{presence_bg = TRUE} argument), only presence records are used
for specifying the folds (recommended). The testing fold comprises only the target \emph{presence} point (optionally,
all background points within the distance are also included when \code{add_bg = TRUE}; this is the
distance that matches the nearest neighbour distance distribution function of training-testing presences and
training-presences and prediction points; often lower than \code{size}).
Any non-target presence points inside the distance are excluded.
All points (presence and background) outside of distance are used for the training set.
The methods cycles through all the presence data, so the number of folds is equal to
the number of presence points in the dataset.

For all other types of data (including presence-absence, count, continuous, and multi-class)
set \code{presence_bg = FALE}, and the function behaves similar to the methods
explained by Milà and colleagues (2022).
}
\examples{
\donttest{
library(blockCV)

# import presence-absence species data
points <- read.csv(system.file("extdata/", "species.csv", package = "blockCV"))
# make an sf object from data.frame
pa_data <- sf::st_as_sf(points, coords = c("x", "y"), crs = 7845)

# load raster data
path <- system.file("extdata/au/bio_5.tif", package = "blockCV")
covar <- terra::rast(path)

nndm <- cv_nndm(x = pa_data,
                column = "occ", # optional
                r = covar,
                size = 350000, # size in metres no matter the CRS
                num_sample = 10000,
                sampling = "regular",
                min_train = 0.1)

}
}
\references{
C. Milà, J. Mateu, E. Pebesma, and H. Meyer, Nearest Neighbour Distance Matching
Leave-One-Out Cross-Validation for map validation, Methods in Ecology and Evolution (2022).
}
\seealso{
\code{\link{cv_buffer}} and \code{\link{cv_spatial_autocor}}
}
