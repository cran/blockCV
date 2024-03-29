% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/v2_environBlock.R
\name{envBlock}
\alias{envBlock}
\title{Use environmental clustering to separate train and test folds}
\usage{
envBlock(
  rasterLayer,
  speciesData,
  species = NULL,
  k = 5,
  standardization = "normal",
  rasterBlock = TRUE,
  sampleNumber = 10000,
  biomod2Format = TRUE,
  numLimit = 0,
  verbose = TRUE
)
}
\arguments{
\item{rasterLayer}{A raster object of covariates to identify environmental groups.}

\item{speciesData}{A simple features (sf) or SpatialPoints object containing species data (response variable).}

\item{species}{Character. Indicating the name of the field in which species data (binary response i.e. 0 and 1) is stored. If \code{speceis = NULL}
the presence and absence data (response variable) will be treated the same and only training and testing records will be counted. This can be used for multi-class responses
such as land cover classes for remote sensing image classification, but it is not necessary. \emph{Do not use this argument when the response variable is
continuous or count data}.}

\item{k}{Integer value. The number of desired folds for cross-validation. The default is \code{k = 5}.}

\item{standardization}{Standardize input raster layers. Three possible inputs are "normal" (the default), "standard" and "none".
See details for more information.}

\item{rasterBlock}{Logical. If TRUE, the clustering is done in the raster layer rather than species data. See details for
more information.}

\item{sampleNumber}{Integer. The number of samples from raster layers to build the clusters.}

\item{biomod2Format}{Logical. Creates a matrix of folds that can be directly used in the \pkg{biomod2} package as
a \emph{DataSplitTable} for cross-validation.}

\item{numLimit}{Integer value. The minimum number of points in each category of data (\emph{train_0},
\emph{train_1}, \emph{test_0} and \emph{test_1}). Shows a message if the number of points
in any of the folds happens to be less than this number.}

\item{verbose}{Logical. To print the report of the recods per fold.}
}
\description{
This function is deprecated and will be removed in future updates! Please use \code{\link{cv_cluster}} instead!
}
\seealso{
\code{\link{cv_cluster}}
}
