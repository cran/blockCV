% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/v2_buffering.R
\name{buffering}
\alias{buffering}
\title{Use distance (buffer) around records to separate train and test folds}
\usage{
buffering(
  speciesData,
  species = NULL,
  theRange,
  spDataType = "PA",
  addBG = TRUE,
  progress = TRUE
)
}
\arguments{
\item{speciesData}{A simple features (sf) or SpatialPoints object containing species data (response variable).}

\item{species}{Character. Indicating the name of the field in which species data (binary response i.e. 0 and 1) is stored. If \code{speceis = NULL}
the presence and absence data (response variable) will be treated the same and only training and testing records will be counted. This can be used for multi-class responses
such as land cover classes for remote sensing image classification, but it is not necessary. \emph{Do not use this argument when the response variable is
continuous or count data}.}

\item{theRange}{Numeric value of the specified range by which the training and testing datasets are separated.
This distance should be in \strong{\emph{metres}} no matter what the coordinate system is. The range can  be explored by \code{\link{spatialAutoRange}}.}

\item{spDataType}{Character input indicating the type of species data. It can take two values, \strong{PA} for \emph{presence-absence} data and \strong{PB} for
\emph{presence-background} data, when \code{species} argument is not \code{NULL}. See the details section for more information on these two approaches.}

\item{addBG}{Logical. Add background points to the test set when \code{spDataType = "PB"}.}

\item{progress}{Logical. If TRUE a progress bar will be shown.}
}
\description{
This function is deprecated and will be removed in future updates! Please use \code{\link{cv_buffer}} instead!
}
\seealso{
\code{\link{cv_buffer}}
}
