% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{pdose_semi_solver}
\alias{pdose_semi_solver}
\title{pdose_semi_solver}
\usage{
pdose_semi_solver(
  B,
  X,
  R,
  A,
  a_dist,
  a_seq,
  lambda,
  bw,
  rho,
  eta,
  gamma,
  tau,
  epsilon,
  btol,
  ftol,
  gtol,
  maxitr,
  verbose,
  ncore
)
}
\arguments{
\item{B}{A matrix of the parameters \code{B}, the columns are subject to the orthogonality constraint}

\item{X}{The covariate matrix}

\item{R}{The perosnalzied medicine reward}

\item{A}{observed dose levels}

\item{a_dist}{A kernel distance matrix for the observed dose and girds of the dose levels}

\item{a_seq}{A grid of dose levels}

\item{lambda}{The penalty for the GCV for the kernel ridge regression}

\item{bw}{A Kernel bandwidth, assuming each variable have unit variance}

\item{rho}{(don't change) Parameter for control the linear approximation in line search}

\item{eta}{(don't change) Factor for decreasing the step size in the backtracking line search}

\item{gamma}{(don't change) Parameter for updating C by Zhang and Hager (2004)}

\item{tau}{(don't change) Step size for updating}

\item{epsilon}{(don't change) Parameter for approximating numerical gradient}

\item{btol}{(don't change) The \code{$B$} parameter tolerance level}

\item{ftol}{(don't change) Estimation equation 2-norm tolerance level}

\item{gtol}{(don't change) Gradient tolerance level}

\item{maxitr}{Maximum number of iterations}

\item{verbose}{Should information be displayed}
}
\value{
The optimizer \code{B} for the esitmating equation.
}
\description{
The pseudo direct learning optimization function for personalized dose finding with dimension reduction.
}
\references{
Zhou, W., Zhu, R., & Zeng, D. (2021). A parsimonious personalized dose-finding model via dimension reduction.
Biometrika, 108(3), 643-659.
DOI: \doi{10.1093/biomet/asaa087}
}
\keyword{internal}
