% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{save_solver}
\alias{save_solver}
\title{semi-save solver \code{C++} function}
\usage{
save_solver(B, X, Y, bw, rho, eta, gamma, tau, epsilon, btol, ftol, gtol,
  maxitr, verbose, ncore)
}
\arguments{
\item{B}{A matrix of the parameters \code{B}, the columns are subject to the orthogonality constraint}

\item{X}{A matrix of the parameters \code{X}}

\item{Y}{A matrix of the parameters \code{Y}}

\item{bw}{Kernel bandwidth for X}

\item{rho}{(don't change) Parameter for control the linear approximation in line search}

\item{eta}{(don't change) Factor for decreasing the step size in the backtracking line search}

\item{gamma}{(don't change) Parameter for updating C by Zhang and Hager (2004)}

\item{tau}{(don't change) Step size for updating}

\item{epsilon}{(don't change) Parameter for apprximating numerical gradient, if \code{g} is not given.}

\item{btol}{(don't change) The \code{$B$} parameter tolerance level}

\item{ftol}{(don't change) Functional value tolerance level}

\item{gtol}{(don't change) Gradient tolerance level}

\item{maxitr}{Maximum number of iterations}

\item{verbose}{Should information be displayed}
}
\description{
Sovling the semi-save estimating equations. This is an internal function and should not be called directly.
}
\references{
Ma, Y. & Zhu, L. (2012). A semiparametric approach to dimension reduction. Journal of the American Statistical Association, 107(497), 168-179.
DOI: \url{https://dx.doi.org/10.1214\%2F12-AOS1072SUPP}.

Wen, Z. & Yin, W., "A feasible method for optimization with orthogonality constraints." Mathematical Programming 142.1-2 (2013): 397-434.
DOI: \url{https://doi.org/10.1007/s10107-012-0584-1}
}
\keyword{internal}
