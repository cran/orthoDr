#include <RcppArmadillo.h>

using namespace Rcpp;

//[[Rcpp::depends(RcppArmadillo)]]

double surv_f(arma::mat,
              arma::mat,
              arma::mat,
              NumericMatrix,
              double,
              NumericVector);

void surv_g(arma::mat B,
            arma::mat& G,
            arma::mat X,
            arma::mat Phit,
            NumericMatrix inRisk,
            double bw,
            NumericVector Fail_Ind,
            double epsilon)
{
	// This function computes the gradiant of the estimation equations

	arma::mat B_new = B;
	double F0 = surv_f(B, X, Phit, inRisk, bw, Fail_Ind);
  int P = B.n_rows;
	int ndr = B.n_cols;

	for (int j = 0; j < ndr; j++)
	{
		for(int i = 0; i < P; i++)
		{
			// small increment
			B_new(i,j) = B(i,j) + epsilon;

			// calculate gradiant
			G(i,j) = (surv_f(B_new, X, Phit, inRisk, bw, Fail_Ind) - F0) / epsilon;

			// reset
			B_new(i,j) = B(i,j);
		}
	}

	return;
}
