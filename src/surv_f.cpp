#include <RcppArmadillo.h>

using namespace Rcpp;

//[[Rcpp::depends(RcppArmadillo)]]

double surv_f(arma::mat B,
              arma::mat X,
              arma::mat Phit,
              NumericMatrix inRisk,
              double bw,
              NumericVector Fail_Ind)
{
	// This function computes the estimation equations and its 2-norm for the survival dimensional reduction model
	// It only implement the dN method, with phi(t)

	int N = X.n_rows;
	int P = X.n_cols;
	int nFail = inRisk.ncol();

	NumericMatrix BX_NM = wrap(X * B);
	NumericMatrix kernel_matrix(N, N);

	for (int i = 0; i < N; i++)
	{
		kernel_matrix(i, i) = 1;
		for (int j = i + 1; j < N; j ++)
		{
			kernel_matrix(j,i) = exp(-sum(pow(BX_NM.row(i)-BX_NM.row(j),2))/bw/bw);
			kernel_matrix(i,j) = kernel_matrix(j,i);
		}
	}

	NumericMatrix TheCenter(nFail, P);
	NumericMatrix X_NM = wrap(X);
	NumericVector TheCond(P);

	double unweighted_sum;
	double weights;

	for(int j=0; j<nFail; j++)
	{
		for(int i=0; i<P; i++)
		{
			unweighted_sum = 0;
			weights = 0;

			for(int k=0; k<N; k++)
			{
				if(inRisk(k,j)==true)
				{
					unweighted_sum = unweighted_sum + X_NM(k,i) * kernel_matrix(k,Fail_Ind[j]-1);
					weights = weights + kernel_matrix(k,Fail_Ind[j]-1);
				}
			}
			TheCond[i] = unweighted_sum/weights;
		}
		TheCenter(j,_) = X_NM(Fail_Ind[j]-1,_) - TheCond;
	}

	arma::mat matrixsum(P, P);
	matrixsum.fill(0);
	arma::mat TheCenter_arma = as<arma::mat>(TheCenter);

	for(int i=0; i<nFail; i++)
	{
		matrixsum = matrixsum + Phit.col(i) * TheCenter_arma.row(i);
	}

	NumericMatrix matrixsum_NM = wrap(matrixsum);
	double ret = sum(pow(matrixsum_NM/N,2));

	return ret;
}
