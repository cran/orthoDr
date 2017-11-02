#include <RcppArmadillo.h>

using namespace Rcpp;

//[[Rcpp::depends(RcppArmadillo)]]

double surv_f(const arma::mat& B,
              const arma::mat& X,
              const arma::mat& Phit,
              const arma::mat& inRisk,
              const arma::vec& Fail_Ind,
              const double bw)
{
  // This function computes the estimation equations and its 2-norm for the survival dimensional reduction model
  // It only implement the dN method, with phi(t)

  int N = X.n_rows;
  int P = X.n_cols;
  int nFail = inRisk.n_cols;

  arma::mat BX = X * B;
  arma::mat kernel_matrix(N, N);

  for (int i = 0; i < N; i++)
  {
    kernel_matrix(i, i) = 1;
    for (int j = i + 1; j < N; j ++)
    {
      kernel_matrix(j,i) = exp(-sum(pow(BX.row(i)-BX.row(j),2))/bw/bw);
      kernel_matrix(i,j) = kernel_matrix(j,i);
    }
  }

  arma::mat TheCenter(nFail, P);
  arma::mat TheCond(1,P);

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
          unweighted_sum = unweighted_sum + X(k,i) * kernel_matrix(k,Fail_Ind[j]-1);
          weights = weights + kernel_matrix(k,Fail_Ind[j]-1);
        }
      }
      TheCond(0,i) = unweighted_sum/weights;
    }
    TheCenter.row(j) = X.row(Fail_Ind[j]-1) - TheCond;
  }

  arma::mat matrixsum(P, P);
  matrixsum.fill(0);

  for(int i=0; i<nFail; i++)
  {
    matrixsum = matrixsum + Phit.col(i) * TheCenter.row(i);
  }

  return accu(pow(matrixsum,2))/N/N;
}

void surv_g(arma::mat& B,
            arma::mat& G,
            const arma::mat& X,
            const arma::mat& Phit,
            const arma::mat& inRisk,
            const arma::vec& Fail_Ind,
            const double bw,
            const double epsilon)
{
  // This function computes the gradiant of the estimation equations

  double F0 = surv_f(B, X, Phit, inRisk, Fail_Ind, bw);
  int P = B.n_rows;
  int ndr = B.n_cols;
  double temp;

  for (int j = 0; j < ndr; j++)
  {
    for(int i = 0; i < P; i++)
    {
      // small increment
      temp = B(i,j);
      B(i,j) += epsilon;

      // calculate gradiant
      G(i,j) = (surv_f(B, X, Phit, inRisk, Fail_Ind, bw) - F0) / epsilon;

      // reset
      B(i,j) = temp;
    }
  }

  return;
}


double dmax(double a, double b)
{
  if (a > b)
    return a;

  return b;
}

double dmin(double a, double b)
{
  if (a > b)
    return b;

  return a;
}


//' @title surv_solver
//' @name surv_solver
//' @description The main optimization function for survival dimensional reduction, the IR-CP method. This is an internal function and should not be called directly.
//' @keywords internal
//' @param B A matrix of the parameters \code{B}, the columns are subject to the orthogonality constraint
//' @param X The covariate matrix
//' @param Phit Phit as defined in Sun et al. (2017)
//' @param inRisk A matrix of indicators shows whether each subject is still alive at each time point
//' @param bw A Kernel bandwidth, assuming each variable have unit variance
//' @param Fail_Ind The locations of the failure subjects
//' @param rho (don't change) Parameter for control the linear approximation in line search
//' @param eta (don't change) Factor for decreasing the step size in the backtracking line search
//' @param gamma (don't change) Parameter for updating C by Zhang and Hager (2004)
//' @param tau (don't change) Step size for updating
//' @param epsilon (don't change) Parameter for approximating numerical gradient
//' @param btol (don't change) The \code{$B$} parameter tolerance level
//' @param ftol (don't change) Estimation equation 2-norm tolerance level
//' @param gtol (don't change) Gradient tolerance level
//' @param maxitr Maximum number of iterations
//' @param verbose Should information be displayed
//' @return The optimizer \code{B} for the esitmating equation.
//' @references Sun, Q., Zhu, R., Wang T. and Zeng D. "Counting Process Based Dimension Reduction Method for Censored Outcomes." (2017) \url{https://arxiv.org/abs/1704.05046} .
//' @references Wen, Z. and Yin, W., "A feasible method for optimization with orthogonality constraints." Mathematical Programming 142.1-2 (2013): 397-434. DOI: \url{https://doi.org/10.1007/s10107-012-0584-1}
//' @examples
//' # This function should be called internally. When having all objects pre-computed, one can call
//' # surv_solver(B, X, Phit, inRisk, bw, Fail.Ind,
//' #             rho, eta, gamma, tau, epsilon, btol, ftol, gtol, maxitr, verbose)
//' # to solve for the parameters B.
//'
// [[Rcpp::export]]

List surv_solver(arma::mat B,
                 const arma::mat& X,
                 const arma::mat& Phit,
                 const arma::mat& inRisk,
                 const arma::vec& Fail_Ind,
                 double bw,
                 double rho,
                 double eta,
                 double gamma,
                 double tau,
                 double epsilon,
                 double btol,
                 double ftol,
                 double gtol,
                 int maxitr,
                 int verbose)
{

  int P = B.n_rows;
  int ndr = B.n_cols;

  arma::mat crit(maxitr,3);
  bool invH = true;
  arma::mat eye2P(2*ndr,2*ndr);

  if(ndr < P/2){
    invH = false;
    eye2P.eye();
  }

  // Initial function value and gradient, prepare for iterations

  double F = surv_f(B, X, Phit, inRisk, Fail_Ind, bw);

  arma::mat G(P, ndr);
  G.fill(0);
  surv_g(B, G, X, Phit, inRisk, Fail_Ind, bw, epsilon);

  //return G;

  arma::mat GX = G.t() * B;
  arma::mat GXT;
  arma::mat H;
  arma::mat RX;
  arma::mat U;
  arma::mat V;
  arma::mat VU;
  arma::mat VX;

  if(invH){
    GXT = G * B.t();
    H = 0.5 * (GXT - GXT.t());
    RX = H * B;
  }else{
    U = join_rows(G, B);
    V = join_rows(B, -G);
    VU = V.t() * U;
    VX = V.t() * B;
  }

  arma::mat dtX = G - B * GX;
  double nrmG = norm(dtX, "fro");

  double Q = 1;
  double Cval = F;

  // main iteration
  int itr;
  arma::mat BP;
  double FP;
  arma::mat GP;
  arma::mat dtXP;
  arma::mat diag_n(P, P);
  arma::mat aa;
  arma::mat S;
  double BDiff;
  double FDiff;
  arma::mat Y;
  double SY;

  if (verbose > 1)
    Rcout << "Initial value,   F = " << F << std::endl;

  for(itr = 1; itr < maxitr + 1; itr++){
    BP = B;
    FP = F;
    GP = G;
    dtXP = dtX;

    int nls = 1;
    double deriv = rho * nrmG * nrmG;

    while(true){
      if(invH){
        diag_n.eye();
        B = solve(diag_n + tau * H, BP - tau * RX);
      }else{
        aa = solve(eye2P + 0.5 * tau * VU, VX);
        B = BP - U * (tau * aa);
      }

      F = surv_f(B, X, Phit, inRisk, Fail_Ind, bw);
      surv_g(B, G, X, Phit, inRisk, Fail_Ind, bw, epsilon);


      if((F <= (Cval - tau*deriv)) || (nls >= 5)){
        break;
      }
      tau = eta * tau;
      nls = nls + 1;
    }

    GX = G.t() * B;

    if(invH){
      GXT = G * B.t();
      H = 0.5 * (GXT - GXT.t());
      RX = H * B;
    }else{
      U = join_rows(G, B);
      V = join_rows(B, -G);
      VU = V.t() * U;
      VX = V.t() * B;
    }

    dtX = G - B * GX; // GX, dtX, nrmG slightly different from those of R code
    nrmG = norm(dtX, "fro");

    S = B - BP;
    BDiff = norm(S, "fro")/sqrt((double) P);
    FDiff = std::abs(FP - F)/(std::abs(FP)+1);

    Y = dtX - dtXP;
    SY = std::abs(accu(S % Y));

    if(itr%2 == 0){
      tau = accu(S % S)/SY;
    }else{
      tau = SY/accu(Y % Y);
    }

    tau = dmax(dmin(tau, 1e10), 1e-20);
    crit(itr-1,0) = nrmG;
    crit(itr-1,1) = BDiff;
    crit(itr-1,2) = FDiff;

    if (verbose > 1 && (itr % 10 == 0) )
      Rcout << "At iteration " << itr << ", F = " << F << std::endl;

    if (itr >= 5) // so I will run at least 5 iterations before checking for convergence
    {
      arma::mat mcrit(5, 3);
      for (int i=0; i<5; i++)
      {
        mcrit.row(i) = crit.row(itr-i-1);
      }

      if ( (BDiff < btol && FDiff < ftol) || (nrmG < gtol) || ((mean(mcrit.col(1)) < btol) && (mean(mcrit.col(2)) < ftol)) )
      {
        if (verbose > 0) Rcout << "converge" << std::endl;
        break;
      }
    }

    double Qp = Q;
    Q = gamma * Qp + 1;
    Cval = (gamma*Qp*Cval + F)/Q;

  }

	if(itr>=maxitr){
		Rcout << "exceed max iteration before convergence ... " << std::endl;
  }

  arma::mat diag_P(ndr,ndr);
  diag_P.eye();
  double feasi = norm(B.t() * B - diag_P, "fro");

  if (verbose > 0){
    Rcout << "number of iterations: " << itr << std::endl;
    Rcout << "norm of functional value: " << F << std::endl;
    Rcout << "norm of gradient: " << nrmG << std::endl;
    Rcout << "norm of feasibility: " << feasi << std::endl;
  }

  List ret;
  ret["B"] = B;
  ret["fn"] = F;
  ret["itr"] = itr;
  ret["converge"] = (itr<maxitr);
  return (ret);
}
