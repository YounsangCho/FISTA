
extern "C"
__host__ void FISTA_S(float *beta, float *X, float *y, float *lambda, float *L, float *Eta, float *tolerance, float *Loss, int *max_iteration, int *N, int *P, int *steps) {

  int max_iter = *max_iteration;
  int n = *N;
  int p = *P;
  int i_k, k;
  
  float lam = *lambda;
  float tol = *tolerance;
  float eta = *Eta;
  float L_prev = *L;
  
  const float One = 1.0, MOne = -1.0;
  const float Zero = 0.0;
  
  size_t sz_np = n*p*sizeof(float);
  size_t sz_n = n*sizeof(float);
  size_t sz_p = p*sizeof(float);
  
  dim3 TPB(32, 1);
  dim3 BPG(1, 1);
  
  unsigned int bpg_p = ceil(((float) p)/TPB.x);
  BPG.x = bpg_p;
  
 
  float *d_y, *d_ymXbp, *d_X, *d_beta_p, *d_XTrbp;
  float *d_bstar, *d_beta, *d_diff_beta;
  float *d_ymXb, *d_beta_prev, *crit;
  float eta_ik, L_cur, pL_cur, RHS, LHS, tnext, t1;
  float h_rbp, h_RHS_1st, h_RHS_2nd, h_rb,h_crit;
  float t = 1.0;
  bool cond;
  
  cudaMalloc(&d_y, sz_n);
  cudaMalloc(&d_ymXbp, sz_n);
  cudaMalloc(&d_X, sz_np);
  cudaMalloc(&d_beta_p, sz_p);
  cudaMalloc(&d_XTrbp, sz_p);
  cudaMalloc(&d_bstar, sz_p);
  cudaMalloc(&d_beta, sz_p);
  cudaMalloc(&d_diff_beta, sz_p);
  cudaMalloc(&d_ymXb, sz_n);
  cudaMalloc(&d_beta_prev, sz_p);
  
  cudaMemcpy(d_y, y, sz_n, HTD);
  cudaMemcpy(d_X, X, sz_np, HTD);
  cudaMemcpy(d_beta_p, beta, sz_p, HTD);
  cudaMemcpy(d_beta_prev, beta, sz_p, HTD);
  
  crit = (float *)malloc(max_iter*sizeof(float));
 
  cublasHandle_t handle;
  cublasOperation_t tran = CUBLAS_OP_T;
  cublasOperation_t ntran = CUBLAS_OP_N;    
 
  cublasCreate(&handle);
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
  for (k = 0; k < max_iter; k++) {
    cublasScopy(handle, n, d_y, 1, d_ymXbp, 1);
    cublasSgemv(handle, ntran, n, p, &MOne, d_X, n, d_beta_p, 1, &One, d_ymXbp, 1);
    cublasSdot(handle, n, d_ymXbp, 1, d_ymXbp, 1, &h_rbp);
    cublasSgemv(handle, tran, n, p, &One, d_X, n, d_ymXbp, 1, &Zero, d_XTrbp, 1);
 
    i_k = -1;
    cond = true;
    while (cond) {
      i_k += 1;
      eta_ik = pow(eta, i_k);
      L_cur = L_prev * eta_ik;
      pL_cur = 1.0 / L_cur;
      cublasScopy(handle, p, d_beta_p, 1, d_bstar, 1);
      cublasSaxpy(handle, p, &pL_cur, d_XTrbp, 1, d_bstar, 1);
      soft_thr_S<<<BPG,TPB>>>(d_bstar, lam/L_cur, d_beta, p);

      cublasScopy(handle, p, d_beta, 1, d_diff_beta, 1);
      cublasSaxpy(handle, p, &MOne, d_beta_p, 1, d_diff_beta, 1);
      cublasSdot(handle, p, d_diff_beta, 1, d_diff_beta, 1, &h_RHS_1st);
      cublasSdot(handle, p, d_diff_beta, 1, d_XTrbp, 1, &h_RHS_2nd);
      RHS = L_cur * h_RHS_1st - 2.0 * h_RHS_2nd;
      
      cublasScopy(handle, n, d_y, 1, d_ymXb, 1);
      cublasSgemv(handle, ntran, n, p, &MOne, d_X,  n, d_beta, 1, &One, d_ymXb, 1);
      cublasSdot(handle, n, d_ymXb, 1, d_ymXb, 1, &h_rb);
      LHS = (h_rb - h_rbp);
      
      cond = (LHS > RHS);
    }
    
    tnext = ( 1.0  + sqrt(1 + 4 * t * t) ) / 2.0;
    cublasScopy(handle, p, d_beta, 1, d_diff_beta, 1);
    cublasSaxpy(handle, p, &MOne, d_beta_prev, 1, d_diff_beta, 1);
    t1 = (t - 1.0) / tnext;
    cublasScopy(handle, p, d_beta, 1, d_beta_p, 1);
    cublasSaxpy(handle, p, &t1, d_diff_beta, 1, d_beta_p, 1);
    cublasSdot(handle, p, d_diff_beta, 1, d_diff_beta, 1, &h_crit);
    crit[k] = sqrtf(h_crit);
    if (crit[k] < tol)
        break;
    t = tnext;
    cublasScopy(handle, p, d_beta, 1, d_beta_prev, 1);
  }
  cublasDestroy(handle);
  
  *steps = k;
  memcpy(Loss, crit, sizeof(float) * max_iter);
  cudaMemcpy(beta, d_beta, sz_p, DTH);

    free(crit);
  cudaFree(d_y);
  cudaFree(d_ymXbp);
  cudaFree(d_X);
  cudaFree(d_beta_p);
  cudaFree(d_XTrbp);
  cudaFree(d_bstar);
  cudaFree(d_beta);
  cudaFree(d_diff_beta);
  cudaFree(d_ymXb);
  cudaFree(d_beta_prev);
}
