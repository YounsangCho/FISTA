#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<stdbool.h>
#include<math.h>
#define blocksize 32
#define HTD cudaMemcpyHostToDevice
#define DTH cudaMemcpyDeviceToHost
#define DTD cudaMemcpyDeviceToDevice
#define sign(i) ((i> 0) ? (1) : (-1))

extern "C"
__global__ void Dgemv(double alpha, double *A, double *x, double *y, int N, int P, bool Bool)
{
    int Row = blockDim.x * blockIdx.x + threadIdx.x;
    double Pvalue = 0.0;
    
    if (Bool == false){
        if (Row < N){
            for (int j = 0; j < P; j++){
                Pvalue += A[Row * P + j] * x[j];
            }
            y[Row] += (alpha * Pvalue);
        }
    }
    
    else{
        if (Row < P){
            for (int i = 0; i < N; i++){
                Pvalue += A[Row + i * P] * x[i];
            }
            y[Row] += (alpha * Pvalue);
        }
    }
}


extern "C"
__global__ void soft_thresh_D(double *x, double alpha, double *res, int length)
{
    double thresh = 0.0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < length)
    {
        thresh = fabs(x[idx]) - alpha;
        res[idx] = (thresh > 0) ? (sign(x[idx])*thresh) : 0.0;
    }
}


extern "C"
__global__ void updating_z_D(double *z, double *x, double *xprev, double t, double tnext, int length)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < length)
    {
        z[idx] = x[idx] + ((t - double(1.0)) / (tnext)) * (x[idx] - xprev[idx]);
    }
}


extern "C"
__global__ void Daxpy(double alpha, double *x, double *y, int length)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < length)
    {
        y[idx] += (alpha * x[idx]);
    }
}


extern "C"
__global__ void vec_prod_D(double *x, double *y, double *z, int length)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx < length)
    {
        z[idx] = x[idx] * y[idx];
    }
}


extern "C"
__global__ void reduce_sum_D(double *res, int length, int mid_length)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx + mid_length < length)
    {
        res[idx] = res[idx] + res[idx + mid_length];
    }
}




extern "C"
__host__ void FISTA_D(double *beta, double *X, double *y, double *lambda, double *L, double *Eta, double *tolerance,
                      double *Loss, int *max_iteration, int *N, int *P, int *steps) {
    int max_iter = *max_iteration;
    int n = *N;
    int p = *P;
    int i, i_k, k;
    
    double lam = *lambda;
    double tol = *tolerance;
    double eta = *Eta;
    double L_prev = *L;
    double One = 1.0, MinusOne = -1.0;
    
    double *d_y, *d_ymXbp, *d_ymXbp2, *d_X, *d_beta_p, *d_XTrbp;
    double *d_bstar, *d_beta, *d_diff_beta, *d_diff_beta2;
    double *d_ymXb, *d_ymXb2, *d_beta_prev, *d_XTrbpd, *crit;
    
    double eta_ik, L_cur, pL_cur, RHS, LHS, tnext, t1;
    double *h_rbp, *h_RHS_1st, *h_RHS_2nd, *h_rb, *h_crit;
    double t = 1.0;
    
    double log_p = (float)log2(p);
    double log_n = (float)log2(n);
    int step_p = ceil(log_p);
    int step_n = ceil(log_n);
    int nbnp;
    int mid_len_n = (int)pow(2, step_n - 1);
    int mid_len_p = (int)pow(2, step_p - 1);
    int mid_tmp_n, mid_tmp_p;
    int len_p, len_n;
    
    bool cond;
    
    size_t sz_np = n * p * sizeof(double);
    size_t sz_n = n * sizeof(double);
    size_t sz_p = p * sizeof(double);
    
    dim3 TPB(32, 1);
    dim3 BPG(1, 1);
    
    unsigned int bpg_p = ceil(float(p) / double(TPB.x));
    unsigned int bpg_n = ceil(float(n) / double(TPB.x));

    h_rbp = (double *) malloc(sizeof(double));
    h_RHS_1st  = (double *) malloc(sizeof(double));
    h_RHS_2nd  = (double *) malloc(sizeof(double));
    h_rb  = (double *) malloc(sizeof(double));
    h_crit = (double *) malloc(sizeof(double));
    
    cudaMalloc(&d_y, sz_n);
    cudaMalloc(&d_ymXbp, sz_n);
    cudaMalloc(&d_ymXbp2, sz_n);
    cudaMalloc(&d_X, sz_np);
    cudaMalloc(&d_beta_p, sz_p);
    cudaMalloc(&d_XTrbp, sz_p);
    cudaMalloc(&d_bstar, sz_p);
    cudaMalloc(&d_beta, sz_p);
    cudaMalloc(&d_diff_beta, sz_p);
    cudaMalloc(&d_diff_beta2, sz_p);
    cudaMalloc(&d_ymXb, sz_n);
    cudaMalloc(&d_ymXb2, sz_n);
    cudaMalloc(&d_beta_prev, sz_p);
    cudaMalloc(&d_XTrbpd, sz_p);
    
    cudaMemcpy(d_y, y, sz_n, HTD);
    cudaMemcpy(d_X, X, sz_np, HTD);
    cudaMemcpy(d_beta_p, beta, sz_p, HTD);
    cudaMemcpy(d_beta_prev, beta, sz_p, HTD);
    
    crit = (double *)malloc(sizeof(double) * max_iter);
    
    for(k = 0; k < max_iter; k++)
    {
        // Calculate r(beta')
        BPG.x = bpg_n;
        cudaMemcpy(d_ymXbp, d_y, sz_n, DTD);
        Dgemv<<<BPG, TPB>>>(MinusOne, d_X, d_beta_p, d_ymXbp, n, p, false);
        vec_prod_D<<<BPG, TPB>>>(d_ymXbp, d_ymXbp, d_ymXbp2, n);

        mid_tmp_n = mid_len_n;
        len_n = n;
        for(i = 0; i < step_n; i++)
        {
          nbnp = ceil(double(mid_tmp_n) / double(TPB.x));
          reduce_sum_D<<<nbnp, TPB>>>(d_ymXbp2, len_n, mid_tmp_n);
          len_n = mid_tmp_n;
          mid_tmp_n /= 2;
        }
        cudaMemcpy(h_rbp, &d_ymXbp2[0], sizeof(double), DTH);

    // Calculate X^T * r(beta')
        cudaMemset(d_XTrbp, 0, sz_p);
        BPG.x = bpg_p;
        Dgemv<<<BPG, TPB>>>(One, d_X, d_ymXbp, d_XTrbp, n, p, true);
        
        //Backtracking Procedure
        i_k = -1;
        cond = true;
        while (cond)
        {
            i_k += 1;
            eta_ik = pow(eta, i_k);
            L_cur = L_prev * eta_ik;
            pL_cur = float(1.0 / L_cur);
            
            BPG.x = bpg_p;
            cudaMemcpy(d_bstar, d_beta_p, sz_p, DTD);
            Daxpy<<<BPG, TPB>>>(pL_cur, d_XTrbp, d_bstar, p);
            soft_thresh_D<<<BPG, TPB>>>(d_bstar, lam / L_cur , d_beta, p);
            
            //RHS
            cudaMemcpy(d_diff_beta, d_beta, sz_p, DTD);
            Daxpy<<<BPG, TPB>>>(MinusOne, d_beta_p, d_diff_beta, p);
            vec_prod_D<<<BPG, TPB>>>(d_diff_beta, d_diff_beta, d_diff_beta2, p);
            vec_prod_D<<<BPG, TPB>>>(d_XTrbp, d_diff_beta,  d_XTrbpd, p);
            
            mid_tmp_p = mid_len_p;
            len_p = p;
            for(i = 0; i < step_p; i++)
            {
                nbnp = ceil(double(mid_tmp_p) / double(TPB.x));
                reduce_sum_D<<<nbnp, TPB>>> (d_diff_beta2, len_p, mid_tmp_p);
                reduce_sum_D<<<nbnp, TPB>>> (d_XTrbpd, len_p, mid_tmp_p);
                len_p = mid_tmp_p;
                mid_tmp_p /= 2;
            }
            cudaMemcpy(h_RHS_1st, &d_diff_beta2[0], sizeof(double), DTH);
            cudaMemcpy(h_RHS_2nd, &d_XTrbpd[0], sizeof(double), DTH);
            
            RHS = L_cur * h_RHS_1st[0] - 2.0 * h_RHS_2nd[0];
            
            
            //LHS
            mid_tmp_n = mid_len_n;
            len_n = n;
            BPG.x = bpg_n;
            cudaMemcpy(d_ymXb, d_y, sz_n, DTD);
            Dgemv<<<BPG, TPB>>>(MinusOne, d_X, d_beta, d_ymXb, n, p, false);
            vec_prod_D<<<BPG, TPB>>>(d_ymXb, d_ymXb, d_ymXb2, n);
            for(i = 0; i < step_n; i++)
            {
                nbnp = ceil(double(mid_tmp_n) / double(TPB.x));
                reduce_sum_D<<<nbnp, TPB>>>(d_ymXb2, len_n, mid_tmp_n);
                len_n = mid_tmp_n;
                mid_tmp_n /= 2;
            }
            cudaMemcpy(h_rb, &d_ymXb2[0], sizeof(double), DTH);
            
            
            LHS = h_rb[0] - h_rbp[0];
            
            cond = (LHS > RHS);
            
        }
        
        tnext = (1.0 + sqrt(1 + 4 * t * t) ) / 2.0;
        BPG.x = bpg_p;
        cudaMemcpy(d_diff_beta, d_beta, sz_p, DTD);
        Daxpy<<<BPG, TPB>>>(MinusOne, d_beta_prev, d_diff_beta, p);
        t1 = (t - 1.0) / tnext;
        cudaMemcpy(d_beta_p, d_beta, sz_p, DTD);
        Daxpy<<<BPG, TPB>>>(t1, d_diff_beta, d_beta_p, p);
        
        vec_prod_D<<<BPG, TPB>>>(d_diff_beta, d_diff_beta, d_diff_beta2, p);
        len_p = p;
        mid_tmp_p = mid_len_p;
        
        for(i = 0; i < step_p; i++)
        {
            nbnp = ceil(float(mid_tmp_p) / float(TPB.x));
            reduce_sum_D<<<nbnp, TPB>>>(d_diff_beta2, len_p, mid_tmp_p);
            len_p = mid_tmp_p;
            mid_tmp_p /= 2;
        }
        
        cudaMemcpy(h_crit, &d_diff_beta2[0], sizeof(double), DTH);
        crit[k] = sqrt(h_crit[0]);
        
        if (crit[k] < tol)
            break;

        t = tnext;
        cudaMemcpy(d_beta_prev, d_beta, sz_p, DTD);
    }
    
    *steps = k;
    memcpy(Loss, crit, sizeof(double) * max_iter);
    cudaMemcpy(beta, d_beta, sz_p, DTH);
    
    free(crit);
    cudaFree(d_y);
    cudaFree(d_ymXbp);
    cudaFree(d_ymXbp2);
    cudaFree(d_X);
    cudaFree(d_beta_p);
    cudaFree(d_XTrbp);
    cudaFree(d_bstar);
    cudaFree(d_beta);
    cudaFree(d_diff_beta);
    cudaFree(d_diff_beta2);
    cudaFree(d_ymXb);
    cudaFree(d_ymXb2);
    cudaFree(d_beta_prev);
    cudaFree(d_XTrbpd);
}


extern "C"
__global__ void Sgemv(float alpha, float *A, float *x, float *y, int N, int P, bool Bool)
{
    int Row = blockDim.x * blockIdx.x + threadIdx.x;
    float Pvalue = 0.0;
    
    if (Bool == false){
        if (Row < N){
            for (int j = 0; j < P; j++){
                Pvalue += A[Row * P + j] * x[j];
            }
            y[Row] += (alpha * Pvalue);
        }
    }
    
    else{
        if (Row < P){
            for (int i = 0; i < N; i++){
                Pvalue += A[Row + i * P] * x[i];
            }
            y[Row] += (alpha * Pvalue);
        }
    }
}


extern "C"
__global__ void soft_thresh_S(float *x, float alpha, float *res, int length)
{
    float thresh = 0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < length)
    {
        thresh = fabs(x[idx]) - alpha;
        res[idx] = (thresh > 0) ? (sign(x[idx])*thresh) : 0.0;
    }
}


extern "C"
__global__ void updating_z_S(float *z, float *x, float *xprev, float t, float tnext, int length)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < length)
    {
        z[idx] = x[idx] + ((t - float(1.0)) / (tnext)) * (x[idx] - xprev[idx]);
    }
}


extern "C"
__global__ void Saxpy(float alpha, float *x, float *y, int length)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < length)
    {
        y[idx] += (alpha * x[idx]);
    }
}


extern "C"
__global__ void vec_prod_S(float *x, float *y, float *z, int length)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx < length)
    {
        z[idx] = x[idx] * y[idx];
    }
}


extern "C"
__global__ void reduce_sum_S(float *res, int length, int mid_length)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx + mid_length < length)
    {
        res[idx] = res[idx] + res[idx + mid_length];
    }
}


extern "C"
__host__ void FISTA_S(float *beta, float *X, float *y, float *lambda, float *L, float *Eta, float *tolerance,
                      float *Loss, int *max_iteration, int *N, int *P, int *steps) {
    int max_iter = *max_iteration;
    int n = *N;
    int p = *P;
    int i, i_k, k;
    
    float lam = *lambda;
    float tol = *tolerance;
    float eta = *Eta;
    float L_prev = *L;
    float One = 1.0, MinusOne = -1.0;
    
    float *d_y, *d_ymXbp, *d_ymXbp2, *d_X, *d_beta_p, *d_XTrbp;
    float *d_bstar, *d_beta, *d_diff_beta, *d_diff_beta2;
    float *d_ymXb, *d_ymXb2, *d_beta_prev, *d_XTrbpd, *crit;
    
    float eta_ik, L_cur, pL_cur, RHS, LHS, tnext, t1;
    float h_rbp=0., h_RHS_1st=0., h_RHS_2nd=0., h_rb=0., h_crit=0.;
    float t = 1.0;
    
    float log_p = (float)log2(p);
    float log_n = (float)log2(n);
    int step_p = ceil(log_p);
    int step_n = ceil(log_n);
    int nbnp;
    int mid_len_n = (int)pow(2, step_n - 1);
    int mid_len_p = (int)pow(2, step_p - 1);
    int mid_tmp_n, mid_tmp_p;
    int len_p, len_n;
    
    bool cond;
    
    size_t sz_np = n * p * sizeof(float);
    size_t sz_n = n * sizeof(float);
    size_t sz_p = p * sizeof(float);
    
    dim3 TPB(blocksize, 1);
    dim3 BPG(1, 1);
    
    unsigned int bpg_p = ceil(float(p) / float(TPB.x));
    unsigned int bpg_n = ceil(float(n) / float(TPB.x));
    
    cudaMalloc(&d_y, sz_n);
    cudaMalloc(&d_ymXbp, sz_n);
    cudaMalloc(&d_ymXbp2, sz_n);
    cudaMalloc(&d_X, sz_np);
    cudaMalloc(&d_beta_p, sz_p);
    cudaMalloc(&d_XTrbp, sz_p);
    cudaMalloc(&d_bstar, sz_p);
    cudaMalloc(&d_beta, sz_p);
    cudaMalloc(&d_diff_beta, sz_p);
    cudaMalloc(&d_diff_beta2, sz_p);
    cudaMalloc(&d_ymXb, sz_n);
    cudaMalloc(&d_ymXb2, sz_n);
    cudaMalloc(&d_beta_prev, sz_p);
    cudaMalloc(&d_XTrbpd, sz_p);
    
    cudaMemcpy(d_y, y, sz_n, HTD);
    cudaMemcpy(d_X, X, sz_np, HTD);
    cudaMemcpy(d_beta_p, beta, sz_p, HTD);
    cudaMemcpy(d_beta_prev, beta, sz_p, HTD);
    
    crit = (float *)malloc(sizeof(float) * max_iter);
    
    for(k = 0; k < max_iter; k++)
    {
        // Calculate r(beta')
        BPG.x = bpg_n;
        cudaMemcpy(d_ymXbp, d_y, sz_n, DTD);
        Sgemv<<<BPG, TPB>>>(MinusOne, d_X, d_beta_p, d_ymXbp, n, p, false);
        vec_prod_S<<<BPG, TPB>>>(d_ymXbp, d_ymXbp, d_ymXbp2, n);

        mid_tmp_n = mid_len_n;
        len_n = n;
        for(i = 0; i < step_n; i++)
        {
          nbnp = ceil(float(mid_tmp_n) / float(TPB.x));
          reduce_sum_S<<<nbnp, TPB>>>(d_ymXbp2, len_n, mid_tmp_n);
          len_n = mid_tmp_n;
          mid_tmp_n /= 2;
        }
        cudaMemcpy(&h_rbp, &d_ymXbp2[0], sizeof(float), DTH);

        // Calculate X^T * r(beta')
        cudaMemset(d_XTrbp, 0, sz_p);
        BPG.x = bpg_p;
        Sgemv<<<BPG, TPB>>>(One, d_X, d_ymXbp, d_XTrbp, n, p, true);
        
        //Backtracking Procedure
        i_k = -1;
        cond = true;
        while (cond)
        {
            i_k += 1;
            eta_ik = pow(eta, i_k);
            L_cur = L_prev * eta_ik;
            pL_cur = float(1.0 / L_cur);
            
            BPG.x = bpg_p;
            cudaMemcpy(d_bstar, d_beta_p, sz_p, DTD);
            Saxpy<<<BPG, TPB>>>(pL_cur, d_XTrbp, d_bstar, p);
            soft_thresh_S<<<BPG, TPB>>>(d_bstar, lam / L_cur , d_beta, p);
            
            //RHS
            cudaMemcpy(d_diff_beta, d_beta, sz_p, DTD);
            Saxpy<<<BPG, TPB>>>(MinusOne, d_beta_p, d_diff_beta, p);
            vec_prod_S<<<BPG, TPB>>>(d_diff_beta, d_diff_beta, d_diff_beta2, p);
            vec_prod_S<<<BPG, TPB>>>(d_XTrbp, d_diff_beta,  d_XTrbpd, p);
            
            mid_tmp_p = mid_len_p;
            len_p = p;
            for(i = 0; i < step_p; i++)
            {
                nbnp = ceil(float(mid_tmp_p) / float(TPB.x));
                reduce_sum_S<<<nbnp, TPB>>> (d_diff_beta2, len_p, mid_tmp_p);
                reduce_sum_S<<<nbnp, TPB>>> (d_XTrbpd, len_p, mid_tmp_p);
                len_p = mid_tmp_p;
                mid_tmp_p /= 2;
            }
            cudaMemcpy(&h_RHS_1st, &d_diff_beta2[0], sizeof(float), DTH);
            cudaMemcpy(&h_RHS_2nd, &d_XTrbpd[0], sizeof(float), DTH);
            
            RHS = L_cur * h_RHS_1st - 2.0 * h_RHS_2nd;
            
            
            //LHS
            mid_tmp_n = mid_len_n;
            len_n = n;
            BPG.x = bpg_n;
            cudaMemcpy(d_ymXb, d_y, sz_n, DTD);
            Sgemv<<<BPG, TPB>>>(MinusOne, d_X, d_beta, d_ymXb, n, p, false);
            vec_prod_S<<<BPG, TPB>>>(d_ymXb, d_ymXb, d_ymXb2, n);
            for(i = 0; i < step_n; i++)
            {
                nbnp = ceil(((float)mid_tmp_n)/TPB.x);
                reduce_sum_S<<<nbnp, TPB>>>(d_ymXb2, len_n, mid_tmp_n);
                len_n = mid_tmp_n;
                mid_tmp_n /= 2;
            }
            cudaMemcpy(&h_rb, &d_ymXb2[0], sizeof(float), DTH);
            LHS = h_rb - h_rbp;
            cond = (LHS > RHS);
        }
        
        tnext =  (1.0 + sqrtf(1 + 4 * t * t) )/2.0;
        
        BPG.x = bpg_p;
        cudaMemcpy(d_diff_beta, d_beta, sz_p, DTD);
        Saxpy<<<BPG, TPB>>>(MinusOne, d_beta_prev, d_diff_beta, p);
        t1 = (t - 1.0)/tnext;
        cudaMemcpy(d_beta_p, d_beta, sz_p, DTD);
        Saxpy<<<BPG, TPB>>>(t1, d_diff_beta, d_beta_p, p);

        vec_prod_S<<<BPG, TPB>>>(d_diff_beta, d_diff_beta, d_diff_beta2, p);
        len_p = p;
        mid_tmp_p = mid_len_p;
        
        for(i = 0; i < step_p; i++)
        {
            nbnp = ceil(((float) mid_tmp_p) / TPB.x);
            reduce_sum_S<<<nbnp, TPB>>>(d_diff_beta2, len_p, mid_tmp_p);
            len_p = mid_tmp_p;
            mid_tmp_p /= 2;
        }
        
        cudaMemcpy(&h_crit, &d_diff_beta2[0], sizeof(float), DTH);
        crit[k] = sqrtf(h_crit);
        
        if (crit[k] < tol)
            break;

        t = tnext;
        cudaMemcpy(d_beta_prev, d_beta, sz_p, DTD);
    }
    
    *steps = k;
    memcpy(Loss, crit, sizeof(float) * max_iter);
    cudaMemcpy(beta, d_beta, sz_p, DTH);
    
    free(crit);
    cudaFree(d_y);
    cudaFree(d_ymXbp);
    cudaFree(d_ymXbp2);
    cudaFree(d_X);
    cudaFree(d_beta_p);
    cudaFree(d_XTrbp);
    cudaFree(d_bstar);
    cudaFree(d_beta);
    cudaFree(d_diff_beta);
    cudaFree(d_diff_beta2);
    cudaFree(d_ymXb);
    cudaFree(d_ymXb2);
    cudaFree(d_beta_prev);
    cudaFree(d_XTrbpd);
}
