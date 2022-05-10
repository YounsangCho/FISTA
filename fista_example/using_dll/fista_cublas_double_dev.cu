#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<stdbool.h>
#include<cublas_v2.h>
#include<math.h>

#define HTD cudaMemcpyHostToDevice
#define DTH cudaMemcpyDeviceToHost
#define DTD cudaMemcpyDeviceToDevice
#define sign(i) ((i > 0) ? (1) : (-1))


extern "C"
__global__ void soft_thr_D(double *x, double alpha, double *S, int length)
{
    double thr = 0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(idx < length)
    {
        thr = fabs(x[idx]) - alpha;
        S[idx] = (thr > 0) ? (sign(x[idx]) * thr) : 0.0;
    }
}

extern "C"
__host__ void FISTA(double *beta, double *X, double *y, double *lambda, double *L, double *Eta, double *tolerance, double *Loss, int *max_iteration, int *N, int *P, int *steps)
{

    int max_iter = *max_iteration;
    int n = *N;
    int p = *P;
    int i_k, k;

    double lam = *lambda;
    double tol = *tolerance;
    double eta = *Eta;
    double L_prev = *L;

    double One = 1.0, MOne = -1.0, Zero=0.0;
    
    size_t sz_1 = sizeof(double);
    size_t sz_np = n * p * sizeof(double);
    size_t sz_n = n * sizeof(double);
    size_t sz_p = p * sizeof(double);
    
    dim3 TPB(32, 1), BPG(1, 1);

    unsigned int bpg_p = (int) ceil(((double) p)/TPB.x);
    BPG.x = bpg_p;

    cublasHandle_t handle;
    cublasOperation_t tran = CUBLAS_OP_T;
    cublasOperation_t ntran = CUBLAS_OP_N;    

    double *d_y, *d_ymXbp, *d_X, *d_beta_p, *d_XTrbp;
    double *d_bstar, *d_beta, *d_diff_beta;
    double *d_ymXb, *d_beta_prev, *crit;
    double *d_One, *d_MOne, *d_Zero;
    double eta_ik, L_cur, pL_cur, RHS, LHS, tnext, t1;
    double h_rbp, h_RHS_1st, h_RHS_2nd, h_rb, h_crit;
    double *d_rbp, *d_RHS_1st, *d_RHS_2nd, *d_rb, *d_crit;
    double t = 1.0, *d_t1, *d_pL_cur;
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
    cudaMalloc(&d_One, sz_1);
    cudaMalloc(&d_Zero, sz_1);
    cudaMalloc(&d_MOne, sz_1);
    cudaMalloc(&d_rbp, sz_1);
    cudaMalloc(&d_rb, sz_1);
    cudaMalloc(&d_RHS_1st, sz_1);
    cudaMalloc(&d_RHS_2nd, sz_1);
    cudaMalloc(&d_crit, sz_1);
    cudaMalloc(&d_t1, sz_1);
    cudaMalloc(&d_pL_cur, sz_1);

    cudaMemcpy(d_One, &One, sz_1, HTD);
    cudaMemcpy(d_MOne, &MOne, sz_1, HTD);
    cudaMemcpy(d_Zero, &Zero, sz_1, HTD);

    cudaMemcpy(d_y, y, sz_n, HTD);
    cudaMemcpy(d_X, X, sz_np, HTD);
    cudaMemcpy(d_beta_p, beta, sz_p, HTD);
    cudaMemcpy(d_beta_prev, beta, sz_p, HTD);
    crit = (double *)malloc(max_iter * sizeof(double));
    
    cublasCreate(&handle);
    cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_DEVICE);
    for (k = 0; k < max_iter; k++)
    {
        cudaMemcpy(d_ymXbp, d_y, sz_n, DTD);
        cublasDgemv(handle, ntran, n, p, d_MOne, d_X, n, 
                    d_beta_p, 1, d_One, d_ymXbp, 1);
        cublasDdot(handle, n, d_ymXbp, 1, d_ymXbp, 1, d_rbp);
        cudaMemcpy(&h_rbp, d_rbp, sizeof(double), DTH);
        cublasDgemv(handle, tran, n, p, d_One, d_X, n, 
                    d_ymXbp, 1, d_Zero, d_XTrbp, 1);
        i_k = -1;
        cond = true;
        while(cond)
        {
            i_k += 1;
            eta_ik = pow(eta, i_k);
            L_cur = L_prev * eta_ik;
            pL_cur = 1.0 / L_cur;
            cudaMemcpy(d_pL_cur, &pL_cur, sizeof(double), HTD);
            cudaMemcpy(d_bstar, d_beta_p, sz_p, DTD);
            cublasDaxpy(handle, p, d_pL_cur, d_XTrbp, 1, d_bstar, 1);
            soft_thr_D<<<BPG, TPB>>>(d_bstar, lam/L_cur, d_beta, p);

            cudaMemcpy(d_diff_beta, d_beta, sz_p, DTD);
            cublasDaxpy(handle, p, d_MOne, d_beta_p, 1, d_diff_beta, 1);
            cublasDdot(handle, p, d_diff_beta, 1, d_diff_beta, 1, d_RHS_1st);
            cublasDdot(handle, p, d_diff_beta, 1, d_XTrbp, 1, d_RHS_2nd);
            cudaMemcpy(&h_RHS_1st, d_RHS_1st, sz_1, DTH);
            cudaMemcpy(&h_RHS_2nd, d_RHS_2nd, sz_1, DTH);
            RHS = L_cur * h_RHS_1st - 2.0 * h_RHS_2nd;
            
            cudaMemcpy(d_ymXb, d_y, sz_n, DTD);
            cublasDgemv(handle, ntran, n, p, d_MOne, d_X, n,
                        d_beta, 1, d_One, d_ymXb, 1);
            cublasDdot(handle, n, d_ymXb, 1, d_ymXb, 1, d_rb);
            cudaMemcpy(&h_rb, d_rb, sz_1, DTH);
            LHS = h_rb - h_rbp;
            cond = (LHS > RHS);
        }
        tnext = ( 1.0  + sqrt(1 + 4 * t * t) ) / 2.0;
        cudaMemcpy(d_diff_beta, d_beta, sz_p, DTD);
        cublasDaxpy(handle, p, d_MOne, d_beta_prev, 1, d_diff_beta, 1);
        t1 = (t - 1.0) / tnext;
        cudaMemcpy(d_t1, &t1, sz_1, HTD);
        cudaMemcpy(d_beta_p, d_beta, sz_p, DTD);
        cublasDaxpy(handle, p, d_t1, d_diff_beta, 1, d_beta_p, 1);
        cublasDdot(handle, p, d_diff_beta, 1, d_diff_beta, 1, d_crit);
        cudaMemcpy(&h_crit, d_crit, sz_1, DTH);
        crit[k] = sqrt(h_crit);

        if (crit[k] < tol)
            break;

        t = tnext;
        L_prev = L_cur;
        cudaMemcpy(d_beta_prev, d_beta, sz_p, DTD);
    }
    *steps = k;
    memcpy(Loss, crit, sizeof(double) * max_iter);
    cudaMemcpy(beta, d_beta, sz_p, DTH);
    cublasDestroy(handle);
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
    cudaFree(d_One);
    cudaFree(d_Zero);
    cudaFree(d_MOne);
    cudaFree(d_rbp);
    cudaFree(d_rb);
    cudaFree(d_RHS_1st);
    cudaFree(d_RHS_2nd);
    cudaFree(d_crit);
    cudaFree(d_t1);
    cudaFree(d_pL_cur);
}


