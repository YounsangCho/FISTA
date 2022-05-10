#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdbool.h>
#include "cublas_v2.h"

#define HTD cudaMemcpyHostToDevice
#define DTH cudaMemcpyDeviceToHost
#define DTD cudaMemcpyDeviceToDevice
#define sign(i) ((i > 0) ? (1) : (-1))

extern "C"
__global__ void soft_thresh_D(double *x, double alpha, double *res, int length)
{
    double thresh = 0.0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(idx < length)
    {
        thresh = fabs(x[idx]) - alpha;
        res[idx] = (thresh > 0) ? (sign(x[idx]) * thresh) : 0.0;
    }
}


extern "C"
__host__ void FISTA_D(double *beta, double *X, double *y, double *lambda, double *L, double *Eta, double *tolerance, double *Loss, int *max_iteration, int *N, int *P, int *steps)
{
    double lam = *lambda;
    double tol = *tolerance;
    double eta = *Eta;
    double L_prev = *L;

    int max_iter = *max_iteration;
    int n = *N;
    int p = *P;
    int i_k, k;

    
    const double One = 1.0, MinusOne = -1.0;
    const double Zero = 0.0;
    
    size_t sz_np = n * p * sizeof(double);
    size_t sz_n = n * sizeof(double);
    size_t sz_p = p * sizeof(double);
    
    dim3 threadsPerBlock(32, 1);
    dim3 blocksPerGrid(1, 1);

    unsigned int bpg_p = ceil(double(p) / double(threadsPerBlock.x));
    blocksPerGrid.x = bpg_p;

    cublasHandle_t handle;
    //cublasStatus_t stat;
    cublasOperation_t tran = CUBLAS_OP_T;
    cublasOperation_t ntran = CUBLAS_OP_N;    

    // Initialization
    double *d_y, *d_ymXbp, *d_X, *d_beta_p, *d_XTrbp; 
    double *d_bstar, *d_beta, *d_diff_beta;
    double *d_ymXb, *d_beta_prev, *crit;
    double eta_ik, L_cur, pL_cur, RHS, LHS, tnext, t1;
    double h_rbp, h_RHS_1st, h_RHS_2nd, h_rb,h_crit;
    double t = 1.0;
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
    
    crit = (double *)malloc(max_iter * sizeof(double));
    
    cublasCreate(&handle);
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    for (k = 0; k < max_iter; k++)
    {

        // Calculate ||r(beta')||^2 / 2
        cublasDcopy(handle, n, d_y, 1, d_ymXbp, 1);
        cublasDgemv(handle, ntran, n, p, &MinusOne, d_X, n, d_beta_p, 1, &One, d_ymXbp, 1);
        cublasDdot(handle, n, d_ymXbp, 1, d_ymXbp, 1, &h_rbp);

        // Calculate X^T * r(beta') {or r(beta')^T * X}
        cublasDgemv(handle, tran, n, p, &One, d_X, n, d_ymXbp, 1, &Zero, d_XTrbp, 1);

        
        // Backtracking Procedure
        i_k = -1;
        cond = true;
        while (cond)
        {
            i_k += 1;
            eta_ik = pow(eta, i_k);
            L_cur = L_prev * eta_ik;
            pL_cur = (double(1.0) / L_cur);
            
            cublasDcopy(handle, p, d_beta_p, 1, d_bstar, 1);
            cublasDaxpy(handle, p, &pL_cur, d_XTrbp, 1, d_bstar, 1);
            
            soft_thresh_D<<<blocksPerGrid, threadsPerBlock>>> (d_bstar, lam / L_cur, d_beta, p);
            
            // RHS
            cublasDcopy(handle, p, d_beta, 1, d_diff_beta, 1);
            cublasDaxpy(handle, p, &MinusOne, d_beta_p, 1, d_diff_beta, 1);
            cublasDdot(handle, p, d_diff_beta, 1, d_diff_beta, 1, &h_RHS_1st);
            cublasDdot(handle, p, d_diff_beta, 1, d_XTrbp, 1, &h_RHS_2nd);
            
            RHS = L_cur * h_RHS_1st - 2.0 * h_RHS_2nd;
            
            // LHS
            cublasDcopy(handle, n, d_y, 1, d_ymXb, 1);
            cublasDgemv(handle, ntran, n, p, &MinusOne, d_X,  n, d_beta, 1, &One, d_ymXb, 1);
            cublasDdot(handle, n, d_ymXb, 1, d_ymXb, 1, &h_rb);

            LHS = (h_rb - h_rbp);

            cond = (LHS > RHS);
        }

        // Updating t
        L_prev = L_cur;
        tnext = ( 1.0  + sqrt(1 + 4 * t * t) ) / 2.0;

        // Updating z_k+1

        cublasDcopy(handle, p, d_beta, 1, d_diff_beta, 1);
        cublasDaxpy(handle, p, &MinusOne, d_beta_prev, 1, d_diff_beta, 1);
        t1 = (t - 1.0) / tnext;
        cublasDcopy(handle, p, d_beta, 1, d_beta_p, 1);
        cublasDaxpy(handle, p, &t1, d_diff_beta, 1, d_beta_p, 1);
        cublasDdot(handle, p, d_diff_beta, 1, d_diff_beta, 1, &h_crit);

        crit[k] = sqrt(h_crit);

        if (crit[k] < tol)
            break;

        t = tnext;
        //L_prev = L_cur;
        cublasDcopy(handle, p, d_beta, 1, d_beta_prev, 1);


    }
    *steps = k;
    
    memcpy(Loss, crit, sizeof(double) * max_iter);
    cudaMemcpy(beta, d_beta, sz_p, DTH);

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
    cublasDestroy(handle);
    
    free(crit);
}




extern "C"
__global__ void soft_thresh_S(float *x, float alpha, float *res, int length)
{
    float thresh = 0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(idx < length)
    {
        thresh = fabs(x[idx]) - alpha;
        res[idx] = (thresh > 0) ? (sign(x[idx]) * thresh) : 0.0;
    }
}


extern "C"
__host__ void FISTA_S(float *beta, float *X, float *y, float *lambda, float *L, float *Eta, float *tolerance, float *Loss, int *max_iteration, int *N, int *P, int *steps)
{

    int max_iter = *max_iteration;
    int n = *N;
    int p = *P;
    int i_k, k;

    float lam = *lambda;
    float tol = *tolerance;
    float eta = *Eta;
    float L_prev = *L;

    const float One = 1.0, MinusOne = -1.0;
    const float Zero = 0.0;
    
    size_t sz_np = n * p * sizeof(float);
    size_t sz_n = n * sizeof(float);
    size_t sz_p = p * sizeof(float);
    
    dim3 threadsPerBlock(32, 1);
    dim3 blocksPerGrid(1, 1);

    unsigned int bpg_p = ceil(((float) p)/threadsPerBlock.x);
    blocksPerGrid.x = bpg_p;

    cublasHandle_t handle;
    //cublasStatus_t stat;
    cublasOperation_t tran = CUBLAS_OP_T;
    cublasOperation_t ntran = CUBLAS_OP_N;    

    // Initialization
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
    
    crit = (float *)malloc(max_iter * sizeof(float));
    
    cublasCreate(&handle);
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    for (k = 0; k < max_iter; k++)
    {

        // Calculate ||r(beta')||^2 / 2
        cublasScopy(handle, n, d_y, 1, d_ymXbp, 1);
        cublasSgemv(handle, ntran, n, p, &MinusOne, d_X, n, d_beta_p, 1, &One, d_ymXbp, 1);
        cublasSdot(handle, n, d_ymXbp, 1, d_ymXbp, 1, &h_rbp);

        // Calculate X^T * r(beta') {or r(beta')^T * X}
        cublasSgemv(handle, tran, n, p, &One, d_X, n, d_ymXbp, 1, &Zero, d_XTrbp, 1);

        
        // Backtracking Procedure
        i_k = -1;
        cond = true;
        while (cond)
        {
            i_k += 1;
            eta_ik = pow(eta, i_k);
            L_cur = L_prev * eta_ik;
            pL_cur = 1.0 / L_cur;
            
            cublasScopy(handle, p, d_beta_p, 1, d_bstar, 1);
            cublasSaxpy(handle, p, &pL_cur, d_XTrbp, 1, d_bstar, 1);
            
            soft_thresh_S<<<blocksPerGrid, threadsPerBlock>>> (d_bstar, lam / L_cur, d_beta, p);
            
            // RHS
            cublasScopy(handle, p, d_beta, 1, d_diff_beta, 1);
            cublasSaxpy(handle, p, &MinusOne, d_beta_p, 1, d_diff_beta, 1);
            cublasSdot(handle, p, d_diff_beta, 1, d_diff_beta, 1, &h_RHS_1st);
            cublasSdot(handle, p, d_diff_beta, 1, d_XTrbp, 1, &h_RHS_2nd);
            
            RHS = L_cur * h_RHS_1st - 2.0 * h_RHS_2nd;
            
            // LHS
            cublasScopy(handle, n, d_y, 1, d_ymXb, 1);
            cublasSgemv(handle, ntran, n, p, &MinusOne, d_X,  n, d_beta, 1, &One, d_ymXb, 1);
            cublasSdot(handle, n, d_ymXb, 1, d_ymXb, 1, &h_rb);

            LHS = (h_rb - h_rbp);

            cond = (LHS > RHS);
        }

        // Updating t
        tnext = ( 1.0  + sqrt(1 + 4 * t * t) ) / 2.0;

        // Updating z_k+1

        cublasScopy(handle, p, d_beta, 1, d_diff_beta, 1);
        cublasSaxpy(handle, p, &MinusOne, d_beta_prev, 1, d_diff_beta, 1);
        t1 = (t - 1.0) / tnext;
        cublasScopy(handle, p, d_beta, 1, d_beta_p, 1);
        cublasSaxpy(handle, p, &t1, d_diff_beta, 1, d_beta_p, 1);
        cublasSdot(handle, p, d_diff_beta, 1, d_diff_beta, 1, &h_crit);

        crit[k] = sqrtf(h_crit);

        if (crit[k] < tol)
            break;

        t = tnext;
        //L_prev = L_cur;
        cublasScopy(handle, p, d_beta, 1, d_beta_prev, 1);


    }
    *steps = k;
    
    memcpy(Loss, crit, sizeof(float) * max_iter);
    cudaMemcpy(beta, d_beta, sz_p, DTH);

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
    cublasDestroy(handle);
    
    free(crit);
}

