#include<cuda.h>
#include<cuda_runtime.h>
#include "cublas_v2.h"

#define HTD cudaMemcpyHostToDevice
#define DTH cudaMemcpyDeviceToHost

extern "C"
void MatMul_cuBLAS(double *A, double *B, double *C, int *N)
{
    double *dA, *dB, *dC;
    int n = *N;
    size_t sz = n*n*sizeof(double);
    const double One = 1.0;

    cublasStatus_t stat;
    cublasHandle_t handle;

    cudaMalloc(&dA, sz);
    cudaMalloc(&dB, sz);
    cudaMalloc(&dC, sz);

    cudaMemcpy(dA, A, sz, HTD);
    cudaMemcpy(dB, B, sz, HTD);
    cudaMemcpy(dC, C, sz, HTD);

    cublasCreate(&handle);
    
    stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, 
                       &One, dA, n, dB, n, &One, dC, n);
    cublasDestroy(handle);
    
    cudaMemcpy(C, dC, sz, DTH);
    
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}