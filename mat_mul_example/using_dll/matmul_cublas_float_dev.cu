#include<cuda.h>
#include<cuda_runtime.h>
#include<cublas_v2.h>

#define HTD cudaMemcpyHostToDevice
#define DTH cudaMemcpyDeviceToHost

extern "C"
void MatMul_cuBLAS(float *A, float *B, float *C, int *N)
{
    float *dA, *dB, *dC;
    int n = *N;
    size_t sz_1 = sizeof(float);
    size_t sz_nn = n*n*sizeof(float);
    float One = 1.0, Zero = 0.0;
    float *d_One, *d_Zero;

    cublasStatus_t stat;
    cublasHandle_t handle;

    cudaMalloc(&d_One, sz_1);
    cudaMalloc(&d_Zero, sz_1);
    cudaMalloc(&dA, sz_nn);
    cudaMalloc(&dB, sz_nn);
    cudaMalloc(&dC, sz_nn);

    cudaMemcpy(d_One, &One, sz_1, HTD);
    cudaMemcpy(d_Zero, &Zero, sz_1, HTD);
    cudaMemcpy(dA, A, sz_nn, HTD);
    cudaMemcpy(dB, B, sz_nn, HTD);

    cublasCreate(&handle);
    cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_DEVICE);
    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, 
                       d_One, dA, n, dB, n, d_Zero, dC, n);
    cublasDestroy(handle);
    
    cudaMemcpy(C, dC, sz_nn, DTH);
    
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}