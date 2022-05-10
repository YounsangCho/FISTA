#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdbool.h>

#define sign(i) ((i> 0) ? (1) : (-1))

extern "C"
__global__ void Dgemv(double alpha, double *A, double *x, double *y, int N, int P, int bl)
{
    int Row = blockDim.x * blockIdx.x + threadIdx.x;
    double Pvalue = 0.0;

    if (bl == 0){
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
__global__ void Sgemv(float alpha, float *A, float *x, float *y, int N, int P, int bl)
{
    int Row = blockDim.x * blockIdx.x + threadIdx.x;
    float Pvalue = 0.0;

    if (bl == 0){
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
__global__ void updating_y_D(double *y, double *x, double *xprev, double t, double tnext, int length)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (idx < length)
  {
    y[idx] = x[idx] + ((t - double(1.0)) / (tnext)) * (x[idx] - xprev[idx]);
  }
}

extern "C"
__global__ void updating_y_S(float *y, float *x, float *xprev, float t, float tnext, int length)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (idx < length)
  {
    y[idx] = x[idx] + ((t - float(1.0)) / (tnext)) * (x[idx] - xprev[idx]);
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
__global__ void Saxpy(float alpha, float *x, float *y, int length)
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
__global__ void vec_prod_S(float *x, float *y, float *z, int length)
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
__global__ void reduce_sum_S(float *res, int length, int mid_length)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx + mid_length < length)
  {
    res[idx] = res[idx] + res[idx + mid_length];
  }
}
