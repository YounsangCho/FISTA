#include <cuda.h>
#include <cuda_runtime.h>

#define HTD cudaMemcpyHostToDevice
#define DTH cudaMemcpyDeviceToHost
#define DTD cudaMemcpyDeviceToDevice

__global__ void d_MatMul(double *A, double *B, double *C, int n)
{
	
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if ((i<n) && (j<n)) {
        double value = 0.0;
		for (int k = 0; k < n; ++k) {
			value += A[i*n+k]*B[k*n+j];
		}
		C[i*n+j] = value;
	}
}

extern "C" void MatMul(double *A, double *B, double *C, int *N)
{

	double *dA, *dB, *dC;
	int n = *N;
  size_t sz_nn = n*n*sizeof(double);

	dim3 TPB(32,32), BPG;
  unsigned int bpg_x = (int) ceil((double)n/TPB.x);
  unsigned int bpg_y = (int) ceil((double)n/TPB.y);
  BPG.x = bpg_x;
  BPG.y = bpg_y;
	
	cudaMalloc(&dA, sz_nn);
	cudaMalloc(&dB, sz_nn);
	cudaMalloc(&dC, sz_nn);
	cudaMemcpy(dA, A, sz_nn, HTD);
	cudaMemcpy(dB, B, sz_nn, HTD);

	d_MatMul<<<BPG, TPB>>>(dA, dB, dC, n);
	cudaMemcpy(C, dC, sz_nn, DTH);

	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);
}

