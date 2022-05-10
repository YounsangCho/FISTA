
__global__ void d_MatMul(double *A, double *B, double *C, int n)
{
	
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if ((i<n) && (j<n)) {
        double Cvalue = 0.0;
		for (int k = 0; k < n; ++k) {
			Cvalue += A[i*n+k]*B[k*n+j];
		}
		C[i*n+j]=Cvalue;
	}
}


