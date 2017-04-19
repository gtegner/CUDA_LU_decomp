#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define check(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ void maxRow(const double **mat, const int N)
{
	int numThreads = blockDim.x * gridDim.x;
	int I = threadIdx.x + blockIdx.x * blockDim.x;

	int maxIndex = I;
	double maxVal = std::numeric_limits<double>::max();
	for (int i = I; i < N; i += numThreads)
	{
		y[i] = a * x[i] + y[i];
	}
	
}

int main() {

	// it's square!
	int N = 1024;
	double** mat;
	int* P;

	for (int i = 0; i < N; i++) {
		check(cudaMalloc(&(mat[i]), N * sizeof(double)));
	}
	check(cudaMalloc(&P, N * sizeof(int)));


}
