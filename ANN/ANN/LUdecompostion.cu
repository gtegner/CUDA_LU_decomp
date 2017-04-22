#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <cuda.h>

#include <sm_30_intrinsics.h>
#ifndef __CUDACC__
#define __CUDACC__
#endif // !__CUDACC__

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <float.h>
#include <algorithm> 

#pragma once
#ifdef __INTELLISENSE__
void __syncthreads();
int __shfl_down(double var, unsigned int delta, int width = warpSize);
int __shfl_down(int var, unsigned int delta, int width = warpSize);
#endif

using namespace std;

#define check(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

//__device__ inline
//double __shfl_down(double var, unsigned int srcLane, int width = 32) {
//	int2 a = *reinterpret_cast<int2*>(&var);
//	a.x = __shfl_down(a.x, srcLane, width);
//	a.y = __shfl_down(a.y, srcLane, width);
//	return *reinterpret_cast<double*>(&a);
//}

__inline__ __device__ void max_ind_warp(int& max_i, double& max_v ) {
	#pragma unroll
	for (int offset = 32 / 2; offset > 0; offset /= 2) {
		double cand_v = __shfl_down(max_v, offset);
		int cand_i = __shfl_down(max_i, offset);
		if (cand_v > max_v) {
			max_v = cand_v;
			max_i = cand_i;
		}
	}
}

__inline__ __device__ void max_ind_block(int& max_i, double& max_v)
{
	__shared__ double max_vs[32];
	__shared__ int max_is[32];

	max_ind_warp(max_i, max_v);

	int lane = threadIdx.x % warpSize;
	int warpID = threadIdx.x / warpSize;

	if (lane == 0) {
		max_vs[warpID] = max_v;
		max_is[warpID] = max_i;
	}

	__syncthreads();

	max_v = (threadIdx.x < blockDim.x / warpSize) ? max_vs[lane] : 0.0;
	max_i = (threadIdx.x < blockDim.x / warpSize) ? max_is[lane] : 0;

	max_ind_warp(max_i, max_v);

}

// Allways call with only one block and 1024 threads!
__global__ void maxColumn(double ** __restrict__ mat, const int c, const int N, int* __restrict__ P)
{
	int numThreads = blockDim.x * gridDim.x;
	int I = threadIdx.x + blockIdx.x * blockDim.x;

	int max_i = I;
	double max_v = DBL_MIN;
	for (int i = 2 * I; i < N; i += numThreads)
	{
		double cand_v = mat[i][c];
		if (cand_v > max_v) {
			max_v = cand_v;
			max_i = i;
		}
	}

	max_ind_block(max_i, max_v);

	if (threadIdx.x == 0) {
		double* tmp = mat[max_i];
		mat[max_i] = mat[c];
		mat[c] = tmp;

		P[c] = max_i;
	}
}

__global__ void compute_L_column (double ** __restrict__ mat, const int row, const int N) {
	double diag = mat[row][row];

	int numThreads = blockDim.x * gridDim.x;
	int I = threadIdx.x + blockIdx.x * blockDim.x;

	for (int i = I; i < N; i += numThreads) {
		mat[i][row] = mat[i][row] / diag;
	}
}

__global__ void L_substitution(const  double ** __restrict__ mat, double* __restrict__ b, const int N) {
	extern __shared__ double _b[];

	int numThreads = blockDim.x * gridDim.x;
	int I = threadIdx.x + blockIdx.x * blockDim.x;

	for (int i = I; i < N; i += numThreads) {
		_b[i] = b[i];
	}

	// on diagonal there are ones which are however not saved
	for (int j = 0; j < N; j++) {
		double b_j = _b[j];
		for (int i = j + 1 + I; i < N; i += numThreads) {
			_b[i] -= mat[i][j] * b_j;
		}
	}

	for (int i = I; i < N; i += numThreads) {
		b[i] = _b[i];
	}
}

__global__ void U_substitution(const  double ** __restrict__ mat, double* __restrict__ b, const int N) {
	extern __shared__ double _b[];

	int numThreads = blockDim.x * gridDim.x;
	int I = threadIdx.x + blockIdx.x * blockDim.x;

	for (int i = I; i < N; i += numThreads) {
		_b[i] = b[i];
	}

	for (int j = N - 1; j > 0; j--) {
		if ((j % numThreads) == I) {
			_b[j] /= mat[j][j];
		}
		__syncthreads();
		double b_j = _b[j];
		for (int i = I; i < j; i += numThreads) {
			_b[i] -= mat[i][j] * b_j;
		}
	}

	for (int i = I; i < N; i += numThreads) {
		b[i] = _b[i];
	}
}

int* LU_decompostion(double** mat, const int N) {

	double** d_mat;
	check(cudaMallocManaged(&d_mat, N * sizeof(double*)));
	for (int i = 0; i < N; i++) {
		check(cudaMalloc(&(d_mat[i]), N * sizeof(double)));
	}

	int* P;
	check(cudaMallocManaged(&P, N * sizeof(int)));

	for (int i = 0; i < N; i++) {
		check(cudaMemcpyAsync(&(d_mat[i]), &(mat[i]), N, cudaMemcpyHostToDevice));
	}

	return P;
}

// deconstructs b during execution
// assume d_mat and d_P are device pointer
void inline solve_LU(const double** d_mat, const int* d_P, double* b, const int N) {
	double* d_b;
	check(cudaMalloc(&d_b, N * sizeof(double)));
	check(cudaMemcpy(&d_b, &b, N, cudaMemcpyHostToDevice));

	unsigned int num_threads = min(256, N);
	unsigned int num_blocks = (N + 255) / 256;

	L_substitution<<<num_blocks, num_threads, N * sizeof(double) >>>(d_mat, d_b, N);
	U_substitution<<<num_blocks, num_threads, N * sizeof(double) >>>(d_mat, d_b, N);

	// TODO permute d_b regarding to d_P

	check(cudaMemcpy(&b, &d_b, N, cudaMemcpyDeviceToHost));
}

int main() {

	// it's square!
	// int N = 1024;



	check(cudaDeviceReset());
}
