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
#include <iostream>
#include <iomanip>

#pragma once
#ifdef __INTELLISENSE__
void __syncthreads();
int __shfl_down(double var, unsigned int delta, int width = warpSize);
int __shfl_down(int var, unsigned int delta, int width = warpSize);
#endif

using namespace std;

ostream& operator<<(ostream& out, pair<int, double**> mat)
{
	int m = mat.first;
	double** arr = mat.second;

	cout << setprecision(5) << fixed << endl;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < m; j++) {
			cout << arr[i][j] << (j + 1 < m ? ",\t" : "\n");
		}
	}
	return out;
}

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
		if (abs(cand_v) > abs(max_v)) {
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

	//  printf("Thread %i: %i %f \n", threadIdx.x, max_i, max_v);

	int lane = threadIdx.x % warpSize;
	int warpID = threadIdx.x / warpSize;

	if (lane == 0) {
		max_vs[warpID] = max_v;
		max_is[warpID] = max_i;
	}

	__syncthreads();

	max_v = (threadIdx.x < 32) ? max_vs[lane] : 0.0;
	max_i = (threadIdx.x < 32) ? max_is[lane] : 0;
	
	// printf("Thread %i: %i %f \n", threadIdx.x, max_i, max_v);

	max_ind_warp(max_i, max_v);

	// printf("Thread %i: %i %f \n", threadIdx.x, max_i, max_v);

}

// Allways call with only one block and 1024 threads!
__global__ void maxColumn(double ** __restrict__ mat, const int c, const int N, int* __restrict__ P)
{
	int numThreads = blockDim.x * gridDim.x;
	int I = threadIdx.x + blockIdx.x * blockDim.x;

	int max_i = I;
	double max_v = 0.0;
	for (int i = c + I; i < N; i += numThreads)
	{
		double cand_v = mat[i][c];
		if (abs(cand_v) >= abs(max_v)) {
			max_v = cand_v;
			max_i = i;
		}
	}

	max_ind_block(max_i, max_v);

	if (threadIdx.x == 0) {
		// printf("%i %f \n", max_i, max_v);
		double* tmp = mat[max_i];
		mat[max_i] = mat[c];
		mat[c] = tmp;

		// printf("max_i %i, P[c] %i, P[max_i] %i \n", max_i, P[c], P[max_i]);
		int itmp = P[c];
		P[c] = P[max_i];
		P[max_i] = itmp;
	}
}

__global__ void compute_L_column (double ** __restrict__ mat, const int col, const int N) {
	double diag = mat[col][col];

	int numThreads = blockDim.x * gridDim.x;
	int I = threadIdx.x + blockIdx.x * blockDim.x;

	for (int i = I + col + 1; i < N; i += numThreads) {
		mat[i][col] /= diag;
	}
}

__global__ void reduce(double ** __restrict__ mat, const int n, const int N) {

	__shared__ double actv[16];
	__shared__ double fact[16];

	//int numThreadsX = blockDim.x * gridDim.x;
	//int numThreadsY = blockDim.y * gridDim.y;

	int I = threadIdx.x + blockIdx.x * blockDim.x;
	int J = threadIdx.y + blockIdx.y * blockDim.y;

	if (I + n + 1 < N && threadIdx.y == 0) {
		actv[threadIdx.x] = mat[n][I + n + 1];
	}

	if (J + n + 1 < N && threadIdx.x == 0) {
		fact[threadIdx.y] = mat[J + n + 1][n];
	}

	__syncthreads();

	if (I + n + 1 < N && J + n + 1 < N) {
		mat[J + n + 1][I + n + 1] -= actv[threadIdx.x] * fact[threadIdx.y];
	}

}

__global__ void L_substitution(double ** __restrict__ mat, double* __restrict__ b, const int N) {
	extern __shared__ double _b[];

	int numThreads = blockDim.x * gridDim.x;
	int I = threadIdx.x + blockIdx.x * blockDim.x;

	//for (int i = I; i < N; i += numThreads) {
	//	printf("%i %f, ", i, b[i]);
	//	_b[i] = b[i];
	//}

	// on diagonal there are ones which are however not saved
	for (int j = 0; j < N; j++) {
		double b_j = _b[j];
		for (int i = j + 1 + I; i < N; i += numThreads) {
			_b[i] -= mat[i][j] * b_j;
		}
	}

	for (int i = I; i < N; i += numThreads) {
		// printf("%i %f, ",i, _b[i]);
		b[i] = _b[i];
	}
}

__global__ void U_substitution(double ** __restrict__ mat, double* __restrict__ b, const int N) {
	extern __shared__ double _b[];

	int numThreads = blockDim.x * gridDim.x;
	int I = threadIdx.x + blockIdx.x * blockDim.x;

	for (int i = I; i < N; i += numThreads) {
		_b[i] = b[i];
	}

	for (int j = N - 1; j >= 0; j--) {
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

__global__ void permute(double* d_b, const int* P, int const N) {

	extern __shared__ double _b[];

	int numThreads = blockDim.x * gridDim.x;
	int I = threadIdx.x + blockIdx.x * blockDim.x;

	//for (int i = I; i < N; i += numThreads) {
	//	printf("%i %f ", i, d_b[i]);
	//}
	// printf("\n ");

	for (int i = I; i < N; i += numThreads) {
		// printf("I=%i, i=%i, P[i]=%i, %f \n", I, i, P[i], d_b[P[i]]);
		_b[i] = d_b[P[i]];
	}

	for (int i = I; i < N; i += numThreads) {
		d_b[i] = _b[i];
	}
}

void LU_decompostion(double** mat, const int N, double** &d_mat, int* &P) {

	check(cudaMallocManaged(&d_mat, N * sizeof(double*)));
	for (int i = 0; i < N; i++) {
		// cudaMalloc
		check(cudaMallocManaged(&(d_mat[i]), N * sizeof(double)));
	}

	check(cudaMallocManaged(&P, N * sizeof(int)));
	for (int i = 0; i < N; i++) {
		P[i] = i;
	}

	for (int i = 0; i < N; i++) {
		check(cudaMemcpyAsync(&(d_mat[i]), &(mat[i]), N, cudaMemcpyHostToDevice));
	}

	cudaDeviceSynchronize();

	unsigned int num_threads = min(256, N);
	unsigned int num_blocks = (N + num_threads - 1) / num_threads;
	dim3 threadsGrid = dim3(16, 16);

	for (int i = 0; i < N; i++) {
		maxColumn<<<1, 1024>>>(d_mat, i, N, P);
		compute_L_column<<<num_blocks, num_threads>>>(d_mat, i, N);
		
		int dim = (N + 16 - 1) / 16;
		dim3 blockgrid = dim3(dim, dim);
		reduce<<<blockgrid, threadsGrid>>>(d_mat, i, N);

		//cudaDeviceSynchronize();
		//auto Mat = make_pair(N, d_mat);
		//cout << endl << Mat << endl;
		//for (int i = 0; i < N; i++) {
		//	printf("%i, ", P[i]);
		//} printf("\n");
	}
}

// deconstructs b during execution
// assume d_mat and d_P are device pointer
void inline solve_LU(double** d_mat, int* d_P, double* b, const int N) {
	double* d_b;
	check(cudaMallocManaged(&d_b, N * sizeof(double)));
	check(cudaMemcpy(d_b, b, N * sizeof(double), cudaMemcpyDefault));

	unsigned int num_threads = min(256, N);
	unsigned int num_blocks = (N + num_threads - 1) / num_threads;

	permute<<<num_blocks, num_threads, N * sizeof(double)>>>(d_b, d_P, N);
	cudaDeviceSynchronize();

	L_substitution<<<num_blocks, num_threads, N * sizeof(double) >>>(d_mat, d_b, N);
	U_substitution<<<num_blocks, num_threads, N * sizeof(double) >>>(d_mat, d_b, N);
	cudaDeviceSynchronize();

	check(cudaMemcpy(b, d_b, N * sizeof(double), cudaMemcpyDefault));
}

void test_maxColumn(double ** __restrict__ mat, const int c, const int N, int* P) {

	maxColumn<<<1,8>>>(mat, c, N, P);

	cudaDeviceSynchronize();

	for(int i = 0; i < N; i++){
		cout << P[i] << ", ";
	}

}

void test_L_column(double ** __restrict__ mat, const int c, const int N) {

	compute_L_column<<<1, 8 >>>(mat, c, N);

	cudaDeviceSynchronize();
	auto Mat = make_pair(N, mat);
	cout << endl << Mat << endl;

}

void test_reduce(double ** __restrict__ mat, const int c, const int N) {
	int* P;
	check(cudaMallocManaged(&P, N * sizeof(int)));
	for (int i = 0; i < N; i++) {
		P[i] = i;
	}

	test_maxColumn(mat, c, N, P);
	test_L_column(mat, c, N);

	dim3 threadsGrid = dim3(4, 4);
	int dim = (N + 4 - 1) / 4;
	dim3 blockgrid = dim3(dim, dim);

	reduce<<<blockgrid, threadsGrid >>>(mat, c, N);

	cudaDeviceSynchronize();
	auto Mat = make_pair(N, mat);
	cout << endl << Mat << endl;
}

void test_LU_decomp(double ** __restrict__ mat, int N) {
	double** d_mat;
	int* P;
	LU_decompostion(mat, N, d_mat, P);

	cudaDeviceSynchronize();
	auto Mat = make_pair(N, d_mat);
	cout << endl << Mat << endl;

	double* b = (double*) calloc(sizeof(double), N);
	for (int i = 0; i < N; i++) {
		b[i] = i;
	}

	cudaDeviceSynchronize();

	solve_LU(d_mat, P, b, N);
	cudaDeviceSynchronize();

	for (int i = 0; i < N; i++) {
		cout << b[i] << ", ";
	}

}

int main() {
	const int N = 5;

	//double mat_[N][N] = {{ 1,2,3 }, 
	//					 { 4,4,6 },
	//					 { 1,2,9 }};


	double mat_[N][N] = { { 17,24, 1, 8,15 },
						  { 23, 5, 7,14,16 },
						  {  4, 6,13,20,22 },
						  { 10,12,19,21, 3 },
						  { 11,18,25, 2, 9 } };
	double** mat;
	check(cudaMallocManaged(&mat, sizeof(double*) * N));
	for (int i = 0; i < N; i++) {
		check(cudaMallocManaged(&(mat[i]), sizeof(double) * N))
		// check(cudaMemcpy(mat[i], mat_[i], 3, cudaMemcpyHostToDevice));
	}
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			mat[i][j] = mat_[i][j];
		}
	}

	// auto Mat = make_pair(N, mat);
	// cout << Mat << endl;

	// test_maxColumn(mat, 2, N);
	// test_L_column(mat, 0, N);
	// test_reduce(mat, 0, N);
	test_LU_decomp(mat, N);

	check(cudaDeviceReset());
}
