#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h> 

#include "../utils/utils.hpp"

// Warp 内规约求和：每步用 __shfl_down_sync 把高 lane 的数据加到当前 lane，
// 多步之后在 lane 0 上得到整 warp 的和（其余 lane 上的值无保证）。
template <unsigned int WarpSize>
__device__ __forceinline__ float warpReduceSum(float sum) {
    if (WarpSize >= 32) sum += __shfl_down_sync(0xffffffff, sum, 16);
    if (WarpSize >= 16) sum += __shfl_down_sync(0xffffffff, sum, 8);
    if (WarpSize >= 8) sum += __shfl_down_sync(0xffffffff, sum, 4);
    if (WarpSize >= 4) sum += __shfl_down_sync(0xffffffff, sum, 2);
    if (WarpSize >= 2) sum += __shfl_down_sync(0xffffffff, sum, 1);
    return sum;
}

// SGEMV: y = A * x，A 为行主序 M×N，x 为 N 维，y 为 M 维。
// 映射：grid.x 为行块，block 内 threadIdx.y 负责不同行，threadIdx.x 为 warp 内 lane（32）。
// 每行沿 N 用 float4 向量化读取 A 与 x；每个 lane 每轮各取一个 float4（4 个 float），
// 点积累加到寄存器后，warp 内 shuffle 规约，由 lane 0 写 y[row]。
// 注意：K >=128 , kIteration 与 launch 的 dimBlock(32,4) 需与数据规模一致。
__global__ void sgemv_kernel(
    float *__restrict__ A,
    float *__restrict__ x,
    float *__restrict__ y,
    const int M,
    const int N) {
    constexpr int kWarpSize = 32;
    
    const int block_row_base = blockIdx.x * blockDim.y; // grid 只有一维
    const int row = block_row_base + threadIdx.y;
    const int lane = threadIdx.x % kWarpSize;

    if (row >= M) {
        return;
    }

    // 指向当前行在 A 中的首地址，后续按 float4 步进遍历列。
    A += static_cast<size_t>(row) * N;

    // 每轮 warp 覆盖 kWarpSize 个 float4，迭代次数与 N 对齐方式见文件头注释。
    int num_vec_iters = (N / kWarpSize) / 4;
    // 至少要有一轮迭代
    if (num_vec_iters == 0) {
        num_vec_iters = 1;
    }

    float dot_partial = 0.f;

#pragma unroll
    for (int iter = 0; iter < num_vec_iters; ++iter) {
        // 当前 lane 在本轮要读的 float4 列块下标（A 与 x 同下标对齐）。
        const int vec_col = iter * kWarpSize + lane;

        const float4 a4 = reinterpret_cast<const float4 *>(A)[vec_col];
        const float4 x4 = reinterpret_cast<const float4 *>(x)[vec_col];

        dot_partial += a4.x * x4.x;
        dot_partial += a4.y * x4.y;
        dot_partial += a4.z * x4.z;
        dot_partial += a4.w * x4.w;
    }

    // 同一 warp 内各 lane 的局部和再相加，结果在 lane 0。
    dot_partial = warpReduceSum<static_cast<unsigned int>(kWarpSize)>(dot_partial);
    if (lane == 0) {
        y[row] = dot_partial;
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("usage: ./main [M] [N]\n");
        exit(0);
    }
    size_t M = atoi(argv[1]);
    size_t N = atoi(argv[2]);

    std::vector<float> h_A(M * N);
    std::vector<float> h_x(N);
    std::vector<float> h_y(M);
    std::vector<float> h_y1(M);

    float* d_A;
    float* d_x;
    float* d_y;

    size_t bytes_A = sizeof(float) * M * N;
    size_t bytes_x = sizeof(float) * N;
    size_t bytes_y = sizeof(float) * M;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_x, bytes_x));
    CUDA_CHECK(cudaMalloc(&d_y, bytes_y));

    // 生成A和x的数据
    std::mt19937 rng(0xC0FFEE);
    std::uniform_real_distribution<float> dist(-0.2f, 0.2f);
    for (auto& v : h_A) v = dist(rng);
    for (auto& v : h_x) v = dist(rng);

    std::fill(h_y.begin(), h_y.end(), 0.0f);
    std::fill(h_y1.begin(), h_y1.end(), 0.0f);

    int nIter = 1000;
    CUDA_CHECK(cudaMemcpy( d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy( d_x, h_x.data(), bytes_x, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy( d_y, h_y.data(), bytes_y, cudaMemcpyHostToDevice));

    dim3 dimGrid(M/4);
    dim3 dimBlock(32,4);

    for (int run = 0 ; run < nIter; run ++ ) {
        sgemv_kernel<<< dimGrid, dimBlock >>>(d_A, d_x, d_y, M, N);
    }

    CUDA_CHECK(cudaMemcpy( h_y.data(), d_y, bytes_y, cudaMemcpyDeviceToHost));

    // cublas
    cublasHandle_t blas_handle;  
    cublasCreate(&blas_handle);
    float alpha = 1.0;
    float beta = 0;
    CUDA_CHECK(cudaMemcpy( d_y, h_y1.data(), bytes_y, cudaMemcpyHostToDevice));
    for (int run = 0 ; run < nIter; run ++ ) {
        cublasSgemv (blas_handle, CUBLAS_OP_T, 
            N, M, &alpha, 
            d_A, N, d_x, 1, &beta, d_y, 1
        );
    }

    CUDA_CHECK(cudaMemcpy( h_y1.data(), d_y, bytes_y, cudaMemcpyDeviceToHost));
    cublasDestroy(blas_handle); 
    
    double eps = 1.e-6;  // machine zero
    bool correct = true;
    for (int i = 0; i < M; i++) {
        double abs_err = fabs(h_y[i] - h_y1[i]);
        double dot_length = M;
        double abs_val = fabs(h_y[i]);
        double rel_err = abs_err / abs_val / dot_length;
        if (rel_err > eps) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                    i, h_y[i], h_y1[i], eps);
            correct = false;
            break;
        }
    }

    printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");
    
    // simple timing (optional)
    {
        // 重置 output 初值，计时只反映 kernel 本身
        CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), bytes_y, cudaMemcpyHostToDevice));
        cudaEvent_t s, e;
        CUDA_CHECK(cudaEventCreate(&s));
        CUDA_CHECK(cudaEventCreate(&e));
        const int iters = 1000;
        CUDA_CHECK(cudaEventRecord(s));
        for (int i = 0; i < iters; ++i) {
            sgemv_kernel<<< dimGrid, dimBlock >>>(d_A, d_x, d_y, M, N);
        }
        CUDA_CHECK(cudaEventRecord(e));
        CUDA_CHECK(cudaEventSynchronize(e));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));

        // calculate FLOPs
        const double avg_ms = ms / iters;
        const double flops = 2.0 * M * N;
        std::printf("avg kernel time: %.3f ms,  %.2f GFLOP/s\n",
                    avg_ms, flops / (avg_ms * 1e-3) / 1e9);
        cudaEventDestroy(s);
        cudaEventDestroy(e);
    }

    // Free Memory
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
    
    return correct ? 0 : 1;
}