#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "../utils/utils.hpp"

#define THREAD_PER_BLOCK 256

// Simplest version of elementwise add
__global__ void vec4_ADD_kernel(
    float* a, float* b, float* c
    ) {
    int idx = (threadIdx.x + blockIdx.x * blockDim.x)*4;
    //c[idx] = a[idx] + b[idx];
    float4 reg_a = FETCH_FLOAT4(a[idx]);
    float4 reg_b = FETCH_FLOAT4(b[idx]);
    float4 reg_c;
    reg_c.x = reg_a.x + reg_b.x;
    reg_c.y = reg_a.y + reg_b.y;
    reg_c.z = reg_a.z + reg_b.z;
    reg_c.w = reg_a.w + reg_b.w;
    FETCH_FLOAT4(c[idx]) = reg_c;
}

// Swish(x) = x * sigmoid(x)
__device__ __forceinline__ float sigmoid_f(float x) {
    return 1.0f / (1.0f + expf(-x));
}

template<
    const int BLOCK_SIZE_M,  // height of block of output that each thread block calculate
    const int BLOCK_SIZE_N,  // width of block of output that each thread block calculate
    const int BLOCK_SIZE_K,  // width of block of input that each thread block load into shared memory
    const int THREAD_TILE_Y, //  height of each thread calculate
    const int THREAD_TILE_X //  width of each thread calculate
    > 
__global__ void SwiGeLU_kernel(
    const float* __restrict__ input,      // [M, K]
    const float* __restrict__ weight_1,   // [K, N]
    const float* __restrict__ weight_2,   // [K, N]
    const float* __restrict__ bias_1,     // [N]
    const float* __restrict__ bias_2,     // [N]
    float* __restrict__ output,           // [M, N]
    const int M, const int K, const int N
    ) {
    static_assert(BLOCK_SIZE_M % THREAD_TILE_Y == 0, "BLOCK_SIZE_M must be divisible by THREAD_TILE_Y");
    static_assert(BLOCK_SIZE_N % THREAD_TILE_X == 0, "BLOCK_SIZE_N must be divisible by THREAD_TILE_X");
    static_assert(BLOCK_SIZE_K % 4 == 0,             "BLOCK_SIZE_K must be multiple of 4");
    static_assert(BLOCK_SIZE_N % 4 == 0,             "BLOCK_SIZE_N must be multiple of 4");
    static_assert(THREAD_TILE_Y % 4 == 0,            "THREAD_TILE_Y must be multiple of 4");
    static_assert(THREAD_TILE_X % 4 == 0,            "THREAD_TILE_X must be multiple of 4");

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int THREAD_X_PER_BLOCK   = BLOCK_SIZE_N / THREAD_TILE_X;
    const int THREAD_Y_PER_BLOCK   = BLOCK_SIZE_M / THREAD_TILE_Y;
    const int THREAD_NUM_PER_BLOCK = THREAD_X_PER_BLOCK * THREAD_Y_PER_BLOCK;

    const int tid = ty * THREAD_X_PER_BLOCK + tx;

    // shared memory for bias
    __shared__ float sb1[BLOCK_SIZE_N];
    __shared__ float sb2[BLOCK_SIZE_N];
    // load bias from global memory to shared memory
    if (tid < BLOCK_SIZE_N / 4) {
        FETCH_FLOAT4(sb1[tid * 4]) = LOAD_FLOAT4(bias_1[blockIdx.x * BLOCK_SIZE_N + tid * 4]);
        FETCH_FLOAT4(sb2[tid * 4]) = LOAD_FLOAT4(bias_2[blockIdx.x * BLOCK_SIZE_N + tid * 4]);
    }

    // allocate shared memory
    __shared__ float sx[BLOCK_SIZE_K][BLOCK_SIZE_M + 4];
    __shared__ float sw1[BLOCK_SIZE_K][BLOCK_SIZE_N + 4];
    __shared__ float sw2[BLOCK_SIZE_K][BLOCK_SIZE_N + 4];

    // set the pointer to the starting position of the current block
    input = &input[(BLOCK_SIZE_M * blockIdx.y)* K];
    weight_1 = &weight_1[BLOCK_SIZE_N * blockIdx.x];
    weight_2 = &weight_2[BLOCK_SIZE_N * blockIdx.x];

    // registers for sub accumulaton
    float acc_x_1[THREAD_TILE_Y][THREAD_TILE_X] = {};
    float acc_x_2[THREAD_TILE_Y][THREAD_TILE_X] = {};

    // registers for sub input and weights
    float frag_x[THREAD_TILE_Y];
    float frag_w1[THREAD_TILE_X];
    float frag_w2[THREAD_TILE_X];

    // the number of threads that need to load one row of a block
    const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
    const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4;

    // row number and col number that needs to be loaded by this thread
    const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;

    const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * 4;
    const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * 4;

    // row stride that thread uses to load multiple rows of a tile
    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;

    int tile_idx = 0;

    // registers for load global memory
    float4 ldg_x_reg;

    do{
        // load input and weight from global memory to shared memory (through register)
        #pragma unroll
        for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
            ldg_x_reg = LOAD_FLOAT4(input[OFFSET(
                A_TILE_ROW_START + i, // row
                A_TILE_COL + tile_idx, // col
                K )]);

            sx[A_TILE_COL  ][A_TILE_ROW_START + i]=ldg_x_reg.x;
            sx[A_TILE_COL+1][A_TILE_ROW_START + i]=ldg_x_reg.y;
            sx[A_TILE_COL+2][A_TILE_ROW_START + i]=ldg_x_reg.z;
            sx[A_TILE_COL+3][A_TILE_ROW_START + i]=ldg_x_reg.w;
        }
        #pragma unroll
        for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
            FETCH_FLOAT4(sw1[B_TILE_ROW_START + i][B_TILE_COL]) = LOAD_FLOAT4(weight_1[OFFSET(
                B_TILE_ROW_START + i + tile_idx, // row
                B_TILE_COL, // col
                N )]);
            FETCH_FLOAT4(sw2[B_TILE_ROW_START + i][B_TILE_COL]) = LOAD_FLOAT4(weight_2[OFFSET(
                B_TILE_ROW_START + i + tile_idx, // row
                B_TILE_COL, // col
                N )]);
        }

        __syncthreads();

        #pragma unroll
        for(int k = 0; k < BLOCK_SIZE_K; ++k){
            // load next tile from shared mem to register
            #pragma unroll
            for(int thread_y = 0; thread_y < THREAD_TILE_Y; thread_y += 4) {
                FETCH_FLOAT4(frag_x[thread_y]) = FETCH_FLOAT4(sx[k][THREAD_TILE_Y * ty + thread_y]);
            }
            #pragma unroll
            for(int thread_x = 0; thread_x < THREAD_TILE_X; thread_x += 4) {
                FETCH_FLOAT4(frag_w1[thread_x]) = FETCH_FLOAT4(sw1[k][THREAD_TILE_X * tx + thread_x]);
                FETCH_FLOAT4(frag_w2[thread_x]) = FETCH_FLOAT4(sw2[k][THREAD_TILE_X * tx + thread_x]);
            }

            // compute input x weight1 and weight2
            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_TILE_Y; ++thread_y) {
                #pragma unroll
                for (int thread_x = 0; thread_x < THREAD_TILE_X; ++thread_x) {
                    acc_x_1[thread_y][thread_x] += frag_x[thread_y] * frag_w1[thread_x];
                    acc_x_2[thread_y][thread_x] += frag_x[thread_y] * frag_w2[thread_x];
                }
            }
    
        }

        __syncthreads();
        tile_idx += BLOCK_SIZE_K;
    
    }while(tile_idx < K);

    const int row_base = blockIdx.y * BLOCK_SIZE_M + ty * THREAD_TILE_Y;
    const int col_base = blockIdx.x * BLOCK_SIZE_N + tx * THREAD_TILE_X;

    #pragma unroll
    for (int y = 0; y < THREAD_TILE_Y; ++y) {
        const int row = row_base + y;
        #pragma unroll
        for (int x = 0; x < THREAD_TILE_X; x += 4) {
            const int col_in_block = tx * THREAD_TILE_X + x;   // [0, BLOCK_SIZE_N)
            const int col          = col_base + x;
            const int idx          = row * N + col;
    
            float4 b1 = FETCH_FLOAT4(sb1[col_in_block]);
            float4 b2 = FETCH_FLOAT4(sb2[col_in_block]);
    
            float a0 = acc_x_1[y][x  ] + b1.x, c0 = acc_x_2[y][x  ] + b2.x;
            float a1 = acc_x_1[y][x+1] + b1.y, c1 = acc_x_2[y][x+1] + b2.y;
            float a2 = acc_x_1[y][x+2] + b1.z, c2 = acc_x_2[y][x+2] + b2.z;
            float a3 = acc_x_1[y][x+3] + b1.w, c3 = acc_x_2[y][x+3] + b2.w;
    
            float4 out;
            out.x = a0 * sigmoid_f(a0) * c0;
            out.y = a1 * sigmoid_f(a1) * c1;
            out.z = a2 * sigmoid_f(a2) * c2;
            out.w = a3 * sigmoid_f(a3) * c3;
    
            FETCH_FLOAT4(output[idx]) = out;
        }
    }
}

bool check(float *out,float *res,int n){
    for(int i=0;i<n;i++){
        if(out[i]!=res[i])
            return false;
    }
    return true;
}

// ====================== CPU reference for SwiGeLU_kernel ======================
// out[i, j] = init_out[i, j] + swish(x @ W1 + b1)[i, j] * (x @ W2 + b2)[i, j]
//           swish(t) = t * sigmoid(t)
static void swiglu_cpu_ref(
    const float* x,        // [M, K]
    const float* w1,       // [K, N]
    const float* w2,       // [K, N]
    const float* b1,       // [N]
    const float* b2,       // [N]
    const float* init_out, // [M, N]
    float*       out,      // [M, N]
    int M, int K, int N)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double a = 0.0, c = 0.0;
            for (int k = 0; k < K; ++k) {
                a += static_cast<double>(x[i * K + k]) * static_cast<double>(w1[k * N + j]);
                c += static_cast<double>(x[i * K + k]) * static_cast<double>(w2[k * N + j]);
            }
            a += static_cast<double>(b1[j]);
            c += static_cast<double>(b2[j]);
            const double sig = 1.0 / (1.0 + std::exp(-a));
            const double swish = a * sig;
            out[i * N + j] = static_cast<float>(static_cast<double>(init_out[i * N + j]) + swish * c);
        }
    }
}

static bool compare_result(const float* gpu, const float* ref, int n,
                           float rtol = 1e-3f, float atol = 1e-4f,
                           int max_print = 10) {
    bool ok = true;
    int printed = 0;
    double max_abs_err = 0.0, max_rel_err = 0.0;
    for (int i = 0; i < n; ++i) {
        const float g = gpu[i], r = ref[i];
        const float abs_err = std::fabs(g - r);
        const float rel_err = abs_err / std::max(std::fabs(r), 1e-20f);
        if (abs_err > atol && rel_err > rtol) {
            if (printed < max_print) {
                std::printf("  mismatch at %d: gpu=%.6f ref=%.6f abs=%.3e rel=%.3e\n",
                            i, g, r, abs_err, rel_err);
                ++printed;
            }
            ok = false;
        }
        max_abs_err = std::max<double>(max_abs_err, abs_err);
        max_rel_err = std::max<double>(max_rel_err, rel_err);
    }
    std::printf("  max_abs_err = %.3e, max_rel_err = %.3e\n", max_abs_err, max_rel_err);
    return ok;
}

int main() {
    constexpr int M = 2048;
    constexpr int K = 512;
    constexpr int N = 2048;

    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 16;
    constexpr int TY = 8;
    constexpr int TX = 8;

    static_assert(M % BM == 0, "M must be divisible by BLOCK_SIZE_M");
    static_assert(N % BN == 0, "N must be divisible by BLOCK_SIZE_N");
    static_assert(K % BK == 0, "K must be divisible by BLOCK_SIZE_K");

    const size_t sz_x  = static_cast<size_t>(M) * K;
    const size_t sz_w  = static_cast<size_t>(K) * N;
    const size_t sz_b  = static_cast<size_t>(N);
    const size_t sz_o  = static_cast<size_t>(M) * N;

    std::vector<float> h_x (sz_x);
    std::vector<float> h_w1(sz_w);
    std::vector<float> h_w2(sz_w);
    std::vector<float> h_b1(sz_b);
    std::vector<float> h_b2(sz_b);
    std::vector<float> h_out_init(sz_o);
    std::vector<float> h_out_gpu (sz_o);
    std::vector<float> h_out_ref (sz_o);

    // use small range random values to avoid exp overflow, and also convenient to compare
    std::mt19937 rng(0xC0FFEE);
    std::uniform_real_distribution<float> dist(-0.2f, 0.2f);
    for (auto& v : h_x ) v = dist(rng);
    for (auto& v : h_w1) v = dist(rng);
    for (auto& v : h_w2) v = dist(rng);
    for (auto& v : h_b1) v = dist(rng);
    for (auto& v : h_b2) v = dist(rng);

    float *d_x, *d_w1, *d_w2, *d_b1, *d_b2, *d_out;
    CUDA_CHECK(cudaMalloc(&d_x,   sz_x * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w1,  sz_w * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w2,  sz_w * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b1,  sz_b * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b2,  sz_b * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, sz_o * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_x,  h_x.data(),        sz_x * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w1, h_w1.data(),       sz_w * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w2, h_w2.data(),       sz_w * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1, h_b1.data(),       sz_b * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2, h_b2.data(),       sz_b * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out, h_out_init.data(), sz_o * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(BN / TX, BM / TY);
    dim3 grid (N / BN,  M / BM);

    SwiGeLU_kernel<BM, BN, BK, TY, TX><<<grid, block>>>(
        d_x, d_w1, d_w2, d_b1, d_b2, d_out, M, K, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out_gpu.data(), d_out, sz_o * sizeof(float), cudaMemcpyDeviceToHost));

    // CPU reference
    swiglu_cpu_ref(h_x.data(), h_w1.data(), h_w2.data(),
                   h_b1.data(), h_b2.data(),
                   h_out_init.data(), h_out_ref.data(),
                   M, K, N);

    std::printf("SwiGeLU correctness test: M=%d K=%d N=%d, block=(%d,%d,%d) thread_tile=(%d,%d)\n",
                M, K, N, BM, BN, BK, TY, TX);
    bool ok = compare_result(h_out_gpu.data(), h_out_ref.data(),
                            static_cast<int>(sz_o),
                            1e-3f, 1e-4f);
    std::printf("Result: %s\n", ok ? "PASS" : "FAIL");

    // simple timing (optional)
    {
        // 重置 output 初值，计时只反映 kernel 本身
        CUDA_CHECK(cudaMemcpy(d_out, h_out_init.data(), sz_o * sizeof(float), cudaMemcpyHostToDevice));
        cudaEvent_t s, e;
        CUDA_CHECK(cudaEventCreate(&s));
        CUDA_CHECK(cudaEventCreate(&e));
        const int iters = 30;
        CUDA_CHECK(cudaEventRecord(s));
        for (int i = 0; i < iters; ++i) {
            // 这里每次都累加到同一块 output 会越积越大，只为看性能，不看数值
            SwiGeLU_kernel<BM, BN, BK, TY, TX><<<grid, block>>>(
                d_x, d_w1, d_w2, d_b1, d_b2, d_out, M, K, N);
        }
        CUDA_CHECK(cudaEventRecord(e));
        CUDA_CHECK(cudaEventSynchronize(e));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
        const double avg_ms = ms / iters;
        // 两个 GEMM + SwiGLU elementwise 的 FLOPs
        const double flops = 2.0 * 2.0 * M * N * K + 5.0 * M * N;
        std::printf("avg kernel time: %.3f ms,  %.2f GFLOP/s\n",
                    avg_ms, flops / (avg_ms * 1e-3) / 1e9);
        cudaEventDestroy(s);
        cudaEventDestroy(e);
    }

    cudaFree(d_x);
    cudaFree(d_w1);
    cudaFree(d_w2);
    cudaFree(d_b1);
    cudaFree(d_b2);
    cudaFree(d_out);

    return ok ? 0 : 1;
}