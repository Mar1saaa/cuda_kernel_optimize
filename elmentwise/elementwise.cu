#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>

#define THREAD_PER_BLOCK 256

// transfer vector
#define FETCH_FLOAT2(pointer) (reinterpret_cast<float2*>(&(pointer))[0])
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

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
    const int BLOCK_M,  // height of block of output that each thread block calculate
    const int BLOCK_N,  // width of block of output that each thread block calculate
    const int BLOCK_K,  // width of block of input that each thread block load into shared memory
    const int THREAD_TILE_N //  length of each thread calculate
    > 
__global__ void SwiGeLU_kernel(
    const float* __restrict__ input,      // [M, K]
    const float* __restrict__ weight_1,   // [K, N]
    const float* __restrict__ weight_2,   // [K, N]
    const float* __restrict__ bias_1,     // [M, N]
    const float* __restrict__ bias_2,     // [M, N]
    float* __restrict__ output,           // [M, N]
    const int M, const int K, const int N
    ) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int THREADS_PER_BLOCK = blockDim.x * blockDim.y;

    const int row = blockIdx.y * BLOCK_M + ty;
    const int col_base = blockIdx.x * BLOCK_N + tx * THREAD_TILE_N;

    __shared__ float tile_a[BLOCK_M][BLOCK_K];
    __shared__ float tile_b1[BLOCK_K][BLOCK_N];
    __shared__ float tile_b2[BLOCK_K][BLOCK_N];

    float acc_1[THREAD_TILE_N] = {0.0f};
    float acc_2[THREAD_TILE_N] = {0.0f};

    for (int tile_k = 0; tile_k < K; tile_k += BLOCK_K) {
        for (int load_idx = tid; load_idx < BLOCK_M * BLOCK_K; load_idx += THREADS_PER_BLOCK) {
            const int r = load_idx / BLOCK_K;
            const int c = load_idx % BLOCK_K;
            const int g_row = blockIdx.y * BLOCK_M + r;
            const int g_col = tile_k + c;
            // reverse
            tile_a[r][c] = (g_row < M && g_col < K) ? input[g_row * K + g_col] : 0.0f;
        }

        for (int load_idx = tid; load_idx < BLOCK_K * BLOCK_N; load_idx += THREADS_PER_BLOCK) {
            const int r = load_idx / BLOCK_N;
            const int c = load_idx % BLOCK_N;
            const int g_row = tile_k + r;
            const int g_col = blockIdx.x * BLOCK_N + c;
            tile_b1[r][c] = (g_row < K && g_col < N) ? weight_1[g_row * N + g_col] : 0.0f;
            tile_b2[r][c] = (g_row < K && g_col < N) ? weight_2[g_row * N + g_col] : 0.0f;
        }
        __syncthreads();

        if (row < M) {
            #pragma unroll
            for (int k_inner = 0; k_inner < BLOCK_K; ++k_inner) {
                const float a_frag = tile_a[ty][k_inner];
                #pragma unroll
                for (int tn = 0; tn < THREAD_TILE_N; ++tn) {
                    const int col = tx * THREAD_TILE_N + tn;
                    acc_1[tn] += a_frag * tile_b1[k_inner][col];
                    acc_2[tn] += a_frag * tile_b2[k_inner][col];
                }
            }
        }
        __syncthreads();
    }

    if (row < M) {
        // Prefer float4 vectorized load/store on contiguous columns.
        #pragma unroll
        for (int tn = 0; tn < THREAD_TILE_N; tn += 4) {
            const int col = col_base + tn;
            if (col + 1 < N && (col & 1) == 0) {
                const int linear_idx = row * N + col;

                const float4 bias1 = FETCH_FLOAT4(bias_1 + linear_idx);
                const float4 bias2 = FETCH_FLOAT4(bias_2 + linear_idx);

                float4 out4;
                const float a0 = acc_1[tn] + bias1.x;
                const float b0 = acc_2[tn] + bias2.x;
                out2.x = a0 * sigmoid_f(a0) * b0;

                const float a1 = acc_1[tn + 1] + bias1.y;
                const float b1 = acc_2[tn + 1] + bias2.y;
                out2.y = a1 * sigmoid_f(a1) * b1;

                const float a2 = acc_1[tn + 2] + bias1.z;
                const float b2 = acc_2[tn + 2] + bias2.z;
                out2.z = a2 * sigmoid_f(a2) * b2;
                
                const float a3 = acc_1[tn + 3] + bias1.w;
                const float b3 = acc_2[tn + 3] + bias2.w;
                out2.w = a3 * sigmoid_f(a3) * b3;

                FETCH_FLOAT4(output + linear_idx)x = out4;
            } else if (col < N) {
                const int linear_idx = row * N + col;
                const float a = acc_1[tn] + bias_1[linear_idx];
                const float b = acc_2[tn] + bias_2[linear_idx];
                output[linear_idx] = a * sigmoid_f(a) * b;
                if (tn + 1 < THREAD_TILE_N && (col + 1) < N) {
                    const int linear_idx_1 = linear_idx + 1;
                    const float a1 = acc_1[tn + 1] + bias_1[linear_idx_1];
                    const float b1 = acc_2[tn + 1] + bias_2[linear_idx_1];
                    output[linear_idx_1] = a1 * sigmoid_f(a1) * b1;
                }
            }
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

int main(){
    const int N=32*1024*1024;
    float *a=(float *)malloc(N*sizeof(float));
    float *b=(float *)malloc(N*sizeof(float));
    float *out=(float *)malloc(N*sizeof(float));
    float *d_a;
    float *d_b;
    float *d_out;
    cudaMalloc((void **)&d_a,N*sizeof(float));
    cudaMalloc((void **)&d_b,N*sizeof(float));
    cudaMalloc((void **)&d_out,N*sizeof(float));
    float *res=(float *)malloc(N*sizeof(float));

    for(int i=0;i<N;i++){
        a[i]=1;
        b[i]=i;
        res[i]=a[i]+b[i];
    }

    cudaMemcpy(d_a,a,N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,b,N*sizeof(float),cudaMemcpyHostToDevice);

    dim3 Grid( N/THREAD_PER_BLOCK/4, 1);
    dim3 Block( THREAD_PER_BLOCK, 1);

    int iter = 2000;
    for(int i=0; i<iter; i++){
        vec4_ADD_kernel<<<Grid,Block>>>(d_a, d_b, d_out);
    }

    cudaMemcpy(out,d_out,N*sizeof(float),cudaMemcpyDeviceToHost);

    if(check(out,res,N))printf("the ans is right\n");
    else{
        printf("the ans is wrong\n");
        for(int i=0;i<N;i++){
            printf("%lf ",out[i]);
        }
        printf("\n");
    }

    cudaFree(d_a);
    cudaFree(d_out);
}