# Introduction of the sgemm kernel
How to use: Execute directly with parameters. Parameters 0, 1, and 2 represent M, K, and N, respectively. Because float4 vector reads are used, all three must be multiples of 4.
If you want to compile it yourself, use this: nvcc xxx.cu -o xxx -lcublas
For ease of understanding, the kernel contains many constants for identification.

## Original Version
The Original is a simplified and optimized version, including implementations of common gemm optimization techniques, such as: loop unrolling, using shared memory, matrix block computation, and data prefetching.

### #pragma unroll
When the number of loop iterations is small and can be roughly determined, #pragma unroll can be used to unroll a for loop into repetitive code, reducing loop control overhead. In fact, the compiler will also evaluate whether to unroll the loop based on the actual situation.

### Shared Memory
Memory units closer to the computation unit, with access latency of 10-30 clock cycles, more than 10 times faster than accessing global memory. Prefetching data into shared memory can achieve a significant speedup.

It's important to note that the memory architecture of Nvidia GPUs differs significantly from that of x86 CPUs. Those interested can consult relevant documentation.

### Matrix Block Computation
Using only one block for the entire matrix multiplication would result in a significant waste of computational resources. Therefore, we divide the multiplying matrices A and B into blocks, with each block responsible for calculating one sub-block. Parallel computation is then performed between multiple blocks.

### Prefetch Data
GPU registers are becoming increasingly large, and memory storage (SM) is becoming more numerous. We can fully utilize the hardware. When allocating memory, we allocate two blocks, A and B. While performing calculations using A, we prefetch data from B. This way, when A finishes its calculation, B has already been read (with a synchronization step in between) and is ready for computation, achieving a space-for-time tradeoff.

It's important to note that if registers are exhausted, global memory will be requested, severely degrading performance. Furthermore, if shared memory space is insufficient, specifying the size of shared memory (e.g., `__shared__ float tile[2048];`) will cause a compiler error; otherwise, it will crash upon overflow.

## V1 Version
TODO

## Hopper Implementation Version
TODO

## Notice
Prior to the Ampere architecture, the instruction set lacked the LDGSTS instruction. There was no direct transfer path between global memory and shared memory; transfer could only be completed through registration. Therefore, when using older architectures prior to Ampere, even without explicit explicit write-in, the compiler would add this step.

The Ampere architecture only supported asynchronous copying from global memory to shared memory, while Hopper supports the reverse copy operation, improving kernel read/write performance across different memory structures.


# Reference
1. https://github.com/NervanaSystems/maxas/wiki/SGEMM
2. https://github.com/Tongkaio/CUDA_Kernel_Samples
3. CUDA 矩阵乘法终极优化指南 - MegEngine Bot的文章 - 知乎 https://zhuanlan.zhihu.com/p/410278370
4. 深入浅出GPU优化系列：GEMM优化（一） - 有了琦琦的棍子的文章 - 知乎 https://zhuanlan.zhihu.com/p/435908830