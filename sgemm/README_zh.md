# The introduction of sgemm kernel
矩阵乘法执行的是两个矩阵相乘并累加到第三个矩阵的操作，据此，我们首先分析一下GEMM的计算访存比
以M * K, K * N两个矩阵相乘举例, 计算结果为M * N的矩阵, 其每个元素计算需要进行K次乘法, K-1次加法, 一共要算M * N个元素, 总的计算量便是 M * N * (K + K - 1) ,约等于2MNK
计算中, 需要读矩阵 MK, KN, 需要写矩阵MN, 若每个元素以 4Bytes 的标准 float存储, 总访存便是 (M * K + K * N + M * N) * 4
假设三个矩阵形状一样, 都是M, N, K均 = N, 这时的计算访存比为 2 * N^3 / 4 * (3N^2) = N / 6, 矩阵规模越大，计算量就越大，可以体现GEMM是计算密集型算子

## 代码食用方式
用nvcc编译, 以./xxx的形式带上参数执行, 参考指令:
nvcc sgemm_original.cu -o sgemm_original -lcublas
参数0 1 2分别是M K N，因为使用了float4向量读，三者必须是4的倍数

## Original Version
Original是一个简略优化版本，包含了gemm常用优化思路的实现，包括: 循环展开, 使用shared memory, 矩阵分块计算, 数据预取

### #pragma unroll
在循环次数小、可大致判断时，可以用#pragma unroll把for循环展开为重复的代码，减少循环控制的开销，实际上，编译器也会根据实际情况评估循环是否要展开

### Shared Memory
离计算单元更近的存储单元，访存延迟在10-30个时钟周期，相比访问global memory快10倍以上，预取数据到shared memory能获得不小的加速幅度。
需注意的是，Nvidia GPU的存储器结构和x86 CPU的设计有不小区别，感兴趣的可以自行查阅相关资料

### Matrix block computation
如果只用一个block去做整个矩阵乘，会让大量的计算资源闲置浪费，因此，我们将相乘的矩阵A B分块，每个block负责计算一个子块，多个block间进行并行计算

### Prefetch Data
GPU的各种寄存器越来越大，SM也越来越多，我们可以充分利用硬件。申请内存时申请A B两块，在用A进行计算时，在B上预取数据。这样，A计算完时，B已读取完毕(中间要加一次同步)，可以计算了，实现以空间换时间。
需注意的是，如果register耗尽，则会去申请global memory，使性能严重下降；而shared memory空间不够时，如果指定了Shared Memory 的大小（例如 __shared__ float tile[2048];）编译器会报错，没有指定，则在溢出时会崩溃。

## V1 Version
TODO

## Hopper Implementation Version
TODO

## Notice
Ampere架构之前，指令集中没有LDGSTS指令，global memory和shared memory之间没有直接传输途径，只能经过register完成传输，因此使用Ampere前的老架构时，即使不明写，编译器也会添加这一步骤
Ampere架构只支持从 global memory 异步拷贝到 shared memory，而 Hopper 支持反向的拷贝操作，提升了 kernel 在不同存储结构上的读写性能。


## Reference
1. 深入浅出GPU优化系列：GEMM优化（一） - 有了琦琦的棍子的文章 - 知乎 https://zhuanlan.zhihu.com/p/435908830
2. https://github.com/NervanaSystems/maxas/wiki/SGEMM
3. CUDA 矩阵乘法终极优化指南 - MegEngine Bot的文章 - 知乎 https://zhuanlan.zhihu.com/p/410278370 