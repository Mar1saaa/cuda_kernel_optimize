# SGEMM kernels

General matrix multiply (GEMM) computes \(C \leftarrow \alpha AB + \beta C\) (here we focus on the core multiply–accumulate pattern). Below is a quick **arithmetic intensity** sketch for square-ish sizes.

## Arithmetic intensity

Let \(A\) be \(M \times K\), \(B\) be \(K \times N\), and \(C\) be \(M \times N\). Each output element needs \(K\) multiplies and about \(K\) adds, so total work is about **\(2MNK\)** flops.

Memory traffic (ignoring caches): read \(MK + KN\) elements and write \(MN\) elements. With `float` (4 bytes), bytes moved scale as **\(4(MK + KN + MN)\)**.

If \(M = N = K = N_0\) for scaling analysis, intensity is about:

\[
\frac{2N_0^3}{4 \cdot 3N_0^2} = \frac{N_0}{6}\ \text{flops/byte}
\]

So **larger matrices favor compute**; GEMM is usually treated as **compute-bound** at sufficient size (still bandwidth-limited when tiny).

## Build and run

Compile with `nvcc` and link cuBLAS for the reference path:

```bash
nvcc sgemm_original.cu -o sgemm_original -lcublas
```

Arguments are **M**, **K**, **N** (in that order). The kernels use **`float4` loads**; **M, K, and N must be multiples of 4**.

```bash
./sgemm_original 4096 4096 4096
```

The sources use named template/block constants for readability.

## Original (`sgemm_original.cu`)

A compact but representative SGEMM with common ideas:

- **`#pragma unroll`**: when trip counts are small and known, unrolling reduces loop overhead; the compiler may still adjust unrolling.
- **Shared memory**: much lower latency than global memory; staging tiles in shared memory is central to throughput.
- **Blocked GEMM**: one thread block computes one \(C\) tile; many blocks run in parallel.
- **Prefetch / double buffering**: while computing from one buffer, prefetch the next tile into the other buffer (with `__syncthreads()` between phases), trading extra shared memory and registers for latency hiding.

**Caveats:** spilling registers to local/global memory hurts badly. If static shared memory is too large, compilation can fail; if you exceed limits dynamically, behavior can be undefined—size tiles to your GPU.

## V1 (`sgemm_v1.cu`)

TODO

## Hopper-oriented variant

TODO

## Architecture note (global ↔ shared)

Before **Ampere**, there was no **LDGSTS**-style path from global memory directly into shared memory in the programming model as we use today; data typically went **global → register → shared** (the compiler may insert stages even if you do not spell them out).

**Ampere** added asynchronous global-to-shared copies. **Hopper** extends the story with more flexible async copy directions, improving how kernels can feed shared memory.

## References

1. 深入浅出 GPU 优化系列：GEMM 优化（一） — 知乎  
   https://zhuanlan.zhihu.com/p/435908830  
2. maxas wiki — SGEMM  
   https://github.com/NervanaSystems/maxas/wiki/SGEMM  
3. CUDA 矩阵乘法终极优化指南 — 知乎  
   https://zhuanlan.zhihu.com/p/410278370  
