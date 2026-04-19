# Element-wise kernels

Element-wise operators apply a function to each element independently (e.g. vector add/mul, Sigmoid, ReLU). They are typical **memory-bound** kernels.

## Arithmetic intensity (example)

Take two \(N \times N\) inputs and one \(N \times N\) output, all `float` (4 bytes). Fused add: about \(N^2\) useful ops (one add per output element); memory traffic is roughly reads of \(2N^2\) floats plus write of \(N^2\) floats, i.e. about \(3N^2 \times 4\) bytes. The **compute-to-bytes ratio** is on the order of \(1 / 12\) flops per byte—very low intensity.

## Practical optimization

Pure element-wise kernels have limited room beyond good launch config (block/grid) and **vectorized loads/stores** (`float2` / `float4`), which often already yields a large fraction of peak memory bandwidth.

A common direction is **kernel fusion**: fewer round-trips through global memory by doing more work per load/store.

## SwiGLU-style fused example

This folder includes a fused-style kernel (not only a trivial add). The reference form:

\[
\mathrm{SwiGLU}(x) = \mathrm{Swish}(xW_1 + b_1) \odot (xW_2 + b_2)
\]

\[
\mathrm{Swish}(t) = t \cdot \sigma(t) = \frac{t}{1 + e^{-t}}
\]

Computation order: inner GEMMs and bias, then Swish on one branch, then element-wise multiply with the other branch. The Swish and Hadamard parts are element-wise; the \(xW\) parts are GEMMs. They are dependent stages; Swish itself is cheap on its own.

The GEMM portion here is a straightforward blocked implementation **without** the prefetch/double-buffer tricks used in the `sgemm/` examples—see that folder if you want to study prefetching.

## Shared memory note

If you pad the leading dimension of a 2D `__shared__` tile for bank-conflict reasons, keep the **row stride (in `float`s) a multiple of 4** wherever you use **`float4` / `FETCH_FLOAT4` / `LOAD_FLOAT4`**, or you will hit **misaligned address** faults.

## Build

From this directory (folder name in the repo is `elmentwise`):

```bash
nvcc -std=c++17 -O2 elementwise.cu -o elementwise_test
# optional: -O3, -arch=sm_XX for your GPU
./elementwise_test
```
