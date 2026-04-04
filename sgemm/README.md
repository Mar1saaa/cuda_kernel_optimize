### The introduce of sgemm kernel

## Original Version

The original version is a slightly optimized version, which includes: introducing a one-dimensional thread tile; using data prefetching; using loop unrolling at appropriate locations; and using FLOAT4 vectorized access. Note that this cannot be used if pointers are not aligned or the data size is not a power of 2.