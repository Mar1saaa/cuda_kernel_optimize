#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// transfer vector (non-const lvalue: read / write vectorized word)
#define FETCH_FLOAT2(pointer) (reinterpret_cast<float2*>(&(pointer))[0])
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

// read-only vectorized load (const float / __restrict__ const float subscripts)
#define LOAD_FLOAT2(pointer) (reinterpret_cast<const float2*>(&(pointer))[0])
#define LOAD_FLOAT4(pointer) (reinterpret_cast<const float4*>(&(pointer))[0])

#define CUDA_CHECK(expr)                                                      \
    do {                                                                      \
        cudaError_t _e = (expr);                                              \
        if (_e != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",              \
                         #expr, __FILE__, __LINE__, cudaGetErrorString(_e));  \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)
