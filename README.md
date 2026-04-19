# PairGEMM

High-performance CUDA kernel for pair-wise batched 4x4 matrix multiplication on NVIDIA T4 GPUs.

## Problem

Given two lists of 4x4 matrices **A** (512 matrices) and **B** (512 matrices), compute:

```
C = sum over all i,j of (A_i * B_j)
```

This produces a single 4x4 output matrix from 262,144 matrix multiplications. The goal is to maximize throughput by minimizing global memory traffic and maximizing arithmetic intensity.

## Key Optimizations

- **Shared memory tiling** — load matrices once, reuse across computations
- **Vectorized loads** — `float4` for aligned, coalesced memory access
- **Block-level reductions** — efficient parallel summation across thread blocks
- **Loop unrolling** — compile-time unrolled inner products




