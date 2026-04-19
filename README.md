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

## Project Structure

```
optimized_kernel.cu   # Optimized kernel (the only file to modify)
main.cu               # Benchmark driver and autograder
naive_kernel.cu/h     # Naive reference implementation
constants.h           # Fixed dimensions (M=N=K=4, NUM_A=NUM_B=512)
utilities.cu/h        # Helper functions
run.sh                # Build and run script
```

## Build & Run

Requires an NVIDIA GPU with `sm_75` support (e.g., T4) and CUDA toolkit.

```bash
chmod u+x run.sh
./run.sh
```

Or manually:

```bash
nvcc -O3 -arch=sm_75 \
    main.cu utilities.cu naive_kernel.cu optimized_kernel.cu \
    -o batched_gemm && ./batched_gemm
```

## Scoring

- **5 pts** — Correctness (output matches reference within tolerance)
- **10 pts** — Performance (full marks at 100x speedup over naive kernel)
