#include <cstdint>
#include "common_utils.h"

using u32 = uint32_t;

static constexpr u32 TILE_SIZE = 16;
static constexpr u32 WARMUP_ROUNDS = 32;

// 只负责在正式 profiling 前让 GPU 进入工作状态；计算结果不会参与正确性校验。
__global__ void warmup(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    u32 row = blockIdx.y * blockDim.y + threadIdx.y;
    u32 col = blockIdx.x * blockDim.x + threadIdx.x;

    float elemSum = 0;
#pragma unroll 1
    for (u32 round = 0; round < WARMUP_ROUNDS; ++round) {
        for (u32 kk = 0; kk < K; kk += TILE_SIZE) {
            u32 aCol = kk + threadIdx.x;
            tileA[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0;

            u32 bRow = kk + threadIdx.y;
            tileB[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0;
            __syncthreads();

            for (u32 t = 0; t < TILE_SIZE; ++t) {
                elemSum += tileA[threadIdx.y][t] * tileB[t][threadIdx.x];
            }
            __syncthreads();
        }
    }

    if (row < M && col < N) {
        C[row * N + col] = elemSum;
    }
}

void warmup_do(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 blockDim{TILE_SIZE, TILE_SIZE};
    dim3 gridDim{CeilDiv<u32>(N, TILE_SIZE), CeilDiv<u32>(M, TILE_SIZE)};

    warmup<<<gridDim, blockDim, 0, nullptr>>>(A, B, C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}
