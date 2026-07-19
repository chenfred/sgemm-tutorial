#include <cstdint>
#include "common_utils.h"

using u32 = uint32_t;

static constexpr u32 TILE_SIZE = 32;
static constexpr u32 BLOCK_SIZE_X = 32;
static constexpr u32 BLOCK_SIZE_Y = 8;
static constexpr u32 OUTPUTS_PER_THREAD = 4;

// 标准 SGEMM 语义：C(M,N) = A(M,K) * B(K,N)，K 为收缩维度。
__global__ void sgemm_v1(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    u32 baseRow = blockIdx.y * TILE_SIZE;
    u32 baseCol = blockIdx.x * TILE_SIZE;

    float regs[OUTPUTS_PER_THREAD] = {};
    for (u32 kOffset = 0; kOffset < K; kOffset += TILE_SIZE) {
        // 1. copy in
#pragma unroll
        for (u32 ii = 0; ii < OUTPUTS_PER_THREAD; ++ii) {
            u32 tileRow = threadIdx.y + BLOCK_SIZE_Y * ii;
            u32 aRow = baseRow + tileRow;
            u32 aCol = kOffset + threadIdx.x;
            tileA[tileRow][threadIdx.x] = (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0;

            u32 bRow = kOffset + tileRow;
            u32 bCol = baseCol + threadIdx.x;
            tileB[tileRow][threadIdx.x] = (bRow < K && bCol < N) ? B[bRow * N + bCol] : 0;
        }
        __syncthreads();

        // 2. compute
#pragma unroll
        for (u32 ii = 0; ii < OUTPUTS_PER_THREAD; ++ii) {
            for (u32 t = 0; t < TILE_SIZE; ++t) {
                regs[ii] += tileA[threadIdx.y + BLOCK_SIZE_Y * ii][t] * tileB[t][threadIdx.x];
            }
        }
        __syncthreads();
    }

    // 3. copy out
#pragma unroll
    for (u32 ii = 0; ii < OUTPUTS_PER_THREAD; ++ii) {
        u32 row = baseRow + BLOCK_SIZE_Y * ii + threadIdx.y;
        u32 col = baseCol + threadIdx.x;
        if (row < M && col < N) {
            C[row * N + col] = regs[ii];
        }
    }
}

void sgemm_v1_do(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 blockDim{BLOCK_SIZE_X, BLOCK_SIZE_Y};
    dim3 gridDim{CeilDiv<u32>(N, TILE_SIZE), CeilDiv<u32>(M, TILE_SIZE)};

    sgemm_v1<<<gridDim, blockDim, 0, nullptr>>>(A, B, C, M, N, K);
}
