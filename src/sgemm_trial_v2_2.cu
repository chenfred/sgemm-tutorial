#include <cstdint>
#include "common_utils.h"

using u32 = uint32_t;

constexpr u32 REG_TILE_X = 3;
constexpr u32 REG_TILE_Y = 4 * REG_TILE_X;

constexpr u32 TILEBASE_X = 32;
constexpr u32 TILEBASE_Y = 8;
constexpr u32 TILE_K = TILEBASE_X;
constexpr u32 TILE_X = TILEBASE_X * REG_TILE_X;
constexpr u32 TILE_Y = TILEBASE_Y * REG_TILE_Y;
constexpr u32 B_TILE_ROWS_PER_THREAD = TILE_K / TILEBASE_Y;

static_assert(TILE_K % TILEBASE_Y == 0);

// 标准 SGEMM 语义：C(M,N) = A(M,K) * B(K,N)，K 为收缩维度。
__global__ void sgemm_trial_v2_2(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float tileA[TILE_Y][TILE_K];
    __shared__ float tileB[TILE_K][TILE_X];

    u32 baseRow = TILE_Y * blockIdx.y;
    u32 baseCol = TILE_X * blockIdx.x;

    float regs[REG_TILE_Y][REG_TILE_X] = {};
    float regX[TILE_X] = {};
    float regY[TILE_Y] = {};
    for (u32 kOffset = 0; kOffset < K; kOffset += TILE_K) {
        // 1. copy in
#pragma unroll
        for (u32 ri = 0; ri < REG_TILE_Y; ++ri) {
            u32 tileRow = TILEBASE_Y * ri + threadIdx.y;
            u32 aRow = baseRow + tileRow;
            u32 aCol = kOffset + threadIdx.x;
            tileA[tileRow][threadIdx.x] = (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0;
        }

#pragma unroll
        for (u32 bi = 0; bi < B_TILE_ROWS_PER_THREAD; ++bi) {
            u32 tileRow = TILEBASE_Y * bi + threadIdx.y;
            u32 bRow = kOffset + tileRow;
#pragma unroll
            for (u32 rj = 0; rj < REG_TILE_X; ++rj) {
                u32 bCol = baseCol + TILEBASE_X * rj + threadIdx.x;
                tileB[tileRow][TILEBASE_X * rj + threadIdx.x] =
                    (bRow < K && bCol < N) ? B[bRow * N + bCol] : 0;
            }
        }
        __syncthreads();


        for (u32 k = 0; k < TILE_K; ++k) {
#pragma unroll
            for (u32 ri = 0; ri < REG_TILE_Y; ++ri) {
                regY[ri] = tileA[TILEBASE_Y * ri + threadIdx.y][k]; //! broadcast
            }
#pragma unroll
            for (u32 rj = 0; rj < REG_TILE_X; ++rj) {
                regX[rj] = tileB[k][TILEBASE_X * rj + threadIdx.x]; //! 刚好32个bank
            }

#pragma unroll
            for (u32 ri = 0; ri < REG_TILE_Y; ++ri) {
#pragma unroll
                for (u32 rj = 0; rj < REG_TILE_X; ++rj) {
                    regs[ri][rj] += regY[ri] * regX[rj];
                }
            }
        }

        __syncthreads();
    }

    // 3. copy out
#pragma unroll
    for (u32 ri = 0; ri < REG_TILE_Y; ++ri) {
#pragma unroll
        for (u32 rj = 0; rj < REG_TILE_X; ++rj) {
            u32 row = baseRow + TILEBASE_Y * ri + threadIdx.y;
            u32 col = baseCol + TILEBASE_X * rj + threadIdx.x;
            if (row < M && col < N) {
                C[row * N + col] = regs[ri][rj];
            }
        }
    }
}

void sgemm_trial_v2_2_do(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 blockDim{TILEBASE_X, TILEBASE_Y};
    dim3 gridDim{CeilDiv<u32>(N, TILE_X), CeilDiv<u32>(M, TILE_Y)};

    sgemm_trial_v2_2<<<gridDim, blockDim, 0, nullptr>>>(A, B, C, M, N, K);
}
