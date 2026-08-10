#include <cstdint>
#include "common_utils.h"

using u32 = uint32_t;

constexpr u32 TILEBASE_X = 32;
constexpr u32 TILEBASE_Y = 8;

constexpr u32 REG_TILE_X = 3;
constexpr u32 REG_TILE_Y = TILEBASE_X / TILEBASE_Y * REG_TILE_X;

constexpr u32 TILE_K = 32;
constexpr u32 TILE_X = TILEBASE_X * REG_TILE_X;
constexpr u32 TILE_Y = TILEBASE_Y * REG_TILE_Y;
constexpr u32 A_TILE_COLS_PER_THREAD = TILE_K / TILEBASE_X;
constexpr u32 B_TILE_ROWS_PER_THREAD = TILE_K / TILEBASE_Y;

// v3_1 只试验 global-to-shared copy；计算和写回逻辑保持与 v2 一致。
constexpr u32 VECTOR_WIDTH = 4;
constexpr u32 THREAD_COUNT = TILEBASE_X * TILEBASE_Y;
constexpr u32 A_TILE_VECTOR_COUNT = TILE_Y * TILE_K / VECTOR_WIDTH;
constexpr u32 B_TILE_VECTOR_COUNT = TILE_K * TILE_X / VECTOR_WIDTH;

static_assert(TILEBASE_X % TILEBASE_Y == 0);
static_assert(TILE_K % TILEBASE_X == 0);
static_assert(TILE_K % TILEBASE_Y == 0);
static_assert(TILE_X == TILE_Y);
static_assert(REG_TILE_Y * A_TILE_COLS_PER_THREAD == REG_TILE_X * B_TILE_ROWS_PER_THREAD);
static_assert(TILE_K % VECTOR_WIDTH == 0);
static_assert(TILE_X % VECTOR_WIDTH == 0);
static_assert(A_TILE_VECTOR_COUNT % THREAD_COUNT == 0);
static_assert(B_TILE_VECTOR_COUNT % THREAD_COUNT == 0);

// 标准 SGEMM 语义：C(M,N) = A(M,K) * B(K,N)，K 为收缩维度。
__global__ void sgemm_trial_v3_1(const float* A, const float* B, float* C, int M, int N, int K) {
    // 显式保证 shared tile 起始地址满足 float4 的 16-byte alignment。
    __shared__ __align__(16) float tileA[TILE_Y][TILE_K];
    __shared__ __align__(16) float tileB[TILE_K][TILE_X];

    u32 baseRow = TILE_Y * blockIdx.y;
    u32 baseCol = TILE_X * blockIdx.x;

    float regs[REG_TILE_Y][REG_TILE_X] = {};
    float regA[REG_TILE_Y] = {};
    float regB[REG_TILE_X] = {};
    for (u32 kOffset = 0; kOffset < K; kOffset += TILE_K) {
        // TODO(v3_1): 只替换下面 A/B 的 cooperative copy。
        // 建议先用 float4 global load + 标量 shared store，再尝试 float4 shared store并比较 SASS。
#pragma unroll
        for (u32 ri = 0; ri < REG_TILE_Y; ++ri) {
            u32 tileRow = TILEBASE_Y * ri + threadIdx.y;
            u32 aRow = baseRow + tileRow;
#pragma unroll
            for (u32 ai = 0; ai < A_TILE_COLS_PER_THREAD; ++ai) {
                u32 tileCol = TILEBASE_X * ai + threadIdx.x;
                u32 aCol = kOffset + tileCol;
                tileA[tileRow][tileCol] = (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0.0f;
            }
        }

#pragma unroll
        for (u32 bi = 0; bi < B_TILE_ROWS_PER_THREAD; ++bi) {
            u32 tileRow = TILEBASE_Y * bi + threadIdx.y;
            u32 bRow = kOffset + tileRow;
#pragma unroll
            for (u32 rj = 0; rj < REG_TILE_X; ++rj) {
                u32 tileCol = TILEBASE_X * rj + threadIdx.x;
                u32 bCol = baseCol + tileCol;
                tileB[tileRow][tileCol] = (bRow < K && bCol < N) ? B[bRow * N + bCol] : 0.0f;
            }
        }
        __syncthreads();

        for (u32 k = 0; k < TILE_K; ++k) {
#pragma unroll
            for (u32 ri = 0; ri < REG_TILE_Y; ++ri) {
                regA[ri] = tileA[TILEBASE_Y * ri + threadIdx.y][k];
            }
#pragma unroll
            for (u32 rj = 0; rj < REG_TILE_X; ++rj) {
                regB[rj] = tileB[k][TILEBASE_X * rj + threadIdx.x];
            }
#pragma unroll
            for (u32 ri = 0; ri < REG_TILE_Y; ++ri) {
#pragma unroll
                for (u32 rj = 0; rj < REG_TILE_X; ++rj) {
                    regs[ri][rj] += regA[ri] * regB[rj];
                }
            }
        }
        __syncthreads();
    }

    // C 的同线程输出并不连续；本轮不试验 vectorized global store。
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

void sgemm_trial_v3_1_do(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 blockDim{TILEBASE_X, TILEBASE_Y};
    dim3 gridDim{CeilDiv<u32>(N, TILE_X), CeilDiv<u32>(M, TILE_Y)};

    sgemm_trial_v3_1<<<gridDim, blockDim, 0, nullptr>>>(A, B, C, M, N, K);
}
