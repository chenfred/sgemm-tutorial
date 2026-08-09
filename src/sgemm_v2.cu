#include <cstdint>
#include "common_utils.h"

using u32 = uint32_t;

constexpr u32 TILEBASE_X = 32;
constexpr u32 TILEBASE_Y = 8;

constexpr u32 REG_TILE_X = 3;
// 匹配 block 的长宽比，使当前 C tile 为正方形，并平衡每线程搬运的 A/B 元素数。
constexpr u32 REG_TILE_Y = TILEBASE_X / TILEBASE_Y * REG_TILE_X;

// K tile 与 blockDim 解耦；取两个轴的整数倍可保持 cooperative load 规则且连续。
constexpr u32 TILE_K = 32;
constexpr u32 TILE_X = TILEBASE_X * REG_TILE_X;
constexpr u32 TILE_Y = TILEBASE_Y * REG_TILE_Y;
constexpr u32 A_TILE_COLS_PER_THREAD = TILE_K / TILEBASE_X;
constexpr u32 B_TILE_ROWS_PER_THREAD = TILE_K / TILEBASE_Y;

static_assert(TILEBASE_X % TILEBASE_Y == 0);
static_assert(TILE_K % TILEBASE_X == 0);
static_assert(TILE_K % TILEBASE_Y == 0);
static_assert(TILE_X == TILE_Y);
static_assert(REG_TILE_Y * A_TILE_COLS_PER_THREAD == REG_TILE_X * B_TILE_ROWS_PER_THREAD);

// 标准 SGEMM 语义：C(M,N) = A(M,K) * B(K,N)，K 为收缩维度。
__global__ void sgemm_v2(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float tileA[TILE_Y][TILE_K];
    __shared__ float tileB[TILE_K][TILE_X];

    u32 baseRow = TILE_Y * blockIdx.y;
    u32 baseCol = TILE_X * blockIdx.x;

    float regs[REG_TILE_Y][REG_TILE_X] = {};
    float regA[REG_TILE_Y] = {};
    float regB[REG_TILE_X] = {};
    for (u32 kOffset = 0; kOffset < K; kOffset += TILE_K) {
        // 每个 warp 沿 K 连续搬运 A；TILE_K 增大时，每线程规则地多搬若干列。
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

        // 每个 warp 沿 N 连续搬运 B；threadIdx.y 分摊 TILE_K 的各行。
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

        // 每个 k 步先把 A/B fragment 放入寄存器，再计算一个 REG_TILE_Y x REG_TILE_X 外积。
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

void sgemm_v2_do(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 blockDim{TILEBASE_X, TILEBASE_Y};
    dim3 gridDim{CeilDiv<u32>(N, TILE_X), CeilDiv<u32>(M, TILE_Y)};

    sgemm_v2<<<gridDim, blockDim, 0, nullptr>>>(A, B, C, M, N, K);
}
