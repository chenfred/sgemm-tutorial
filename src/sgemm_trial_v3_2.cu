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

// 与 v3_1 使用相同的展平映射；v3_2 只把完整、对齐 chunk 改为 float4 搬运。
constexpr u32 VECTOR_WIDTH = 4;
constexpr u32 THREAD_COUNT = TILEBASE_X * TILEBASE_Y;
constexpr u32 A_TILE_VECTOR_COUNT = TILE_Y * TILE_K / VECTOR_WIDTH;
constexpr u32 B_TILE_VECTOR_COUNT = TILE_K * TILE_X / VECTOR_WIDTH;
constexpr u32 A_TILE_VECTORS_PER_THREAD = A_TILE_VECTOR_COUNT / THREAD_COUNT;
constexpr u32 B_TILE_VECTORS_PER_THREAD = B_TILE_VECTOR_COUNT / THREAD_COUNT;

static_assert(TILEBASE_X % TILEBASE_Y == 0);
static_assert(TILE_K % TILEBASE_X == 0);
static_assert(TILE_K % TILEBASE_Y == 0);
static_assert(TILE_X == TILE_Y);
static_assert(TILE_K % VECTOR_WIDTH == 0);
static_assert(TILE_X % VECTOR_WIDTH == 0);
static_assert(A_TILE_VECTOR_COUNT % THREAD_COUNT == 0);
static_assert(B_TILE_VECTOR_COUNT % THREAD_COUNT == 0);
static_assert(A_TILE_VECTOR_COUNT == B_TILE_VECTOR_COUNT);

// 标准 SGEMM 语义：C(M,N) = A(M,K) * B(K,N)，K 为收缩维度。
__global__ void sgemm_trial_v3_2(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ __align__(16) float tileA[TILE_Y][TILE_K];
    __shared__ __align__(16) float tileB[TILE_K][TILE_X];

    u32 baseRow = TILE_Y * blockIdx.y;
    u32 baseCol = TILE_X * blockIdx.x;

    float regs[REG_TILE_Y][REG_TILE_X] = {};
    float regA[REG_TILE_Y] = {};
    float regB[REG_TILE_X] = {};
    for (u32 kOffset = 0; kOffset < K; kOffset += TILE_K) {
        u32 threadId = threadIdx.y * TILEBASE_X + threadIdx.x;

#pragma unroll
        for (u32 copyIndex = 0; copyIndex < A_TILE_VECTORS_PER_THREAD; ++copyIndex) {
            u32 vectorIndex = threadId + copyIndex * THREAD_COUNT;
            u32 scalarIndex = vectorIndex * VECTOR_WIDTH;
            u32 tileRow = scalarIndex / TILE_K;
            u32 tileCol = scalarIndex % TILE_K;
            u32 aRow = baseRow + tileRow;
            u32 aCol = kOffset + tileCol;

            const bool vectorInBounds = aRow < M && aCol + VECTOR_WIDTH <= K;
            if (vectorInBounds) {
                const float* globalAddress = &A[aRow * K + aCol];
                const bool globalAddressAligned =
                    reinterpret_cast<std::uintptr_t>(globalAddress) % alignof(float4) == 0;
                if (globalAddressAligned) {
                    float4 value = *reinterpret_cast<const float4*>(globalAddress);
                    *reinterpret_cast<float4*>(&tileA[tileRow][tileCol]) = value;
                    continue;
                }
            }

#pragma unroll
            for (u32 vi = 0; vi < VECTOR_WIDTH; ++vi) {
                u32 scalarCol = aCol + vi;
                tileA[tileRow][tileCol + vi] =
                    (aRow < M && scalarCol < K) ? A[aRow * K + scalarCol] : 0.0f;
            }
        }

#pragma unroll
        for (u32 copyIndex = 0; copyIndex < B_TILE_VECTORS_PER_THREAD; ++copyIndex) {
            u32 vectorIndex = threadId + copyIndex * THREAD_COUNT;
            u32 scalarIndex = vectorIndex * VECTOR_WIDTH;
            u32 tileRow = scalarIndex / TILE_X;
            u32 tileCol = scalarIndex % TILE_X;
            u32 bRow = kOffset + tileRow;
            u32 bCol = baseCol + tileCol;

            const bool vectorInBounds = bRow < K && bCol + VECTOR_WIDTH <= N;
            if (vectorInBounds) {
                const float* globalAddress = &B[bRow * N + bCol];
                const bool globalAddressAligned =
                    reinterpret_cast<std::uintptr_t>(globalAddress) % alignof(float4) == 0;
                if (globalAddressAligned) {
                    float4 value = *reinterpret_cast<const float4*>(globalAddress);
                    *reinterpret_cast<float4*>(&tileB[tileRow][tileCol]) = value;
                    continue;
                }
            }

#pragma unroll
            for (u32 vi = 0; vi < VECTOR_WIDTH; ++vi) {
                u32 scalarCol = bCol + vi;
                tileB[tileRow][tileCol + vi] =
                    (bRow < K && scalarCol < N) ? B[bRow * N + scalarCol] : 0.0f;
            }
        }
        __syncthreads();

        for (u32 k = 0; k < TILE_K; ++k) {
            // shared-to-register load 与外积计算保持和 v3_1 完全一致。
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

void sgemm_trial_v3_2_do(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 blockDim{TILEBASE_X, TILEBASE_Y};
    dim3 gridDim{CeilDiv<u32>(N, TILE_X), CeilDiv<u32>(M, TILE_Y)};

    sgemm_trial_v3_2<<<gridDim, blockDim, 0, nullptr>>>(A, B, C, M, N, K);
}
