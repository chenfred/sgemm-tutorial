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

// 每个线程按 cooperative-copy 映射预取 A 的 next tile；越界元素在进入寄存器时补零。
__device__ __forceinline__ void load_reg_tile_a(float (&prefetchA)[REG_TILE_Y][A_TILE_COLS_PER_THREAD], const float* A,
                                                u32 M, u32 K, u32 baseRow, u32 kOffset) {
#pragma unroll
    for (u32 ri = 0; ri < REG_TILE_Y; ++ri) {
        u32 aRow = baseRow + TILEBASE_Y * ri + threadIdx.y;
#pragma unroll
        for (u32 ai = 0; ai < A_TILE_COLS_PER_THREAD; ++ai) {
            u32 aCol = kOffset + TILEBASE_X * ai + threadIdx.x;
            prefetchA[ri][ai] = (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0.0f;
        }
    }
}

// 把已完成边界处理的 A 预取值写入 shared[stage]，不再访问 global memory。
__device__ __forceinline__ void store_shared_tile_a(__shared__ float tileA[2][TILE_Y][TILE_K],
                                                    const float (&prefetchA)[REG_TILE_Y][A_TILE_COLS_PER_THREAD],
                                                    u32 stage) {
#pragma unroll
    for (u32 ri = 0; ri < REG_TILE_Y; ++ri) {
        u32 tileRow = TILEBASE_Y * ri + threadIdx.y;
#pragma unroll
        for (u32 ai = 0; ai < A_TILE_COLS_PER_THREAD; ++ai) {
            u32 tileCol = TILEBASE_X * ai + threadIdx.x;
            tileA[stage][tileRow][tileCol] = prefetchA[ri][ai];
        }
    }
}

// 每个线程按 cooperative-copy 映射预取 B 的 next tile；越界元素在进入寄存器时补零。
__device__ __forceinline__ void load_reg_tile_b(float (&prefetchB)[B_TILE_ROWS_PER_THREAD][REG_TILE_X], const float* B,
                                                u32 K, u32 N, u32 baseCol, u32 kOffset) {
#pragma unroll
    for (u32 bi = 0; bi < B_TILE_ROWS_PER_THREAD; ++bi) {
        u32 bRow = kOffset + TILEBASE_Y * bi + threadIdx.y;
#pragma unroll
        for (u32 rj = 0; rj < REG_TILE_X; ++rj) {
            u32 bCol = baseCol + TILEBASE_X * rj + threadIdx.x;
            prefetchB[bi][rj] = (bRow < K && bCol < N) ? B[bRow * N + bCol] : 0.0f;
        }
    }
}

// 把已完成边界处理的 B 预取值写入 shared[stage]，不再访问 global memory。
__device__ __forceinline__ void store_shared_tile_b(__shared__ float tileB[2][TILE_K][TILE_X],
                                                    const float (&prefetchB)[B_TILE_ROWS_PER_THREAD][REG_TILE_X],
                                                    u32 stage) {
#pragma unroll
    for (u32 bi = 0; bi < B_TILE_ROWS_PER_THREAD; ++bi) {
        u32 tileRow = TILEBASE_Y * bi + threadIdx.y;
#pragma unroll
        for (u32 rj = 0; rj < REG_TILE_X; ++rj) {
            u32 tileCol = TILEBASE_X * rj + threadIdx.x;
            tileB[stage][tileRow][tileCol] = prefetchB[bi][rj];
        }
    }
}

__device__ __forceinline__ void compute_tile_c(float (&regs)[REG_TILE_Y][REG_TILE_X],
                                               __shared__ const float tileA[2][TILE_Y][TILE_K],
                                               __shared__ const float tileB[2][TILE_K][TILE_X], u32 stage) {
    float regA[REG_TILE_Y];
    float regB[REG_TILE_X];

    for (u32 k = 0; k < TILE_K; ++k) {
#pragma unroll
        for (u32 ri = 0; ri < REG_TILE_Y; ++ri) {
            regA[ri] = tileA[stage][TILEBASE_Y * ri + threadIdx.y][k];
        }
#pragma unroll
        for (u32 rj = 0; rj < REG_TILE_X; ++rj) {
            regB[rj] = tileB[stage][k][TILEBASE_X * rj + threadIdx.x];
        }
#pragma unroll
        for (u32 ri = 0; ri < REG_TILE_Y; ++ri) {
#pragma unroll
            for (u32 rj = 0; rj < REG_TILE_X; ++rj) {
                regs[ri][rj] += regA[ri] * regB[rj];
            }
        }
    }
}

struct SharedLayout {
    float tileA[2][TILE_Y][TILE_K];
    float tileB[2][TILE_K][TILE_X];
};

// v3 使用普通 LDG register prefetch 和双 shared stage 实现 double buffering。
// 标准 SGEMM 语义：C(M,N) = A(M,K) * B(K,N)，K 为收缩维度。
__global__ void sgemm_v3(const float* A, const float* B, float* C, int M, int N, int K) {
    extern __shared__ SharedLayout dynamicShared[];
    auto& tileA = dynamicShared[0].tileA;
    auto& tileB = dynamicShared[0].tileB;

    // next tile 的 global-load 结果跨越 current compute 保持存活；编译器应优先将这些数组标量化到寄存器。
    float prefetchA[REG_TILE_Y][A_TILE_COLS_PER_THREAD];
    float prefetchB[B_TILE_ROWS_PER_THREAD][REG_TILE_X];
    float regs[REG_TILE_Y][REG_TILE_X] = {};

    const u32 baseRow = TILE_Y * blockIdx.y;
    const u32 baseCol = TILE_X * blockIdx.x;

    u32 readStage = 0;

    // Prologue：第一块之前没有 current compute，只能直接 load -> store -> barrier。
    load_reg_tile_a(prefetchA, A, M, K, baseRow, 0);
    load_reg_tile_b(prefetchB, B, K, N, baseCol, 0);
    store_shared_tile_a(tileA, prefetchA, readStage);
    store_shared_tile_b(tileB, prefetchB, readStage);
    __syncthreads();

    for (u32 kOffset = TILE_K; kOffset < K; kOffset += TILE_K) {
        const u32 writeStage = readStage ^ 1;

        // 发射 next tile 的 global load，暂不写入 shared，使结果与后续 current compute 没有数据依赖。
        load_reg_tile_a(prefetchA, A, M, K, baseRow, kOffset);
        load_reg_tile_b(prefetchB, B, K, N, baseCol, kOffset);

        // current 只读取 shared[readStage]，位于 next LDG 和依赖其结果的 STS 之间。
        compute_tile_c(regs, tileA, tileB, readStage);

        // next 只写入 shared[writeStage]，不会覆盖本轮读取的 current stage。
        store_shared_tile_a(tileA, prefetchA, writeStage);
        store_shared_tile_b(tileB, prefetchB, writeStage);

        // 同时保证 next 已写完、current 已读完；之后才能交换并在下一轮复用两个 stage。
        __syncthreads();
        readStage = writeStage;
    }

    // Epilogue：消费最后一个已准备好的 stage；此后不再复用 shared，因此不需要 barrier。
    compute_tile_c(regs, tileA, tileB, readStage);

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

void sgemm_v3_do(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 blockDim{TILEBASE_X, TILEBASE_Y};
    dim3 gridDim{CeilDiv<u32>(N, TILE_X), CeilDiv<u32>(M, TILE_Y)};

    CUDA_CHECK(cudaFuncSetAttribute(sgemm_v3, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SharedLayout)));

    sgemm_v3<<<gridDim, blockDim, sizeof(SharedLayout), nullptr>>>(A, B, C, M, N, K);
}
