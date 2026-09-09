#include "common_utils.h"
#include <cstdint>

using u32 = uint32_t;

constexpr u32 TILEBASE_X = 32;
constexpr u32 TILEBASE_Y = 8;

constexpr u32 REG_TILE_X = 3;
// 匹配 block 的长宽比，使当前 C tile 为正方形，并平衡每线程搬运的 A/B 元素数。
constexpr u32 REG_TILE_Y = TILEBASE_X / TILEBASE_Y * REG_TILE_X;

// K tile 与 blockDim 解耦；取两个轴的整数倍可保持 cooperative load 规则且连续。
constexpr u32 TILE_K = 32;
constexpr u32 STAGES = 2; // 双 stage 对照实验，保持“计算后补入”的循环结构。
constexpr u32 TILE_X = TILEBASE_X * REG_TILE_X;
constexpr u32 TILE_Y = TILEBASE_Y * REG_TILE_Y;
constexpr u32 A_TILE_COLS_PER_THREAD = TILE_K / TILEBASE_X;
constexpr u32 B_TILE_ROWS_PER_THREAD = TILE_K / TILEBASE_Y;

static_assert(TILEBASE_X % TILEBASE_Y == 0);
static_assert(TILE_K % TILEBASE_X == 0);
static_assert(TILE_K % TILEBASE_Y == 0);
static_assert(TILE_X == TILE_Y);
static_assert(REG_TILE_Y * A_TILE_COLS_PER_THREAD == REG_TILE_X * B_TILE_ROWS_PER_THREAD);

// 本文件使用 SM80+ 的 inline PTX；每次复制 4 字节，保留 v3 的线程映射。
// PTX
// 参考：https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async
__device__ __forceinline__ void copy_async_float(float* dstShared, const float* srcGlobal) {
    // CUDA 指针是 generic address；PTX 的 shared 操作数需要 shared 空间内的地址。
    const u32 sharedAddr = static_cast<u32>(__cvta_generic_to_shared(dstShared));
    // .ca：允许各级缓存；.shared.global：目标 shared、源 global；4：复制一个
    // float。 %0/%1 对应下面两个输入；"r" 是 32-bit 寄存器，"l" 是 64-bit
    // 寄存器。 无输出操作数；volatile 保留汇编，"memory"
    // 告知编译器存在内存副作用。 "memory" 不是 GPU 同步指令，复制的完成仍需
    // wait。
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" : : "r"(sharedAddr), "l"(srcGlobal) : "memory");
}

// 每个线程沿用 v3 的 A cooperative-copy 映射，直接发起 global ->
// shared[stage]。
__device__ __forceinline__ void issue_async_tile_a(float tileA[STAGES][TILE_Y][TILE_K], const float* A, u32 M, u32 K,
                                                   u32 baseRow, u32 kOffset, u32 stage) {
#pragma unroll
    for (u32 ri = 0; ri < REG_TILE_Y; ++ri) {
        u32 tileRow = TILEBASE_Y * ri + threadIdx.y;
        u32 aRow = baseRow + tileRow;
#pragma unroll
        for (u32 ai = 0; ai < A_TILE_COLS_PER_THREAD; ++ai) {
            u32 tileCol = TILEBASE_X * ai + threadIdx.x;
            u32 aCol = kOffset + tileCol;
            if (aRow < M && aCol < K) {
                copy_async_float(&tileA[stage][tileRow][tileCol], &A[aRow * K + aCol]);
            } else {
                // 保持边界处理直观：不构造越界 global 指针，直接写零。
                tileA[stage][tileRow][tileCol] = 0.0f;
            }
        }
    }
}

// B 的布局仍为 K×N，沿 K 分块，沿 N 分配 C tile 的列。
__device__ __forceinline__ void issue_async_tile_b(float tileB[STAGES][TILE_K][TILE_X], const float* B, u32 K, u32 N,
                                                   u32 baseCol, u32 kOffset, u32 stage) {
#pragma unroll
    for (u32 bi = 0; bi < B_TILE_ROWS_PER_THREAD; ++bi) {
        u32 tileRow = TILEBASE_Y * bi + threadIdx.y;
        u32 bRow = kOffset + tileRow;
#pragma unroll
        for (u32 rj = 0; rj < REG_TILE_X; ++rj) {
            u32 tileCol = TILEBASE_X * rj + threadIdx.x;
            u32 bCol = baseCol + tileCol;
            if (bRow < K && bCol < N) {
                copy_async_float(&tileB[stage][tileRow][tileCol], &B[bRow * N + bCol]);
            } else {
                tileB[stage][tileRow][tileCol] = 0.0f;
            }
        }
    }
}

__device__ __forceinline__ void compute_tile_c(float (&regs)[REG_TILE_Y][REG_TILE_X],
                                               const float tileA[STAGES][TILE_Y][TILE_K],
                                               const float tileB[STAGES][TILE_K][TILE_X], u32 stage) {
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
    float tileA[STAGES][TILE_Y][TILE_K];
    float tileB[STAGES][TILE_K][TILE_X];
};

// v4_2：沿用 v4_1 的搬运映射和计算，采用预填满、计算后补入的环形流水线。
// C(M,N) = A(M,K) * B(K,N)。当前双 stage 占用 48 KiB shared。
__global__ void sgemm_trial_v4_2(const float* A, const float* B, float* C, int M, int N, int K) {
    extern __shared__ SharedLayout dynamicShared[];
    auto& tileA = dynamicShared[0].tileA;
    auto& tileB = dynamicShared[0].tileB;

    float regs[REG_TILE_Y][REG_TILE_X] = {};

    const u32 baseRow = TILE_Y * blockIdx.y;
    const u32 baseCol = TILE_X * blockIdx.x;

    const u32 tileCount = static_cast<u32>(K) / TILE_K + (K % TILE_K != 0);

    // Prologue：预填满两个 stage；每个 tile 的 A/B 搬运单独 commit 为一组。
    static_assert(STAGES == 2, "下面的等待分支按双 stage 编写");
    for (u32 preload = 0; preload < STAGES && preload < tileCount; ++preload) {
        issue_async_tile_a(tileA, A, M, K, baseRow, preload * TILE_K, preload);
        issue_async_tile_b(tileB, B, K, N, baseCol, preload * TILE_K, preload);
        asm volatile("cp.async.commit_group;" ::: "memory");
    }

    // 主循环只有一个目标：按顺序计算全部 K tile，每轮恰好计算一块。
    for (u32 computeTile = 0; computeTile < tileCount; ++computeTile) {
        const u32 stage = computeTile % STAGES;
        const u32 remaining = tileCount - computeTile;

        // 等最老的 current，允许后续一组继续搬运；PTX 的等待参数必须是立即数。
        // 最后一块必须等待全部。边界线程没有有效 copy 也会提交空组。
        if (remaining >= 2) {
            asm volatile("cp.async.wait_group 1;" ::: "memory");
        } else {
            asm volatile("cp.async.wait_group 0;" ::: "memory");
        }
        __syncthreads(); // 所有线程的 current 已就绪，允许跨线程读取。

        compute_tile_c(regs, tileA, tileB, stage);

        // 所有线程读完后，在原位置补入新块：消费 tile 0 后，stage 0 搬入 tile 2。
        // 预取进度完全由计算进度推导；没有新块时不再覆盖，无需这个 barrier。
        const u32 prefetchTile = computeTile + STAGES;
        if (prefetchTile < tileCount) {
            __syncthreads();
            issue_async_tile_a(tileA, A, M, K, baseRow, prefetchTile * TILE_K, stage);
            issue_async_tile_b(tileB, B, K, N, baseCol, prefetchTile * TILE_K, stage);
            asm volatile("cp.async.commit_group;" ::: "memory");
        }
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

void sgemm_trial_v4_2_do(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 blockDim{TILEBASE_X, TILEBASE_Y};
    dim3 gridDim{CeilDiv<u32>(N, TILE_X), CeilDiv<u32>(M, TILE_Y)};

    CUDA_CHECK(
        cudaFuncSetAttribute(sgemm_trial_v4_2, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SharedLayout)));

    sgemm_trial_v4_2<<<gridDim, blockDim, sizeof(SharedLayout), nullptr>>>(A, B, C, M, N, K);
}
