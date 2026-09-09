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
__device__ __forceinline__ void issue_async_tile_a(float tileA[2][TILE_Y][TILE_K], const float* A, u32 M, u32 K,
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
__device__ __forceinline__ void issue_async_tile_b(float tileB[2][TILE_K][TILE_X], const float* B, u32 K, u32 N,
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
                                               const float tileA[2][TILE_Y][TILE_K],
                                               const float tileB[2][TILE_K][TILE_X], u32 stage) {
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

// v4 仅将 v3 的 LDG -> prefetch registers -> STS 替换为
// cp.async，保留双缓冲。 标准 SGEMM 语义：C(M,N) = A(M,K) * B(K,N)，K
// 为收缩维度。
__global__ void sgemm_v4(const float* A, const float* B, float* C, int M, int N, int K) {
    extern __shared__ SharedLayout dynamicShared[];
    auto& tileA = dynamicShared[0].tileA;
    auto& tileB = dynamicShared[0].tileB;

    float regs[REG_TILE_Y][REG_TILE_X] = {};

    const u32 baseRow = TILE_Y * blockIdx.y;
    const u32 baseCol = TILE_X * blockIdx.x;

    u32 readStage = 0;

    // Prologue：issue -> commit -> wait -> block barrier，先准备第一块。
    issue_async_tile_a(tileA, A, M, K, baseRow, 0, readStage);
    issue_async_tile_b(tileB, B, K, N, baseCol, 0, readStage);
    // 将本线程发起的 A/B copies 提交为一个 group；commit 不等待完成。
    asm volatile("cp.async.commit_group;" ::: "memory");
    // 0 表示不允许留下未完成的已提交 group；只等待本线程发起的 copies。
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    // 所有线程完成 copy/边界补零后，才可读取其他线程搬运的数据。
    __syncthreads();

    for (u32 kOffset = TILE_K; kOffset < K; kOffset += TILE_K) {
        const u32 writeStage = readStage ^ 1;

        // next 直接搬到另一个 shared stage，与 current compute 无数据依赖。
        issue_async_tile_a(tileA, A, M, K, baseRow, kOffset, writeStage);
        issue_async_tile_b(tileB, B, K, N, baseCol, kOffset, writeStage);
        asm volatile("cp.async.commit_group;" ::: "memory");

        // 保持 v3 的计算循环；在 commit 与 wait 之间提供 copy/compute 重叠窗口。
        compute_tile_c(regs, tileA, tileB, readStage);

        asm volatile("cp.async.wait_group 0;" ::: "memory");
        // 同时保证所有线程的 next 已准备好、current 已读完，下一轮才能复用旧
        // stage。 commit/wait/barrier 均由整个 block 执行，边界线程也不能提前
        // return。
        __syncthreads();
        readStage = writeStage;
    }

    // Epilogue：消费最后一个已准备好的 stage；此后不再复用 shared，因此不需要
    // barrier。
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

void sgemm_v4_do(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 blockDim{TILEBASE_X, TILEBASE_Y};
    dim3 gridDim{CeilDiv<u32>(N, TILE_X), CeilDiv<u32>(M, TILE_Y)};

    CUDA_CHECK(
        cudaFuncSetAttribute(sgemm_v4, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SharedLayout)));

    sgemm_v4<<<gridDim, blockDim, sizeof(SharedLayout), nullptr>>>(A, B, C, M, N, K);
}
