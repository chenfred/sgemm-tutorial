#include <cstdint>

#include "common_utils.h"

namespace {

using u32 = uint32_t;

constexpr u32 TILE_SIZE = 32;
constexpr u32 BLOCK_SIZE_X = 32;
constexpr u32 BLOCK_SIZE_Y = 8;
constexpr u32 OUTPUTS_PER_THREAD = 4;

// 历史上的原 sgemm_v2<false>：外层逐个处理 accumulator，内层遍历 K tile。
// 该循环顺序不主动暴露 4 条 accumulator 链之间的 ILP；当前 sgemm_v1 也采用了这一顺序。
// 标准 SGEMM 语义：C(M,N) = A(M,K) * B(K,N)，K 为收缩维度。
__global__ void sgemm_trial_v1_2(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    const u32 baseRow = blockIdx.y * TILE_SIZE;
    const u32 baseCol = blockIdx.x * TILE_SIZE;

    float regs[OUTPUTS_PER_THREAD] = {};
    for (u32 kOffset = 0; kOffset < static_cast<u32>(K); kOffset += TILE_SIZE) {
#pragma unroll
        for (u32 outputIndex = 0; outputIndex < OUTPUTS_PER_THREAD; ++outputIndex) {
            const u32 tileRow = threadIdx.y + BLOCK_SIZE_Y * outputIndex;
            const u32 aRow = baseRow + tileRow;
            const u32 aCol = kOffset + threadIdx.x;
            tileA[tileRow][threadIdx.x] =
                (aRow < static_cast<u32>(M) && aCol < static_cast<u32>(K)) ? A[aRow * K + aCol] : 0.0f;

            const u32 bRow = kOffset + tileRow;
            const u32 bCol = baseCol + threadIdx.x;
            tileB[tileRow][threadIdx.x] =
                (bRow < static_cast<u32>(K) && bCol < static_cast<u32>(N)) ? B[bRow * N + bCol] : 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (u32 outputIndex = 0; outputIndex < OUTPUTS_PER_THREAD; ++outputIndex) {
            for (u32 kInner = 0; kInner < TILE_SIZE; ++kInner) {
                regs[outputIndex] +=
                    tileA[threadIdx.y + BLOCK_SIZE_Y * outputIndex][kInner] * tileB[kInner][threadIdx.x];
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (u32 outputIndex = 0; outputIndex < OUTPUTS_PER_THREAD; ++outputIndex) {
        const u32 row = baseRow + BLOCK_SIZE_Y * outputIndex + threadIdx.y;
        const u32 col = baseCol + threadIdx.x;
        if (row < static_cast<u32>(M) && col < static_cast<u32>(N)) {
            C[row * N + col] = regs[outputIndex];
        }
    }
}

}  // namespace

void sgemm_trial_v1_2_do(const float* A, const float* B, float* C, int M, int N, int K) {
    const dim3 blockDim{BLOCK_SIZE_X, BLOCK_SIZE_Y};
    const dim3 gridDim{CeilDiv<u32>(N, TILE_SIZE), CeilDiv<u32>(M, TILE_SIZE)};

    sgemm_trial_v1_2<<<gridDim, blockDim, 0, nullptr>>>(A, B, C, M, N, K);
}
