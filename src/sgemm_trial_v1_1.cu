#include <cstdint>

#include "common_utils.h"

namespace {

using u32 = uint32_t;

constexpr u32 TILE_SIZE = 32;
constexpr u32 K_TILE_SIZE = 8;
constexpr u32 BLOCK_SIZE_X = 16;
constexpr u32 BLOCK_SIZE_Y = 16;
constexpr u32 THREADS_PER_BLOCK = BLOCK_SIZE_X * BLOCK_SIZE_Y;
constexpr u32 OUTPUTS_PER_AXIS = TILE_SIZE / BLOCK_SIZE_X;

// 历史上的原 sgemm_v1：每个线程计算 2x2 个输出，沿 K 方向每次加载 8 个元素。
// 标准 SGEMM 语义：C(M,N) = A(M,K) * B(K,N)，K 为收缩维度。
__global__ void sgemm_trial_v1_1(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float tileA[TILE_SIZE][K_TILE_SIZE];
    __shared__ float tileB[K_TILE_SIZE][TILE_SIZE];

    const u32 threadIndex = threadIdx.y * blockDim.x + threadIdx.x;

    constexpr u32 COPY_A_BLOCK_SIZE_X = K_TILE_SIZE;
    constexpr u32 COPY_A_BLOCK_SIZE_Y = THREADS_PER_BLOCK / COPY_A_BLOCK_SIZE_X;
    const u32 copyAThreadX = threadIndex % COPY_A_BLOCK_SIZE_X;
    const u32 copyAThreadY = threadIndex / COPY_A_BLOCK_SIZE_X;

    constexpr u32 COPY_B_BLOCK_SIZE_X = TILE_SIZE;
    constexpr u32 COPY_B_BLOCK_SIZE_Y = THREADS_PER_BLOCK / COPY_B_BLOCK_SIZE_X;
    const u32 copyBThreadX = threadIndex % COPY_B_BLOCK_SIZE_X;
    const u32 copyBThreadY = threadIndex / COPY_B_BLOCK_SIZE_X;

    const u32 baseRow = blockIdx.y * TILE_SIZE;
    const u32 baseCol = blockIdx.x * TILE_SIZE;

    float regs[OUTPUTS_PER_AXIS][OUTPUTS_PER_AXIS] = {};
    for (u32 kOffset = 0; kOffset < static_cast<u32>(K); kOffset += K_TILE_SIZE) {
        for (u32 tileRow = copyAThreadY; tileRow < TILE_SIZE; tileRow += COPY_A_BLOCK_SIZE_Y) {
            for (u32 tileCol = copyAThreadX; tileCol < K_TILE_SIZE; tileCol += COPY_A_BLOCK_SIZE_X) {
                const u32 aRow = baseRow + tileRow;
                const u32 aCol = kOffset + tileCol;
                tileA[tileRow][tileCol] =
                    (aRow < static_cast<u32>(M) && aCol < static_cast<u32>(K)) ? A[aRow * K + aCol] : 0.0f;
            }
        }

        for (u32 tileRow = copyBThreadY; tileRow < K_TILE_SIZE; tileRow += COPY_B_BLOCK_SIZE_Y) {
            for (u32 tileCol = copyBThreadX; tileCol < TILE_SIZE; tileCol += COPY_B_BLOCK_SIZE_X) {
                const u32 bRow = kOffset + tileRow;
                const u32 bCol = baseCol + tileCol;
                tileB[tileRow][tileCol] =
                    (bRow < static_cast<u32>(K) && bCol < static_cast<u32>(N)) ? B[bRow * N + bCol] : 0.0f;
            }
        }
        __syncthreads();

        for (u32 outputRow = 0; outputRow < OUTPUTS_PER_AXIS; ++outputRow) {
            for (u32 outputCol = 0; outputCol < OUTPUTS_PER_AXIS; ++outputCol) {
                for (u32 kInner = 0; kInner < K_TILE_SIZE; ++kInner) {
                    regs[outputRow][outputCol] += tileA[threadIdx.y + outputRow * BLOCK_SIZE_Y][kInner] *
                                                        tileB[kInner][outputCol * BLOCK_SIZE_X + threadIdx.x];
                }
            }
        }
        __syncthreads();
    }

    for (u32 outputRow = 0; outputRow < OUTPUTS_PER_AXIS; ++outputRow) {
        for (u32 outputCol = 0; outputCol < OUTPUTS_PER_AXIS; ++outputCol) {
            const u32 row = baseRow + outputRow * BLOCK_SIZE_Y + threadIdx.y;
            const u32 col = baseCol + outputCol * BLOCK_SIZE_X + threadIdx.x;
            if (row < static_cast<u32>(M) && col < static_cast<u32>(N)) {
                C[row * N + col] = regs[outputRow][outputCol];
            }
        }
    }
}

}  // namespace

void sgemm_trial_v1_1_do(const float* A, const float* B, float* C, int M, int N, int K) {
    const dim3 blockDim{BLOCK_SIZE_X, BLOCK_SIZE_Y};
    const dim3 gridDim{CeilDiv<u32>(N, TILE_SIZE), CeilDiv<u32>(M, TILE_SIZE)};

    sgemm_trial_v1_1<<<gridDim, blockDim, 0, nullptr>>>(A, B, C, M, N, K);
}
