#include <cstdint>
#include "common_utils.h"

using u32 = uint32_t;

__device__ __forceinline__ float GetValue(const float* Mat, u32 N, u32 row, u32 col) { return Mat[row * N + col]; }
__device__ __forceinline__ void SetValue(float* Mat, u32 N, u32 row, u32 col, float val) { Mat[row * N + col] = val; }

// 标准 SGEMM 语义：C(M,N) = A(M,K) * B(K,N)，K 为收缩维度。
__global__ void sgemm_v2(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float tileA[32][32];
    __shared__ float tileB[32][32];

    u32 baseRow = blockIdx.y * 32;
    u32 baseCol = blockIdx.x * 32;

    float regs[4] = {}; // 需要初始化为0
    for (u32 kOffset = 0; kOffset < K; kOffset += 32) {
        // 1. copy in
        #pragma unroll
        for (u32 ii = 0; ii < 4; ++ii) {
            u32 aRow = baseRow + 8 * ii + threadIdx.y;
            u32 aCol = kOffset + threadIdx.x;
            tileA[threadIdx.y + 8 * ii][threadIdx.x] = (aRow < M && aCol < K) ? GetValue(A, K, aRow, aCol) : 0;
            u32 bRow = kOffset + 8 * ii + threadIdx.y;
            u32 bCol = baseCol + threadIdx.x;
            tileB[threadIdx.y + 8 * ii][threadIdx.x] = (bRow < K && bCol < N) ? GetValue(B, N, bRow, bCol) : 0;
        }
        __syncthreads();

        // 2. compute
        for (u32 t = 0; t < 32; ++t) {
            #pragma unroll
            for (u32 ii = 0; ii < 4; ++ii) {
                regs[ii] += tileA[threadIdx.y + 8 * ii][t] * tileB[t][threadIdx.x];
            }
        }
        __syncthreads();
    }

    // 3. copy out
    #pragma unroll
    for (u32 ii = 0; ii < 4; ++ii) {
        u32 row = baseRow + 8 * ii + threadIdx.y;
        u32 col = baseCol + threadIdx.x;
        if (row < M && col < N) {
            SetValue(C, N, row, col, regs[ii]);
        }
    }
}

void sgemm_v2_do(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 blockDim{32, 8};
    dim3 gridDim{CeilDiv<u32>(N, 32), CeilDiv<u32>(M, 32)};

    sgemm_v2<<<gridDim, blockDim, 0, nullptr>>>(A, B, C, M, N, K);
}
