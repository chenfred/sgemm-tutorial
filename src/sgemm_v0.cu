#include <cstdint>
#include "common_utils.h"

static constexpr uint32_t TILE_SIZE = 16;

// 标准 SGEMM 语义：C(M,N) = A(M,K) * B(K,N)，K 为收缩维度。
static __global__ void sgemm_v0(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y; // 沿 M（C 的行）
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 沿 N（C 的列）

    float elemSum = 0;
    for (int kk = 0; kk < K; kk += TILE_SIZE) {
        // 1. copy in
        int aCol = kk + threadIdx.x;
        tileA[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0;
        int bRow = kk + threadIdx.y;
        tileB[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0;
        __syncthreads();

        // 2. compute
        for (int ti = 0; ti < TILE_SIZE; ++ti) {
            elemSum += tileA[threadIdx.y][ti] * tileB[ti][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = elemSum;
    }
}

void sgemm_v0_do(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 blockDim{TILE_SIZE, TILE_SIZE};
    dim3 gridDim{CeilDiv<uint32_t>(N, blockDim.x), CeilDiv<uint32_t>(M, blockDim.y)};

    sgemm_v0<<<gridDim, blockDim, 0, nullptr>>>(A, B, C, M, N, K);
}
