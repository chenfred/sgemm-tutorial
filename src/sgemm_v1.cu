#include <cstdint>
#include "common_utils.h"

constexpr uint32_t NAIVE_TILE_SIZE = 16;

__global__ void sgemm_v1(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float tileA[NAIVE_TILE_SIZE][NAIVE_TILE_SIZE];
    __shared__ float tileB[NAIVE_TILE_SIZE][NAIVE_TILE_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float elemSum = 0;
    for (int ni = 0; ni < N; ni += NAIVE_TILE_SIZE) {
        // 1. copy in
        int aCol = ni + threadIdx.x;
        tileA[threadIdx.y][threadIdx.x] = (row < M && aCol < N) ? A[row * N + aCol] : 0;
        int bRow = ni + threadIdx.y;
        tileB[threadIdx.y][threadIdx.x] = (bRow < N && col < K) ? B[bRow * K + col] : 0;
        __syncthreads();

        // 2. compute
        for (int ti = 0; ti < NAIVE_TILE_SIZE; ++ti) {
            elemSum += tileA[threadIdx.y][ti] * tileB[ti][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < K) {
        C[row * K + col] = elemSum;
    }
}

void sgemm_v1_do(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 blockDim{NAIVE_TILE_SIZE, NAIVE_TILE_SIZE};
    dim3 gridDim{CeilDiv<uint32_t>(K, blockDim.x), CeilDiv<uint32_t>(M, blockDim.y)};

    sgemm_v1<<<gridDim, blockDim, 0, nullptr>>>(A, B, C, M, N, K);
}
