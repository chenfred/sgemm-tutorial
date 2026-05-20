#include <cstdint>
#include "common_utils.h"

using u32 = uint32_t;

constexpr u32 TILE_LEN = 128;
constexpr u32 NUM_THREADS_PER_AXIS = 16;
constexpr u32 NUM_THREADS_PER_BLOCK = NUM_THREADS_PER_AXIS * NUM_THREADS_PER_AXIS; // 16*16=256 threads per block
constexpr u32 REGS_LEN = TILE_LEN / NUM_THREADS_PER_AXIS; // 8*8 = TILE_LEN*TILE_LEN/NUM_THREADS_BLOCK

__device__ constexpr u32 constexpr_min(u32 a, u32 b) { return a < b ? a : b; }
__device__ constexpr u32 constexpr_max(u32 a, u32 b) { return a > b ? a : b; }

__global__ void sgemm_v1(const float* A, const float* B, float* C, int M, int N, int K) {
    constexpr u32 Am = TILE_LEN;
    constexpr u32 Bk = TILE_LEN;
    constexpr u32 NS = 8;

    __shared__ float tileA[Am][NS];
    __shared__ float tileB[NS][Bk];

    // 线程在整个block下的唯一id，用于后续线程重排
    u32 tidx = threadIdx.y * blockDim.x + threadIdx.x;
    // 为拷贝tileA重排线程
    constexpr u32 BLOCKDIM_X_CPA = constexpr_min(NS, NUM_THREADS_PER_BLOCK);
    constexpr u32 BLOCKDIM_Y_CPA = constexpr_max(NUM_THREADS_PER_BLOCK / BLOCKDIM_X_CPA, 1u);
    u32 tx_a = tidx % BLOCKDIM_X_CPA;
    u32 ty_a = tidx / BLOCKDIM_X_CPA;
    // 为拷贝tileB重排线程
    constexpr u32 BLOCKDIM_X_CPB = constexpr_min(Bk, NUM_THREADS_PER_BLOCK);
    constexpr u32 BLOCKDIM_Y_CPB = constexpr_max(NUM_THREADS_PER_BLOCK / BLOCKDIM_X_CPB, 1u);
    u32 tx_b = tidx % BLOCKDIM_X_CPB;
    u32 ty_b = tidx / BLOCKDIM_X_CPB;

    // 当前block处理的第一个元素在C[r0][c0]，最后一个元素在C[r0+TILE_LEN-1][c]
    u32 r0 = blockIdx.y * TILE_LEN;
    u32 c0 = blockIdx.x * TILE_LEN;

    float regs[REGS_LEN][REGS_LEN] = {};
    for (u32 ni = 0; ni < N; ni += NS) {
        // 1. copy in
        // tileA
        for (u32 i = ty_a; i < Am; i += BLOCKDIM_Y_CPA) {
            for (u32 j = tx_a; j < NS; j += BLOCKDIM_X_CPA) {
                u32 ar = r0 + i;
                u32 ac = ni + j;
                tileA[i][j] = (ar < M && ac < N) ? A[ar * N + ac] : 0;
            }
        }
        // #pragma unroll
        // for (u32 i = tidx; i < Am * NS; i += NUM_THREADS_BLOCK) {
        //     u32 r = i / NS;
        //     u32 c = i % NS;
        //     u32 ar = r0 + r;
        //     u32 ac = ni + c;
        //     tileA[r][c] = (ar < M && ac < N) ? A[ar * N + ac] : 0;
        // }

        // tileB
        for (u32 i = ty_b; i < NS; i += BLOCKDIM_Y_CPB) {
            for (u32 j = tx_b; j < Bk; j += BLOCKDIM_X_CPB) {
                u32 br = ni + i;
                u32 bc = c0 + j;
                tileB[i][j] = (br < N && bc < K) ? B[br * K + bc] : 0;
            }
        }
        // #pragma unroll
        // for (u32 i = tidx; i < NS * Bk; i += NUM_THREADS_BLOCK) {
        //     u32 r = i / Bk;
        //     u32 c = i % Bk;
        //     u32 br = ni + r;
        //     u32 bc = c0 + c;
        //     tileB[r][c] = (br < N && bc < K) ? B[br * K + bc] : 0;
        // }
        __syncthreads();

        // 2. compute
        for (u32 i = 0; i * blockDim.y < Am; i++) {
            for (u32 j = 0; j * blockDim.x < Bk; j++) {
                for (u32 p = 0; p < NS; p++) {
                    regs[i][j] += tileA[threadIdx.y + i * blockDim.y][p] * tileB[p][j * blockDim.x + threadIdx.x];
                }
            }
        }
        __syncthreads();
    }

// 3. copy out
    for (u32 i = 0; i < REGS_LEN; i++) {
        for (u32 j = 0; j < REGS_LEN; ++j) {
            u32 r = r0 + i * blockDim.y + threadIdx.y;
            u32 c = c0 + j * blockDim.x + threadIdx.x;
            if (r < M && c < K) {
                C[r * K + c] = regs[i][j];
            }
        }
    }
}

void sgemm_v1_do(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 blockDim{NUM_THREADS_PER_AXIS, NUM_THREADS_PER_AXIS};
    dim3 gridDim{CeilDiv<u32>(K, TILE_LEN), CeilDiv<u32>(M, TILE_LEN)};

    sgemm_v1<<<gridDim, blockDim, 0, nullptr>>>(A, B, C, M, N, K);
}
