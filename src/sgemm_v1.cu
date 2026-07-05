#include <cstdint>
#include "common_utils.h"

using u32 = uint32_t;

constexpr u32 TILE_LEN = 32;                                                          // 一个 block 处理 C 的 TILE_LEN×TILE_LEN 个元素
constexpr u32 NUM_THREADS_PER_AXIS = 16;
constexpr u32 NUM_THREADS_PER_BLOCK = NUM_THREADS_PER_AXIS * NUM_THREADS_PER_AXIS;    // 16*16=256 threads per block
constexpr u32 REGS_LEN = TILE_LEN / NUM_THREADS_PER_AXIS;                             // 2，每线程在 M/N 方向各负责 REGS_LEN 个元素

__device__ constexpr u32 constexpr_min(u32 a, u32 b) { return a < b ? a : b; }
__device__ constexpr u32 constexpr_max(u32 a, u32 b) { return a > b ? a : b; }

// 标准 SGEMM 语义：C(M,N) = A(M,K) * B(K,N)，K 为收缩维度。
__global__ void sgemm_v1(const float* A, const float* B, float* C, int M, int N, int K) {
    constexpr u32 BM = TILE_LEN; // block 覆盖 C 的 M 方向 32 行
    constexpr u32 BN = TILE_LEN; // block 覆盖 C 的 N 方向 32 列
    constexpr u32 KS = 8;        // 沿 K（收缩）方向每次推进的步长

    __shared__ float tileA[BM][KS]; // A 的行片段：M 方向 × K 方向
    __shared__ float tileB[KS][BN]; // B 的列片段：K 方向 × N 方向

    // 线程在整个block下的唯一id，用于后续线程重排
    u32 tidx = threadIdx.y * blockDim.x + threadIdx.x;
    // 为拷贝tileA重排线程
    constexpr u32 BLOCKDIM_X_CPA = constexpr_min(KS, NUM_THREADS_PER_BLOCK);
    constexpr u32 BLOCKDIM_Y_CPA = constexpr_max(NUM_THREADS_PER_BLOCK / BLOCKDIM_X_CPA, 1u);
    u32 tx_a = tidx % BLOCKDIM_X_CPA;
    u32 ty_a = tidx / BLOCKDIM_X_CPA;
    // 为拷贝tileB重排线程
    constexpr u32 BLOCKDIM_X_CPB = constexpr_min(BN, NUM_THREADS_PER_BLOCK);
    constexpr u32 BLOCKDIM_Y_CPB = constexpr_max(NUM_THREADS_PER_BLOCK / BLOCKDIM_X_CPB, 1u);
    u32 tx_b = tidx % BLOCKDIM_X_CPB;
    u32 ty_b = tidx / BLOCKDIM_X_CPB;

    // 当前 block 处理 C[r0..r0+TILE_LEN-1][c0..c0+TILE_LEN-1]，r0 沿 M，c0 沿 N
    u32 r0 = blockIdx.y * TILE_LEN; // M 方向起点
    u32 c0 = blockIdx.x * TILE_LEN; // N 方向起点

    float regs[REGS_LEN][REGS_LEN] = {};
    for (u32 kk = 0; kk < K; kk += KS) {
        // 1. copy in
        // tileA: 加载 A[r0+i][kk+j]，A 布局 M×K
        for (u32 i = ty_a; i < BM; i += BLOCKDIM_Y_CPA) {
            for (u32 j = tx_a; j < KS; j += BLOCKDIM_X_CPA) {
                u32 ar = r0 + i;
                u32 ac = kk + j;
                tileA[i][j] = (ar < M && ac < K) ? A[ar * K + ac] : 0;
            }
        }
        // #pragma unroll
        // for (u32 i = tidx; i < BM * KS; i += NUM_THREADS_PER_BLOCK) {
        //     u32 r = i / KS;
        //     u32 c = i % KS;
        //     u32 ar = r0 + r;
        //     u32 ac = kk + c;
        //     tileA[r][c] = (ar < M && ac < K) ? A[ar * K + ac] : 0;
        // }

        // tileB: 加载 B[kk+i][c0+j]，B 布局 K×N
        for (u32 i = ty_b; i < KS; i += BLOCKDIM_Y_CPB) {
            for (u32 j = tx_b; j < BN; j += BLOCKDIM_X_CPB) {
                u32 br = kk + i;
                u32 bc = c0 + j;
                tileB[i][j] = (br < K && bc < N) ? B[br * N + bc] : 0;
            }
        }
        // #pragma unroll
        // for (u32 i = tidx; i < KS * BN; i += NUM_THREADS_PER_BLOCK) {
        //     u32 r = i / BN;
        //     u32 c = i % BN;
        //     u32 br = kk + r;
        //     u32 bc = c0 + c;
        //     tileB[r][c] = (br < K && bc < N) ? B[br * N + bc] : 0;
        // }
        __syncthreads();

        // 2. compute
        for (u32 i = 0; i * blockDim.y < BM; i++) {
            for (u32 j = 0; j * blockDim.x < BN; j++) {
                for (u32 p = 0; p < KS; p++) {
                    regs[i][j] += tileA[threadIdx.y + i * blockDim.y][p] * tileB[p][j * blockDim.x + threadIdx.x];
                }
            }
        }
        __syncthreads();
    }

    // 3. copy out
    for (u32 i = 0; i < REGS_LEN; i++) {
        for (u32 j = 0; j < REGS_LEN; ++j) {
            u32 r = r0 + i * blockDim.y + threadIdx.y; // M 方向
            u32 c = c0 + j * blockDim.x + threadIdx.x; // N 方向
            if (r < M && c < N) {
                C[r * N + c] = regs[i][j];
            }
        }
    }
}

void sgemm_v1_do(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 blockDim{NUM_THREADS_PER_AXIS, NUM_THREADS_PER_AXIS};
    dim3 gridDim{CeilDiv<u32>(N, TILE_LEN), CeilDiv<u32>(M, TILE_LEN)};

    sgemm_v1<<<gridDim, blockDim, 0, nullptr>>>(A, B, C, M, N, K);
}
