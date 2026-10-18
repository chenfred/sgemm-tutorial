#include <cstdint>
#include <mma.h>
#include "common_utils.h"

namespace wmma = nvcuda::wmma;

static constexpr int WARP_THREADS = 32;
static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 8;

// TF32 WMMA 入门：一个 block = 一个 warp，计算一个 16×16 的 C tile；A/B/C 均以 float 存储。
static __global__ void sgemm_v5(const float* A, const float* B, float* C, int M, int N, int K) {

    const uint32_t r = blockIdx.y * WMMA_M;
    const uint32_t c = blockIdx.x * WMMA_N;

    // TODO 1：声明 16×16×8 的 A/B fragment（precision::tf32、row_major）与 float accumulator，并清零。
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> bFrag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> cFrag;
    wmma::fill_fragment(cFrag, 0.0);

#pragma unroll
    for (uint32_t k = 0; k < K; k += WMMA_K) {
        // TODO 2：load_matrix_sync 加载 A/B，并用 __float_to_tf32 显式转换输入精度。
        wmma::load_matrix_sync(aFrag, A + r * K + k, K);
        wmma::load_matrix_sync(bFrag, B + k * N + c, N);
        for (uint32_t i = 0; i < aFrag.num_elements; ++i) {
            aFrag.x[i] = wmma::__float_to_tf32(aFrag.x[i]);
        }
        for (uint32_t i = 0; i < bFrag.num_elements; ++i) {
            bFrag.x[i] = wmma::__float_to_tf32(bFrag.x[i]);
        }
        // TODO 3：沿 K 以 WMMA_K 为步长，用 mma_sync 累加，再用 store_matrix_sync 写回 row-major C。
        wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);

        __syncthreads(); //? 要加吗？那前面load以后，mma以前要不要加呢？
    }
    wmma::store_matrix_sync(C + r * N + c, cFrag, N, wmma::mem_row_major);

    // 32 个线程共同执行 WMMA；leading dimension 按元素计，float 行跨度须为 4 的倍数，首地址须 32-byte 对齐。
    // 当前只支持 M/N 为 16 的倍数、K 为 8 的倍数；尾块处理之后再补。
}

void sgemm_v5_do(const float* A, const float* B, float* C, int M, int N, int K) {
    const dim3 blockDim{WARP_THREADS};
    const dim3 gridDim{static_cast<unsigned int>(CeilDiv(N, WMMA_N)), static_cast<unsigned int>(CeilDiv(M, WMMA_M))};

    sgemm_v5<<<gridDim, blockDim>>>(A, B, C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}
