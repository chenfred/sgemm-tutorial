#include <cstdio>
#include <cuda_profiler_api.h>
#include <string_view>
#include <vector>
#include "common_utils.h"
#include "sgemm_func.h"
#include "sgemm_verify.h"

struct SgemmImplementation {
    const char* name;
    sgemm_func_t func;
};

struct SgemmResult {
    float* d_C = nullptr;
    float elapsed = 0;
};

static const std::vector<SgemmImplementation> IMPLEMENTATIONS = {
    // {"sgemm_v0", sgemm_v0_do},
    // {"sgemm_v1", sgemm_v1_do},
    {"sgemm_v2", sgemm_v2_do},
    {"sgemm_v3", sgemm_v3_do},
    {"sgemm_trial_v4_1", sgemm_trial_v4_1_do},
};

static void test_sgemm(int M, int N, int K, bool dry_run) {
    // 标准 SGEMM 语义：C(M,N) = A(M,K) * B(K,N)
    const size_t bytes_a = static_cast<size_t>(M) * K * sizeof(float);
    const size_t bytes_b = static_cast<size_t>(K) * N * sizeof(float);
    const size_t bytes_c = static_cast<size_t>(M) * N * sizeof(float);

    std::vector<float> h_A(static_cast<size_t>(M) * K);
    std::vector<float> h_B(static_cast<size_t>(K) * N);
    std::vector<float> h_C(static_cast<size_t>(M) * N);
    for (size_t i = 0; i < h_A.size(); ++i) {
        h_A[i] = static_cast<float>(i % 100) / 100.0f;
    }
    for (size_t i = 0; i < h_B.size(); ++i) {
        h_B[i] = static_cast<float>(i % 100) / 100.0f;
    }

    std::vector<float> golden;
    if (!dry_run) {
        golden = sgemm_golden(h_A, h_B, M, N, K);
    }

    float *d_A, *d_B, *d_warmup;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_a));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_b));
    CUDA_CHECK(cudaMalloc(&d_warmup, bytes_c));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_a, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes_b, cudaMemcpyHostToDevice));

    std::vector<SgemmResult> results(IMPLEMENTATIONS.size());
    for (auto& result : results) {
        CUDA_CHECK(cudaMalloc(&result.d_C, bytes_c));
    }

    CudaTimer timer;

    // 每个 Application Replay pass 都会重新运行 warmup；只在正式 kernel 期间启用 profiler。
    for (size_t i = 0; i < IMPLEMENTATIONS.size(); ++i) {
        warmup_do(d_A, d_B, d_warmup, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());

        if (!dry_run) {
            timer.tic();
        }
        CUDA_CHECK(cudaProfilerStart());
        IMPLEMENTATIONS[i].func(d_A, d_B, results[i].d_C, M, N, K);
        if (!dry_run) {
            results[i].elapsed = timer.toc();
        }
        // 不依赖 cudaProfilerStop、D2H 或 cudaFree 的隐式行为，明确等待正式 kernel 完成。
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaProfilerStop());
    }

    for (size_t implementationIndex = 0; implementationIndex < IMPLEMENTATIONS.size(); ++implementationIndex) {
        CUDA_CHECK(cudaMemcpy(h_C.data(), results[implementationIndex].d_C, bytes_c, cudaMemcpyDeviceToHost));

        if (!dry_run) {
            const bool pass = sgemm_verify(h_C, golden, M, N);
            float elapsed = results[implementationIndex].elapsed;
            float gflops = (2.0f * M * N * K) / (elapsed / 1000.0f) / 1e9f;
            printf("[%s]  M=%-5d N=%-5d K=%-5d  time=%8.3f ms  GFLOPS=%8.2f  %s",
                   IMPLEMENTATIONS[implementationIndex].name, M, N, K, elapsed, gflops, pass ? "PASS" : "FAIL");
            printf("\n");
        }
    }

    for (auto& result : results) {
        CUDA_CHECK(cudaFree(result.d_C));
    }
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_warmup));
}

int main(int argc, char** argv) {
    const bool dry_run = argc == 2 && std::string_view(argv[1]) == "--dry-run";
    if (argc > 1 && !dry_run) {
        printf("Usage: %s [--dry-run]\n", argv[0]);
        return 1;
    }

    // 当前仅保留用于对比 v0/v1 NCU 报告的基准尺寸。
    int cases[][3] = {
        {1024, 4096, 1024},
    };

    for (auto& c : cases) {
        test_sgemm(c[0], c[1], c[2], dry_run);
    }

    if (!dry_run) {
        printf("\nAll tests done.\n");
    }

    return 0;
}
