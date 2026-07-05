#include <cmath>
#include <string>
#include "common_utils.h"
#include "sgemm_func.h"

static void test_sgemm(int M, int N, int K, sgemm_func_t sgemm_func, std::string caseName = "testcase") {
    // 标准 SGEMM 语义：C(M,N) = A(M,K) * B(K,N)
    const size_t bytes_a = M * K * sizeof(float);
    const size_t bytes_b = K * N * sizeof(float);
    const size_t bytes_c = M * N * sizeof(float);

    float* h_A = (float*)malloc(bytes_a);
    float* h_B = (float*)malloc(bytes_b);
    float* h_C = (float*)malloc(bytes_c);
    for (int i = 0; i < M * K; i++)
        h_A[i] = float(i % 100) / 100.0f;
    for (int i = 0; i < K * N; i++)
        h_B[i] = float(i % 100) / 100.0f;

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_a));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_b));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_c));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_a, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_b, cudaMemcpyHostToDevice));

    CudaTimer timer;
    timer.tic();
    sgemm_func(d_A, d_B, d_C, M, N, K);
    float elapsed = timer.toc();

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes_c, cudaMemcpyDeviceToHost));

    // CPU reference: C(M,N) = A(M,K) * B(K,N)
    int errors = 0;
    for (int i = 0; i < M && errors < 10; i++) {
        for (int j = 0; j < N && errors < 10; j++) {
            float sum = 0.0f;
            for (int p = 0; p < K; p++)
                sum += h_A[i * K + p] * h_B[p * N + j];
            float actual = h_C[i * N + j];
            if (fabs(actual - sum) > 1e-3f) {
                printf("[%s]  Mismatch at (%d,%d): actual %f != expected %f\n", caseName.c_str(), i, j, actual, sum);
                errors++;
            }
        }
    }

    float gflops = (2.0f * M * N * K) / (elapsed / 1000.0f) / 1e9f;
    bool pass = (errors == 0);
    printf("[%s]  M=%-5d N=%-5d K=%-5d  time=%8.3f ms  GFLOPS=%8.2f  %s", caseName.c_str(), M, N, K, elapsed, gflops,
           pass ? "PASS" : "FAIL");
    if (!pass)
        printf("[%s]  errors=%d", caseName.c_str(), errors);
    printf("\n");

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
}

int main() {
    // 覆盖：可被 tile 整除的尺寸，以及非方阵(M≠N≠K)且不能被 tile size 整除的尺寸
    int cases[][3] = {
        {1024, 4096, 1024},
        {1000, 2000, 1500},
    };

    for (auto& c : cases) {
        test_sgemm(c[0], c[1], c[2], sgemm_naive_do, "sgemm_naive");
        test_sgemm(c[0], c[1], c[2], sgemm_v1_do, "sgemm_v1");
    }

    printf("\nAll tests done.\n");

    return 0;
}
