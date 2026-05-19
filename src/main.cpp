#include "common_utils.h"

#include <cmath>

void sgemm_kernel_do(const float* A, const float* B, float* C, int M, int N, int K);

static void test_sgemm(int M,int N,int K){
    const size_t bytes_a = M * N * sizeof(float);
    const size_t bytes_b = N * K * sizeof(float);
    const size_t bytes_c = M * K * sizeof(float);

    float *h_A = (float*)malloc(bytes_a);
    float *h_B = (float*)malloc(bytes_b);
    float *h_C = (float*)malloc(bytes_c);
    for (int i = 0; i < M * N; i++) h_A[i] = float(i % 100) / 100.0f;
    for (int i = 0; i < N * K; i++) h_B[i] = float(i % 100) / 100.0f;

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_a));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_b));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_c));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_a, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_b, cudaMemcpyHostToDevice));

    CudaTimer timer;
    timer.tic();
    sgemm_kernel_do(d_A, d_B, d_C, M, N, K);
    float elapsed = timer.toc();

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes_c, cudaMemcpyDeviceToHost));

    // CPU reference
    int errors = 0;
    for (int i = 0; i < M && errors < 10; i++) {
        for (int k = 0; k < K && errors < 10; k++) {
            float sum = 0.0f;
            for (int j = 0; j < N; j++) sum += h_A[i * N + j] * h_B[j * K + k];
            float actual = h_C[i * K + k];
            if (fabs(actual - sum) > 1e-3f) {
                printf("  Mismatch at (%d,%d): %f != %f\n", i, k, actual, sum);
                errors++;
            }
        }
    }

    float gflops = (2.0f * M * N * K) / (elapsed / 1000.0f) / 1e9f;
    bool pass = (errors == 0);
    printf("  M=%-5d N=%-5d K=%-5d  time=%8.3f ms  GFLOPS=%8.2f  %s",
           M, N, K, elapsed, gflops, pass ? "PASS" : "FAIL");
    if (!pass) printf("  errors=%d", errors);
    printf("\n");

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
}

int main(){
    test_sgemm(1024, 1024, 1024);
    printf("\nAll tests done.\n");
    return 0;
}
