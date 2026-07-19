#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include "sgemm_verify.h"

namespace {

    bool is_close(float output, float golden, float rtol, float atol) {
        if (output == golden) {
            return true;
        }
        if (!std::isfinite(output) || !std::isfinite(golden)) {
            return false;
        }

        return std::fabs(output - golden) <= atol + rtol * std::fabs(golden);
    }

    size_t matrix_elements(int rows, int cols) { return static_cast<size_t>(rows) * static_cast<size_t>(cols); }

} // namespace

std::vector<float> sgemm_golden(const std::vector<float>& A, const std::vector<float>& B, int M, int N, int K) {
    if (M <= 0 || N <= 0 || K <= 0) {
        throw std::invalid_argument("sgemm_golden requires positive M, N and K");
    }
    if (A.size() != matrix_elements(M, K) || B.size() != matrix_elements(K, N)) {
        throw std::invalid_argument("sgemm_golden input size does not match M, N and K");
    }

    std::vector<float> golden(matrix_elements(M, N), 0.0f);

    // 每一行 C 相互独立，按行分配给 OpenMP 线程；K→N 顺序让 B 和 C 都连续访问。
#pragma omp parallel for schedule(static)
    for (int row = 0; row < M; ++row) {
        float* golden_row = golden.data() + static_cast<size_t>(row) * N;
        const float* a_row = A.data() + static_cast<size_t>(row) * K;

        for (int inner = 0; inner < K; ++inner) {
            const float a = a_row[inner];
            const float* b_row = B.data() + static_cast<size_t>(inner) * N;

#pragma omp simd
            for (int col = 0; col < N; ++col) {
                golden_row[col] += a * b_row[col];
            }
        }
    }

    return golden;
}

bool sgemm_verify(const std::vector<float>& output, const std::vector<float>& golden, int M, int N, float rtol,
                  float atol) {
    if (M <= 0 || N <= 0) {
        std::fprintf(stderr, "SGEMM verification failed: M and N must be positive\n");
        return false;
    }
    if (rtol < 0.0f || atol < 0.0f) {
        std::fprintf(stderr, "SGEMM verification failed: rtol and atol must be non-negative\n");
        return false;
    }

    const size_t expected_size = matrix_elements(M, N);
    if (output.size() != expected_size || golden.size() != expected_size) {
        std::fprintf(stderr,
                     "SGEMM verification failed: size mismatch, output=%zu, golden=%zu, expected=%zu for M=%d N=%d\n",
                     output.size(), golden.size(), expected_size, M, N);
        return false;
    }

    std::int64_t mismatch_count = 0;
#pragma omp parallel for reduction(+ : mismatch_count) schedule(static)
    for (std::int64_t index = 0; index < static_cast<std::int64_t>(expected_size); ++index) {
        if (!is_close(output[index], golden[index], rtol, atol)) {
            ++mismatch_count;
        }
    }

    if (mismatch_count == 0) {
        return true;
    }

    std::fprintf(stderr, "SGEMM verification failed: %lld/%zu elements mismatch (rtol=%g, atol=%g)\n",
                 static_cast<long long>(mismatch_count), expected_size, static_cast<double>(rtol),
                 static_cast<double>(atol));

    constexpr size_t MAX_REPORTED_MISMATCHES = 8;
    size_t reported = 0;
    for (size_t index = 0; index < expected_size && reported < MAX_REPORTED_MISMATCHES; ++index) {
        const float actual = output[index];
        const float expected = golden[index];
        if (!is_close(actual, expected, rtol, atol)) {
            const float absolute_error = std::fabs(actual - expected);
            const float tolerance = atol + rtol * std::fabs(expected);
            std::fprintf(
                stderr, "  mismatch[%zu] at (%zu,%zu): output=% .9g, golden=% .9g, abs_error=%g, tolerance=%g\n",
                reported, index / static_cast<size_t>(N), index % static_cast<size_t>(N), static_cast<double>(actual),
                static_cast<double>(expected), static_cast<double>(absolute_error), static_cast<double>(tolerance));
            ++reported;
        }
    }

    return false;
}
