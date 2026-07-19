#pragma once

#include <vector>

// 采用 PyTorch torch.testing.assert_close 对 FP32 的默认混合容差。
inline constexpr float FP32_RTOL = 1.3e-6f;
inline constexpr float FP32_ATOL = 1.0e-5f;

std::vector<float> sgemm_golden(const std::vector<float>& A, const std::vector<float>& B, int M, int N, int K);

bool sgemm_verify(const std::vector<float>& output, const std::vector<float>& golden, int M, int N, float rtol = FP32_RTOL,
                  float atol = FP32_ATOL);
