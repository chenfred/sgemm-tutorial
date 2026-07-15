#pragma once

#include <functional>

using sgemm_func_t = std::function<void(const float*, const float*, float*, int, int, int)>;

void sgemm_naive_do(const float* A, const float* B, float* C, int M, int N, int K);
void sgemm_v1_do(const float* A, const float* B, float* C, int M, int N, int K);
void sgemm_v2_do(const float* A, const float* B, float* C, int M, int N, int K);
void sgemm_v2_do_woILP(const float* A, const float* B, float* C, int M, int N, int K);
