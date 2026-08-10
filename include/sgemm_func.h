#pragma once

#include <functional>

using sgemm_func_t = std::function<void(const float*, const float*, float*, int, int, int)>;

void warmup_do(const float* A, const float* B, float* C, int M, int N, int K);
void sgemm_v0_do(const float* A, const float* B, float* C, int M, int N, int K);
void sgemm_v1_do(const float* A, const float* B, float* C, int M, int N, int K);
void sgemm_v2_do(const float* A, const float* B, float* C, int M, int N, int K);
void sgemm_trial_v1_1_do(const float* A, const float* B, float* C, int M, int N, int K);
void sgemm_trial_v1_2_do(const float* A, const float* B, float* C, int M, int N, int K);
void sgemm_trial_v2_1_do(const float* A, const float* B, float* C, int M, int N, int K);
void sgemm_trial_v2_2_do(const float* A, const float* B, float* C, int M, int N, int K);
void sgemm_trial_v3_1_do(const float* A, const float* B, float* C, int M, int N, int K);
void sgemm_trial_v3_2_do(const float* A, const float* B, float* C, int M, int N, int K);
