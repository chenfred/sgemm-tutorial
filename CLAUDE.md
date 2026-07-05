# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目性质

这是一个**学习 CUDA core 编程**的练习仓库，以单精度矩阵乘（SGEMM）为载体，通过实现多个版本（naive → 优化版）来对比和分析性能。当前已实现 `sgemm_naive` 与 `sgemm_v1` 两版。改动以「新增一个优化版本 + 对比性能」为主要工作模式。

## 关键约定：标准的 M/N/K 语义

本仓库的 GEMM 定义是 **C(M,N) = A(M,K) × B(K,N)**，其中 **K 是收缩（reduce/求和）维度**（cuBLAS 标准约定）。

- `A` 布局 M×K，`B` 布局 K×N，`C` 布局 M×N（见 `src/main.cpp` 的 `bytes_*`、初始化范围与 CPU 参考实现）。
- kernel 内沿 K 方向循环累加（`for (kk = 0; kk < K; kk += TILE)`），每次迭代加载一块 A 的行片段（M×K）与一块 B 的列片段（K×N）到 shared memory。
- grid 配置：`gridDim.x = CeilDiv(N, tile)`（C 的列），`gridDim.y = CeilDiv(M, tile)`（C 的行）。

**修改任何 kernel 前必须重新核对此语义**——边界判断、tile 加载、grid 维度、host 侧 `bytes`/初始化范围都依赖 M/N/K 各自的几何含义。改语义时务必把「矩阵大小」出现的每一处（bytes、`malloc` 后的初始化循环、CPU 参考、kernel 索引）同步改到，漏掉初始化范围会缓冲区溢出。

## 添加新 kernel 版本的开发循环

仓库围绕「新增版本对比」组织，CMake 用 `file(GLOB ... CONFIGURE_DEPENDS src/*.cpp src/*.cu)` 自动收集源码，因此新增一个版本**无需改 CMake**：

1. 新建 `src/sgemm_vN.cu`：实现 `__global__` kernel + host 封装 `sgemm_vN_do(const float* A, const float* B, float* C, int M, int N, int K)`。
2. 在 `include/sgemm_func.h` 中加上 `sgemm_vN_do` 的声明。
3. 在 `src/main.cpp` 的 `main()` 里加一行 `test_sgemm(M, N, K, sgemm_vN_do, "sgemm_vN");`。
4. `scripts/build.sh --run` 编译并跑对比；`scripts/profile.sh -o sgemm_vN` 生成 ncu 报告。

`main.cpp` 用 `using sgemm_func_t = std::function<void(const float*, const float*, float*, int, int, int)>;` 统一所有 kernel 的签名，单个 `test_sgemm()` 同时负责计时（`CudaTimer`）、GFLOPS 计算、CPU 三重循环参考校验。

## 常用命令

```bash
scripts/build.sh              # 增量配置 + 编译，产物 build/sgemm
scripts/build.sh --clean      # 删 build/ 后从零配置编译
scripts/build.sh --run        # 编译后运行，输出 time/GFLOPS/PASS|FAIL
scripts/profile.sh -o sgemm_vN   # Nsight Compute 全量报告 → report/sgemm_vN.ncu-rep
```

也可直接 `cmake -S . -B build && cmake --build build`。脚本均通过 `SCRIPT_DIR`/`PROJECT_ROOT` 实现任意目录可执行。

**没有独立的 test 框架**——测试逻辑内置于 `main.cpp`：GPU 输出与 CPU 参考逐元素比较，绝对误差 `< 1e-3` 判 PASS。默认跑两组尺寸：`1024×4096×1024`（可被 tile 整除）与 `1000×2000×1500`（非方阵 M≠N≠K 且不能被 tile size 整除，用于检验边界）。新增版本时保留这两类尺寸。

## 编译目标与工具链

- `CMakeLists.txt` 锁定 **`CMAKE_CUDA_ARCHITECTURES=120`**（Blackwell / RTX 50 系）。换 GPU 时必须改这里。
- C++17 / CUDA 17；CMake 生成 `build/compile_commands.json` 供 clangd（`.clangd` 会过滤掉 nvcc 专用参数）。
- CUDA 编译单元额外加 `-lineinfo`，供 ncu 做 源码/SASS 行号映射。
- `report/`（ncu 报告）与 `build/` 已在 `.gitignore`。

## 编码风格（沿用现有代码）

4 空格缩进、不用 tab；kernel 用小写下划线（`sgemm_naive`），host 封装加 `_do` 后缀（`sgemm_naive_do`）；宏全大写（`CUDA_CHECK`）；tile size 等用 `constexpr`（`TILE_SIZE`/`TILE_LEN`）。改 kernel 优先保证边界正确，再谈性能。

## 协作方式

当用户说「请指导我 / 请引导我」完成某任务时，采用**老师式协作**：一步步讲解如何实现、测试、生成报告、分析结果，并等用户实践反馈，不要直接代为跑完全流程并给出结论。
