# Repository Guidelines

## 项目结构与模块组织

本仓库用于练习 CUDA 单精度矩阵乘法（SGEMM）算子实现。

- `src/main.cpp`：主程序、输入数据初始化、GPU kernel 调用、CPU 参考结果校验与性能输出。
- `src/sgemm_naive.cu`：naive SGEMM kernel 实现。
- `src/sgemm_v1.cu`：SGEMM 优化版本实现。
- `include/common_utils.h`：通用 CUDA 检查宏、计时器和辅助函数。
- `include/sgemm_func.h`：SGEMM kernel host 侧调用声明。
- `scripts/build.sh`：CMake 编译脚本，支持 `--clean` 和 `--run`。
- `scripts/profile.sh`：Nsight Compute profiling 脚本，报告输出到 `report/`。
- `CMakeLists.txt`：CMake 构建入口，自动收集 `src/*.cpp` 与 `src/*.cu`。
- `.clangd`：clangd 配置，使用 `build/compile_commands.json` 并过滤 nvcc 专用参数。
- `README.md`：项目目标说明。

当前还没有独立的 `tests/` 或 `assets/` 目录。新增测试时优先使用清晰目录，例如 `tests/`。

## SGEMM 语义约定

本仓库采用标准（cuBLAS）约定：**C(M,N) = A(M,K) × B(K,N)**，其中 **K 是收缩（求和）维度**。

- `A` 布局 M×K，`B` 布局 K×N，`C` 布局 M×N。
- kernel 内沿 K 方向分块累加，每次把 A 的一块行片段与 B 的一块列片段加载进 shared memory。
- grid 配置：`gridDim.x = CeilDiv(N, tile)`（C 的列），`gridDim.y = CeilDiv(M, tile)`（C 的行）。

修改 kernel 时必须据此核对 `M`、`N`、`K` 三个维度的边界与索引含义；若改动语义，host 侧的 `bytes`、初始化范围、CPU 参考实现要一并同步，漏掉会缓冲区溢出。

## 构建、测试与开发命令

使用 CMake 在 `build/` 目录中配置和编译：

```bash
scripts/build.sh
scripts/build.sh --clean
scripts/build.sh --run
```

命令说明：

- `scripts/build.sh`：配置并增量编译，产物位于 `build/sgemm`。
- `scripts/build.sh --clean`：删除 `build/` 后从 0 配置和编译。
- `scripts/build.sh --run`：编译成功后运行默认 SGEMM 测试。
- `scripts/profile.sh -o <name>`：使用 Nsight Compute 生成 `report/<name>.ncu-rep`。

## 编码风格与命名约定

- 使用 C++/CUDA，保持现有简洁风格。
- 缩进使用 4 个空格，不使用 tab。
- CUDA kernel 使用小写加下划线命名，例如 `sgemm_naive`。
- host 侧封装函数使用描述性名称，例如 `sgemm_naive_do`。
- 宏使用全大写，例如 `CUDA_CHECK`。
- 常量使用 `constexpr`，名称可沿用现有全大写风格，例如 naive 版的 `TILE_SIZE`、v1 版的 `TILE_LEN` / `BM` / `BN` / `KS`。

修改 kernel 时优先保证边界条件正确，再考虑性能优化。

## 测试指南

当前测试逻辑内置在 `src/main.cpp` 中：GPU 输出会与 CPU 三重循环参考实现比较，绝对误差 `< 1e-3` 判 PASS。默认跑两组尺寸：`1024×4096×1024`（可被 tile 整除）与 `1000×2000×1500`（非方阵 M≠N≠K 且不能被 tile 整除，用于检验边界）。

开发时建议至少运行：

```bash
scripts/build.sh --run
```

新增 kernel 或优化版本时，应覆盖非方阵尺寸，例如 `M != N != K`，并包含不能被 tile size 整除的尺寸。测试输出应保持明确的 `PASS` / `FAIL` 标记。

## 提交与 Pull Request 规范

提交信息采用中文 Conventional Commits 风格，格式 `<type>(<scope>): <中文描述>`，例如：

```text
feat(v1): 新增寄存器分块优化版本
refactor(scripts): 统一收纳工具脚本
bugfix(naive): 修复 K 方向边界判断
```

常见 type：`feat`、`refactor`、`bugfix`、`docs`、`agent` 等。

PR 应包含：

- 改动目的和主要实现说明。
- 构建与运行命令，以及关键输出结果。
- 若涉及性能优化，说明测试矩阵尺寸、GPU 型号和性能变化。
- 若修复正确性问题，说明触发问题的输入尺寸。

## Codex 工作指引

- 优先阅读现有代码风格后再修改。
- 不要重置或覆盖用户未提交的改动。
- 对 CUDA kernel 的修改必须按「SGEMM 语义约定」核对 `M`、`N`、`K` 三个维度的边界与索引含义。
- 若修改构建系统，保持 `scripts/build.sh` 可用，并同步更新本文档。
- `scripts/*.sh` 应能从任意当前目录执行；新增脚本时参考现有脚本的 `SCRIPT_DIR` / `PROJECT_ROOT` 写法。
- 当用户说“请指导我”“请引导我”完成某项任务时，采用老师式协作：一步步说明如何实现、如何测试、如何生成报告、如何分析结果，并等待用户实践和反馈。不要直接代替用户执行完整流程、分析报告并给出最终结论。
