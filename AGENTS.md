# Repository Guidelines

## 项目结构与模块组织

本仓库用于练习 CUDA 单精度矩阵乘法（SGEMM）算子实现。

- `src/main.cpp`：主程序、输入数据初始化、GPU kernel 调用、正确性校验编排与性能输出。
- `src/verify.cpp`：OpenMP 多线程 CPU golden 计算与混合容差校验。
- `src/warmup.cu`：正式 kernel 前的 GPU warmup；配合 Application Replay 覆盖每个 NCU 采集 pass。
- `src/sgemm_v0.cu`：16×16 shared-memory tiling 基线版本。
- `src/sgemm_v1.cu`：32×32 shared-memory tiling、每线程计算 4 个输出的优化版本。
- `include/common_utils.h`：通用 CUDA 检查宏、计时器和辅助函数。
- `include/sgemm_func.h`：SGEMM kernel host 侧调用声明。
- `include/sgemm_verify.h`：CPU golden/verify 声明与 FP32 默认容差。
- `scripts/build.sh`：CMake 编译脚本，支持 `--clean` 和 `--run`。
- `scripts/profile.sh`：Nsight Compute profiling 脚本，报告输出到 `ncu-rep/`。
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
- `scripts/build.sh --run [PROGRAM_ARG...]`：编译成功后运行 SGEMM，并把后续参数传给程序。
- `scripts/profile.sh -o <name>`：使用 Nsight Compute 生成 `ncu-rep/<name>.ncu-rep`。

## 编码风格与命名约定

- 使用 C++/CUDA，保持现有简洁风格。
- 缩进使用 4 个空格，不使用 tab。
- `for`、`if` 等控制流即使只有单条语句也必须使用花括号。
- CUDA kernel 使用小写加下划线命名，例如 `sgemm_v0`。
- host 侧封装函数使用描述性名称，例如 `sgemm_v0_do`。
- 宏使用全大写，例如 `CUDA_CHECK`。
- 常量使用 `constexpr`，名称沿用现有全大写风格，例如 `TILE_SIZE`、`BLOCK_SIZE_X`、`OUTPUTS_PER_THREAD`。

修改 kernel 时优先保证边界条件正确，再考虑性能优化。

## 测试指南

当前测试由 `src/main.cpp` 编排，CPU 参考计算与校验位于 `src/verify.cpp`。GPU 输出按 `|out-golden| <= atol + rtol*|golden|` 比较，FP32 默认 `rtol=1.3e-6`、`atol=1e-5`。为简化 v0/v1 的 NCU 报告，当前默认只跑 `1024×4096×1024`。

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
