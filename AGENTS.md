# Repository Guidelines

## 项目结构与模块组织

本仓库用于练习 CUDA 单精度矩阵乘法（SGEMM）算子实现。

- `src/main.cpp`：主程序、输入数据初始化、GPU kernel 调用、CPU 参考结果校验与性能输出。
- `src/kernel.cu`：CUDA SGEMM kernel 实现与 launch 封装。
- `include/common_utils.h`：通用 CUDA 检查宏、计时器和辅助函数。
- `CMakeLists.txt`：CMake 构建入口，自动收集 `src/*.cpp` 与 `src/*.cu`。
- `.clangd`：clangd 配置，使用 `build/compile_commands.json` 并过滤 nvcc 专用参数。
- `README.md`：项目目标说明。

当前还没有独立的 `tests/` 或 `assets/` 目录。新增测试或脚本时，优先使用清晰目录，例如 `tests/`、`scripts/`。

## 构建、测试与开发命令

使用 CMake 在 `build/` 目录中配置和编译：

```bash
cmake -S . -B build
cmake --build build
./build/sgemm
```

命令说明：

- `cmake -S . -B build`：生成构建文件和 `build/compile_commands.json`。
- `cmake --build build`：编译 `sgemm`，产物位于 `build/sgemm`。
- `./build/sgemm`：运行默认 `1024 x 1024 x 1024` SGEMM 测试，输出耗时、GFLOPS 和 PASS/FAIL。

## 编码风格与命名约定

- 使用 C++/CUDA，保持现有简洁风格。
- 缩进使用 4 个空格，不使用 tab。
- CUDA kernel 使用小写加下划线命名，例如 `sgemm_naive`。
- host 侧封装函数使用描述性名称，例如 `sgemm_kernel_do`。
- 宏使用全大写，例如 `CUDA_CHECK`。
- 常量使用 `constexpr`，名称可沿用现有全大写风格，例如 `NAIVE_TILE_SIZE`。

修改 kernel 时优先保证边界条件正确，再考虑性能优化。

## 测试指南

当前测试逻辑内置在 `src/main.cpp` 中：GPU 输出会与 CPU 三重循环参考实现比较。

开发时建议至少运行：

```bash
cmake --build build
./build/sgemm
```

新增 kernel 或优化版本时，应覆盖非方阵尺寸，例如 `M != N != K`，并包含不能被 tile size 整除的尺寸。测试输出应保持明确的 `PASS` / `FAIL` 标记。

## 提交与 Pull Request 规范

当前 git 历史没有提交记录，因此尚无既有提交格式。建议使用简短、动词开头的提交信息：

```text
Add tiled SGEMM kernel
Fix SGEMM output bounds check
Add CMake build configuration
```

PR 应包含：

- 改动目的和主要实现说明。
- 构建与运行命令，以及关键输出结果。
- 若涉及性能优化，说明测试矩阵尺寸、GPU 型号和性能变化。
- 若修复正确性问题，说明触发问题的输入尺寸。

## Codex 工作指引

- 优先阅读现有代码风格后再修改。
- 不要重置或覆盖用户未提交的改动。
- 对 CUDA kernel 的修改必须检查 `M`、`N`、`K` 三个维度的边界含义。
- 若修改构建系统，保持 `cmake -S . -B build && cmake --build build` 可用，并同步更新本文档。
