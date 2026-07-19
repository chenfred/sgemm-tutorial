# 当前项目状态

更新时间：2026-07-19。

## 不变量与代码结构

- SGEMM 语义固定为 `C(M,N) = A(M,K) × B(K,N)`；A/B/C 分别为 M×K、K×N、M×N，K 是收缩维度。
- 当前教学基线只保留 `sgemm_v0` 与 `sgemm_v1`。新增版本需要在 `include/sgemm_func.h` 声明，并注册到 `src/main.cpp` 的 `IMPLEMENTATIONS`。
- v0：16×16 shared-memory tile，每线程计算一个输出。
- v1：32×32 shared-memory tile，block 为 32×8，每线程沿 M 方向计算 4 个输出。
- `src/warmup.cu` 提供独立 warmup；`src/verify.cpp` 提供 OpenMP 多线程 CPU golden 与 verify。
- 默认只运行 `1024×4096×1024`，以保持 NCU 报告简单。开发新 kernel 时必须临时加入至少一个 M/N/K 不相等且不能被 tile 整除的用例。

## 工具链与构建

- 环境是 Ubuntu 22.04 / WSL2、RTX 5080、Compute Capability 12.0（84 SM）。
- 当前工具链为 CUDA 13.2.2、NVCC 13.2.86、Nsight Compute 2026.1.1；CMake 锁定 `CMAKE_CUDA_ARCHITECTURES=120`。
- CMake 默认使用 Release，host C++ 链接 OpenMP，CUDA 源文件保留 `-lineinfo -g` 供 NCU 源码/SASS 关联。
- 常用入口：`scripts/build.sh [--clean] [--run [PROGRAM_ARG...]]`。

## 正确性与执行模式

- 普通模式先调用一次 `sgemm_golden`，再逐 kernel D2H 并调用 `sgemm_verify`。
- FP32 混合容差采用 `|out-golden| <= atol + rtol*|golden|`，默认 `rtol=1.3e-6`、`atol=1e-5`；Inf 只有完全相等才通过，NaN 默认失败。
- verify 并行统计全部错误；失败时再顺序扫描，稳定地只打印前 8 个错误位置和值。
- `--dry-run` 仍执行数据生成、H2D、warmup、正式 kernel 和 D2H，但跳过 CPU golden/verify 与应用侧计时输出。
- 正式 kernel 后显式 `cudaDeviceSynchronize()`，再调用 `cudaProfilerStop()`；不要依赖 Stop、D2H 或 cudaFree 的隐式同步。

## Profiling 基线

- 命令：`scripts/profile.sh -o sgemm.v0v1.0719`。
- 报告：`ncu-rep/sgemm.v0v1.0719.ncu-rep`（被 gitignore，只在本地保存）。
- 脚本使用 `--set full --clock-control none --profile-from-start off --replay-mode application`，只采集 profiler Start/Stop 内的具名 v0/v1 kernel。
- 2026-07-19 报告共 40 个 Application Replay pass：v0/v1 SM 频率为 2.949968/2.936141 GHz，差约 0.47%；DRAM 频率差约 0.0018%；Duration 为 1.683520/0.705600 ms，v1 约快 2.39 倍。

## 工作区注意事项

- `draft.md` 是用户维护的任务草稿；发现未提交修改时不得覆盖或清理。
- `build/`、`ncu-rep/` 是本地产物。不要把二进制报告当作可提交的项目记忆。
- 仓库级指引继续使用根目录 `AGENTS.md`，以符合 Codex 默认发现规则；不要移动到 `.agents/AGENTS.md`。除此入口外，agent 记忆、计划、交接和内部分析统一放在 `.agents/`。
- `.claude/` 与 `CLAUDE.md` 已在提交 `07bdb05` 中清理并由用户验收，后续不要重新创建 Claude 专用配置或重复文档。
