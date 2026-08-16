# 当前项目状态

更新时间：2026-08-16。

## 不变量与代码结构

- SGEMM 语义固定为 `C(M,N) = A(M,K) × B(K,N)`；A/B/C 分别为 M×K、K×N、M×N，K 是收缩维度。
- 当前 `IMPLEMENTATIONS` 注册 `sgemm_v2`、`sgemm_trial_v3_3` 与 `sgemm_trial_v3_4`，用于 double buffering
  学习期的正确性对照；v0/v1 暂时注释，其余 trial 源码和 host 声明保留。
- v0：16×16 shared-memory tile，每线程计算一个输出。
- v1：32×32 shared-memory tile，block 为 32×8，每线程沿 M 方向计算 4 个输出。
- `sgemm_trial_v2_1` 已提交：block 为 32×8、K tile 为 32，`REG_TILE_X/Y` 控制二维 thread tile，默认 4×4。B cooperative load 已与 `REG_TILE_Y` 解耦，X/Y 的 1..4 组合全部通过快速完整输出校验；当前计算顺序仍是 `ri -> rj -> t`。
- `sgemm_trial_v2_2` 已实现 `t -> regA/regB -> ri/rj` 外积结构，当前教学参数为 `REG_TILE_X=3`、`REG_TILE_Y=12`，形成 `96x96` block C tile。默认尺寸完整校验 PASS；快速 benchmark 重复结果约 `0.361 ms / 23.8 TFLOPS`，ptxas 为 96 registers/thread、24 KiB static shared、无 spill。固定 `4x4` 时 v2_1/v2_2 性能在约 1% 内，说明显式循环换序本身没有新增收益。
- 正式版 `sgemm_v2` 基于 `sgemm_trial_v2_2`：block 为 32×8、C tile 为 96×96、每线程
  register tile 为 12×3、K tile 默认为 32。A/B cooperative load 均支持 `TILE_K` 分别按
  `TILEBASE_X/Y` 的整数倍扩展；默认尺寸和非整除用例 `127×259×137` 均 PASS，临时设为
  `TILE_K=64` 的同一非整除用例也 PASS，验证后已恢复 32。
- `sgemm_trial_v3_1/v3_2` 构成 global-to-shared vectorization 的单变量对照：两者使用相同的
  展平 chunk 映射、边界语义、shared layout 和计算循环；v3_1 每个 4-float chunk 显式执行
  4 次标量 load/store，v3_2 只在完整且 16-byte 对齐时换成 `float4`，否则使用同样的标量
  fallback。默认尺寸和非整除用例 `127×259×137` 均 PASS。SASS 确认 v3_1 copy-in 只有
  `LDG.E/STS`，v3_2 主路径生成 `LDG.E.128/STS.128`；两者分别使用 96/102 registers/thread，
  static shared 均为 25600 bytes。单次应用计时不稳定，不作为向量化收益结论。
- `sgemm_trial_v3_3` 已实现同步 global-to-shared copy 的双 shared stage ping-pong：prologue 准备 stage 0，
  steady state 同步加载 next 后计算 current，barrier 同时保护 next 完成和 current 生命周期，epilogue 计算最后
  一个 stage。它验证 stage/synchronization，但没有明确构造同一 warp 内的 LDG/compute 重叠。
- `sgemm_trial_v3_4` 已实现普通 LDG register-prefetch DB：将 copy 拆成 `global -> prefetch registers`、
  current compute、`prefetch registers -> shared[next]`，并保留双 shared stage 和 block barrier。用户已验证
  正确性且初测性能良好；“无 spill”目前是合理推测，仍待编译资源信息或 NCU 确认。
- `sgemm_trial_v1_1` 恢复自 `df41736` 中的原 `sgemm_v1`：block 为 16×16、每线程计算 2×2
  输出、K tile 为 8。`sgemm_trial_v1_2` 对应原 `sgemm_v2<false>`；它与当前 v1 都采用
  accumulator 外层、K 内层的源码循环顺序。两个 trial 已临时注册并通过 `1024×4096×1024`
  和非整除边界用例 `37×53×29` 的 FP32 正确性校验。
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
- 用户主要在 VS Code 终端或 Windows Terminal 中阅读 Codex 输出；交互与面向终端的说明默认使用纯文本公式，不使用 LaTeX/MathJax，具体约束见根目录 `AGENTS.md`。
- `.claude/` 与 `CLAUDE.md` 已在提交 `07bdb05` 中清理并由用户验收，后续不要重新创建 Claude 专用配置或重复文档。
