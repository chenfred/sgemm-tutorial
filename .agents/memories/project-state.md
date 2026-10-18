# 当前项目状态

更新时间：2026-10-18。

本地代码基线为提交 `83b6fae`，用户随后完成 v5 WMMA 入门实现。2026-10-18 构建和 clang-format 检查通过；
本机默认 v0～v5 在 `1024×4096×1024` 全部 PASS。用户已采集下述 NCU 报告；历史资源/性能数据需按机器与输入配置区分。

## 不变量与代码结构

- SGEMM 语义固定为 `C(M,N) = A(M,K) × B(K,N)`；A/B/C 分别为 M×K、K×N、M×N，K 是收缩维度。
- 当前 `IMPLEMENTATIONS` 按顺序注册 `sgemm_v0` 至 `sgemm_v5`，共用一个测试路径，CLI 只保留 `--dry-run`。
- 用户已完成 v5 的 fragment、global load、TF32 转换、mma 与 global store：一个 block=32 threads，
  WMMA shape=16×16×8，FP32 累加，host 接口仍为 float。A/B 均为 row-major，输入从 global 直接加载至 fragment，
  当前没有 shared tiling/DB。仅支持 M/N 为 16 的倍数、K 为 8 的倍数；没有尾块处理。
- 按用户要求删除了 `wmma_learning`、`--wmma`、专用实现列表/用例与 NaN 预填充；普通入口直接对比 v0～v5。
  统一输入改为 `(i % 100) / 128.0f`，可由 TF32 精确表示，沿用既有 FP32 容差；整除不等于能消除输入量化误差。
  v5 是用户亲自学习的实现，后续优先静态指导，不自动代写优化。当前循环中仍保留多余的 `__syncthreads()`，
  已说明纯 global→fragment→MMA 路径在 load/mma 前后都不需要额外 block barrier，留给用户自行清理。
- 正式 v4 按用户要求从 trial_v4_1 复制，仅改 kernel/wrapper 名和版本注释；保留两个 trial 的源码与声明，
  默认不运行 trial。教学主线为 v2 register tiling -> v3 LDG 预取双缓冲 -> v4 cp.async 双缓冲。
- 正式 v4 / `sgemm_trial_v4_1` 按用户要求采用少量 inline PTX 入门：沿用 v3 的 32×8 block、96×96 C tile、
  12×3 register tile、K tile=32、双 shared stage（48 KiB）和计算循环，
  以 4-byte `cp.async.ca.shared.global` 替换 prefetch registers + STS；每轮 issue/commit 后计算 current，
  再 wait_group 0 + block barrier。尾块用普通 shared store 补零，不构造越界 global 指针。
- `sgemm_trial_v4_2` 当前固定 `STAGES=2`：先预填满，主循环每轮计算一块，再向原 stage 补入
  `computeTile+STAGES`。消费前 remaining>=2 时 wait_group 1，否则 wait_group 0；就绪和复用使用分开的
  block barrier，无独立计算 epilogue。该版本曾试过三 stage，但当前源码不能仅修改 STAGES 就自动适配等待逻辑。
- 已学习 cp.async/commit/wait、每线程分组和 memory clobber。用户试过 barrier -> wait 且数次 PASS，
  但该顺序缺少跨线程完成保证；当前正式 v4 / v4_1 已恢复 wait -> barrier。细节见 learning-notes.md。
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
- 正式 `sgemm_v3` 由 `sgemm_trial_v3_4` 转正：将 copy 拆成 `global -> prefetch registers`、current compute、
  `prefetch registers -> shared[next]`，并保留双 shared stage 和 block barrier。默认尺寸和多组非整除尺寸均
  PASS；重复的 trial_v3_4 源码与声明已删除。v3 是 v4 的普通 LDG/STS 对照基线。
- `sgemm_trial_v1_1` 恢复自 `df41736` 中的原 `sgemm_v1`：block 为 16×16、每线程计算 2×2
  输出、K tile 为 8。`sgemm_trial_v1_2` 对应原 `sgemm_v2<false>`；它与当前 v1 都采用
  accumulator 外层、K 内层的源码循环顺序。两个 trial 已临时注册并通过 `1024×4096×1024`
  和非整除边界用例 `37×53×29` 的 FP32 正确性校验。
- `src/warmup.cu` 提供独立 warmup；`src/verify.cpp` 提供 OpenMP 多线程 CPU golden 与 verify。
- 默认只运行 `1024×4096×1024`，以保持 NCU 报告简单。开发新 kernel 时必须临时加入至少一个 M/N/K 不相等且不能被 tile 整除的用例。

## v5 NCU 报告结论（2026-10-18，RTX 5080）

- 报告 `ncu-rep/sgemm.v1v2v3v4v5.ncu-rep` 实际包含 v0～v5，尺寸为 `1024×4096×1024`；
  NCU 2026.1.1、Application Replay、clock-control none，已嵌入源码。此次仅读取既有报告，没有重新采集或修改 kernel。
- Duration：v1 703.904 us、v2 339.392 us、v3 328.192 us、v4 412.160 us、v5 540.032 us。
  v5 仅快于 v0/v1，慢于 v2～v4，不能描述为已经超越优化后的 CUDA Core 实现。
- v5 L2 Throughput 90.55% 的最大组成项是 tag requests；sector throughput 约 51.46%，DRAM Throughput 5.54%。
  主因应描述为 L2 请求处理压力与 global load 等待，不能把 90.55% 当成 L2 字节带宽或 DRAM 饱和。
- v5 逻辑 global A/B 读取共 2 GiB，v2～v4 共 348 MiB，约 5.89 倍；L1 global-load sector 请求量为
  3 GiB 对 348 MiB，约 8.83 倍，说明 v5 还有访存事务放大。上述均不等于 DRAM 流量。
  缓存后 L2→L1 read bytes 为 1,380,490,336 对 364,904,448，约 3.78 倍。
- v5 一 warp/block 被 24 blocks/SM 限制，理论 occupancy 50%，实测 48.49%；No Eligible 79.58%。
  报告已确认 HMMA TF32 路径以及循环中的 BAR；barrier 等待占比仅约 0.32%，不是主要性能解释。
- Tensor active cycles 56.84% 不等于算术峰值利用率；TF32 ops 的 pct-of-peak 为 12.54%。
  Source 中 TF32 转换位置的大量 long-scoreboard 采样是等待前序 LD 的寄存器结果，不能据此认定转换算术慢。
- 两个 subagent 的报告解读与静态流量模型已经互相核对，详细指标、口径和复查命令保存于
  `.agents/analysis/sgemm-v5-ncu-findings.md`；当前 SM 频率差仅约 1.7%，不解释主要性能差距。
- 保留 v5 为 WMMA 入门基线；下一实验优先多 warp/block + shared tiling（例如 4 warps/32×32 C tile），
  显式复用 A/B、合并搬运，再考虑接回 cp.async。仅把多个独立 warp 打包进 block 不会自动减少重复读取。

## v4 历史实验与待验证项

- 2026-09 的实验机器是 RTX 3060 Laptop / sm_86、NVCC 13.2.78，与当前本机不同。
- v4_1 的默认尺寸及 `37×53×13`、`70×131×32`、`127×259×45`、`191×257×97` 全部 PASS，SASS 有 LDGSTS。
  v3/v4_1 均为 128 registers/thread、无 spill；一次计时约 1.680/2.203 ms，不构成稳定快慢结论。
- v4_2 双 stage 为 48 KiB shared、100 registers/thread、无 spill；默认尺寸七次 PASS，尚未重跑边界集。
  v4_1/v4_2 时间中位数 2.296/2.251 ms，约差 2%；未锁频、版本顺序固定且波动明显，视为接近。
- v4_2 历史三 stage 为 72 KiB shared、80 registers/thread、无 spill；默认和七组边界 PASS。
  曾出现明显退化，但没有稳定证据确定主因；该配置已改回双 stage，不将三 stage 的边界结果视为当前版本复测。
- memcheck 曾因 WDDM debugger interface 初始化失败，未完成有效检查，racecheck 也未执行。
  这是另一台机器的工具限制，不能直接认定当前 RTX 5080 也无法检查。
- 当前 RTX 5080 的 v4 默认尺寸已在接入 v5 骨架后重测 PASS；非整除尺寸、资源、sanitizer 和稳定 NCU 对照未重测。

## 工具链与构建

- 2026-10-18 只读核验：本机 RTX 5080 / Compute Capability 12.0、NVCC 13.2.86、NCU 2026.1.1。
  沙箱内 NVML 报 GPU access blocked，沙箱外 nvidia-smi 查询正常。
- 当前 CMake 配置 `CMAKE_CUDA_ARCHITECTURES=86;120`，同时覆盖两台机器；不要误恢复成仅 86 或仅 120。
  历史上仅编译 sm_120 曾让 RTX 3060 报 no kernel image。
- CMake 默认使用 Release，host C++ 链接 OpenMP，CUDA 源文件保留 `-lineinfo -g` 供 NCU 源码/SASS 关联。
- 仓库已有兼容 clang-format 14 的 `.clang-format`，后续修改 C++/CUDA 直接按仓库配置格式化，无需早期临时兼容配置。
- 常用入口：`scripts/build.sh [--clean] [--run [PROGRAM_ARG...]]`。

## 正确性与执行模式

- 普通模式先调用一次 `sgemm_golden`，再逐 kernel D2H 并调用 `sgemm_verify`。
- FP32 混合容差采用 `|out-golden| <= atol + rtol*|golden|`，默认 `rtol=1.3e-6`、`atol=1e-5`；Inf 只有完全相等才通过，NaN 默认失败。
- verify 并行统计全部错误；失败时再顺序扫描，稳定地只打印前 8 个错误位置和值。
- `--dry-run` 仍执行数据生成、H2D、warmup、正式 kernel 和 D2H，但跳过 CPU golden/verify 与应用侧计时输出。
  普通运行使用 `scripts/build.sh --run`；已无 WMMA 专用参数。
- 正式 kernel 后显式 `cudaDeviceSynchronize()`，再调用 `cudaProfilerStop()`；不要依赖 Stop、D2H 或 cudaFree 的隐式同步。

## Profiling 工作流与历史基线

- 当前脚本使用 `--set full --import-source yes --clock-control none --profile-from-start off --replay-mode application`。
  通过 profiler Start/Stop 采集 `IMPLEMENTATIONS` 中的 kernel（当前 v0～v5），不限定 v0/v1 名字；每个 pass 各自重新 warmup。
- 历史命令：`scripts/profile.sh -o sgemm.v0v1.0719`；对应报告 `ncu-rep/sgemm.v0v1.0719.ncu-rep`
  被 gitignore，仅本地保存。当前默认实现已变多，不能假定重跑该命令仍只采 v0/v1。
- 2026-07-19 报告共 40 个 Application Replay pass：v0/v1 SM 频率为 2.949968/2.936141 GHz，差约 0.47%；DRAM 频率差约 0.0018%；Duration 为 1.683520/0.705600 ms，v1 约快 2.39 倍。

## 工作区注意事项

- 正式版快速复习说明为 `docs/sgemm_v0.md` 至 `docs/sgemm_v4.md`，逐版对应源码的切分、搬运、计算、同步和边界。
  `docs/README.md` 提供版本参数总览并区分早期 NCU 材料；根 README 已链接，后续改正式版时同步维护这些说明。

- `draft.md` 是用户维护的任务草稿；发现未提交修改时不得覆盖或清理。
- `build/`、`ncu-rep/` 是本地产物。不要把二进制报告当作可提交的项目记忆。
- 仓库级指引继续使用根目录 `AGENTS.md`，以符合 Codex 默认发现规则；不要移动到 `.agents/AGENTS.md`。除此入口外，agent 记忆、计划、交接和内部分析统一放在 `.agents/`。
- 用户主要在 VS Code 终端或 Windows Terminal 中阅读 Codex 输出；交互与面向终端的说明默认使用纯文本公式，不使用 LaTeX/MathJax，具体约束见根目录 `AGENTS.md`。
- `.claude/` 与 `CLAUDE.md` 已在提交 `07bdb05` 中清理并由用户验收，后续不要重新创建 Claude 专用配置或重复文档。
