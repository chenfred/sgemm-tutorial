# SGEMM 学习与性能结论

## 当前 v0/v1 的证据

数据来自 `ncu-rep/sgemm.v0v1.0719.ncu-rep`，尺寸为 1024×4096×1024：

| 指标 | v0 | v1 |
|---|---:|---:|
| NCU Duration | 1.683520 ms | 0.705600 ms |
| executed instructions | 472.91 M | 282.76 M |
| shared load SASS | 167.77 M | 67.11 M |
| global load SASS | 16.78 M | 8.39 M |
| issued warp/scheduler/cycle | 0.28 | 0.41 |
| no eligible cycle | 71.50% | 59.08% |
| MIO Throttle | 21.94 | 12.37 |
| FMA pipe elapsed | 9.96% | 22.37% |
| achieved occupancy | 98.67% | 80.08% |
| registers/thread | 40 | 48 |

v1 的核心收益不是提高 occupancy，而是每线程计算 4 个输出后减少线程、地址/循环和 shared-memory 指令，同时让一个 B 值服务四条独立累加链。总指令减少约 40.2%，FMA 利用率和实际发射率提高，即使 occupancy 下降仍快约 2.39 倍。

## Warp、TLP、ILP 的正确认识

- Active warp 只是已驻留；Eligible warp 才表示下一条指令依赖和资源均已就绪；Issued warp 是本周期真正被调度的 warp。
- v0 每 scheduler 有约 11.84 个 active warp，却有 71.50% 周期没有 eligible warp，证明“线程多、occupancy 高”不保证流水线被填满。多个 warp 可能同时堵在 shared/MIO 队列、scoreboard 或 barrier。
- TLP 用其他 warp 隐藏当前 warp 的等待；ILP 用同一 warp 内不同累加链的独立指令隐藏依赖。若共享硬件队列已拥堵，继续增加 TLP 只会增加排队；更有效的是让每次 load 支持更多 FMA。
- Register tiling 不只是“取消连续指令依赖”。它的第一性收益是寄存器数据复用和降低 load/地址指令密度，ILP 是由多个独立累加器自然产生的附加收益。

## v2_1 当前实验结论（2026-07-31）

- `sgemm_trial_v2_1` 固定 `blockDim=32x8`、`TILE_K=32`，用 `REG_TILE_X/Y` 定义每线程二维输出，默认 `4x4`。
- B cooperative load 曾错误依赖 `REG_TILE_Y`：Y 小于 4 时不能完整初始化 32 行 B tile。现在使用独立的 `B_TILE_ROWS_PER_THREAD=TILE_K/TILEBASE_Y=4`，X/Y 的 `1..4` 组合均通过完整输出校验。
- 快速 benchmark 对 `1..4 x 1..4` 做了 20 次 warmup、5 批各 100 launches 的筛选；当前循环结构下 `4x4` 最快，约 17.51 TFLOPS，是 `1x1` 的 3.69 倍。该数据未锁频且不是 NCU 结果，只作为趋势证据。
- Y 的收益显著高于 X：基础 C tile 为 `8x32`，固定 X 时增大 Y 不增加每 block 的 B tile，却让它服务更多输出行并减少纵向 blocks 重复发出的 B global loads。
- v2_1 的计算顺序仍为 `ri -> rj -> t`。它实现了 block 级 global-to-shared 复用，但没有在源码上显式实现 `t -> regA/regB -> ri/rj` 的 shared-to-register 外积复用；因此 4x4 不是最终结论。
- 下一实验 `sgemm_trial_v2_2` 只改变计算循环为 t 外层的小型外积，固定其余布局与参数，再用正确性、多次 benchmark、NCU 的 LDS/FFMA、MIO Throttle、registers、occupancy 和 spill 建立证据链。

## 历史失败实验及保留价值

- 旧 agent 文档分析的是已经删除的早期 v1：当时约 55 registers/thread、7.60e8 executed instructions、occupancy 约 65%，比旧 naive 慢。那些数值不适用于当前 v1；保留的教训是：复杂线程重排和索引可能让非 FMA 指令暴涨，静态上“复用更高”不等于最终更快。
- 已删除的 v2 与 `v2<0>` 曾尝试通过交换/拆分 reduce 循环消除循环间依赖，但 SASS 和性能没有形成有意义差异，推断编译器已经展开并重排独立累加器。以后不要只凭 CUDA 源码顺序判断 ILP，必须检查 SASS、registers、动态指令和真实时间。
- 静态分析用于提出假设，NCU 用于证伪；顶层 Compute/Memory SOL 接近 100% 不等于 FP32 或 DRAM 已满，必须下钻到 L1/TEX、DRAM 和具体 FMA pipeline。

## 下一学习方向

1. 写 v2 前先熟练从 NCU 的 SOL → Scheduler → Warp State → Memory → Occupancy → Source/SASS 建立证据链。
2. 主线优先二维 register tiling，例如每线程 4×4 输出，用 `TM+TN` 个 shared 标量支持 `TM×TN` 个 FMA；目标是继续降低 shared load/FMA 和 MIO Throttle。
3. v2 正确后再做少量 BM/BN/BK/TM/TN 单变量实验，并始终检查 spill、registers、occupancy 和真实时间。
4. 再优化 cooperative/vectorized global load；只有 Long Scoreboard、Barrier 或 load/compute 气泡成为突出问题后，才进入 double buffering/async copy。
5. v1 的 Details 硬件计数虽提示 shared store 约 1.2-way conflict，但 Source/SASS 中三处 shared 访问均为 `Wavefronts Shared = Ideal`、`Excessive = 0`，手工地址映射也确认 store 连续、A load 广播、B load 连续。该计数更可能包含不可归因的 L1TEX 仲裁，不应再把 padding/layout 当作 v2 前实验或主要优化方向。

面向学习者的完整说明以 `docs/ncu-gui-sgemm-analysis.md` 和 `docs/v1-to-v2-learning-roadmap.md` 为准；本文件只保存跨会话决策与易丢失经验。
