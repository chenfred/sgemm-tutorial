# SGEMM 学习与性能结论

## 当前学习策略

- 默认先广度、后深度：优先实践典型 CUDA kernel 语法、优化套路和可复用能力；正确性、核心机制和基本性能证据成立后即进入下一主题。
- 参数解耦/穷举、极限微调和逐条 SASS 考古先记录后延；仅在正确性、spill、下一主题受阻或结果严重违背预期时提前深挖。
- 当前 `BX=32,BY=8,BK=32` 与 `TM=12,TN=3` 已足以完成二维 register tiling 学习，不继续搜索 `BX/BY/BK/TM/TN`；待典型优化模式大致实践一遍后再回到系统调参与自动搜索。

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

## v2_2 Outer Product 与 tile 扫描（2026-08-09）

- 正式版 `sgemm_v2` 已从 v2_2 收敛：固定 `BX=32,BY=8,BK=32,TM=12,TN=3`，形成 256-thread、96×96 block tile。`BK` 在概念和加载代码上已与 `BX` 解耦，但用 `BK % BX == 0`、`BK % BY == 0` 保持 cooperative load 规整；不继续做参数搜索。
- v2_2 已显式实现 `k -> regA/regB -> ri/rj` outer product。固定 `4x4` 时，多轮交错 benchmark 得到 v2_1 约 `0.516 ms`、v2_2 约 `0.520 ms`，差异在约 1% 内；ptxas 分别使用 56/48 registers，无 spill。这说明 v2_1 的源码循环顺序已被编译器优化成效果接近的机器指令，显式换序本身没有形成新性能收益。
- 在 `M=1024,N=4096,K=1024`、随机输入、30 次 warmup、9 批各 100 launches 的快速扫描中，v2_2 的代表结果为：`1x4=0.762 ms`、`2x4=0.615 ms`、`4x4=0.528 ms`、`2x8=0.436 ms`、`3x8=0.485 ms`、`3x12=0.369 ms`、`4x16=0.557 ms`。各配置抽样校验 PASS，当前 `3x12` 的重复结果稳定在 `0.360～0.362 ms`，约 23.8 TFLOPS，相对同进程 v2_1 快约 1.43 倍。
- `REG_TILE_Y=4*REG_TILE_X` 同时让 block C tile 成为正方形，并在理想模型中平衡 A/B 的 global-to-shared 重复加载：`A loads/output` 约为 `1/X`，`B loads/output` 约为 `4/Y`。在编译器用 `LDS.128` 搬 A、标量 LDS 搬 B 的当前代码生成下，它也近似平衡两侧 shared-load SASS 数量。
- `3x12` 是复用和资源代价的当前拐点：36 accumulators/thread、96 registers/thread、24 KiB static shared、无 spill。继续到 `4x16` 虽进一步降低理想 load/FMA，但增加到 64 accumulators、128 registers、32 KiB shared，且 grid 仅 256 blocks，低并行度、尾波、长 live range 和更大的展开代码抵消收益。
- 以上是未锁频、输入驻留后反复 launch 的快速 benchmark；使用同进程 v2_1 交错计时和多轮重复控制频率漂移，适合作为当前尺寸的参数选择，但正式跨版本结论仍应由简化 NCU 指标验证。

## 历史失败实验及保留价值

- 旧 agent 文档分析的是已经删除的早期 v1：当时约 55 registers/thread、7.60e8 executed instructions、occupancy 约 65%，比旧 naive 慢。那些数值不适用于当前 v1；保留的教训是：复杂线程重排和索引可能让非 FMA 指令暴涨，静态上“复用更高”不等于最终更快。
- 已删除的 v2 与 `v2<0>` 曾尝试通过交换/拆分 reduce 循环消除循环间依赖，但 SASS 和性能没有形成有意义差异，推断编译器已经展开并重排独立累加器。以后不要只凭 CUDA 源码顺序判断 ILP，必须检查 SASS、registers、动态指令和真实时间。
- 静态分析用于提出假设，NCU 用于证伪；顶层 Compute/Memory SOL 接近 100% 不等于 FP32 或 DRAM 已满，必须下钻到 L1/TEX、DRAM 和具体 FMA pipeline。

## Vectorized Copy 学习结论（2026-08-13）

- `float4` 本身只是对满足 16-byte 对齐的连续 4 个 FP32 做一次宽 load/store；算好合法、连续、对齐的首地址后，语法并不复杂。
- 实际复杂度来自为宽访问创造条件：原先每线程一个标量的 cooperative copy 可能需要重新分工，还必须处理行跨度对齐、矩阵尾部和标量 fallback。向量化减少搬运指令，但可能增加索引、分支和寄存器，性能不保证提高。
- `sgemm_trial_v3_1/v3_2` 已足以完成该技巧的学习：相同展平映射下分别生成标量 `LDG.E/STS` 和向量 `LDG.E.128/STS.128`。不继续扩展 shared vectorized load 或更多微调版本，下一主线进入 double buffering。

## Double Buffering 当前认识（2026-08-17）

- v3_3 的双 shared stage 能隔离 current consumer 与 next producer，但其同步 copy 对同一 warp 仍是依赖链
  `LDG -> STS -> current compute`；明确的普通指令软件流水应拆成 `next LDG -> current LDS/FMA -> next STS`。
- steady state 末尾的 block barrier 有两个职责：等待所有 next shared store 完成，并等待所有 warp 读完 current，
  防止下一轮复用旧 stage 时覆盖慢 warp 尚未消费的数据。最后一个 stage 计算后不再复用 shared，因此无需 barrier。
- 正式 v3 每线程保存 A/B 各 12 个 next 值，共约 24 个长生命周期 FP32 临时值，并用
  `next LDG -> current LDS/FMA -> next STS` 构造同 warp ILP。它已通过默认和多组非整除尺寸，成为后续
  `cp.async` 的普通 LDG/STS 基线；重复 trial_v3_4 已删除。
- 原计划使用 C pipeline primitives；2026-09-08 按用户要求改为少量 inline PTX，已实现 v4_1 的 4-byte copy，
  保持 v3 tile、线程映射和计算不变，便于同时入门 async 和 PTX。

## cp.async 学习进展（2026-09-08）

- 本轮已讨论并确认普通 cp.async 的发起粒度、分组、等待参数及 block 同步职责。cp.async 执行时就发起搬运，
  commit 不是启动开关；commit 前复制可能正在进行或已经完成，但不能未经完成同步就读取结果。
- commit_group 按线程把此前尚未提交的全部 cp.async 组成新组；分组边界由 commit 位置决定，无显式 group ID，
  不能向已提交组追加。没有新复制时仍生成空组，视为已完成；不同线程的组相互独立。
- wait_group N 按线程等待已提交组，允许最新至多 N 个组仍未完成；N 是编译期常量，不是组编号或等待组数。
  wait_group 0 不包含未 commit 的复制；wait_all 等价于 commit_group + wait_group 0。
- 当前双缓冲只有一批 next 待完成，不能直接将 wait_group 0 改成 1。正确消费顺序为各线程 wait，然后
  __syncthreads，再跨线程读取。block barrier 单独不保证 cp.async 完成；warp 调度也不能替代同步语义。
- 用户曾把循环末尾改为 barrier -> wait，反馈数次 PASS。该顺序缺少跨线程复制完成保证，PASS 仅说明测试未暴露
  问题；较长计算窗口可能掩盖错误，但未验证实际原因。最新源码已恢复 wait -> barrier，保留用户其他格式修改。
- inline PTX 的 volatile 与 "memory" 是编译器约束；"memory" 描述未显式列出的内存副作用，当前 copy/commit/wait
  封装应保留，不能当作硬件 barrier 或完成等待。并非所有 inline PTX 都必须写 memory clobber。
- 普通 cp.async 每线程每条复制 4/8/16 字节（.ca 支持三种，.cg 仅 16），源/目标需按复制宽度对齐；每线程大小与
  warp 合并访问是不同概念。v4_1 每线程每 stage 24 次 4-byte copy，warp 地址连续；不能仅改立即数为 16，
  否则产生重叠/不对齐/边界问题。工业实现按布局常用 16-byte copy、多 stage，搬运映射可独立于计算映射；
  CUTLASS SM80 SGEMM 示例也有元素宽度 copy。TMA 整 tile 搬运属于后续主题，当前不展开。
- v3 已通过普通 LDG prefetch 提供延迟隐藏；cp.async 不保证更快。当前 v3/v4_1 都为 128 registers，无 spill，
  删除源码 prefetch 数组未兑现寄存器数量收益。单次 v4_1 耗时多约 31%，不是稳定结论；4-byte 指令开销、
  shared 资源竞争和编译调度只是候选原因。应用单次计时区间含 profiler 启动和 host wrapper，需警惕提交间隙。
- 参考：[PTX cp.async](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async)、
  [CUTLASS SM80 SGEMM](https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/sgemm_sm80.cu)。

## 下一学习方向

1. v3 普通 LDG register-prefetch DB 已完成并转正；后续资源报告可补充，但不再阻塞主线。
2. v4_1 已实现且本轮完成基本同步语义讨论；下一步在正确同步版本上补稳定计时、简化 NCU 对照，条件允许时补
   memcheck/racecheck。定位到足以解释核心机制后收束，暂不扩展 16-byte 重排、多 stage 搜索或 TMA。
3. 保持 v2 的 `BX/BY/BK/TM/TN` 不变，不在 DB 学习期重新搜索超参数或展开逐条 SASS 考古。
4. v1 的 Details 硬件计数虽提示 shared store 约 1.2-way conflict，但 Source/SASS 中三处 shared 访问均为
   `Wavefronts Shared = Ideal`、`Excessive = 0`；不再把 padding/layout 当作当前主要优化方向。

面向学习者的完整说明以 `docs/ncu-gui-sgemm-analysis.md` 和 `docs/v1-to-v2-learning-roadmap.md` 为准；本文件只保存跨会话决策与易丢失经验。
