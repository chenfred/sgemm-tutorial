# 如何阅读 Nsight Compute 报告（ncu-rep）分析 kernel 瓶颈

面向：在 Windows 下用 Nsight Compute GUI 打开 `report/sgemm.ncu-rep`，定位 `sgemm_naive` 与 `sgemm_v1` 的性能瓶颈。
配套：先用 `scripts/profile.sh -o sgemm` 生成报告（已含 `--import-source on`，GUI 能看到源码）。

---

## 0. 打开报告与界面认识

- 启动 Nsight Compute GUI → `File → Open` 选 `sgemm.ncu-rep`。
- **左侧 Results 列表**：本次有 4 个 kernel 实例（两个尺寸 × 两个 kernel）。靠 **Grid 维度**区分尺寸：
  - `sgemm_naive (256,64,1)` / `sgemm_v1 (128,32,1)` → `1024×4096×1024`
  - `sgemm_naive (125,63,1)` / `sgemm_v1 (63,32,1)` → `1000×2000×1500`
- **中间主区**：顶部一排 **Section 标签**（Speed of Light、Compute Workload、Memory Workload、Scheduler、Warp State、Occupancy、Source…），每个标签是一类指标的可视化。
- 阅读顺序建议：**Speed of Light → Compute/Memory Workload → Scheduler/Warp State → Occupancy → Source**（从宏观到微观）。

---

## 1. Speed of Light（SOL）——先看这里，定大方向

GUI 顶部两条最长的条：**Compute (SM) Throughput** 与 **Memory Throughput**，各表示该子系统相对自身峰值的利用率。

**判读规则**：
- 谁的 % 高，谁就是当前瓶颈（更接近它的天花板）。
- 两者都高（如本次 naive：Compute 92%、Memory 92%）→ kernel 已把硬件压到接近饱和，**性能受限于硬件吞吐**，不是「没活干」。
- 再往下看一组小条：`DRAM / L1-TEX / L2` Throughput，定位是哪一级内存忙。

**本次举例**（`sgemm_naive` @1024×4096）：
- Compute 92.35% / Memory 92.35% / **L1/TEX 97.06%** / DRAM 7.19% / L2 22.95%。
- 解读：瓶颈不在 DRAM（数据都在 cache），而在 **L1/TEX pipe 97%** —— 即 shared memory 访问把这条流水线打满了。

> ⚠️ **这里有个大坑**：SOL 区右侧有个 `Duration`，**不要用它比较两个 kernel 的真实速度**。见第 6 节。

---

## 2. Compute Workload Analysis —— 算力是否喂得满

关键指标：
- **IPC（Instructions Per Cycle）**：`Executed Ipc Active`。每周期每调度器发射的指令数。sm_120 每 SMP 每周期单发，理论 IPC 上限低；实际看到 1.0–2.0 量级。
- **Issue Slots Busy**：发射槽利用率。低（如 naive 26.8%）说明大量周期没在发射指令（在 stall）。
- **SM Busy**：SM 整体忙碌度。

**本次**：naive IPC=1.13 / Issue 26.8%；**v1 IPC=1.71 / Issue 42.5%**。→ v1 的指令发射效率其实更好（这点颠覆了静态直觉，见主报告 Q3）。

---

## 3. Memory Workload Analysis —— 数据从哪来、卡在哪

关键指标：
- **各级 Hit Rate**：`L1/TEX Hit Rate`、`L2 Hit Rate`。命中率高 = 数据在 cache 复用得好。
- **Mem Pipes Busy / Max Bandwidth**：内存管线饱和度。
- **Memory Throughput（Gbyte/s）**：实际带宽。

**本次对比**（很有意思）：
| | L1/TEX Hit | L2 Hit | DRAM % |
|---|---|---|---|
| naive | **7.3%** | 93.9% | 7.2% |
| v1 | **79.4%** | 98.0% | 3.1% |

v1 的 L1 命中率高得多（block 大、L1 内复用多），但 L1/TEX pipe 本身也 94.8% 饱和，省下的流量换不成更多算力。

---

## 4. Scheduler Statistics + Warp State Statistics —— 为什么发射不满

这是定位「为什么 Issue Slots Busy 低」的核心。

**Scheduler Statistics**：
- **Eligible Warps Per Scheduler**：平均有多少 warp 处于可发射状态。理想 ≥ 2–4；本次 naive 1.33、v1 1.14，**都偏低**。
- **No Eligible %**：多少比例的周期**没有任何 warp 可发射**（全在 stall）。naive **71.8%**、v1 57.5%。这个值高 = 延迟没藏住。

**Warp State Statistics**（GUI 是按 stall 原因分色的堆叠柱状图）—— **最重要的一张图**。常见 stall 原因：
| stall 原因 | 含义 | 通常对应 |
|---|---|---|
| `Long Scoreboard` | 等内存（global/shared load 未返回）| 访存依赖、未 coalesced、cache miss |
| `Short Scoreboard` | 等 shared memory / 寄存器依赖 | shared bank conflict、密集 shared 访问 |
| `Wait` | 等 `__syncthreads` / barrier | 同步过多（如 v1 的 KS=8）|
| `Barrier` | 显式屏障 | 同上 |
| `Not Selected` | warp 就绪但没被选中（有别的 warp 先发）| 占用率够、无碍 |
| `LG Throttle` | load/store 单元限流 | 内存指令太多 |
| `MIO Throttle` | ALU 单元限流 | 整数/地址指令太多 |

> **怎么用**：占比最大的那种 stall 就是主因。例如若 `Short Scoreboard` 占大头，说明被 shared memory 访问卡住（和 SOL 里 L1/TEX 97% 互相印证）。

**本次**（从 `Warp Cycles Per Issued Instruction` 侧面看）：naive 42、v1 18.2 —— naive 每条指令平均等 42 周期，说明 stall 很重，靠 98.7% 占用率的多 warp 轮转掩盖。

---

## 5. Occupancy —— 为什么 SM 上 warp 不够多

GUI 这里通常有个**柱状图显示 Block Limit 的各项限制**：`Warps / Registers / Shared Mem / Barriers`，谁先顶住上限，谁就是占用率的瓶颈。

关键指标：
- **Theoretical Occupancy**：理论上限。
- **Achieved Occupancy**：实际跑出来的。
- **Block Limit Registers / Shared Mem / Warps**：各项各自允许的 block 数，**最小的那个就是真瓶颈**。

**本次**：
| | naive | v1 |
|---|---|---|
| Registers Per Thread | 40 | **55** |
| Block Limit Registers | 6 | **4** |
| Theoretical Occupancy | 100% | **66.67%** |
| Achieved Occupancy | 98.7% | **65.0%** |

→ v1 的占用率被 **Registers（55/thread）** 顶到 66.7%。这是 v1 慢的次要主因，一眼能看出来：`Block Limit Registers=4` 是那一排里最小的。

---

## 6. 三个必须避开的坑

1. **别用 ncu 的 `Duration` 比 kernel 快慢**。ncu 用 kernel replay 多次重放来采指标，会大幅拖慢；且本次 `SM Frequency` 在 1.55–2.28 GHz 波动（thermal/power throttling）。**比真实速度用 `main.cpp` 的 `cudaEvent` 计时**（naive 1.788ms vs v1 1.986ms）。ncu 只用它的**归一化 %、IPC、occupancy、stall**。
2. **`No Eligible` 高 ≠ 一定坏**。要看是被什么 stall 顶住、占用率够不够补。naive No Eligible 71.8% 但靠占用率补回来了。
3. **`Block Limit` 那一排要看最小项**。不是「寄存器多就好」，而是哪一项先顶住上限。v1 就是栽在 Registers。

---

## 7. 实战：用本报告判断「v1 为何慢于 naive」的完整流程

1. **SOL**：两者 Compute/Memory 都 ~92–94%、L1/TEX ~95–97% → 都撞 shared-memory 带宽墙，差别在细节。
2. **Compute**：v1 IPC 1.71 > naive 1.13 → v1 发射效率更好，**不是 ILP 问题**。
3. **Instruction Statistics**：v1 Executed Instructions 7.6e8 vs naive 4.7e8 → **v1 多 62% 指令**（线程重排 + 复杂索引的开销）。
4. **Occupancy**：v1 被 Registers(55) 顶到 65% vs naive 98.7% → **占用率掉 1/3**。
5. 合起来：指令 +62%、IPC +51%、占用率 −1/3 → 净效果慢 ~11%。和 `cudaEvent` 实测吻合。

→ 这个流程可复用到任何 kernel：**SOL 定方向 → Warp State 定 stall 类型 → Occupancy 定资源瓶颈 → Instruction/Source 定代码热点**。

---

## 8. Source 视图（按源码行定位热点）

`Source Counters` / `Source` section：因为 profile 时加了 `--import-source on`，GUI 能把指标映射到 `sgemm_v1.cu` 的每一行。
- 可以看到每行的 `Stall`、内存访问次数、采样热度。
- 用它定位「哪几行代码贡献了那 62% 的多余指令」或「哪行 shared 访问最热」。
- 用法：选中 kernel → Source 标签 → 按某指标排序，看最热的行。

---

## 9. 对比两个 kernel

Nsight Compute GUI 支持：
- 左侧 Results 里**选中两个结果 → 右键 → Compare**，会并排显示指标差异（非常适合 naive vs v1 这种 A/B 对比）。
- 或分别打开两个 report 文件做 compare。
- 对比时盯「相对差异」+「各自瓶颈项」，而不是绝对耗时。
