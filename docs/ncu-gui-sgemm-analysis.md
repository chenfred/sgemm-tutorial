# 使用 Nsight Compute GUI 分析 `sgemm_v0` 与 `sgemm_v1`

本文配套报告：`ncu-rep/sgemm.v0v1.0719.ncu-rep`。

配套报告采集环境为 NVIDIA GeForce RTX 5080、Compute Capability 12.0、Nsight Compute 2026.1.1。输入统一为：

```text
M = 1024, N = 4096, K = 1024
C(M,N) = A(M,K) × B(K,N)
```

本文面向第一次系统使用 NCU GUI 的读者。目标不是背诵所有指标，而是学会一套可复用的分析顺序。

## 1. 先看结论，再学习如何得到它

当前报告支持以下结论：

1. v1 的优化方向正确。在频率可比的正式报告中，NCU Duration 为 `1.684 ms → 0.706 ms`，约加速 2.39 倍。
2. v1 的核心不是“occupancy 更高”。它的 achieved occupancy 反而从 98.67% 降到 80.08%。
3. 核心收益是每线程计算 4 个输出，使总线程数降为四分之一，并提高 tile 内的数据复用：
   - global-load SASS 指令减半；
   - shared-load SASS 指令降到 v0 的 40%；
   - 总动态指令减少约 40%；
   - FMA pipeline 利用率从 9.96% 提高到 22.37%；
   - 每 scheduler 发射率从 0.28 提高到 0.41 warp/cycle。
4. v1 仍未把 FP32 FMA 资源打满。当前最突出的压力仍是 shared-memory/MIO 路径：L1/TEX throughput 约 97%，最大 stall 是 MIO Throttle。
5. 下一步应提高 shared→register 数据的复用率，即二维 register tiling；double buffering 不是当前第一优先级。

后续各节会说明怎样从 GUI 中独立得到这些结论。

## 2. 报告为什么只剩两个 kernel

两个版本的映射如下：

| kernel | block | grid | block tile | 每线程输出 |
|---|---:|---:|---:|---:|
| `sgemm_v0` | `(16,16,1)` | `(256,64,1)` | `16×16` | 1 |
| `sgemm_v1` | `(32,8,1)` | `(128,32,1)` | `32×32` | 4（M 方向 `4×1`） |

两者计算的是同一个矩阵乘法，逻辑工作量相同：

```text
2 × M × N × K = 8,589,934,592 FLOP
```

v0 启动 `16,384 × 256 = 4,194,304` 个线程，每线程写一个 C 元素。v1 启动 `4,096 × 256 = 1,048,576` 个线程，每线程写四个 C 元素。这是后面理解指令数变化的第一把钥匙。

## 3. 如何重新生成并打开报告

先编译、验证正确性，再采样：

```bash
scripts/build.sh
./build/sgemm
scripts/profile.sh -o sgemm.v0v1.0719
```

从命令行打开 GUI：

```bash
ncu-ui ncu-rep/sgemm.v0v1.0719.ncu-rep
```

也可以先启动 `ncu-ui`，再使用 `File → Open` 打开报告。

报告中应只有两个结果：`sgemm_v0` 和 `sgemm_v1`。程序用 `cudaProfilerStart/Stop` 只标记正式 kernel，warmup 不会出现在报告里；这里没有“第几次调用”的数字约定。

脚本使用 Application Replay。本次 NCU 2026.1.1 的 `full` 采集进度显示 40 个 pass；具体数量由 NCU 根据版本、section 和指标组合决定，不应写进代码。每个 pass 都会重新启动一次 `build/sgemm --dry-run`，所以每个正式 kernel 在每个 pass 中都会重新 warmup。`--dry-run` 仍执行数据生成、H2D、kernel 和 D2H，只跳过 CPU golden、正确性校验和应用侧计时输出；在 profiling 前必须先单独运行 `./build/sgemm`，确认两项均为 `PASS`。

## 4. 先处理两个测量陷阱

### 4.1 NCU 采集期间的应用侧计时不能用

脚本使用 `--set full`，需要数十个 application replay pass。NCU 会在不同进程中收集不同指标，再把它们合并成一个结果；这与普通的一次 kernel launch 不是同一种执行环境。

当前脚本传入 `--dry-run`，程序会跳过 CPU golden/verify 和 CUDA Event 输出，避免把校验成本带进 replay，也避免展示没有比较意义的应用侧时间。如果手工改掉该选项，profiling 期间打印的时间仍不能当作真实单次 kernel 时间。分析报告时应看：

```text
Details → GPU Speed Of Light Throughput → Duration
```

日常性能比较则看带 warmup 和多次迭代的 CUDA Event benchmark。当前 `main.cpp` 会在每个正式 kernel 前做长 warmup，但仍只计时一次，适合教学验证，不是最终严谨 benchmark。

### 4.2 频率不同会污染 Duration

NVIDIA 官方说明：应用中的第一个 kernel 经常处于较低时钟，replay 的不同 pass 也可能处于不同频率。因此比较 Duration 前，先同时检查：

- `SM Frequency`
- `DRAM Frequency`
- `Duration`

默认 Kernel Replay 只在第一次原始 launch 前执行应用的 warmup，后面的 metric pass 会单独重放正式 kernel；因此仅在源码里 warmup 一次仍可能出现低频。改用 Application Replay 后，正式报告为：

| 指标 | v0 | v1 | 差异 |
|---|---:|---:|---:|
| SM Frequency | 2.950 GHz | 2.936 GHz | 约 0.47% |
| DRAM Frequency | 14.9866 GHz | 14.9868 GHz | 小于 0.01% |
| NCU Duration | 1.684 ms | 0.706 ms | v1 约快 2.39 倍 |

连续三份 Application Replay 报告中的 v0/v1 频率与 Duration 都接近，可以进行入门级对比。动态 Boost 仍受温度、功耗和后台负载影响，换环境后应重新检查本表，而不是永久相信一次结果。

### 4.3 为什么当前脚本不再使用 NCU 锁频

`scripts/profile.sh` 显式使用：

```bash
--clock-control none
--profile-from-start off
--replay-mode application
```

`none` 表示 NCU 不修改 GPC 或 memory frequency，所以脚本既不会建立锁频，也不存在退出后忘记解锁的问题。控制台中的 “Running with unmodified GPU clocks” 警告是有意选择，不是错误。

本机升级后的 NCU 2026.1.1 提供 `base`、`boost`、`force-boost`、`none`、`reset`，且默认值已是 `boost`。脚本仍显式选择 `none`，避免工具版本升级悄悄改变实验策略；`force-boost` 也不是当前教学分析的默认选择。

为什么不使用 `base` 等锁频模式？NVIDIA 文档明确说明，实际时钟仍可能因驱动支持程度而变化。本机重复实验也观察到：即使采集 warmup，`base` 下的频率仍会随 replay 模式和 pass 数变化。它有动作，但没有实现本任务需要的稳定可比频率。

Application Replay 的代价是采集时间从约十几秒增加到约一分钟；收益是每个 pass 都重新执行 warmup，而且报告仍只包含 v0/v1。只有两者频率接近时才比较 Duration。

### 4.4 NCU 有没有“前 m 次只 warmup，后面自动采集”的参数

没有一个参数能直接表达这套语义：

- `--launch-skip m` 只忽略应用本来就会发射的前 m 个匹配 kernel，不会替应用额外执行它们；
- `--launch-count n` 只限制收集多少个匹配结果；
- Kernel Replay 的 pass 数由 NCU 根据指标集决定，但每个 pass 都属于指标采集，不能指定前几次 pass 只 warmup、不计入结果；
- Application Replay 同样由 NCU 决定 pass 数，但会为每个 pass 重新启动整个应用。于是应用中的 warmup 会自然重跑，这正是当前脚本采用的机制。

因此当前方案把职责分开：应用定义“如何 warmup、哪些 kernel 是正式区间”，NCU 决定 `full` 指标集需要多少个 pass。

### 4.5 为什么没有改成 Range Replay

NCU 的 Range Replay 确实能把 H2D、D2H 和 CPU 校验留在重放范围之外。实测过的代码边界如下：

```cpp
cudaMemcpy(/* H2D */);

for (const auto& implementation : IMPLEMENTATIONS) {
    warmup_do(/* ... */);
    cudaDeviceSynchronize();

    cudaProfilerStart();
    implementation.func(/* 单次正式 kernel launch */);
    cudaDeviceSynchronize();
    cudaProfilerStop();
}

cudaMemcpy(/* D2H */);
sgemm_verify(/* ... */);
```

对应命令的核心是：

```bash
ncu --set full --clock-control none --replay-mode range ./build/sgemm
```

这套布局能够正确运行：进程只启动一次，NCU 捕获两个 range，每个 range 里只有一个正式 kernel，最后 v0/v1 都能通过校验。官方定义也说明，一个 range 包含 Start/Stop 之间所有线程发出的 CUDA API 调用和 kernel；收集到的指标属于整个 range，而不是其中某个独立 kernel。参见 [Range Replay](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#range-replay) 与 [CLI `--replay-mode`](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#command-line-options)。

但它不适合当前“同一份 `full` 报告精确比较两个短 kernel”的目标，原因有三点：

1. 范围外的 warmup 不会随每个 metric pass 重放。`none + full` 实测中，range 0 的 v0 为 `2.951 GHz / 1.766 ms`，range 1 的 v1 却降为 `1.744 GHz / 1.364 ms`；DRAM frequency 也从 `14.987 GHz` 降到 `8.882 GHz`。
2. 把顺序改成 v1→v0 后，第一个 v1 为 `2.946 GHz`，第二个 v0 降为 `2.248 GHz`。这证明主要是 range 的位置效应，而不是 v1 天生只能低频。
3. `base` 在三指标短采集中看似稳定，但 `full` 中 v0/v1 又变成 `1.556/1.126 GHz`。不同指标来自不同 pass，短采集稳定不能证明 `full` 报告内部可比。

还有两个 NCU 2026.1.1 的使用细节：

- `--profile-from-start off` 与 Range Replay 不能同时使用；Range Replay 已经由 Start/Stop 定义边界。
- `--import-source yes` 与 Range Replay 的组合会被 CLI 拒绝。报告仍能在本机从原路径解析 CUDA/SASS 关联，但不会把源码永久嵌入报告。

Profiler Start/Stop 产生的结果名也只是 `range`，`launch__kernel_name` 为空；只能靠结果 ID、顺序或 block/grid 配置映射 v0/v1。相比之下，本次 `none + Application Replay + full` 报告得到：v0/v1 的 SM frequency 为 `2.950/2.936 GHz`，DRAM frequency 均约 `14.987 GHz`，Duration 为 `1.684/0.706 ms`，并保留具名的两个 kernel。因此当前默认脚本继续使用 Application Replay；`--dry-run` 只裁掉 CPU golden/verify 与应用侧计时，不改变被分析的 H2D、kernel、D2H 执行链。

## 5. 第一次打开 GUI：只做这六步

不同 NCU 版本的按钮位置可能略有差异，但页面名称基本一致。

1. 在结果列表或顶部 launch 下拉框中选择 `sgemm_v0`。
2. 打开 `Details`，展开 `GPU Speed Of Light Throughput`。
3. 把 v0 加为 Baseline。
4. 切换到 `sgemm_v1`，观察相对 baseline 的变化。
5. 依次展开 `Compute Workload Analysis`、`Scheduler Statistics`、`Warp State Statistics`、`Memory Workload Analysis`、`Occupancy`。
6. 最后打开 `Source`，选择 CUDA-C/SASS 关联视图。

不要从 Raw 页开始。Raw 有数千个计数器，适合验证具体假设，不适合作为第一次阅读的入口。

## 6. 固定使用这一条分析主线

每次看一个新 kernel，都按下面顺序问问题：

```text
1. Duration 是否真的变好，频率是否可比？
2. 最忙的是 DRAM、L1/TEX、FMA，还是其他 pipeline？
3. scheduler 是否经常找不到 eligible warp？
4. warp 为什么不能发射？
5. 代码执行了多少 global/shared 指令？
6. registers/shared memory 是否限制 occupancy，是否 spill？
7. Source/SASS 是否符合源码意图？
8. 哪一个最小实验能验证当前推断？
```

这条主线比“看到红色就优化”可靠。NCU 的颜色和估算建议只是线索，不是最终裁决。

## 7. GPU Speed Of Light：先找最忙的数据通路

关键值如下：

| 指标 | v0 | v1 |
|---|---:|---:|
| Compute (SM) Throughput | 96.71% | 96.71% |
| Memory Throughput | 96.71% | 96.71% |
| L1/TEX Cache Throughput | 97.22% | 97.40% |
| L2 Cache Throughput | 24.54% | 31.20% |
| DRAM Throughput | 2.85% | 4.74% |
| FMA pipe，elapsed cycles | 9.96% | 22.37% |

最容易犯的错误是：

> “Compute Throughput 接近 100%，所以 FP32 计算单元已经满了。”

这是错误的。顶层 Compute/Memory throughput 是多个子单元的汇总，常由最忙的子单元主导。本报告里真正接近满载的是 L1/TEX/LSU 路径；FMA pipeline 只有 9.96% 和 22.37%。

第二个常见错误是：

> “Memory Throughput 接近 100%，所以 DRAM 带宽满了。”

也是错误的。DRAM Throughput 只有约 2.9%～4.7%。这里的高 memory/LSU 利用率主要来自 shared memory 使用的片上数据通路，而不是显存带宽。

因此本节结论是：

```text
不是 DRAM-bound；也不是 FP32 FMA-bound；主要压力在 L1/TEX/LSU/shared-memory 路径。
```

## 8. Compute Workload 中几个指标分别表示什么

### 8.1 Executed IPC Active

SM 处于 active 周期时，平均每周期执行多少条指令。它排除了完全不活跃周期，适合观察活跃阶段的执行效率。

### 8.2 Executed IPC Elapsed

以所有 elapsed cycles 为分母，包括没有活动的周期。它通常不高于 Active 版本，更接近整个 kernel 时间范围的平均执行密度。

### 8.3 Issued IPC Active

scheduler 在 active 周期平均发射多少条指令。Issued 与 Executed 非常接近通常是正常现象；二者差异较大时才需要进一步检查重放、取消或架构行为。

### 8.4 Issue Slots Busy

可用发射槽中实际被使用的比例。低值表示 scheduler 经常没有合适指令可发射，但它不直接告诉你原因；原因要去 Scheduler 和 Warp State 看。

### 8.5 SM Busy

至少有某种 SM 工作活动的周期占比。它不等于 FMA 利用率，也不等于 occupancy。

当前对比：

| 指标 | v0 | v1 |
|---|---:|---:|
| Executed IPC Active | 1.14 | 1.64 |
| Executed IPC Elapsed | 1.13 | 1.62 |
| Issue Slots Busy | 28.34% | 40.62% |
| SM Busy | 36.45% | 40.62% |
| Executed Instructions | 472.91 M | 282.76 M |

v1 不仅 IPC 更高，而且总指令更少。二者一起出现才是强证据：相同数学工作用更少指令完成，同时 scheduler 发射得更有效。

单独看到 IPC 上升不能判定优化成功。如果新版本 IPC 更高、总指令却翻倍，最终时间仍可能更差。

## 9. Scheduler Statistics：active warp 多不等于能发射

先区分三个概念：

- `Active Warps`：已经驻留在 scheduler 上、尚未结束的 warp。
- `Eligible Warps`：下一条指令已经就绪，本周期有资格被选择的 warp。
- `Issued Warp`：本周期真正选中并发射的 warp。

当前数据：

| 指标（每 scheduler） | v0 | v1 |
|---|---:|---:|
| Active Warps | 11.84 | 9.62 |
| Eligible Warps | 1.30 | 1.70 |
| Issued Warp / cycle | 0.28 | 0.41 |
| One or More Eligible | 28.50% | 40.92% |
| No Eligible | 71.50% | 59.08% |

v0 几乎拥有最大数量的 active warps，却有 71.50% 的周期找不到 eligible warp。这证明“线程很多、occupancy 很高”仍不能保证填满流水线；这些 warp 可能一起被 shared-memory 队列、scoreboard 或 barrier 卡住。

v1 的 active warps 较少，但 eligible warps 更多，实际发射率更高。这是 ILP、较低指令压力与较少等待共同作用的结果。

## 10. Warp State：为什么 warp 没有 eligible

这里的数值表示平均每发射一条指令，对应多少个 warp cycle 处于某种状态。不要把它直接读成时间百分比。

| stall reason | v0 | v1 | 入门解释 |
|---|---:|---:|---|
| MIO Throttle | 21.94 | 12.37 | MIO 指令队列满；shared-memory 指令常走这条路径 |
| Long Scoreboard | 6.46 | 1.80 | 等待较长延迟的数据依赖，常与 global/local memory 有关 |
| Barrier | 5.43 | 2.60 | warp 到达同步点后等待其他 warp |
| Not Selected | 3.51 | 3.16 | 已 eligible，但本周期选了别的 warp；不一定是坏事 |
| Wait | 1.99 | 0.96 | 等待固定延迟的执行依赖 |
| Short Scoreboard | 0.81 | 0.96 | 等待较短延迟的 MIO/shared 等依赖 |

`Warp Cycles Per Issued Instruction` 从 v0 的 41.54 降到 v1 的 23.50。也就是说，v1 发射两条相邻指令之间的平均等待显著缩短。

不能只凭 `MIO Throttle` 这个名字就断定 shared memory 是根因。这里还有两组交叉证据：

- L1/TEX active throughput 接近 98%；
- shared-load 指令数远高于 global-load 指令数。

三项互相印证后，“shared-memory/MIO 压力”才是可靠推断。

## 11. Memory Workload：比较完成同样工作用了多少指令

对于这两个 kernel，跨版本最直观的是动态 SASS 指令数：

| SASS 指令 | v0 | v1 | v1/v0 |
|---|---:|---:|---:|
| global load | 16.78 M | 8.39 M | 50% |
| global store | 0.131 M | 0.131 M | 100% |
| shared load | 167.77 M | 67.11 M | 40% |
| shared store | 16.78 M | 8.39 M | 50% |
| 全部 executed instructions | 472.91 M | 282.76 M | 59.8% |

这张表揭示了 v1 的核心优化：

1. 每线程处理 4 个输出，总线程数降为四分之一。
2. 一个加载到 shared/register 的值支持更多 FMA。
3. cooperative load、地址计算、循环控制和 shared 指令都随之减少。
4. 输出元素数量不变，所以 global store 指令数不变。

`L1/TEX Hit Rate` 对 shared memory 不是“是否命中缓存”的总评分。shared memory 使用 L1/TEX 相关硬件通路，但不按普通 L1 cache hit/miss 的方式理解。不要因为 v1 的 hit rate 约 0.1% 就说 shared memory 没生效。

### 11.1 v1 的 shared-store bank conflict 是一个次级实验点

NCU 规则报告：v1 的 8,388,608 次 shared-store 请求产生约 1,382,161 个 bank conflict，约占 9,785,076 个 shared-store wavefront 的 14.13%，平均每个请求约产生 1.17 个 wavefront（约 1.2-way conflict）。

这是值得验证的线索，但不要立即把所有 shared 数组都改成 `TILE_SIZE+1`：源码看上去是按行连续写，编译器又进行了展开和 SASS 重排。正确做法是：

1. 在 Source 页定位 `STS` 和相关 source line。
2. 重复采样，确认 conflict 数稳定存在。
3. 分别只改变 tileA 或 tileB 的 layout/padding。
4. 每次同时看 conflict、MIO Throttle、registers 和真实时间。
5. 只有时间稳定下降才保留修改。

NCU 给出的“Estimated Speedup”是局部上限估算，不是承诺。

## 12. Occupancy：v1 为什么更低却更快

| 指标 | v0 | v1 |
|---|---:|---:|
| registers/thread | 40 | 48 |
| static shared/block | 2.05 KB | 8.19 KB |
| theoretical occupancy | 100% | 83.33% |
| achieved occupancy | 98.67% | 80.08% |
| active warps/SM | 47.36 | 38.44 |
| local spilling | 0 | 0 |

v1 用更多寄存器保存四个累加结果，用更大的 shared tile 提高复用，因此 occupancy 下降。这是合理的资源交换。

判断这种交换是否成功，要看：

- 是否出现 local spill：当前为 0；
- eligible warp 和 issue rate 是否恶化：实际上改善；
- 最终时间是否下降：约快 2.4 倍。

所以“把 occupancy 拉回 100%”不是当前优化目标。Occupancy 是隐藏延迟的一种手段，不是最终成绩。

## 13. Source 页：连接 CUDA、SASS 与指标

打开 v1 的 `Source` 页，选择 CUDA-C/SASS 关联视图。先只识别以下指令：

| SASS | 含义 |
|---|---|
| `FFMA` | FP32 fused multiply-add |
| `LDS` / `LDS.128` | 从 shared memory 读取标量/更宽数据 |
| `STS` | 写 shared memory |
| `LDG` / `STG` | global load/store |
| `BAR` | block 同步 |
| `BRA` | 分支/循环控制 |

当前 v1 的 compute 主体可以看到大量交错的 `LDS`、`LDS.128` 与 `FFMA`。这说明编译器没有机械地逐行执行 CUDA 源码，而是展开并重新调度了循环，在多个独立累加结果之间寻找 ILP。

建议完成三个练习：

1. 找到 v0 内层循环对应的 `LDS + FFMA`。
2. 找到 v1 四个累加器对应的交错 `FFMA`。
3. 对照 Launch Statistics 的 40/48 registers，思考更多展开为什么需要更多寄存器。

源码提供“可能的独立性”，最终机器是否利用它必须看 SASS 和性能计数器。

## 14. PM Sampling 中的 PM 是什么

PM 是 Performance Monitor。PM Sampling 按固定间隔对硬件 performance-monitor counter 采样，用来观察 kernel 执行期间指标随时间的变化，而不是只得到整个 kernel 的一个汇总值。

Details 中 PM Sampling 的几列主要是采样配置：

- `Maximum Buffer Size`：设备侧采样缓冲区上限；
- `Maximum Sampling Interval`：最大采样间隔，本报告约 3 μs；
- `# Pass Groups`：这些 sampling metric 被分成多少组采集。

这些值本身不是瓶颈。真正有用的是 GUI 中的时间轴：例如观察 SM frequency、FMA、L1/TEX 是否在 kernel 前后发生阶段性变化。

对于约 1～4 ms 的 kernel，PM Sampling 能提供不少样本，但它仍是采样而非逐周期真值。第一次分析时先看汇总 sections；需要回答“瓶颈是否只发生在某个阶段”时再看 PM 时间轴。

## 15. 把证据整理成一页结论

| 版本 | 已解决什么 | 证据 | 仍有什么问题 |
|---|---|---|---|
| v0 | 正确的 shared-memory tiling 基线 | correctness PASS，DRAM 仅 2.85% | 每线程只算 1 个输出；shared load 167.8 M；MIO Throttle 21.94；FMA 9.96% |
| v1 | 4×1 thread tile，提高复用并减少线程/指令 | shared load 67.1 M；总指令少 40.2%；issue 0.41；约快 2.39× | L1/TEX 97.40%；MIO 12.37；FMA 仅 22.37%；存在 shared-store conflict 线索 |

因此下一版的首要假设应是：

> 在 M、N 两个方向做二维 register tiling，让每次 shared load 支持更多 FMA，进一步降低 shared/MIO 指令压力。

## 16. 一次 45 分钟的 GUI 实践

不要边读边漫无目的点页面。按下面顺序做一次：

1. 记录 v0/v1 的 Duration、SM/DRAM Frequency。
2. 记录 L1/TEX、DRAM、FMA pipeline。
3. 记录 Executed Instructions、IPC、Issue Slots Busy。
4. 记录 active/eligible/issued warps。
5. 记录前五个 stall reason。
6. 记录 global/shared load/store 指令。
7. 记录 registers、shared/block、occupancy、spill。
8. 在 Source 页各找一处 `LDG`、`LDS`、`STS`、`FFMA`、`BAR`。
9. 用三句话写出“证据 → 推断 → 下一实验”。

如果能不看本文重新解释“为什么 v0 occupancy 更高却更慢”，你已经掌握了这份报告最重要的部分。

## 17. 命令行复核

GUI 用于交互学习，CLI 便于复制关键表格：

```bash
ncu --import ncu-rep/sgemm.v0v1.0719.ncu-rep \
    --page details \
    --print-summary per-kernel
```

查看 v1 的 CUDA/SASS 关联：

```bash
ncu --import ncu-rep/sgemm.v0v1.0719.ncu-rep \
    --page source \
    --print-source cuda,sass \
    --kernel-name regex:sgemm_v1
```

Raw CSV 是“列式宽表”，通常应由脚本提取所需 metric，不建议手工阅读整个输出。

## 18. 参考资料

### 简体中文优先

1. [NVIDIA Nsight Compute 产品与入门页](https://developer.nvidia.cn/nsight-compute)
2. [CUDA 编程模型接口：shared-memory tiled matmul](https://developer.nvidia.cn/blog/cuda-programming-model-interface-cn/)
3. [在 CUDA C/C++ 中使用共享内存](https://developer.nvidia.cn/blog/using-shared-memory-cuda-cc/)
4. [NVIDIA 加速计算学习路径](https://www.nvidia.cn/training/learning-path/accelerated-computing/)

### 英文官方资料

1. [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)：replay、clock control、metrics 与 reproducibility。
2. [Nsight Compute GUI User Guide](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)：各页面、baseline、source correlation。
3. [Nsight Compute CLI Guide](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)：`--replay-mode`、`--clock-control`、import 与过滤。
4. [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)：coalescing、shared memory、occupancy 与 benchmark 方法。
5. [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)：warp scheduling、memory hierarchy 与执行模型。
