# 使用 Nsight Compute GUI 分析 naive、v1 与 v2

本文配套报告：`report/sgemm_nv1v2_2.ncu-rep`。

目标不是背指标，而是学会一条可复用的分析链：

```text
确认实验可比
  → SOL 定位忙碌子系统
  → Scheduler 判断延迟是否藏住
  → Warp State 找“为什么不能发射”
  → Memory / Compute 找具体管线
  → Occupancy 检查资源约束
  → Source / SASS 回到代码
  → 提出一个可证伪的优化实验
```

本文针对 Nsight Compute 2025.3.x。不同版本的按钮位置可能略有变化，但页面和指标含义基本一致。

## 1. 先理解这份报告记录了什么

报告包含同一组矩阵尺寸 `M=1024, N=4096, K=1024` 的四次 kernel launch：

| GUI 中的 kernel | 源码入口 | block | grid | 每线程输出 |
|---|---|---:|---:|---:|
| `sgemm_naive` | `sgemm_naive_do` | `(16,16,1)` | `(256,64,1)` | 1 |
| `sgemm_v1` | `sgemm_v1_do` | `(16,16,1)` | `(128,32,1)` | 4（2×2） |
| `sgemm_v2<1>` | `sgemm_v2<true>` | `(32,8,1)` | `(128,32,1)` | 4（4×1） |
| `sgemm_v2<0>` | `sgemm_v2<false>` | `(32,8,1)` | `(128,32,1)` | 4（4×1） |

这里的模板参数非常重要：

- `v2<1>` 对应 `OptForILP=true`，源码把 `t` 放外层，把四个累加器的更新放内层。
- `v2<0>` 对应 `OptForILP=false`，源码把累加器 `ii` 放外层，每个累加器先完成整个 `t` 循环。

四个 kernel 都完成同样的 SGEMM 数学工作，因此可以比较“为了完成相同工作，硬件付出了什么代价”。

## 2. 打开报告与建立 baseline

在带 Nsight Compute GUI 的机器上：

1. 启动 `ncu-ui`，选择 `File → Open`。
2. 打开 `report/sgemm_nv1v2_2.ncu-rep`。
3. 报告包含多个结果时，默认先显示 **Summary Page**。
4. 在顶部 Launch 下拉框或 Summary 表中选择 `sgemm_naive`。
5. 点击 `Compare → Add Baseline`，把 naive 设为 baseline。
6. 再选择 v1 或 v2，Details 和 Raw 页面会显示相对 baseline 的变化。

第一次阅读建议分三轮比较：

```text
naive → v1       看“做更多输出但代码复杂化”的代价
naive → v2<1>    看 v2 的整体收益来自哪里
v2<1> → v2<0>    看循环顺序是否真的改变机器代码和性能
```

不要一次选很多 kernel 后只盯颜色。每一轮只回答一个问题。

## 3. 阅读前必须知道的四个陷阱

### 3.1 NCU 的 Duration 不是稳定 benchmark

`--set full` 往往需要多次 replay kernel 来收集不同硬件计数器。不同 launch 的 GPU 频率也可能不同。本报告中：

| kernel | NCU Duration | SM Frequency |
|---|---:|---:|
| naive | 3.633 ms | 1.50 GHz |
| v1 | 2.334 ms | 2.28 GHz |
| v2&lt;0&gt; | 0.916 ms | 2.26 GHz |
| v2&lt;1&gt; | 0.925 ms | 2.25 GHz |

naive 与 v1 的频率差异很大，所以不能用这一列精确计算版本加速比。正确做法是：

- 日常 benchmark：预热后用 CUDA Event 重复执行几十到几百次，报告中位数或稳定均值。
- NCU：解释为什么快或慢，重点看归一化利用率、指令、流量、stall 和资源。
- v2 两版本的频率接近，报告时间可作为弱证据，但一次测量的约 1% 差距仍不足以下定论。

### 3.2 顶层 Compute 和 Memory 同时很高，不等于 FP32 与 DRAM 同时饱和

SOL 顶层值由多个子指标汇总，常常由其中最忙的一条管线主导。本报告的 v2 同时显示约 96% 的 Compute/Memory，但：

- FP32/FMA pipe 只有约 21%。
- DRAM 只有约 4%～6%。
- L1/TEX active throughput 约 98%。

所以必须展开 breakdown。不能看到 `Compute (SM) Throughput=96%` 就直接宣布“算力瓶颈”。

### 3.3 Hit Rate 和 Throughput 不是同一件事

- Hit Rate：请求有多少在某级 cache 命中。
- Throughput：某条数据通路相对其峰值有多忙。

shared memory 使用 L1/TEX 相关数据通路，但 shared memory 本身不是靠 cache hit/miss 工作。因此 v2 的 `L1/TEX Hit Rate≈0.1%` 不代表 shared memory 没起作用；应结合 shared load 数、L1/TEX throughput 和 MIO stall 解读。

### 3.4 stall 很多不一定影响最终性能

某个 warp stall 时，如果 scheduler 仍有其他 eligible warp 可发射，延迟就被隐藏了。官方指南建议：先确认 scheduler 是否经常没有 eligible warp，再追 stall 原因。

## 4. 第一站：Summary 与 Speed Of Light

### 4.1 Summary Page 做什么

Summary 用于快速回答：

- 哪些 kernel 被采集？
- launch 配置是否一致或符合预期？
- 哪些自动规则被 NCU 标为高优先级？

自动规则是线索，不是裁判。规则不知道你的算法结构，也不知道一个修改会不会增加寄存器或破坏其他管线。

### 4.2 Speed Of Light 做什么

进入 **Details Page → GPU Speed Of Light Throughput**，先看：

- `Compute (SM) Throughput`
- `Memory Throughput`
- `DRAM Throughput`
- `L1/TEX Cache Throughput`
- `L2 Cache Throughput`

本报告的关键值：

| 指标 | naive | v1 | v2&lt;1&gt; | v2&lt;0&gt; |
|---|---:|---:|---:|---:|
| Compute/Memory 顶层 SOL | 88.26% | 94.18% | 96.43% | 96.82% |
| DRAM Throughput | 9.95% | 2.95% | 5.52% | 4.02% |
| L1/TEX Throughput（active） | 97.07% | 94.83% | 97.51% | 97.79% |
| L2 Throughput | 21.00% | 57.88% | 28.29% | 28.50% |

第一层结论只能写成：

> 四个 kernel 都不是 DRAM 带宽受限；最忙的内存侧资源是 L1/TEX 路径。还需要 Compute Workload、Scheduler 和 Warp State 确认它是否真的限制发射，以及流量主要来自什么指令。

注意这句话保留了验证空间，没有把相关性直接写成因果关系。

## 5. 第二站：Compute Workload 与 Instruction Statistics

### 5.1 不要把 IPC 当作 GFLOPS

IPC 统计发射/执行的所有 SASS 指令，包括：

- FFMA；
- shared/global load/store；
- 地址计算；
- 比较和分支；
- barrier；
- 循环控制。

因此 IPC 高可能来自更多非计算指令。要同时看总指令数和各 pipe 利用率。

| 指标 | naive | v1 | v2&lt;1&gt; | v2&lt;0&gt; |
|---|---:|---:|---:|---:|
| Executed IPC Active | 1.13 | 1.71 | 1.80 | 1.64 |
| Issue Slots Busy | 25.63% | 42.56% | 44.61% | 40.67% |
| Executed Instructions | 468.7 M | 759.6 M | 311.4 M | 282.8 M |
| FMA pipe 利用率（elapsed） | 8.54% | 9.20% | 21.40% | 21.20% |

这里能得到三个重要结论：

1. v1 的 IPC 比 naive 高，但完成相同工作执行了约 62% 更多指令；高 IPC 并没有自动变成高有效算力。
2. v2 的总指令明显减少，同时 FMA pipe 利用率提高到约 21%，说明 v2 把更多执行能力用于有效乘加。
3. v2&lt;0&gt; 比 v2&lt;1&gt; 少约 9.2% 动态指令，但 IPC、occupancy 和 eligible warps 更低；多个效应互相抵消，最终时间接近。

在 GUI 中展开 Compute Workload 的 pipeline breakdown。看到某个 pipe 很高时，再回 Source/SASS 确认是什么指令造成的。

## 6. 第三站：Scheduler Statistics

Scheduler 页面回答：“每个 scheduler 手里有多少 warp？有多少已经 ready？多少周期能发射？”

| 指标 | naive | v1 | v2&lt;1&gt; | v2&lt;0&gt; |
|---|---:|---:|---:|---:|
| Active Warps / Scheduler | 11.84 | 7.80 | 11.45 | 9.58 |
| Eligible Warps / Scheduler | 1.34 | 1.14 | 2.24 | 1.69 |
| One or More Eligible | 28.19% | 42.85% | 45.12% | 40.92% |
| No Eligible | 71.81% | 57.15% | 54.88% | 59.08% |
| Issued Warp / Scheduler | 0.28 | 0.43 | 0.45 | 0.41 |

理解方式：

- `Active Warps` 是驻留且尚未结束的 warp，不代表现在可以发射。
- `Eligible Warps` 是依赖和资源都满足、下一条指令可发射的 warp。
- `No Eligible` 高，说明很多周期 scheduler 找不到可发射 warp，延迟隐藏仍不充分。
- `Not Selected` 较高通常表示 ready warp 多于发射槽，是竞争而不是阻塞，不应首先优化。

本报告里 v2&lt;1&gt; 的 eligible warp 最多、发射率最高，说明它在调度层面最好；但这不代表总时间必然最低，因为它也执行了更多循环控制等指令。

## 7. 第四站：Warp State，找出不能发射的原因

在 **Warp State Statistics** 中，每个数通常表示“每条已发射指令平均经历多少个该状态的 warp-cycle”。它不是时间百分比，多个状态也不能随意相加成 wall time。

常见状态的可靠解释：

| 状态 | 表示 warp 在等什么 | 本项目中首先检查 |
|---|---|---|
| `MIO Throttle` | MIO 指令队列满；shared memory、部分特殊/分支指令会使用该路径 | shared load/store 数、L1/TEX、Source 热点 |
| `Long Scoreboard` | 等待 L1TEX scoreboard 管理的较长延迟操作，常见为 global/local/texture/surface load | global load、cache、合并访问 |
| `Short Scoreboard` | 等待 MIO scoreboard 管理的操作，常见为 shared memory 依赖 | shared 访问、bank conflict、load 后立即使用 |
| `Barrier` | warp 到达 CTA barrier，等待其他 warp | `__syncthreads()` 两侧负载是否均衡 |
| `Wait` | 等待固定延迟执行依赖 | 相关 FMA 链、其他执行依赖；它不等同于 `__syncthreads()` |
| `LG Throttle` | local/global memory 指令队列满 | global/local 指令密度、spill |
| `Math Pipe Throttle` | 目标数学管线过度订阅 | 对应算术 pipe utilization |
| `Not Selected` | warp 已 ready，但本周期选择了别的 warp | 通常说明 TLP 足够，不是首要坏事 |

本报告每条 issued instruction 的主要 stall（只列最大的几项）：

| kernel | MIO Throttle | Long Scoreboard | Barrier | Short Scoreboard | Not Selected |
|---|---:|---:|---:|---:|---:|
| naive | 22.39 | 6.41 | 5.59 | 0.74 | 3.71 |
| v1 | 6.71 | 2.34 | 2.44 | 1.29 | 1.67 |
| v2&lt;1&gt; | 11.70 | 1.01 | 3.59 | 2.48 | 3.96 |
| v2&lt;0&gt; | 12.73 | 1.42 | 2.43 | 0.96 | 3.13 |

对这几个 SGEMM，`MIO Throttle + L1/TEX 高利用率 + 大量 shared load` 三项互相印证：shared-memory/MIO 路径是当前最值得优化的方向。不能仅凭 `MIO Throttle` 名称就下结论；是代码结构和其他计数器让这个推断成立。

## 8. 第五站：Memory Workload，数清楚数据指令

对 SGEMM，单看 GByte/s 容易受频率和执行时间影响。更适合跨版本比较的是“完成同样工作执行了多少 load/store 指令”。

| SASS 动态指令 | naive | v1 | v2&lt;1&gt; | v2&lt;0&gt; |
|---|---:|---:|---:|---:|
| global load | 16.78 M | 8.39 M | 8.39 M | 8.39 M |
| global store | 0.131 M | 0.131 M | 0.131 M | 0.131 M |
| shared load | 167.77 M | 150.99 M | 67.11 M | 67.11 M |
| shared store | 16.78 M | 8.39 M | 8.39 M | 8.39 M |

这是理解版本演进最有价值的一张表：

- v1 的更大 block tile 把 global load 指令减半。
- 但 v1 的 shared load 只比 naive 少约 10%，同时引入大量循环/地址指令，所以收益没有兑现。
- v2 把 shared load 降到 naive 的约 40%，这是 v2 明显进步的核心证据。
- v2 的 L1/TEX 仍接近满载，说明下一步仍应提高“每次 shared→register 读取能支持多少 FMA”的复用率。

### 8.1 如何判断 shared bank conflict

不要看到任意 `bank conflicts` 计数非零就立即加 padding。GUI 的 Memory Workload 表和 Source 页面中应同时检查：

- shared requests 与 wavefronts；
- excessive wavefronts / conflict ratio；
- 哪一条 SASS `LDS/STS` 被归因；
- 修改 padding 后实际时间和 MIO/short-scoreboard 是否改善。

本报告的 `derived__memory_l1_wavefronts_shared_excessive` 为 0，因此没有证据表明 bank conflict 是当前主因。v2 的计算访问模式本身也很规整：A 对 warp 是广播，B 对 warp 是连续 bank。

## 9. 第六站：Occupancy，检查 TLP 的资源上限

Occupancy 是 active warps 相对硬件最大值的比例。它是隐藏延迟的手段，不是性能目标。

| 指标 | naive | v1 | v2&lt;1&gt; | v2&lt;0&gt; |
|---|---:|---:|---:|---:|
| Registers / Thread | 40 | 55（按 56 分配） | 40 | 48 |
| Static Shared / Block | 2 KiB | 2 KiB | 8 KiB | 8 KiB |
| Register Block Limit | 6 | 4 | 6 | 5 |
| Theoretical Occupancy | 100% | 66.67% | 100% | 83.33% |
| Achieved Occupancy | 98.67% | 64.99% | 95.43% | 80.06% |

GUI 中看 `Block Limit Registers/Shared Mem/Warps`，最小值是驻留 block 数的限制项：

- v1 被寄存器限制到 4 blocks/SM。
- v2&lt;0&gt; 被寄存器限制到 5 blocks/SM。
- v2&lt;1&gt; 可以达到 6 blocks/SM。

但 v2&lt;0&gt; 并没有因为 occupancy 较低就显著变慢，这正好说明“100% occupancy 不是必要条件”。只有当 eligible warp 不足、相关 stall 无法隐藏时，occupancy 下降才会转化为明确损失。

可以在 `Tools → Occupancy Calculator` 中改变 block size、寄存器数和 shared memory，观察理论驻留 block/warp 如何变化。它只能计算资源上限，不能预测最终性能。

## 10. 第七站：Source 与 SASS，把指标对应回代码

报告由 `--import-source yes` 生成，GUI 可以显示源码；即使源码路径在另一台机器不可用，也可以查看导入的 source/SASS。

建议操作：

1. 选择某个 kernel，进入 **Source Page**。
2. 在 `View` 中同时显示 CUDA-C 和 SASS。
3. `Navigate By` 依次选择：
   - Instructions Executed；
   - Warp Stall Sampling (Not Issued)；
   - Attributed Stalls；
   - Memory 指标。
4. 点击最热的 CUDA 行，观察对应生成了多少 `LDS`、`FFMA`、地址计算和分支。
5. 对 v2 两版本使用 `Compare → Source Comparison`。

注意“stall 归因”常指向产生 scoreboard 的 producer 指令，而不一定是后来真正等待它的 consumer 行。

## 11. v2&lt;1&gt; 与 v2&lt;0&gt;：编译器是否把实验优化掉了

结论：**没有编译成完全相同的代码，但手写循环顺序没有带来稳定收益。**

证据如下：

| | v2&lt;1&gt; | v2&lt;0&gt; |
|---|---:|---:|
| 动态总指令 | 311.4 M | 282.8 M |
| 寄存器/线程 | 40 | 48 |
| Achieved Occupancy | 95.43% | 80.06% |
| Eligible warps/scheduler | 2.24 | 1.69 |
| NCU 单次时间 | 925.15 µs | 916.10 µs |

进一步用 `cuobjdump --dump-sass build/sgemm` 检查当前二进制：

- `v2<0>` 的计算主体静态展开出 128 条 FFMA，使用更多临时寄存器；编译器对这些指令重新调度，并非机械保留源码中的四条长依赖链。
- `v2<1>` 的主体更紧凑，静态可见约 32 条 FFMA，并保留循环执行，因此寄存器少但动态循环控制指令更多。
- 两者数学上的 FFMA 数相同，只是静态展开、调度、寄存器生命期和循环开销不同。

所以不能用源码循环顺序直接推断机器 ILP。正确实验流程是：

```text
改源码循环/pragma
  → 检查 SASS 是否真的变化
  → 检查 registers、instruction count、occupancy
  → 重复 benchmark 判断性能差异是否超过噪声
  → 用 scheduler/stall 解释差异
```

本实验的合理结论不是“ILP 无效”，也不是“编译器完全优化掉了”，而是：

> 四个累加器已经给编译器提供了可利用的独立性；仅交换两层循环，编译器会选择不同的展开与调度策略，但当前 kernel 的主压力仍在 shared-memory/MIO 路径，因此这项局部改动没有产生显著加速。

## 12. 对四个 kernel 的最终诊断

### naive

- 优点：简单，40 registers，接近满 occupancy。
- 主要问题：每个 thread 只有一个输出；shared load 达 167.8 M，MIO Throttle 22.39，FMA pipe 仅 8.54%。
- 优化方向：增大 register tile，提高 shared 数据对多个输出的复用。

### v1

- 做对了：block 输出 tile 扩大到 32×32，global load 指令减半。
- 没兑现：shared load 仍有 151.0 M；总指令升到 759.6 M；55 registers 把 occupancy 压到约 65%。
- 优化方向：简化索引/循环，把 register tiling 写成显式外积，减少 shared load 和控制指令。

### v2&lt;1&gt;

- 进步：shared load 降到 67.1 M，总指令降到 311.4 M，FMA pipe 提升到 21.4%，40 registers 保持高 occupancy。
- 剩余瓶颈：L1/TEX 约 97.5%，MIO Throttle 11.70；4×1 thread tile 主要复用 B，A 方向仍需较多 shared 读取。
- 优化方向：学习 2D thread/register tile，让寄存器中的 A 和 B 都被多个 FMA 复用。

### v2&lt;0&gt;

- 编译器生成更激进的静态展开，动态指令更少。
- 代价是 48 registers、较低 occupancy 和更少 eligible warps。
- 与 v2&lt;1&gt; 性能接近，适合作为“源码、SASS、资源、性能不一一对应”的教学案例，不适合作为继续微调循环顺序的理由。

## 13. 每次分析都填写的实验记录模板

```markdown
### 实验名称

- 唯一改动：
- 正确性尺寸：
- benchmark 尺寸、预热、重复次数：
- baseline 时间 / GFLOPS：
- 新版本时间 / GFLOPS：

#### 预先假设

- 我预计哪个指标变化？为什么？

#### NCU 证据

- SOL：
- FMA / LSU / MIO pipe：
- global/shared 指令或字节：
- registers / occupancy：
- eligible / no eligible：
- top stalls：
- SASS 是否符合源码意图：

#### 结论

- 假设被支持还是被证伪：
- 下一次只改变什么：
```

## 14. 附录：推荐资料

### 简体中文优先

1. [NVIDIA Nsight Compute 中文产品页](https://developer.nvidia.cn/nsight-compute)：先建立工具能回答什么问题的整体认识。
2. [NVIDIA CUDA 编程手册系列：CUDA 编程模型接口](https://developer.nvidia.cn/blog/cuda-programming-model-interface-cn/)：包含 tiled 矩阵乘和 shared memory 复用。
3. [在 CUDA C/C++ 中使用共享内存](https://developer.nvidia.cn/blog/using-shared-memory-cuda-cc/)：shared memory、同步和合并访问入门。
4. [NVIDIA 加速计算学习路径](https://www.nvidia.cn/training/learning-path/accelerated-computing/)：有中文 CUDA C++ 基础课程，也列出了 Nsight 分析课程。
5. [CUDA C++ Programming Guide PDF（NVIDIA 中国镜像）](https://docs.nvidia.cn/cuda/pdf/CUDA_C_Programming_Guide.pdf) 与 [Best Practices Guide PDF](https://docs.nvidia.cn/cuda/pdf/CUDA_C_Best_Practices_Guide.pdf)：镜像页面主体仍以英文为主，但属于最可靠的参考手册。

### 英文官方资料

1. [Nsight Compute UI User Guide](https://docs.nvidia.com/nsight-compute/2025.3/NsightCompute/index.html)：GUI 页面、Baseline、Source Comparison 的权威说明。
2. [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/2025.3/ProfilingGuide/index.html)：SOL、Scheduler、Warp State 和每种 stall 的定义。
3. [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)：profiling、coalescing、occupancy、shared memory 和 async copy。
4. [CUTLASS: Fast Linear Algebra in CUDA C++](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)：理解生产级 GEMM 的 block/warp/thread 分层 tiling 与 double buffering。

阅读文档时，以当前 CUDA/Nsight 版本的官方定义为准。论坛和博客适合建立直觉，但不能替代指标定义。

### 不打开 GUI 时如何复核报告

GUI 是主要学习工具，但下面两条命令适合确认数值或保存实验记录：

```bash
# 按 kernel 打印各 section 的摘要
ncu --import report/sgemm_nv1v2_2.ncu-rep \
    --page details \
    --print-summary per-kernel

# 导出报告中采集的原始 metrics；列数很多，通常再用脚本筛选
ncu --import report/sgemm_nv1v2_2.ncu-rep \
    --page raw \
    --csv \
    --print-units base
```

CLI 导出和 GUI 读取的是同一份 `.ncu-rep`，很适合把关键指标纳入版本化的实验表格；不要为了方便而只截 GUI 图片、丢失 kernel 名称和测量条件。
