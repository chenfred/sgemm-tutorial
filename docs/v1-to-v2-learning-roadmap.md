# 从 v1 到 v2：学习路线与实验顺序

## 1. 先给结论

从学习效率和基础能力建设看，当前最值得做的下一步是：

> **二维 register tiling：每线程计算 `TM×TN` 个输出，把 A、B 的 shared-memory 数据先读入寄存器，再做小型外积。**

不建议把主要时间继续投入到“交换 reduce 循环、手工排列几条 FMA”的孤立 ILP 实验，也不建议现在就把 double buffering 塞进 v1。

推荐顺序：

```text
可靠 benchmark 与实验方法
  → 看懂当前 v1 的 SASS、registers 与 scheduler
  → 验证 shared-store bank-conflict 线索
  → 2D register tiling（v2）
  → 少量 tile 形状实验
  → cooperative/vectorized global load
  → double buffering / async copy
  → warp tiling、Tensor Core、CUTLASS
```

## 2. 先校正一个关键认识：register tiling 不只是为了 ILP

当前 v1 每线程有四个累加结果：

```cpp
float regs[4] = {};
```

它同时产生三种收益：

1. **数据复用**：一个 B 值可以参与四个输出的 FMA。
2. **减少指令与线程**：每线程写四个 C 元素，总线程数降到 v0 的四分之一。
3. **提供 ILP**：四条累加依赖链相互独立，编译器可以交错调度 `FFMA`，隐藏单条依赖链的 latency。

报告中最强的量化证据是：

| 指标 | v0 | v1 |
|---|---:|---:|
| shared load | 167.77 M | 67.11 M |
| global load | 16.78 M | 8.39 M |
| executed instructions | 472.91 M | 282.76 M |
| FMA pipe（elapsed） | 9.96% | 22.37% |
| issued warp/scheduler | 0.28 | 0.41 |
| MIO Throttle | 21.94 | 12.37 |

所以更准确的表述是：

> register tiling 通过“让寄存器中的数据支持更多计算”提高复用，同时自然产生多个独立累加链；ILP 是重要收益之一，但不是唯一目的。

如果只增加独立变量，却没有提高数据复用、减少 shared/global 指令或改善真实时间，这只是“为 ILP 而 ILP”。

## 3. 多线程为什么没有自动把流水线填满

v0 已经有非常高的线程级并行性：

- theoretical occupancy 100%；
- achieved occupancy 98.67%；
- 每 scheduler 平均 11.84 个 active warps，接近硬件上限 12。

但它仍然只有：

- 1.30 个 eligible warps/scheduler；
- 0.28 issued warp/cycle；
- 71.50% 的周期没有 eligible warp。

原因是“线程之间没有数据依赖”不等于“它们使用的硬件资源互不冲突”。很多 warp 可以同时遇到同一种限制：

```text
大量 shared load
  → MIO 指令队列接近满载
  → 多个 warp 一起 MIO Throttle
  → active warp 很多，但下一条指令都没准备好
  → scheduler 仍找不到 eligible warp
```

v1 的 occupancy 降到 80.08%，active warps 也更少，但由于 shared 指令压力下降、四条累加链提供 ILP，它反而有 1.70 个 eligible warps 和 0.41 的发射率。

这正是 TLP 与 ILP 的边界：

- TLP 用其他 warp 隐藏当前 warp 的等待；
- ILP 用同一线程/warp 内的独立指令隐藏依赖 latency；
- 如果共享硬件队列本身已经拥堵，继续增加 TLP 可能只会让更多 warp 一起排队；
- 最有效的办法往往是减少对该队列的请求数量，让每个请求产生更多有效计算。

## 4. 为什么不继续只研究循环顺序

当前 v1 的源码写成四个输出分别完成 K 循环，但 Source/SASS 中可以看到编译器进行了大规模展开和重新调度，并交错放置 `LDS`、`LDS.128` 与 `FFMA`。

这意味着：

- 源码的循环先后顺序不等于最终硬件执行顺序；
- 四个累加器已经把独立性暴露给编译器；
- 仅交换两层循环，可能主要改变 unroll、寄存器分配和调度，而不会改变算法的数据复用比；
- 当前最大 stall 仍是 MIO Throttle，而不是 Math Pipe Throttle。

可以把循环顺序保留为一个小型编译器实验，但不应把它作为下一个版本的主线。

正确实验方法是同时比较：

```text
真实时间
动态指令数
registers/thread
occupancy 与 spill
LDS/FFMA 的 SASS 排列
eligible warp 与 stall
```

## 5. 为什么现在也不应先上 double buffering

double buffering 主要解决：

```text
global→shared 搬运下一块数据
与
使用当前 shared tile 进行计算
无法重叠
```

它通常针对 global-load latency、Long Scoreboard 和同步阶段气泡。当前 v1 的报告却显示：

| 指标 | v1 |
|---|---:|
| DRAM Throughput | 4.74% |
| Long Scoreboard | 1.80 |
| Barrier | 2.60 |
| MIO Throttle | 12.37 |
| L1/TEX Throughput | 97.40% |
| shared load | 67.11 M |

因此目前更像是 shared-memory 指令密度过高，而不是 DRAM 带宽耗尽或 global latency 成为第一主因。

double buffering 还会带来：

- shared buffer 近似翻倍；
- 更复杂的 producer/consumer 同步；
- 可能增加寄存器与地址状态；
- 可能降低驻留 block 数；
- 如果没有异步 copy，只写两个数组并不会自动重叠。

它可能最终有收益，但现在学习它容易出现“代码复杂很多，主瓶颈却没变”。先减少 shared→register 请求，再让 global→shared 与 compute 重叠，因果关系更清晰。

## 6. 阶段 0：先把测量闭环补齐

### 6.1 当前已经完成的部分

- `main.cpp` 通过实现列表自动在每个正式 kernel 前运行独立 warmup。
- `cudaProfilerStart/Stop` 精确标记正式 kernel，不依赖 launch 次数或版本名过滤。
- `profile.sh` 使用 Application Replay，使 NCU 根据指标集决定的每个采集 pass 都重新执行 warmup。
- `--clock-control none` 不修改 GPU 时钟；正式报告中 v0/v1 的 SM frequency 为 2.950/2.936 GHz，只差约 0.47%，DRAM frequency 均约为 14.99 GHz。
- 当前报告只含 v0/v1，适合初学者逐项比较。
- 普通运行会先由多线程 `sgemm_golden` 生成一次 CPU 参考结果，再由 `sgemm_verify` 用 FP32 混合容差检查各 kernel；`--dry-run` 只用于 profiler，跳过这两步但保留数据生成、H2D、kernel 和 D2H。

### 6.2 写 v2 前仍应补的 benchmark

当前程序只对正式 kernel 计时一次。要可靠区分 1%～5% 的变化，建议以后把 correctness 与 benchmark 分开：

1. 分配、初始化和 H2D copy 只做一次。
2. 每个 kernel warmup 到频率和耗时进入稳态（可先以 5～10 次为起点，再用数据验证）。
3. CUDA Event 包住 50～200 次连续 launch。
4. 总时间除以迭代数。
5. 整个过程重复若干轮，记录中位数和波动区间。
6. correctness 单独执行，不进入性能计时。

建议记录：

```text
GPU、driver、CUDA、编译选项
M/N/K、warmup 策略、测量次数
中位时间、波动范围、GFLOPS
registers、occupancy、spill
```

### 6.3 正确性不能只剩当前一个尺寸

为让 NCU 报告简单，`main.cpp` 当前只保留 `1024×4096×1024`。但开发 v2 时，必须临时恢复至少一个非整除、非方阵用例，例如：

```text
M=1000, N=2000, K=1500
```

它用于验证 M/N/K 边界、补零和 grid 映射。优化通过后，再决定 profiling 入口是否只保留主尺寸。

## 7. 阶段 1：先做一个小型 bank-conflict 实验

当前 NCU 规则指出，v1 的 shared store 存在约 1.2-way conflict：

- shared-store requests：8.39 M；
- bank conflicts：约 1.38 M；
- 约占 shared-store wavefront 的 14.13%。

这适合作为 v2 前的小练习，因为改动范围小，能训练“指标 → Source → 实验”的闭环。

推荐步骤：

1. 重复采两次报告，确认数值不是采样偶然性。
2. 在 Source/SASS 中找到 `tileA`、`tileB` 对应的 `STS`。
3. 分别只给一个 shared tile 加 padding 或改变 layout。
4. 每次检查 conflict、MIO Throttle、registers 和真实时间。
5. 如果 conflict 降低但时间不变或变差，记录原因并撤销，不为指标而保留代码。

不要一开始同时改 tileA、tileB、vector width 和 block layout，否则无法知道是哪项产生效果。

## 8. 阶段 2：掌握源码到 SASS 的最小集合

可以在 NCU Source 页学习，也可以使用：

```bash
cuobjdump --dump-sass build/sgemm
```

先只识别：

- `FFMA`：有效 FP32 乘加；
- `LDS` / `LDS.128`：shared load；
- `STS`：shared store；
- `LDG` / `STG`：global memory；
- `BAR`：同步；
- `BRA`：循环/分支。

对当前 v1 回答五个问题：

1. reduce 循环是否完全或部分展开？
2. 四个累加结果的 `FFMA` 是否交错出现？
3. 一段 compute 中 `LDS` 与 `FFMA` 的比例大约是多少？
4. 为什么 v1 使用 48 registers，而 v0 是 40？
5. 是否出现 local load/store，也就是 register spill？当前答案应为没有。

不需要一开始理解每个 SASS modifier。能把“加载、同步、计算、写回”四段对应起来就足够进入 v2。

## 9. 阶段 3：v2 的核心——二维 register tiling

### 9.1 从多个点积转向寄存器小外积

当前 v1 是 `4×1` thread tile。v2 可先尝试 `4×4`：

```cpp
float accum[TM][TN] = {};

for (int k = 0; k < BK; ++k) {
    float regA[TM];
    float regB[TN];

    for (int i = 0; i < TM; ++i) {
        regA[i] = tileA[...][k];
    }

    for (int j = 0; j < TN; ++j) {
        regB[j] = tileB[k][...];
    }

    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            accum[i][j] += regA[i] * regB[j];
        }
    }
}
```

一轮 K 中的逻辑复用为：

```text
shared values loaded: TM + TN
FMA:                  TM × TN
```

当 `TM=TN=4`：

```text
8 个 shared 标量值 → 16 个 FMA
```

这同时复用 A 与 B，并自然产生 16 条累加链。它是“数据复用带来 ILP”，比人为堆独立临时变量更有迁移价值。

### 9.2 推荐的教学起点

可以从以下配置开始，不承诺第一次就是 RTX 5080 的最优值：

```text
BM = 64
BN = 64
BK = 8 或 16
TM = 4
TN = 4
block = 16 × 16 = 256 threads
```

映射关系：

```text
16 个 thread-y × 每线程 4 行 = 64 行
16 个 thread-x × 每线程 4 列 = 64 列
```

每个 block 计算 C 的 `64×64` tile。实现时要分别画出：

- block tile 的 M/N 范围；
- 每线程 `TM×TN` 输出在 tile 内的位置；
- 每轮需要装入的 A(`BM×BK`) 与 B(`BK×BN`)；
- 256 个线程怎样 cooperative load 这两个 shared tile。

先在纸上验证索引，再写代码。二维 thread tile 的主要 bug 通常来自映射，而不是 FMA 本身。

### 9.3 推荐实现顺序

1. 只写线程到 C tile 的映射，并用小矩阵检查每个输出是否唯一覆盖。
2. 写 global→shared cooperative load，先用标量加载。
3. 正确处理 M/N/K 边界与 K 尾块补零。
4. 写 shared→register 的 `regA/regB`。
5. 写 `TM×TN` 外积。
6. 先跑非方阵 correctness，再做性能测试。
7. 最后才加 `#pragma unroll` 和模板参数。

## 10. v2 应该让哪些指标变化

| 指标 | 期望方向 | 原因 |
|---|---|---|
| shared load / FMA | 明显下降 | A、B 都在寄存器中复用 |
| MIO Throttle | 下降 | shared 指令密度降低 |
| FMA pipeline | 上升 | 更多周期用于有效乘加 |
| executed instructions / FLOP | 下降 | 地址、循环、load 摊薄 |
| registers/thread | 上升 | 16 accumulators + regA/regB |
| occupancy | 可能下降 | 合理代价，不要求 100% |
| eligible warp / issue rate | 不应明显恶化 | ILP 应补偿部分 TLP 损失 |
| local spilling | 必须为 0 | spill 会破坏 register tiling 收益 |
| elapsed time | 稳定下降 | 唯一最终验收标准 |

如果 registers 增加、occupancy 下降，但 eligible warps 足够且时间明显下降，这是成功。不要为了恢复 100% occupancy 删除有效的寄存器复用。

如果 FMA pipeline 上升但总时间不降，要检查：

- 动态指令是否增加太多；
- global/shared load 是否失去合并访问；
- register spill；
- active blocks 是否过低；
- 边界和 predication 是否过重；
- bank conflict 是否恶化。

## 11. 阶段 4：做少量、有问题意识的参数实验

v2 正确后，把以下值做成编译期参数：

```text
BM, BN, BK, TM, TN
```

不要盲目穷举。先做少量、每次只回答一个问题的实验：

| 实验 | 想回答的问题 |
|---|---|
| `TM×TN = 4×2, 4×4, 8×4` | 更多复用何时被寄存器压力抵消？ |
| `BK = 8, 16, 32` | 更少 barrier 是否值得更多 shared/展开？ |
| `BM×BN = 64×64, 64×128` | block tile 复用和 wave 数怎样权衡？ |

每个实验记录：

- benchmark 中位时间/GFLOPS；
- registers、shared memory、occupancy、spill；
- shared load、MIO Throttle、FMA pipe；
- active/eligible/issued warp；
- SASS 是否按预期展开。

目标不是找到一个神奇数字，而是能解释曲线为什么先升后降。

## 12. 阶段 5：再优化 global→shared

二维 register tile 稳定后，再优化 cooperative load：

1. 用一维 `tid` 明确分配 tile 元素。
2. 检查每个 warp 是否访问连续 global 地址。
3. 查看 sectors/request，确认访问合并。
4. 对齐且边界允许时尝试 `float4` 或更宽加载。
5. 边缘 tile 保留正确的 predicated/scalar 路径。

向量化的价值往往是减少 load 指令和地址计算，不一定让 DRAM Throughput 上升。当前 DRAM 并未饱和，评价仍以总时间和指令数为准。

## 13. 何时才进入 double buffering

满足以下条件后再学 double buffering，收益更容易解释：

- 2D register tiling 已正确且没有 spill；
- shared load/FMA 比已经明显改善；
- global→shared 访问已合并；
- 报告中 Long Scoreboard、Barrier 或 load/compute 阶段气泡成为更突出的问题；
- shared memory 翻倍后仍有可接受的驻留 block 数。

先学习普通 ping-pong buffer 的 prologue/main-loop/epilogue，再学习真正的异步 global→shared copy：

```text
单缓冲：load k → sync → compute k → sync → load k+1

双缓冲：compute buffer 0 的 k
        同时准备 buffer 1 的 k+1
        然后交换 buffer
```

“定义两个 shared 数组”不等于产生重叠。必须有能异步推进的 copy 机制和正确的 producer/consumer 同步。

加入后重点观察：

- Long Scoreboard、Barrier 是否下降；
- eligible warp 和 issue rate 是否上升；
- shared memory/寄存器增加是否降低 occupancy；
- 总时间是否改善。

如果 MIO/shared 仍是主墙，而 global latency 并不突出，double buffering 收益有限也是有效结论。

## 14. 建议分四次学习，不要一次写完

### 第一次：只复盘 v0/v1 报告

- 完成 NCU 文档中的 45 分钟练习。
- 能解释 occupancy、active warp、eligible warp 的区别。
- 写出 v1 加速的三条证据。

### 第二次：只看 SASS 与 bank conflict

- 找到 v1 的 `LDS/STS/FFMA/BAR`。
- 做一个单变量 shared layout 实验。
- 无论加速与否，都写明实验结论。

### 第三次：只实现正确的 2D register tile

- 先画索引映射。
- 用可整除和不可整除尺寸验证。
- 不加 vector load、double buffer 或过多模板。

### 第四次：做 profiler 闭环

- 可靠 benchmark。
- 新旧 NCU baseline 对比。
- 检查 shared load/FMA、MIO、FMA pipe、registers、spill。
- 决定下一次只改变哪一个参数。

## 15. v2 的验收标准

### 正确性

- 语义仍为 `C(M,N)=A(M,K)×B(K,N)`。
- `1024×4096×1024` PASS。
- 至少一个 `M != N != K` 且不能被 tile 整除的尺寸 PASS。
- 无越界、race，K 尾块正确补零或 predication。

### 代码结构

- `BM/BN/BK/TM/TN` 含义清楚。
- 能明确指出 global→shared、shared→register、register compute 三层移动。
- compute 主体是 `regA × regB` 外积。
- cooperative load 与 thread tile 映射能画图解释。

### 性能证据

- benchmark 有 warmup、多次迭代和波动范围。
- 对比真实时间，不使用 NCU 运行时程序打印的数秒耗时。
- shared load/FMA 比下降。
- MIO Throttle 有合理变化。
- FMA pipeline 或有效指令比例提高。
- registers/occupancy 代价可解释。
- local spill 为 0。
- Source/SASS 与设计意图一致。

达到这些标准后，再做 double buffering，你会清楚它在流水化哪一层、应该改变哪些指标，也能判断复杂度是否值得。

## 16. 推荐阅读顺序

### 简体中文

1. [CUDA 编程模型接口：shared-memory tiled matmul](https://developer.nvidia.cn/blog/cuda-programming-model-interface-cn/)
2. [在 CUDA C/C++ 中使用共享内存](https://developer.nvidia.cn/blog/using-shared-memory-cuda-cc/)
3. [NVIDIA Nsight Compute 产品与入门页](https://developer.nvidia.cn/nsight-compute)
4. [NVIDIA 加速计算学习路径](https://www.nvidia.cn/training/learning-path/accelerated-computing/)

### 英文官方资料

1. [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)：memory optimization、occupancy、benchmark、async copy。
2. [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)：scheduler、stall、replay、clock control。
3. [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)：warp scheduling、memory hierarchy、asynchronous SIMT。
4. [CUTLASS: Fast Linear Algebra in CUDA C++](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)：写完 2D register tile 后再读分层 tiling 与流水化。

最终路线可以概括成一句话：

> **先让每次数据搬运产生更多计算，再让数据搬运与计算重叠。**

二维 register tiling 解决前半句；double buffering 解决后半句。
