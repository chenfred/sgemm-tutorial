# 从 v2 到 v3：学习路线与实验顺序

## 1. 先给结论

从学习效率和基础能力建设看，v2 之后不建议继续把主要时间投入到“交换 reduce 循环、增加几个独立累加器”的孤立 ILP 实验，也不建议立刻把 double buffering 塞进当前 kernel。

更合适的顺序是：

```text
可靠 benchmark 与实验方法
  → 看懂 SASS 和寄存器/occupancy
  → 2D register tiling（显式外积）
  → 参数化并比较 tile 形状
  → vectorized/coalesced load
  → 再学习 double buffering / async copy
  → 最后接触 warp tiling、Tensor Core、CUTLASS
```

对 v3，推荐主题是：

> **二维寄存器 tiling：每线程计算 `TM×TN` 个输出，显式把 A、B 从 shared memory 读入寄存器，再做小型外积。**

这一步既继续发展 ILP，又直接减少当前报告暴露出的 shared-memory/MIO 压力，比单纯调整 reduce 循环更有学习价值。

## 2. 为什么现在不应只继续研究 ILP

v2 已有四个独立累加器：

```cpp
float regs[4];
```

`v2<1>` 与 `v2<0>` 的报告证明：

- 编译器确实根据循环结构生成了不同 SASS；
- 一个版本用较少寄存器和更多动态指令，另一个更充分展开但占更多寄存器；
- 两者最终性能仅相差约 1%，没有稳定的数量级收益；
- v2 的 FMA pipe 约 21%，而 L1/TEX active throughput 约 98%，最大 stall 是 MIO Throttle。

这说明当前最重要的问题不是“有没有四条独立累加链”，而是：

> 每次从 shared memory 取出的数据，只支持了多少有效 FMA？

v2 的 `4×1` thread tile 会让一个 B 值服务四个输出，但没有同时在 N 方向形成寄存器复用。继续改循环顺序不会改变这个数据复用比。

## 3. 为什么也不应马上上 double buffering

double buffering 主要解决“搬运下一块数据”与“计算当前块数据”不能重叠的问题。典型目标是降低：

- global-memory load latency；
- barrier 前后的空等；
- load/compute 串行阶段带来的 pipeline 气泡。

而当前 v2 的证据是：

- DRAM throughput 只有约 4%～6%；
- Long Scoreboard 约 1.0～1.4 cycles/issued instruction，不是最大项；
- Barrier 约 2.4～3.6；
- MIO Throttle 约 12，L1/TEX 约 98%；
- shared load 仍有 67.1 M。

因此 double buffering 可能有帮助，但它不会自动减少 compute 阶段的 shared→register 流量，甚至会增加 shared memory、寄存器、同步和代码复杂度。如果先做它，很容易出现“代码复杂很多，但不知道为什么没变快”。

正确的学习顺序是先把单缓冲 compute tile 写清楚并提高寄存器复用；当报告显示 global load、Long Scoreboard 或 barrier 成为更明显的限制时，再用 double buffering 解决它们。

## 4. 阶段 0：先修好测量闭环

在写 v3 前，先把 benchmark 与 correctness 分开。当前 `main.cpp` 每个 kernel 只计时一次，第一次 launch、频率状态和偶然噪声都会影响结果。

建议的 benchmark 结构：

1. 分配和初始化只做一次。
2. kernel 预热 5～10 次。
3. CUDA Event 包住连续 50～200 次 kernel launch。
4. 除以迭代数得到单次时间。
5. 整个 benchmark 重复若干轮，报告中位数或稳定区间。
6. correctness 单独跑，继续覆盖可整除和不可整除尺寸。
7. 保留固定的主性能尺寸，并至少再选一个不同形状，避免只对单个矩阵过拟合。

建议记录：

```text
GPU 型号、驱动、CUDA 版本、编译选项
M/N/K、预热次数、测量次数
平均/中位时间、GFLOPS、正确性
```

这一阶段的能力目标是：能可靠地区分 1% 噪声和 10% 优化。

## 5. 阶段 1：掌握源码到 SASS 的映射

v2 两版本已经是很好的练习材料。

使用：

```bash
cuobjdump --dump-sass build/sgemm
```

或者在 NCU Source Page 中并排看 CUDA-C/SASS，重点识别：

- `FFMA`：有效浮点乘加；
- `LDS/STS`：shared load/store；
- global load/store；
- `BAR`：同步；
- 循环分支和地址计算；
- `.reuse` 等编译器调度提示。

你不需要立即读懂每一个编码位。先能回答以下问题：

1. 内层循环是否展开？
2. 一轮 K tile 有多少 FFMA、LDS 和分支？
3. 独立累加器是否在 SASS 中交错？
4. 修改后寄存器为什么增加或减少？
5. 有没有 local load/store，也就是 register spill？

完成标准：能够解释 `v2<0>` 为什么动态指令少却使用更多寄存器，以及为什么这不保证更快。

## 6. 阶段 2：v3 的核心——二维 register tiling

### 6.1 从点积思维转向小外积

当前 v2 每线程计算 `4×1` 输出。v3 可以先尝试 `4×4`：

```cpp
float accum[TM][TN] = {};

for (int k = 0; k < BK; ++k) {
    float regA[TM];
    float regB[TN];

    // shared → registers
    for (int i = 0; i < TM; ++i)
        regA[i] = tileA[...][k];
    for (int j = 0; j < TN; ++j)
        regB[j] = tileB[k][...];

    // register outer product
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j)
            accum[i][j] += regA[i] * regB[j];
}
```

一次 K 迭代中：

```text
shared load: TM + TN
FMA:         TM × TN
```

当 `TM=TN=4`：

```text
8 次 shared 标量读取 → 16 次 FMA
```

相比只沿一个方向扩展，这同时复用 A 和 B，并自然提供 16 条累加链。这是“数据复用带来 ILP”，而不是为了 ILP 人为堆临时变量。

### 6.2 推荐一个便于学习的起点

可以从以下逻辑配置开始，不要求第一次就最优：

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
16 threads × 4 outputs/thread = 64 rows
16 threads × 4 outputs/thread = 64 cols
```

它的优点是与 v1 的 `(16,16)` thread layout 容易对照，而且每个 thread tile 是规则的 4×4。实现时必须重新设计 cooperative load，不能直接照搬 v1 的复杂循环。

注意：这个配置是教学起点，不是对 RTX 5080 的最优承诺。最终选择必须由 benchmark 和 NCU 决定。

### 6.3 v3 必须回答的指标问题

与 v2 比较时，预期方向是：

| 指标 | 期望变化 | 原因 |
|---|---|---|
| shared load / 有效 FMA | 明显下降 | A/B 都在寄存器中复用 |
| MIO Throttle | 下降 | shared 指令密度降低 |
| FMA pipe utilization | 上升 | 更多时间用于乘加 |
| registers/thread | 上升 | 16 个 accumulator + A/B 临时值 |
| occupancy | 可能下降 | 合理代价，不要求保持 100% |
| local spill | 必须保持 0 | spill 会破坏 register tiling 收益 |
| elapsed time/GFLOPS | 明显改善才算成功 | 指标只是解释，不是目标 |

如果 registers 增加、occupancy 降低，但 eligible warps 仍足够且性能上升，这是成功；不要为了恢复 100% occupancy 把有效 register tile 删除。

## 7. 阶段 3：参数化与小规模设计空间实验

v3 正确后，不要立刻手写 v4。先把以下值改成编译期模板参数：

```text
BM, BN, BK, TM, TN
```

只测试少量有明确目的的组合，例如：

| 实验 | 想回答的问题 |
|---|---|
| `TM×TN = 4×2, 4×4, 8×4` | 更多寄存器复用何时被寄存器压力抵消？ |
| `BK = 8, 16, 32` | 更少 barrier 是否值得更多 shared/寄存器？ |
| `BM×BN = 64×64, 64×128` | block tile 复用与 wave 数如何权衡？ |

每次只改变一个维度，并记录：

- 时间/GFLOPS；
- registers、shared memory、occupancy；
- shared load、FMA pipe、MIO Throttle；
- SASS 展开和 spill。

能力目标不是找到一个神奇数字，而是能解释性能曲线为什么先升后降。

## 8. 阶段 4：global load 与 cooperative loading

二维 register tiling 稳定以后，再优化 global→shared：

1. 用一维 `tid` 明确分配 tile 中的元素。
2. 验证 warp 访问连续地址，确保合并访问。
3. 在满足地址和矩阵边界对齐时尝试 `float4` 向量化。
4. 对边缘 tile 保留正确的标量或 predicated 路径。
5. 用 NCU 检查 sectors/request、global load 指令数，而不是只看源码看起来是否连续。

向量化的主要价值可能是减少指令和地址计算，并不保证显存带宽上升；当前 DRAM 并未饱和，评价时要看整体 kernel 时间。

## 9. 阶段 5：这时再学 double buffering

### 9.1 先学概念模型

单缓冲：

```text
load tile k → sync → compute tile k → sync → load tile k+1
```

双缓冲：

```text
buffer 0: compute tile k
buffer 1: load tile k+1
下一轮交换 buffer
```

但“数组写成两份”不等于实现了重叠。必须确认加载操作可以与计算并行推进，并正确管理 producer/consumer 同步。

### 9.2 分两步学习

1. **普通 ping-pong shared buffer**：先理解 buffer 所有权、边界 tile、同步位置和 prologue/main-loop/epilogue。
2. **异步 global→shared copy**：再学习 `cuda::memcpy_async` / `cuda::pipeline` 或对应底层机制，让 copy 真正与计算重叠。

加入 double buffering 后，重点观察：

- Long Scoreboard、Barrier 是否下降；
- eligible warps 和 issue rate 是否上升；
- shared memory 翻倍是否降低驻留 block；
- registers 是否增加；
- 总时间是否改善。

如果 MIO/shared throughput 仍是主墙而 global latency 已经不突出，double buffering 可能收益有限。这也是一个有效实验结论。

## 10. v3 前建议补齐的基础知识

按优先级排序：

### 必须掌握

- warp、scheduler、eligible warp、scoreboard；
- latency 与 throughput 的区别；
- TLP、ILP 与 occupancy 的相互补偿；
- global coalescing、32-byte sectors；
- shared memory bank、broadcast 和 conflict；
- register allocation、occupancy limit、spill；
- block tile、thread tile 和算术强度；
- CUDA Event 的可靠 benchmark 方法。

### v3 过程中掌握

- 编译期模板参数和 `#pragma unroll`；
- SASS 中的 FFMA/LDS/STS/branch/barrier；
- outer-product register tiling；
- NCU baseline、Source Comparison、stall attribution。

### v3 后再深入

- double buffering 与 async copy；
- warp-level tiling 和 shuffle；
- Roofline 的分层版本（DRAM/L2/shared）；
- CUTLASS 的 threadblock/warp/thread hierarchy；
- Tensor Core、WMMA/MMA/cuTile。

Tensor Core 应当放在后面，不是因为它不重要，而是因为当前 CUDA-core SGEMM 正好能训练数据移动、寄存器、调度和 profiler 推理。过早切换到高层 MMA 会绕过这些学习目标。

## 11. v3 的验收标准

实现 v3 前先写下假设，实现后逐项验收：

### 正确性

- 标准语义仍为 `C(M,N)=A(M,K)×B(K,N)`。
- 可整除尺寸 PASS。
- `M != N != K` 且各维不能被 tile 整除时 PASS。
- 无越界、无 race，边界 tile 正确补零或 predication。

### 代码结构

- `BM/BN/BK/TM/TN` 含义清楚。
- global→shared、shared→register、register compute 三层数据移动可明确指出。
- compute 主体是显式 `regA × regB` 外积。
- 线程映射和 cooperative load 不依赖难以解释的魔法索引。

### 性能证据

- benchmark 有预热和重复测量。
- 对比 v2 的时间/GFLOPS，而不是只对比 NCU Duration。
- shared load/FMA 比下降。
- FMA pipe 利用率提高或至少有效工作比例提高。
- MIO Throttle 有合理变化解释。
- registers/occupancy 的代价可解释。
- local spilling 为 0。
- SASS 与设计意图一致。

达到这些标准后，再做 double buffering，你会清楚它在流水化哪一层、应该改变哪些指标，也能判断它是否值得保留。

## 12. 推荐阅读顺序

### 简体中文

1. [CUDA 编程模型接口：shared-memory tiled matmul](https://developer.nvidia.cn/blog/cuda-programming-model-interface-cn/)
2. [在 CUDA C/C++ 中使用共享内存](https://developer.nvidia.cn/blog/using-shared-memory-cuda-cc/)
3. [NVIDIA Nsight Compute 中文产品与入门页](https://developer.nvidia.cn/nsight-compute)
4. [NVIDIA 加速计算学习路径](https://www.nvidia.cn/training/learning-path/accelerated-computing/)

### 英文官方资料

1. [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)：先读 memory optimization、occupancy、async copy。
2. [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/2025.3/ProfilingGuide/index.html)：配合本仓库报告逐项查定义。
3. [CUTLASS: Fast Linear Algebra in CUDA C++](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)：在写完二维 register tile 后再读，会更容易理解分层 tiling 和 double buffering。
4. [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)：异步 SIMT、`cuda::pipeline` 和硬件执行模型的权威参考。

最终路线可以概括为：

> **先让每次数据搬运产生更多计算，再让数据搬运和计算重叠。**

二维 register tiling 解决前半句；double buffering 解决后半句。按这个顺序学习，性能现象更容易解释，能力也更可迁移。
