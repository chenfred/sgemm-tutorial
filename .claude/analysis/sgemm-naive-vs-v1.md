# SGEMM naive vs v1 性能分析（静态推测 + ncu 实测修正）

- 日期：2026-07-05 初版（静态）；2026-07-06 用 `report/sgemm.ncu-rep` 实测修正
- 对象：`src/sgemm_naive.cu`、`src/sgemm_v1.cu`（标准语义 `C(M,N)=A(M,K)×B(K,N)`，`sm_120` / CC 12.0 / 84 SM）
- **本文最有价值的地方不是结论，而是「静态分析哪里猜错了、ncu 怎么纠正的」这个过程。** 详见第六节。

## 实测基准

真实耗时来自 `main.cpp` 的 `cudaEvent` 计时（**不是** ncu 的 Duration，原因见第六节）：

| kernel | 1024×4096×1024 | 1000×2000×1500 |
|---|---|---|
| sgemm_naive | 1.788 ms / 4805 GFLOPS | 1.296 ms / 4628 GFLOPS |
| sgemm_v1    | 1.986 ms / 4326 GFLOPS | 1.422 ms / 4219 GFLOPS |

**v1 比 naive 慢约 10–11%。** 两者都只有 ~4.5 TFLOPS，离 FP32 CUDA core 峰值很远 —— 都未优化好。

---

## Q1. naive 的瓶颈在哪里

### 静态推测（初版）
单累加器 `elemSum`、每线程只算 1 个 C 元素、tile 16×16 → ILP=1、计算密度低。

### ncu 实测（@1024×4096×1024）
| 指标 | 值 | 含义 |
|---|---|---|
| L1/TEX Cache Throughput | **97.06%** | shared memory（走 L1/TEX pipe）几乎打满 |
| Compute (SM) Throughput | 92.35% | SM 接近饱和 |
| Memory Throughput | 92.35% | 内存子系统接近饱和 |
| DRAM Throughput | 7.19% | 几乎不打 DRAM（数据在 cache）|
| Issued IPC | 1.13 | 偏低 |
| Warp Cycles / Issued Inst | **42.0** | 每条指令平均要等 42 周期 |
| No Eligible | **71.8%** | 71.8% 的周期没有任何 warp 可发射 |

**实测结论**：naive 真正的瓶颈是 **L1/TEX pipe（shared memory 访问带宽）被打满 97%**。每个 FMA 要 2 次 shared load，shared 访问成了天花板；warp 大量时间在等 shared 数据（Warp Cycles/Inst=42、No Eligible 71.8%），靠 98.7% 的高占用率（很多 warp）轮转掩盖。ILP=1 是真，但不是主导 —— 主导是 **shared memory 访问压力**。

---

## Q2. v1 相对 naive 改了什么；当时的意图

| 项 | naive | v1 |
|---|---|---|
| block 覆盖的 C tile | 16×16 | **32×32**（`BM=BN=32`） |
| 每线程算的 C 元素数 | 1 | **4**（`regs[2][2]`） |
| K 方向步进 | 16 | **8**（`KS`） |
| shared tile 形状 | `tileA/B[16][16]` | `tileA[32][8]`、`tileB[8][32]` |
| shared 用量 | 2KB | 2KB |
| `blockDim` | (16,16)=256 | (16,16)=256（**没变**）|
| 加载方式 | 线程 `(tx,ty)` 直搬 | **线程重排**分别搬 A/B |

**意图**（教科书 SGEMM 优化第一步）：放大 tile 提升 A/B 复用、用寄存器做 C 累加器提升计算/访存比与 ILP、重排线程让 global 加载 coalesced。方向正确。

---

## Q3. v1 为什么反而更慢

### 静态推测（初版，**部分错误**）
初版把「compute 循环把 `p` 放最内层 → 4 个累加器串行归约 → ILP=1」列为**首要嫌疑**。

### ncu 实测：ILP 假说被推翻 ❌

| 指标 | naive | v1 | 解读 |
|---|---|---|---|
| Issued IPC Active | 1.13 | **1.71** | v1 指令吞吐反而高 51%！|
| Issue Slots Busy | 26.8% | **42.5%** | v1 发射槽利用率更高 |
| Warp Cycles / Issued Inst | 42.0 | **18.2** | v1 每 instruction 等得更少 |
| No Eligible | 71.8% | **57.5%** | v1 stall 更少 |

→ **v1 的 ILP/发射效率明显比 naive 好**，IPC 1.71 vs 1.13。「ILP=1 拖累吞吐」这个假说**被实测证伪**。原因见第六节解释：GPU 藏延迟主要靠多 warp（TLP），只要占用率够，单 warp ILP=1 不是问题；而 v1 虽占用率低，但 warp 内 stall 少。

### ncu 实测：真正的主因 ✅

**主因 1：v1 的总指令数多了 62%**
| | naive | v1 |
|---|---|---|
| Executed Instructions | 4.69e8 | **7.60e8**（+62%） |

多出来的几乎全是**非 FMA 的整数/地址指令**：线程重排的取模除法（`tidx % / `）、多层循环边界（`i*blockDim.y`）、复杂 shared 索引、copy-in 的嵌套循环。naive 的 copy-in 是一条直赋值、compute 是单层循环，指令极简。

**主因 2：占用率从 100% 掉到 66.7%（被寄存器拖累）**
| | naive | v1 |
|---|---|---|
| Registers Per Thread | 40 | **55** |
| Block Limit Registers | 6 | **4** |
| Theoretical Occupancy | 100% | **66.67%** |
| Achieved Occupancy | 98.7% | **65.0%** |

v1 的 `regs[2][2]` + 复杂索引让每线程用到 55 个寄存器，导致每 SM 只能驻留 4 个 block（`64K / (55×256) ≈ 4.5`）→ 占用率 66.7%。这是 `Block Limit Registers` 顶住的直接结果。

### 算术验证（把两个主因合起来，能解释 11% 的差距）

```
时间 ∝ 指令数 / (IPC × 占用率因子)
v1/naive ≈ (7.60/4.69) / (1.71/1.13) = 1.62 / 1.51 ≈ 1.07
再叠加占用率下降（98.7%→65%）对延迟隐藏的削弱 → ≈ 1.10
实测 = 1.986/1.788 = 1.11  ✓
```

**定论**：v1 慢的根因是 **「指令开销爆炸（+62%）」+「寄存器把占用率压到 66.7%」**，二者合力抵消了 v1 在 ILP/IPC 上的改进。ILP 不是问题，`KS=8` 的 sync 翻倍也不是主导。

---

## Q4（实测补充）两者共同的墙：L1/TEX 带宽

| | naive | v1 |
|---|---|---|
| L1/TEX Cache Throughput | **97.1%** | **94.8%** |
| Compute (SM) SOL | 92.4% | 94.1% |
| DRAM Throughput | 7.2% | 3.1% |
| L1/TEX Hit Rate | 7.3% | 79.4% |

两个 kernel 的 **L1/TEX pipe 都接近打满（~95–97%）** —— shared memory 访问是共同的天花板。这也解释了为什么两者 GFLOPS 都不高：**每个 FMA 要 2 次 shared load，shared 带宽顶住了**。v1 的 L1/TEX hit rate 79% 远高于 naive 的 7%（block 大、L1 内复用好），但 SOL 已经饱和，省下的流量换不成更多算力。

---

## Q5. 给 v2 的建议（按实测重排）

1. **降低 shared memory 访问压力（首要）**：放大 register tile（4×4 或 8×8），让每次 shared load 喂更多 FMA → 把 L1/TEX 95% 的压力降下来。这才是 SGEMM 提算力的关键。
2. **压寄存器 / 提占用率**：v1 被寄存器（55）卡到 66.7%。v2 要么精简索引让寄存器回落到能上 6 block/SM，要么干脆追求更激进的 ILP（大 register tile）来抵消低占用率 —— 两者只能选一条，别像 v1 一样两头不到岸。
3. **削减指令开销**：线程重排太重，改用向量化加载（`float4`）+ 更直白的索引，把那 62% 的多余指令砍掉。
4. **终极方向**：上 **Tensor Core**（`wmma`/`mma`）。FP32 CUDA core 的 SGEMM 天花板就在这（~几个 TFLOPS），Tensor Core 能再高一个数量级。

---

## Q6. 方法论：为什么静态分析会错，ncu 怎么纠错

这次最大的教训：

1. **静态分析容易把 CPU 思维套到 GPU 上**。初版「单线程 ILP=1 → 1/延迟」是把 GPU 当单核 CPU 想了。GPU 藏延迟的主力是 **TLP（多 warp 轮转）**，不是单线程 ILP；只要占用率够，ILP=1 不致命。ncu 的 `IPC=1.71 > 1.13` 直接打脸了这个假说。

2. **静态分析看不见「指令数」**。静态分析数的是「逻辑上的 FMA/访存」，但编译后的真实指令数（含地址计算、循环管理、取模除法）只有 ncu 的 `Executed Instructions` 才告诉你。v1 多 62% 指令这件事，静态完全看不到。

3. **静态分析看不见「占用率被寄存器顶住」**。寄存器用量要靠 `nvcc` 编译后的 `Registers Per Thread`（ncu Launch Statistics）才知道；占用率的瓶颈因素要看 `Block Limit *` 那一排。这些只有 profile 才有。

4. **ncu 的 Duration 不能直接用来比速度**。本次 naive 的 ncu Duration=3.11ms、v1=2.33ms（似乎 v1 快），但 `SM Frequency` 在 1.55–2.28 GHz 之间波动（thermal/power throttling），且 ncu 用 kernel replay 采集指标会大幅拖慢执行。**比真实速度要用 `cudaEvent` 计时**（main.cpp 的 1.788 vs 1.986）；ncu 只用它的**归一化指标**（%、IPC、occupancy、stall）来定位瓶颈结构。

→ **结论：静态分析用来「提出假设」，ncu 用来「证伪/证实」。两者都要，且 ncu 是最终裁判。**

> 配套阅读：`如何阅读ncu-rep.md`（GUI 操作）、`理论补课路线图.md`（看懂这些指标需要的理论知识）。
