# SGEMM v2 尝试计划：从二维输出映射走向真正的 Outer-Product Register Tiling

> 更新时间：2026-07-31。下面的“当前进度与下一步”是现行执行计划；后面的原始教学检查点保留为原理与 NCU 分析参考，其中未勾选的旧清单不再表示当前实际进度。

## 0. 当前结论

`sgemm_trial_v2_1` 已经完成二维输出映射、二维累加器和可调 `REG_TILE_X/REG_TILE_Y`，但计算循环仍是：

```text
ri -> rj -> t
```

因此它已经实现了：

- 一个 block 计算更大的 C tile；
- A/B 从 global memory 搬到 shared memory后的 block 级复用；
- 每线程用 `REG_TILE_Y * REG_TILE_X` 个寄存器累加输出。

但它还没有在源码结构上显式实现：

```text
t -> 先加载 regA/regB -> ri/rj 外积
```

所以当前最重要的下一步不是继续扩大 `REG_TILE_X/Y`，也不是加入 double buffering，而是用 `sgemm_trial_v2_2` 只改变计算循环，验证：

1. shared-to-register 的 A/B 复用是否真正减少 LDS 指令；
2. `4x4` 的 16 条独立累加链是否提高 ILP 和 FMA pipeline 利用率；
3. 收益是否足以抵消额外寄存器、指令调度和可能的 occupancy 代价。

## 0.1 已完成任务

- [x] 建立 `32x8`、256 threads/block 的线程布局。
- [x] 使用标准 SGEMM 语义 `C(M,N) = A(M,K) * B(K,N)`。
- [x] 实现二维 thread tile：线程 `(tx,ty)` 负责：

  ```text
  row = baseRow + ty + 8 * ri
  col = baseCol + tx + 32 * rj
  ```

- [x] 使用 `regs[REG_TILE_Y][REG_TILE_X]` 保存二维输出累加器。
- [x] 将 tile 大小参数化：

  ```text
  TILE_X = 32 * REG_TILE_X
  TILE_Y = 8  * REG_TILE_Y
  TILE_K = 32
  ```

- [x] 修复 B cooperative load 错误依赖 `REG_TILE_Y` 的问题。B 的 K 方向加载行数现在独立使用：

  ```text
  B_TILE_ROWS_PER_THREAD = TILE_K / TILEBASE_Y = 4
  ```

  因此 `REG_TILE_Y=1..4` 时都会完整初始化 `tileB[32][TILE_X]`。

- [x] 扫描 `REG_TILE_X/Y=1..4` 的 16 个组合，全部通过完整输出校验。
- [x] `X=3`、`Y=3` 会让 M/N 遇到非整除 tile，相关输出边界已在扫描中覆盖。
- [x] 最终恢复默认 `REG_TILE_X=4`、`REG_TILE_Y=4`。
- [x] 默认尺寸 `M=1024,N=4096,K=1024` 工程回归 PASS。
- [x] 建立对 Y 方向收益的解释：增大 Y 合并纵向 C blocks，使同一 B tile 服务更多输出行，并减少重复发出的 B global loads。

尚未完成：

- [ ] 用 `K` 不能整除 32 的用例验证 K 尾块。
- [ ] 用随机输入完成至少两组 `M != N != K` 且三维非整除测试。
- [ ] 对当前 v2 执行 Compute Sanitizer。
- [ ] 采集 v2_1/v2_2 的 NCU 对比报告。
- [ ] 检查 registers/thread、spill、occupancy、LDS/FFMA 和真实 SASS。

## 0.2 已完成的 X/Y 性能筛选

测试条件：

```text
M=1024, N=4096, K=1024
20 次 warmup
5 个计时批次，每批 100 次 kernel
表中为批次单次耗时中位数换算的 TFLOPS
```

| REG_TILE_Y \ REG_TILE_X | 1 | 2 | 3 | 4 |
|---:|---:|---:|---:|---:|
| 1 | 4.75 | 5.02 | 5.38 | 5.63 |
| 2 | 7.28 | 9.28 | 10.10 | 10.28 |
| 3 | 10.06 | 12.84 | 12.88 | 13.49 |
| 4 | 11.74 | 14.91 | 16.74 | **17.51** |

当前筛选结论：

- `4x4` 最快，约为 `1x1` 的 `3.69x`。
- 增大 Y 的收益明显高于增大 X，因为基础 block tile 是 `32列 x 8行`；在 `1x1` 时，每个 K tile 的 B global load 量约为 A 的 4 倍。
- 这不是一个纯 ILP 实验。X/Y 同时改变了 block tile、global-to-shared 复用、grid block 数、累加器数量和寄存器压力。
- 该结果只适用于当前 `ri -> rj -> t` 版本。改成 outer product 后，最优 X/Y 需要重新确认，不能直接把 `4x4` 当成最终结论。
- 测试未锁定 GPU 频率，也不是 NCU 数据；适合作为快速筛选，不作为最终性能证据。

## 0.3 下一步主线：实现 `sgemm_trial_v2_2`

当前工作区已经准备了 `sgemm_trial_v2_2` 的副本、声明和临时注册；它目前与 v2_1 只有名字不同，尚未实现核心优化。

### 步骤 1：严格控制变量

v2_2 第一版保持以下内容与 v2_1 完全相同：

```text
blockDim = 32x8
REG_TILE_X = 4
REG_TILE_Y = 4
TILE_K = 32
shared-memory layout
global-to-shared cooperative load
边界判断
C 写回映射
```

只修改 shared-to-register 和 FMA 的计算循环。暂时不要混入：

- vectorized load；
- shared-memory padding/转置；
- double buffering；
- async copy；
- block/warp tile 重排。

### 步骤 2：改成 t 外层的小型外积

目标结构：

```cpp
for (u32 t = 0; t < TILE_K; ++t) {
    float regA[REG_TILE_Y];
    float regB[REG_TILE_X];

    // 先从 shared 读取当前 t 对应的 4 个 A 和 4 个 B。

    // 再让 regA[4] x regB[4] 更新 regs[4][4]。
}
```

对于每个 `t`，源码层面的目标是：

```text
shared loads: 4 个 A + 4 个 B = 8
FMA:          4 x 4 = 16
shared 标量/FMA = 8 / 16 = 0.5
```

作为对照，v2_1 按源码循环结构需要：

```text
shared loads: 2 x 4 x 4 = 32
FMA:          4 x 4 = 16
shared 标量/FMA = 2
```

编译器可能消除或重排其中一部分 load，所以这些源码计数只是待验证假设，最终以 SASS 和 NCU 动态计数为准。

### 步骤 3：先完成正确性验收

至少测试：

```text
1024 x 4096 x 1024
37 x 53 x 29
70 x 131 x 45
```

验收项：

- [ ] v2_1 和 v2_2 三组用例全部 PASS。
- [ ] M、N、K 三个方向的尾块都被覆盖。
- [ ] v2_1/v2_2 除计算循环外没有意外语义差异。
- [ ] 如运行 Compute Sanitizer，无越界和 race 报告。

### 步骤 4：普通 benchmark 对比

先比较固定 `4x4` 的 v2_1/v2_2，不立刻重新扫描 16 种 X/Y：

```text
v2_1: ri -> rj -> t
v2_2: t -> regA/regB -> ri/rj
```

要求：

- [ ] 使用相同输入、warmup、launch 次数和计时方法。
- [ ] 每个版本至少记录中位数及 min/max。
- [ ] 不把单次应用计时或 NCU replay 时间当成最终结果。
- [ ] 如果差异落在运行波动内，先记为“无稳定差异”，不要挑最好的一次。

### 步骤 5：采集最小、可解释的 NCU 对比

建议报告只保留 v2_1/v2_2，名称可用：

```bash
scripts/profile.sh -o sgemm.v2_1v2_2.outer-product
```

按以下顺序分析：

1. `Duration`、SM/DRAM Frequency：先确认时间和时钟可比。
2. shared-load SASS / FFMA：验证 shared load/FMA 是否下降。
3. executed instructions：确认没有被额外地址和循环指令抵消。
4. MIO Throttle：判断 shared/MIO 压力是否下降。
5. FMA pipeline、eligible warp、issued warp：判断 ILP 是否转化为计算吞吐。
6. registers/thread、theoretical/achieved occupancy：量化资源代价。
7. local load/store：必须确认没有 register spill。
8. Source/SASS：确认外积展开、实际 LDS/FFMA 顺序和动态计数。

### 步骤 6：outer product 成立后重新选择 X/Y

先筛选少量有代表性的组合：

```text
2x2
2x4
4x2
4x4
```

只有相邻结果仍有明显趋势时，再完整扫描 `1..4 x 1..4`。最终保留版本必须同时满足：

- [ ] Duration 稳定更好。
- [ ] 无 spill。
- [ ] shared load/FMA 符合设计或能够解释。
- [ ] registers 与 occupancy 的交换合理。
- [ ] 不依赖偶然的频率差异。

## 0.4 完成 outer product 之后的决策门

只有 v2_2 的正确性、普通 benchmark 和 NCU 证据链完整后，才根据新的第一瓶颈选择后续主题：

```text
global load/地址指令仍突出
    -> 优化 cooperative load，必要时试 vectorized load

Long Scoreboard、Barrier 或 load/compute 阶段气泡突出
    -> 学习 double buffering / async copy

寄存器压力、spill 或 eligible warp 恶化
    -> 缩小 register tile，检查展开与地址计算

FP32 pipeline 已接近主要上限
    -> 总结标量 CUDA Core 路线，再决定是否进入 warp tiling/Tensor Core
```

在这个决策门之前，不继续盲目放大 register tile。

---

## 原始教学检查点与分析参考

## 1. 这次要学习什么

v1 已经是一个 `4x1` 的一维 register tiling：

```text
TM = 4
TN = 1
```

每个线程计算同一列上的 4 个输出，并用 `regs[4]` 保存四条累加链。它能复用 B，但不能让同一个 A 同时服务多个输出列。

v2 的核心目标是把 thread tile 扩展到二维：

```text
v2-A：TM=4，TN=2，每线程计算 8 个输出
v2-B：TM=4，TN=4，每线程计算 16 个输出
```

每轮 K 计算先从 shared memory 读取少量 A、B 到寄存器，再做一个小型外积：

```text
regA[TM] x regB[TN] -> accum[TM][TN]
```

这次实验主要回答三个问题：

1. 同时复用 A 和 B，能否继续降低 shared load/FMA？
2. 更多独立累加器带来的 ILP，能否提高 eligible warp 和 FMA pipeline 利用率？
3. register/shared-memory 占用增加到什么程度后，会被 occupancy 或 spill 抵消？

## 2. 本轮明确不做什么

为了让实验结果容易解释，v2 暂时不加入：

- double buffering；
- async copy；
- `float4` 或其他显式向量化；
- shared-memory 转置或 padding；
- warp tiling；
- Tensor Core；
- 大量模板参数或自动搜索；
- 只为改变 ILP 而交换循环顺序。

v1 的 Details 页虽然报告了 shared-store conflict，但 Source/SASS 中三处 shared 访问均满足：

```text
L1 Wavefronts Shared = L1 Wavefronts Shared Ideal
L1 Wavefronts Shared Excessive = 0
```

手工地址分析也确认其访问分别是连续访问或 broadcast。因此，本轮不做 bank-conflict padding 实验。

## 3. 为什么先保留当前 block 形状

第一版继续使用：

```text
blockDim.x = 32
blockDim.y = 8
threads/block = 256
BM = 32
BK = 32
```

主要原因不是认为它一定最优，而是它便于学习：

- 一个 warp 的 `threadIdx.x` 正好是 `0..31`；
- 同一个 warp 的 `threadIdx.y` 固定；
- A 的 compute load 可以保持 broadcast；
- B 的 compute load 可以保持横向连续；
- C 的 store 可以保持横向连续；
- 与 v1 对比时，不会同时改变 warp 映射和 threads/block。

v1、v2-A、v2-B 的主要变化如下：

| 版本 | TM | TN | BM | BN | BK | 每线程输出 | block 输出 |
|---|---:|---:|---:|---:|---:|---:|---:|
| v1 | 4 | 1 | 32 | 32 | 32 | 4 | 32x32 |
| v2-A | 4 | 2 | 32 | 64 | 32 | 8 | 32x64 |
| v2-B | 4 | 4 | 32 | 128 | 32 | 16 | 32x128 |

这里：

```text
BM = blockDim.y * TM
BN = blockDim.x * TN
```

不要把 `tileA`、`tileB` 恰好都是 `[32][32]` 当成不变量。它们的逻辑形状分别是：

```text
tileA[BM][BK]
tileB[BK][BN]
```

因此：

```text
v2-A：tileA[32][32]，tileB[32][64]
v2-B：tileA[32][32]，tileB[32][128]
```

## 4. 核心优化的直观解释

### 4.1 v1 的复用

对于一个固定的 `k`，v1 的一个线程理想情况下需要：

```text
4 个 A
1 个 B
4 个 FMA
```

因此：

```text
shared 标量/FMA = (4 + 1) / 4 = 1.25
```

### 4.2 v2-A 的复用

`4x2` thread tile 对一个固定的 `k` 需要：

```text
4 个 A
2 个 B
8 个 FMA
```

因此：

```text
shared 标量/FMA = (4 + 2) / 8 = 0.75
```

### 4.3 v2-B 的复用

`4x4` thread tile 对一个固定的 `k` 需要：

```text
4 个 A
4 个 B
16 个 FMA
```

因此：

```text
shared 标量/FMA = (4 + 4) / 16 = 0.50
```

这是 v2 的核心优化。ILP 会从 4 条累加链增加到 8 或 16 条，但它是数据复用自然产生的收益，不是单独堆出来的独立指令。

## 5. 检查点 0：先画线程到输出的映射

这一阶段不要写 kernel。

v2-A 使用：

```text
TM = 4
TN = 2
```

线程 `(tx, ty)` 负责：

```text
row = blockRow + ty + i * 8
col = blockCol + tx + j * 32

i = 0..3
j = 0..1
```

例如：

```text
blockIdx = (0, 0)
tx = 5
ty = 3
```

它负责的行是：

```text
3, 11, 19, 27
```

负责的列是：

```text
5, 37
```

共得到 8 个输出坐标。

### 本阶段验收问题

- [ ] 能列出任意 `(tx, ty)` 对应的全部输出坐标。
- [ ] 能解释为什么 256 个线程恰好覆盖 `32x64` 个输出。
- [ ] 能证明线程之间没有重复写同一个 C 元素。
- [ ] 能证明 block 内没有漏掉 C 元素。
- [ ] 能解释为什么固定 `j` 写 C 时，一个 warp 的地址连续。

只有这些问题都能回答后，才开始写 v2-A。

## 6. 检查点 1：创建最小 v2-A 骨架

预计涉及：

```text
src/sgemm_v2.cu
include/sgemm_func.h
src/main.cpp
```

建议先硬编码教学参数，不要立刻模板化：

```cpp
static constexpr u32 BM = 32;
static constexpr u32 BN = 64;
static constexpr u32 BK = 32;
static constexpr u32 TM = 4;
static constexpr u32 TN = 2;
static constexpr u32 BLOCK_SIZE_X = 32;
static constexpr u32 BLOCK_SIZE_Y = 8;
```

kernel 内先只声明：

```cpp
__shared__ float tileA[BM][BK];
__shared__ float tileB[BK][BN];
float accum[TM][TN] = {};
```

host wrapper 的 grid 应为：

```text
grid.x = CeilDiv(N, BN)
grid.y = CeilDiv(M, BM)
```

### 本阶段验收问题

- [ ] 能解释 `tileA` 为什么是 `BM x BK`。
- [ ] 能解释 `tileB` 为什么是 `BK x BN`。
- [ ] 能解释 grid.x 为什么使用 N，grid.y 为什么使用 M。
- [ ] 代码可以编译，但此时还不要求计算正确。

## 7. 检查点 2：实现 global 到 shared 的 cooperative load

v2-A 每轮 K tile 需要搬运：

```text
A tile：32x32 = 1024 个 float
B tile：32x64 = 2048 个 float
总计：3072 个 float
```

共有 256 个线程，平均每线程搬运：

```text
A：4 个 float
B：8 个 float
```

为了延续 v1、让地址映射容易理解，可以先使用两层小循环：

```text
每线程加载 4 个 A
每个 A 对应一个 tileRow

每线程加载 4 行 x 2 列组的 B
列位置为 tx 和 tx+32
```

逻辑位置：

```text
tileRow = ty + i * 8

A:
  global row = blockRow + tileRow
  global col = kOffset + tx
  shared     = tileA[tileRow][tx]

B:
  global row = kOffset + tileRow
  global col = blockCol + tx + j * 32
  shared     = tileB[tileRow][tx + j * 32]
```

边界条件必须分别检查：

```text
A：aRow < M 且 aCol < K
B：bRow < K 且 bCol < N
```

越界位置写入 shared 的值必须是 0，确保 K 尾块正确。

第一次实现先使用普通标量 load/store。不要为了减少源码行数引入 `float4`。

### 为什么这些访问容易分析

对于固定的 `i`、`j`：

- warp 的 A global load 横向读取连续 K；
- warp 的 A shared store 横向写入连续 bank；
- warp 的 B global load 横向读取连续 N；
- warp 的 B shared store横向写入连续 bank。

### 本阶段验收问题

- [ ] 画出一个 warp 在固定 `i`、`j` 时的 32 个 global 地址。
- [ ] 画出对应的 32 个 shared 下标。
- [ ] 能解释 global load 为什么合并。
- [ ] 能解释 shared store 为什么没有地址型 bank conflict。
- [ ] 每次 cooperative load 后有一次 `__syncthreads()`。

## 8. 检查点 3：实现 shared 到 register 的 4x2 外积

计算循环要以 `t` 为外层，因为每轮需要先取出当前 K 位置的 A、B 片段：

```cpp
for (u32 t = 0; t < BK; ++t) {
    float regA[TM];
    float regB[TN];

    // 从 shared 读取 4 个 A。
    // 从 shared 读取 2 个 B。

    // regA[4] 与 regB[2] 做外积，更新 accum[4][2]。
}
```

寄存器片段的逻辑下标：

```text
regA[i] = tileA[ty + i * 8][t]
regB[j] = tileB[t][tx + j * 32]
```

外积：

```text
对每个 i=0..3：
    对每个 j=0..1：
        accum[i][j] += regA[i] * regB[j]
```

这里的关键不是循环写法，而是确认：

```text
4 次 A shared load + 2 次 B shared load
支持 8 次 FMA
```

K tile 计算完成后保留第二次 `__syncthreads()`，防止下一轮覆盖仍在读取的 shared tile。

### 本阶段验收问题

- [ ] 能指出同一个 `regA[i]` 被哪两个 FMA 复用。
- [ ] 能指出同一个 `regB[j]` 被哪四个 FMA 复用。
- [ ] 能数出 8 条互不相同的累加链。
- [ ] 能解释为什么这是真正的数据复用，而不只是循环换序。

## 9. 检查点 4：写回 C 并验证正确性

写回位置必须与检查点 0 完全一致：

```text
row = blockRow + ty + i * 8
col = blockCol + tx + j * 32
```

每个输出都要分别检查：

```text
row < M
col < N
```

开发时把 `sgemm_v2_do` 临时加入 `IMPLEMENTATIONS`，与 v0/v1 一起校验。

至少测试：

```text
主尺寸：M=1024, N=4096, K=1024
小型尾块：M=37, N=53, K=29
跨多个 tile：M=70, N=131, K=45
```

后两个尺寸用于覆盖：

- M/N/K 不相等；
- M、N 不能整除 block tile；
- K 不能整除 BK；
- N 跨过 64 列边界；
- M、K 跨过 32 元素边界。

### 本阶段验收标准

- [ ] 三组尺寸全部 PASS。
- [ ] Compute Sanitizer 没有越界或 race 线索；如本轮决定运行它，再单独记录命令和结果。
- [ ] 临时测试用例不会误留在只用于简洁 profiling 的默认入口中。
- [ ] 能用自己的话解释 M/N/K 三个边界判断。

## 10. 检查点 5：先做普通性能比较

先在 NCU 外运行，避免把 profiler replay 时间当成真实性能：

```bash
scripts/build.sh --run
```

当前程序的单次计时只适合粗看。如果 v1/v2-A 差异很小，不要急于下结论；后续应把 benchmark 改成 warmup、多次 launch、CUDA Event 取中位数，但不要把 benchmark 重构混进 v2 kernel 的第一笔改动。

记录：

```text
v1 时间/GFLOPS
v2-A 时间/GFLOPS
重复运行的波动范围
```

不要因为 occupancy 下降就直接判定失败。最终先看时间，再解释资源变化。

## 11. 检查点 6：采集只含 v1/v2-A 的 NCU 报告

正确性通过后，为保持报告简单，profiling 时只注册 v1 与 v2-A。报告名可使用：

```bash
scripts/profile.sh -o sgemm.v1v2.v2a
```

不要使用 NCU 中被 replay 放大的应用侧计时输出作为性能结论。

### 第一组：结果是否真的更快

查看：

```text
Duration
SM Frequency
DRAM Frequency
```

先确认时钟差异足够小，再比较 Duration。

### 第二组：核心优化是否发生

重点查看：

```text
shared-load SASS instructions
FFMA instructions
shared load / FMA
executed instructions
MIO Throttle
FMA pipeline utilization
```

期望：

- shared load/FMA 下降；
- MIO Throttle 下降；
- 更多执行份额进入 FFMA；
- 每个输出摊到的地址、循环和 shared-load 指令减少。

### 第三组：代价是否可接受

查看：

```text
registers/thread
static shared memory/block
theoretical occupancy
achieved occupancy
active warps/scheduler
eligible warps/scheduler
issued warp/cycle
local load/store
```

必须确认：

```text
local load = 0
local store = 0
```

如果 occupancy 下降，但 eligible warp、issue rate 和 Duration 改善，则资源交换是成功的。

### 第四组：源码与 SASS 是否符合设计

在 Source 页：

- 关闭 `[%]`，查看绝对值；
- 展开 compute 行对应的 `LDS/LDS.128/FFMA`；
- 确认外积确实展开；
- 确认没有意外的 local load/store；
- 比较 `L1 Wavefronts Shared` 与 `Ideal`；
- 只有 `Excessive > 0` 时才分析可修复的地址型 bank conflict。

## 12. v2-A 的预期变化与风险

| 项目 | v1 | v2-A 期望 |
|---|---:|---:|
| 每线程输出 | 4 | 8 |
| 独立累加器 | 4 | 8 |
| block 输出 | 32x32 | 32x64 |
| A shared tile | 4 KB | 4 KB |
| B shared tile | 4 KB | 8 KB |
| 总 static shared | 8 KB | 12 KB |
| shared 标量/FMA | 1.25 | 0.75 |
| grid.x | `CeilDiv(N,32)` | `CeilDiv(N,64)` |
| registers/thread | 48 | 上升 |
| occupancy | 约 80% | 可能下降 |

可能失败的原因：

- `accum[4][2]` 和输入片段使 registers/thread 增长过多；
- 编译器生成了额外地址计算；
- 循环没有按预期展开；
- shared load/FMA 没有在 SASS 中实际下降；
- occupancy 下降后 eligible warp 不足；
- 出现 register spill；
- 新的 B tile 访问产生了意外的 excessive wavefront。

## 13. 检查点 7：把 TN 从 2 扩展到 4

只有 v2-A 正确，并且能解释它的 NCU 指标后，才进行 v2-B。

保持以下参数不变：

```text
blockDim = 32x8
BM = 32
BK = 32
TM = 4
```

只修改：

```text
TN：2 -> 4
BN：64 -> 128
tileB：32x64 -> 32x128
accum：4x2 -> 4x4
```

输出列映射扩展为：

```text
col = blockCol + tx + j * 32
j = 0..3
```

重复与 v2-A 完全相同的：

- 非整除正确性测试；
- 普通性能测试；
- NCU 指标检查；
- SASS 检查；
- spill 检查。

报告名可使用：

```bash
scripts/profile.sh -o sgemm.v1v2.v2b
```

## 14. 如何决定保留 4x2 还是 4x4

不要预设 4x4 一定更快。

### 选择 4x4 的条件

- Duration 稳定优于 4x2；
- shared load/FMA 进一步下降；
- 没有 spill；
- registers 和 occupancy 代价能够解释；
- eligible warp 和 issue rate 没有严重恶化。

### 选择 4x2 的条件

- 4x4 的 registers/thread 明显过高；
- 4x4 发生 spill；
- occupancy/eligible warp 下降抵消了复用收益；
- 4x4 的真实时间没有稳定改善。

即使最后保留 4x2，4x4 仍然是成功的教学实验，因为它能展示 register tiling 的收益不是无限增长的。

## 15. 常见结果的诊断顺序

### 情况 A：shared load/FMA 下降，时间也下降

说明核心优化成立。继续比较 4x2 与 4x4。

### 情况 B：shared load/FMA 下降，但时间不变

依次检查：

1. registers/thread；
2. occupancy 和 active blocks；
3. eligible warp 与 issue rate；
4. 地址/循环指令是否增加；
5. FMA pipeline 是否提高；
6. 时钟与普通 benchmark 波动。

### 情况 C：shared load/FMA 没有下降

检查 Source/SASS：

- `regA/regB` 是否真正复用；
- 编译器是否重复发射 LDS；
- 外积循环是否展开；
- 源码循环结构是否妨碍优化。

不要仅凭 CUDA 源码认为复用已经发生。

### 情况 D：出现 local load/store

这是 register spill。先减小 `TN`，不要立刻增加 occupancy 参数或加入更多优化。

### 情况 E：Source 页出现 `Shared > Ideal`

展开到具体 `LDS/STS`，按 warp 的活动线程地址计算 bank。只有定位到具体指令后才调整布局，不要给所有 shared 数组盲目 padding。

## 16. v2 完成后的下一方向

当最佳二维 register tile 已经确定后，下一步建议顺序是：

```text
可靠的多次 benchmark
  -> cooperative load 的指令与地址计算优化
  -> 在对齐和边界允许时尝试 vectorized global load
  -> 再看新的 NCU 瓶颈
  -> Long Scoreboard/Barrier/阶段气泡突出时学习 double buffering
```

double buffering 不是固定的“下一个版本”。只有 register tiling 已经降低 shared load/FMA，并且 global-to-shared 搬运延迟成为更明显的问题时，它才有清晰的优化对象。

## 17. 最终验收清单

### 理解

- [ ] 能解释 v1 为什么是 `4x1` register tiling。
- [ ] 能解释 v2 为什么是二维 register tiling。
- [ ] 能画出 block tile、thread tile、A tile、B tile。
- [ ] 能解释 A、B 分别在哪个方向被复用。
- [ ] 能解释 ILP 是如何由多个累加器自然产生的。

### 正确性

- [ ] 主尺寸 PASS。
- [ ] 至少两个非方阵、非整除尺寸 PASS。
- [ ] M/N/K 边界检查含义正确。
- [ ] 没有越界、race 或错误的 K 尾块。

### 性能证据

- [ ] 普通 benchmark 有可重复结果。
- [ ] NCU 报告中 v1/v2 时钟接近。
- [ ] shared load/FMA 下降。
- [ ] MIO Throttle、FMA pipeline 的变化可以解释。
- [ ] registers、shared memory、occupancy 代价可以解释。
- [ ] local spilling 为 0。
- [ ] 最终选择 4x2 或 4x4 有 Duration 证据支持。

### 实验纪律

- [ ] v2 没有混入 double buffering、vector load 等额外变量。
- [ ] 4x2 到 4x4 只改变 TN、BN 及对应数组/循环边界。
- [ ] 没有为了让单个指标更漂亮而保留无真实性能收益的修改。

## 18. 推荐的协作方式

按以下节奏逐步推进，不一次写完整个 v2：

1. 用户先完成检查点 0 的线程映射。
2. Codex 检查映射并解释可能的错误。
3. 用户完成 kernel 骨架和 cooperative load。
4. Codex 只审查当前阶段的索引、边界与访存。
5. 用户完成外积和写回。
6. 双方一起完成正确性验证。
7. 用户生成报告并先给出自己的分析。
8. Codex 根据报告补充或纠正结论。

这样每个阶段都能明确知道“为什么正确”和“为什么更快或更慢”，避免只得到一份能运行但无法解释的代码。
