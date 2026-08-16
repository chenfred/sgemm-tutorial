# SGEMM v3 Double Buffering 学习计划

更新时间：2026-08-17。

## 1. 当前检查点

- [x] `sgemm_trial_v3_3` 使用两份 shared tile，实现同步 copy 的 ping-pong stage。
- [x] 修正 K tile 偏移，删除无效末尾 barrier，并通过现有正确性测试。
- [x] 恢复与 v2 相同的 `96x96` block tile 和 `12x3` register tile。
- [x] 建立 `sgemm_trial_v3_4`，将 global load 与 shared store 拆开，以寄存器保存 next tile。
- [x] 把 current compute 放到 next LDG 与依赖它的 STS 之间。
- [x] 默认尺寸与多组非整除边界尺寸均通过正确性校验。
- [x] 将实现转正为 `src/sgemm_v3.cu`，默认入口保留 v2/v3 对照，并删除重复的 trial_v3_4。

本阶段已经完成。正式 v3 保留普通 `LDG -> register -> STS` 软件流水作为下一阶段 `cp.async` 的基线；
后续计划见 `sgemm-v4-cp-async-plan.md`。历史 NCU 报告仍可用于确认资源和 spill，但不再阻塞进入下一学习点。

## 2. 两个 trial 分别在验证什么

### v3_3：同步 shared ping-pong

当前结构：

```text
global -> register -> shared[next]
compute shared[current]
barrier + swap
```

对同一个 warp，依赖 global load 结果的 shared store 位于 compute 前面，所以源码没有明确构造单 warp 内的 load/compute 重叠。它仍可依靠不同 warp 的进度差异形成部分重叠。

v3_3 的价值是先验证：

- 两个 shared stage 的索引切换；
- prologue / steady state / epilogue；
- barrier 和 shared 生命周期；
- 尾块补零。

### v3_4：普通 LDG register prefetch

目标结构：

```text
global -> prefetch registers       发射 next tile 的 LDG
compute shared[current]            执行与 next 数据无依赖的 LDS/FMA
prefetch registers -> shared[next] 执行依赖 LDG 结果的 STS
barrier + swap
```

这里仍没有异步复制 API。它利用的是同一个 warp 内互不依赖指令的 ILP：如果 next LDG 尚未完成，调度器仍有机会发射 current tile 的 LDS/FMA。

注意：源码顺序只是构造机会，不保证编译器最终保留预期调度。正确性和基本性能测通后，只需局部确认 SASS 中 next LDG 是否早于一段 current LDS/FMA，不做逐条考古。

## 3. 每线程预取数据形状

保持 v2 参数不变：

```text
blockDim       = 32 x 8
TILE_X/Y/K     = 96 / 96 / 32
REG_TILE_X/Y   = 3 / 12
```

每线程负责搬运：

```text
A：REG_TILE_Y x A_TILE_COLS_PER_THREAD = 12 x 1 = 12 floats
B：B_TILE_ROWS_PER_THREAD x REG_TILE_X = 4 x 3 = 12 floats
合计                                               24 floats
```

建议直接使用与 cooperative copy 循环一致的数组形状：

```cpp
float prefetchA[REG_TILE_Y][A_TILE_COLS_PER_THREAD];
float prefetchB[B_TILE_ROWS_PER_THREAD][REG_TILE_X];
```

数组只保存数值。目标 shared 下标仍由 `ri/ai/bi/rj`、`threadIdx` 和 `writeStage` 在 store helper 中计算，避免额外保存地址。

## 4. 推荐实现顺序

### 第一步：只拆 A 的 load/store

把 `load_tile_a()` 拆成：

```cpp
load_tile_a_to_registers(prefetchA, A, M, K, baseRow, kOffset);
store_tile_a_to_shared(tileA, prefetchA, writeStage);
```

global load helper 负责边界判断，越界时把 `0.0f` 写进 prefetch register；shared store helper 不再做 global 边界判断。

先在原位置连续调用两个 helper，保持行为完全不变，然后编译并测试。这一步只验证拆分是否正确。

### 第二步：同样拆 B

建立：

```cpp
load_tile_b_to_registers(prefetchB, B, K, N, baseCol, kOffset);
store_tile_b_to_shared(tileB, prefetchB, writeStage);
```

仍在原位置连续调用，重新测试。此时 v3_4 应与 v3_3 结果一致，但还没有形成明确重叠。

### 第三步：整理 prologue

第一块 tile 没有可供重叠的 current compute，因此顺序保持简单：

```text
load tile 0 -> prefetch registers
store prefetch registers -> shared[0]
__syncthreads()
readStage = 0
```

### 第四步：重排 steady state

将循环改为：

```cpp
for (u32 kOffset = TILE_K; kOffset < K; kOffset += TILE_K) {
    u32 writeStage = readStage ^ 1;

    load_tile_a_to_registers(prefetchA, ... kOffset);
    load_tile_b_to_registers(prefetchB, ... kOffset);

    compute_tile_c(regs, tileA, tileB, readStage);

    store_tile_a_to_shared(tileA, prefetchA, writeStage);
    store_tile_b_to_shared(tileB, prefetchB, writeStage);

    __syncthreads();
    readStage = writeStage;
}
```

关键不变量：

```text
compute 只读 shared[readStage]
store   只写 shared[writeStage]
readStage != writeStage
```

### 第五步：保留 epilogue

循环中每次计算 current，同时准备 next。因此退出循环时还有最后一个已准备但未计算的 stage：

```cpp
compute_tile_c(regs, tileA, tileB, readStage);
```

之后直接写 C，不需要末尾 `__syncthreads()`。

## 5. Barrier 为什么仍然必要

steady state 末尾的 barrier 同时保证：

1. 所有线程已经把 next tile 写完，下一轮才能读取它；
2. 所有 warp 已经读完 current tile，下一轮才能把旧 current stage 当作写入目标复用。

它必须位于两个 shared store 之后、stage 交换之前。不能因为 load 先进入了线程私有寄存器就删除 block barrier。

## 6. 正确性测试

至少覆盖：

```text
1024 x 4096 x 1024  正常多 stage
37   x 53   x 13    单个 K 尾块
70   x 131  x 32    恰好一个完整 K stage
127  x 259  x 45    两个 stage，第二块为尾块
191  x 257  x 97    多 stage，M/N/K 均非整除
```

重点排查：

- prefetch 数组下标与原 cooperative copy 映射不一致；
- 边界补零留在了 shared store 之后；
- store 错写到 `readStage`；
- stage 提前交换；
- 遗漏 epilogue；
- 某些线程绕过 `__syncthreads()`。

普通 PASS 后再运行：

```text
compute-sanitizer --tool memcheck
compute-sanitizer --tool racecheck
```

## 7. 性能验证与停止条件

v3.4 会让约 24 个预取值跨越整个 current compute，主要风险是寄存器压力和 spill。优先查看：

```text
Duration / SM Frequency
Registers Per Thread
Local Load/Store / Spill
Active Blocks / Achieved Occupancy
Long Scoreboard
Eligible Warps / No Eligible
```

学习验收标准：

- [ ] 能解释为什么 next LDG 与 current LDS/FMA 没有数据依赖。
- [ ] 能解释为什么 next STS 必须等待对应 LDG，但可以放在 compute 后面。
- [ ] 所有边界用例 PASS，sanitizer 无越界和 race。
- [ ] 确认是否 spill，并用基础性能数据判断收益。
- [ ] 无论是否加速，都记录结论并收束 register-prefetch DB 学习。

如果寄存器路线已经理解，再把 async global-to-shared copy 作为后续独立 trial；不要在 v3_4 中同时混入两种机制。
