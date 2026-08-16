# SGEMM v4 cp.async 学习与开发计划

更新时间：2026-08-17。

## 1. 这一阶段只学习什么

目标是在正式 v3 的基础上，把：

```text
global -> registers -> shared[next]
```

替换成硬件支持的：

```text
global --------async--------> shared[next]
       current shared tile -> registers/FMA
```

本阶段需要掌握四件事：

1. 发起异步 global-to-shared copy；
2. 用 commit 把一批 A/B copy 组成一个 stage；
3. 在读取目标 shared stage 前 wait；
4. 用 block barrier 处理跨线程可见性以及 shared stage 的安全复用。

以下内容暂不进入主线：inline PTX、TMA、warp specialization、三重缓冲、重新搜索 tile 参数、为了 16-byte copy 重排全部线程映射。它们先等典型优化模式学完后再回看。

## 2. 先建立正确的概念边界

这里的 `cp.async` 不是 host 侧的 `cudaMemcpyAsync`：

- `cudaMemcpyAsync` 在 kernel 外由 host 发起，通常讨论 stream 与 host/device 或 device/device 传输；
- 本阶段讨论 kernel 内部的 global-to-shared 异步复制；在 SM80+ 上可由 `cp.async`/`LDGSTS` 路径实现；
- 它不是一台完全独立、一次搬完整二维 tile 的通用 DMA。warp 仍需发射若干 copy 指令，但数据不再先占用普通中间寄存器，并且 copy 完成前可以执行无依赖的 current-tile 计算。

v3 和 v4 的核心区别：

```text
v3: next LDG -> 24 个长期存活的 prefetch 值 -> next STS
v4: next async global-to-shared copy -------------> shared[next]
```

所以 v4 的预期收益不只是“更异步”，还包括删除 v3 每线程 A/B 合计 24 个 prefetch 值的长生命周期，降低 register pressure。实际是否加速仍需测试，不能只凭机制下结论。

## 3. 速成资料与阅读顺序

### 3.1 中文优先：预计 40～60 分钟

1. [NVIDIA 中文版：异步数据拷贝](https://developer.nvidia.cn/blog/c-expansion-cn/)
   - 在页面内搜索 `B.26. Asynchronous Data Copies`。
   - 重点读 copy-and-compute、pipeline、alignment；不读后面的 warp entanglement 细节。
2. [CUDA 编程指南中文翻译：4.11 异步数据拷贝](https://bearneck.github.io/cuda-programming-guide-zh/chapters/24-async-copies/)
   - 重点看 `CUDA C 原语` 代码，以及多 stage pipeline 示例。
   - 这是社区翻译，若与当前 CUDA 13.2 文档冲突，以 NVIDIA 当前英文文档为准。
3. [PTX-Notes：cp.async 系列](https://github.com/xuwilight/PTX-Notes/blob/main/docs/cp.async.md)
   - 只读开头、Async-group mechanism 和 Non-bulk copy。
   - 先理解 4/8/16-byte copy、commit group、wait group；暂时跳过 TMA、bulk 和 memory proxy 深挖。

没有把泛讲“CUDA 异步/stream”的热门中文视频列入必看，因为它们大多讲 host `cudaMemcpyAsync`，会混淆本阶段的 kernel 内 copy。先读上面三份针对性资料更快。

### 3.2 官方材料：边开发边查

1. [CUDA Programming Guide：Pipelines](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html)
   - 官方当前语义与示例；同时展示 `cuda::pipeline` 和 C primitives。
2. [CUDA Programming Guide：Advanced Kernel Programming](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html)
   - 查 pipeline primitives 的职责和限制。
3. [CUDA Best Practices：Asynchronous Copy](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#asynchronous-copy-from-global-memory-to-shared-memory)
   - 看同步 copy 与 async copy 的数据路径、register pressure 和 4/8/16-byte 对比。
4. [NVIDIA CUDA Sample：globalToShmemAsyncCopy](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/3_CUDA_Features/globalToShmemAsyncCopy)
   - 这是官方矩阵乘示例。只借鉴 API 与同步结构，不直接照搬它的完整 kernel。
5. [GTC 2025：CUDA Techniques to Maximize Memory Bandwidth and Hide Latency](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72683/)
   - 可选英文视频；约 16 分钟处开始演示 async copy、commit 和 wait。

## 4. API 选择

第一个版本 `sgemm_trial_v4_1` 使用：

```cpp
#include <cuda_pipeline.h>

__pipeline_memcpy_async(dstShared, srcGlobal, sizeof(float));
__pipeline_commit();
__pipeline_wait_prior(0);
```

选择它的原因：

- 与 `cp.async + commit_group + wait_group` 心智模型一一对应；
- 每个线程可以沿用 v3 当前的 cooperative-copy 下标，各自发起多个 4-byte copy；
- 最适合做 v3/v4 单变量对照；
- 不需要第一天同时理解 shared pipeline state、producer/consumer role 等 C++ API 结构。

机制掌握后，可选择用下面的现代 C++ API 重写一次，但不把它当成新的性能优化版本：

```cpp
#include <cuda/pipeline>

auto pipeline = cuda::make_pipeline();
pipeline.producer_acquire();
cuda::memcpy_async(dstShared, srcGlobal, sizeof(float), pipeline);
pipeline.producer_commit();
pipeline.consumer_wait();
pipeline.consumer_release();
```

## 5. 开发步骤

### 第一步：建立 v4_1 骨架

- 从正式 `src/sgemm_v3.cu` 复制为 `src/sgemm_trial_v4_1.cu`；
- kernel/host wrapper 改名并补声明；
- 临时在 `IMPLEMENTATIONS` 中保留 v3/v4_1；
- 不改变 `TILEBASE_X/Y`、`TILE_X/Y/K`、`REG_TILE_X/Y`、shared layout 和 compute loop。

先原样编译、校验，确保复制和改名没有引入变化。

### 第二步：只改 copy 的数据路径

删除 v3 的四个 helper：

```text
load_reg_tile_a
store_shared_tile_a
load_reg_tile_b
store_shared_tile_b
```

改成两个发起 async copy 的 helper：

```text
issue_async_tile_a(A, tileA[stage], ...)
issue_async_tile_b(B, tileB[stage], ...)
```

第一版每次仍只复制一个 `float`。不要在这里同时做 `float4`、展平重排或 tile 搜索；4-byte copy 已足以学习 `cp.async` 的完整同步模型。

边界策略保持直观：

```cpp
if (globalIndexValid) {
    __pipeline_memcpy_async(&sharedElement, &globalElement, sizeof(float));
} else {
    sharedElement = 0.0f;
}
```

不要构造越界的 global 指针再期待 zero-fill 替你兜底。等基本版本正确后，再单独学习 primitive 的 `zfill` 参数是否值得使用。

### 第三步：先做单 stage 正确性实验

先把 prologue 写成：

```text
issue tile 0 async copies
commit
wait for all committed copies
__syncthreads
compute tile 0
```

这一版还没有 overlap，但能独立验证：

- copy 是否真的写入正确 shared 地址；
- commit/wait 的基本语义；
- async 完成后仍为什么需要 block 级同步。

必须先让默认尺寸和至少一个非整除尺寸 PASS，再进入 double buffering。

### 第四步：改成两 stage overlap

保持 v3 的双 shared stage，结构改为：

```text
Prologue:
    issue async copy tile 0 -> shared[0]
    commit
    wait_prior(0)
    __syncthreads

Steady state:
    issue async copy next -> shared[writeStage]
    commit
    compute shared[readStage]
    wait_prior(0)
    __syncthreads
    swap readStage/writeStage

Epilogue:
    compute last shared[readStage]
```

这里的顺序不能随意改：

- `commit` 后才能把本轮 A/B copies 当作同一个已提交 stage；
- `compute current` 放在 `commit` 和 `wait` 之间，才提供实际重叠窗口；
- `wait_prior(0)` 只保证当前线程发起的 async copies 完成；
- 随后的 `__syncthreads()` 让所有线程都到达，并保证下一轮跨线程读取的新 shared tile 已准备好；
- 同一个 barrier 还确保所有 warp 已读完 current stage，下一轮才可覆盖它。

所有线程必须以相同次数、收敛地执行 commit/wait/barrier。不要让某个线程因边界条件提前 return。

### 第五步：确认编译器真的生成异步路径

只做最小静态验证：

```bash
cuobjdump --dump-sass build/sgemm | rg 'sgemm_trial_v4_1|LDGSTS'
```

验收点：

- v4_1 能看到 `LDGSTS` 一类 global-to-shared 指令；
- v3 仍是普通 `LDG` 与 `STS`；
- 检查 ptxas 或 NCU 的 registers/thread 与 local spill，不做逐条 SASS 考古。

如果没有生成预期指令，优先检查：目标架构、源/目标地址空间、copy size/alignment 和 API 用法，而不是马上改 tile 参数。

## 6. 正确性与工具验证

沿用 v3 的测试矩阵：

```text
1024 x 4096 x 1024  正常多 stage
37   x 53   x 13    单个 K 尾块
70   x 131  x 32    恰好一个完整 K stage
127  x 259  x 45    两个 stage，第二块为尾块
191  x 257  x 97    多 stage，M/N/K 均非整除
```

普通 PASS 后运行：

```bash
compute-sanitizer --tool memcheck build/sgemm
compute-sanitizer --tool racecheck build/sgemm
```

重点排查：

- wait 前读取 async copy 的目标 shared 地址；
- `wait` 后缺少 block barrier，导致线程读取别的线程尚未完成的 copy；
- 下一轮过早复用 current stage；
- 尾块跳过 copy 后没有把对应 shared 元素置零；
- commit/wait/barrier 次数因分支而不一致。

## 7. 性能对照与 NCU 阅读顺序

固定默认尺寸，只比较正式 v3 与 v4_1：

1. `Duration` 与 `SM Frequency`：先确认时间和频率可比；
2. `Registers Per Thread`、`Local Load/Store`：验证删除 prefetch registers 后是否降低资源压力且没有 spill；
3. `Achieved Occupancy / Active Warps`：只作为资源变化结果，不把 occupancy 当最终目标；
4. `Long Scoreboard`：看 global load 等待是否下降；
5. `MIO Throttle`、`Barrier`：判断瓶颈是否转移到 shared pipeline 或同步；
6. `Eligible Warps / No Eligible`、FMA pipeline：判断 copy/compute overlap 是否让计算资源获得更多可发射工作。

预期可能出现三种合理结果：

- 更快且 registers/thread 下降：机制和资源收益都兑现；
- registers/thread 下降但时间近似：v3 已通过 ILP/TLP 隐藏大部分延迟，学习仍然成功；
- 更慢：4-byte async 指令数、commit/wait/barrier 或现有 shared/compute 瓶颈抵消收益，先用上述指标定位，不立刻扩大参数搜索。

## 8. 本阶段完成标准

- [ ] 能区分 kernel 内 `cp.async` 与 host `cudaMemcpyAsync`。
- [ ] 能解释 issue、commit、wait、block barrier 四者分别保证什么。
- [ ] `sgemm_trial_v4_1` 通过默认与非整除尺寸，并通过 memcheck/racecheck。
- [ ] SASS 确认生成异步 global-to-shared 路径，且没有 register spill。
- [ ] 用一份简化 NCU 报告比较 v3/v4_1 的时间、寄存器和主要 stall。
- [ ] 无论是否加速，都记录原因并收束；不要立刻进入 TMA 或超参数搜索。
- [ ] 若机制清楚且证据充分，再决定是否转正为 `sgemm_v4`。
