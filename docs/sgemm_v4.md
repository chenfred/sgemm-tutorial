# v4：用 cp.async 把下一块直接搬到 shared

[源码](../src/sgemm_v4.cu) · [上一版 v3](sgemm_v3.md) · [版本总览](README.md)

正式 v4 从 trial_v4_1 复制而来。它保留 v3 的双缓冲结构，只把“global → 预取寄存器 → shared”改成 cp.async 的直接搬运，方便逐段对照。

## 切分与 v3 一样

矩阵仍为行存储的 `C(M,N)=A(M,K)×B(K,N)`。

| 项目 | 当前配置 |
|---|---|
| block | x=32、y=8，256 个线程 |
| C tile / K tile | 96×96 / 32 |
| 每线程输出 | 12×3，行间隔 8、列间隔 32 |
| 每线程每个 K tile 搬运 | A 12 个 float、B 12 个 float |
| shared | 两套 A[96][32]、B[32][96]，共 48 KiB |

grid 按 N/96、M/96 向上取整。输出坐标仍为 `row=baseRow+ty+8*ri`、`col=baseCol+tx+32*rj`。

compute helper 仍从 shared 取 12 个 A、3 个 B，更新 36 个累加器。删除的是 prefetchA/B 中转数组，算乘加所需的寄存器仍然存在。

## 一条 cp.async 做什么

`copy_async_float()` 中：

```cpp
const u32 sharedAddr = static_cast<u32>(__cvta_generic_to_shared(dstShared));
asm volatile("cp.async.ca.shared.global [%0], [%1], 4;"
             :: "r"(sharedAddr), "l"(srcGlobal) : "memory");
```

它让本线程发起一次 4 字节 global → shared 复制，也就是一个 float。

- shared 指针先转换为 shared 地址空间内的地址。
- %0、%1 对应两个输入；r 对应 32 位地址操作数，l 对应 64 位地址操作数。
- .ca 是缓存策略，shared.global 表示目标和源地址空间。
- volatile 和 memory 是给编译器的约束；memory 告知它这里有隐式内存副作用，并不等待 GPU 搬完。

每线程会发起多条复制；一个 warp 的相邻线程仍然沿 A 的 K 或 B 的 N 连续访问。每线程搬 4 字节，不代表每个线程独立产生一次显存事务。

当前每次只有 4 字节，不能直接把它改成 16：需要同时改线程分工、对齐和边界，否则相邻线程会覆盖重叠区域。这个路径要求 SM80 或更新架构。

## 发起、提交、等待、同步

这四件事各管一段：

| 操作 | 当前代码里负责什么 |
|---|---|
| cp.async | 发起复制；这时就可能开始搬运 |
| commit_group | 将本线程此前未提交的 A/B 复制归为一组 |
| wait_group 0 | 等本线程所有已提交组完成 |
| __syncthreads() | 等所有线程走完各自的等待，再跨线程读 shared |

commit 不负责启动或等待复制。group 是每线程的，不是整个 block 共用一个组。wait_group 0 也不会等尚未 commit 的复制；wait_all 才相当于先 commit 再 wait_group 0。

## 主循环怎么走

```text
开头：
  发起 tile 0 → shared[0]
  commit → wait 0 → barrier

每轮：
  发起 next → shared[另一个 stage]
  commit
  计算 current
  wait 0 → barrier
  交换 readStage

结尾：
  计算最后一块
  写回 C
```

计算夹在发起和等待之间，给搬运留出时间。循环的 kOffset 仍表示 next；最后一次计算放在循环外，和 v3 一样。

不能把 wait 和 barrier 交换：先 barrier 后 wait，只说明大家都到达了“准备等待”的位置，快线程仍可能读到慢线程尚未搬完的数据。循环末尾 barrier 还保证 current 都读完了，下一轮才可复用旧 stage。

此版每轮只预取一组 next，所以 wait_group 0 正合适。改成 1 会允许 next 未完成就开始读。多个 stage 下允许保留多少组，要和预填充、消费次序以及收尾一起设计。

## 尾块和容易误判的地方

A 检查 M/K，B 检查 K/N；有效位置执行 cp.async，无效位置普通 shared store 写零，不构造越界 global 指针。随后 block barrier 也保证这些补零写入对其他线程可见。任何线程都不能因自己的输出越界而跳过同步。

删除预取数组不等于最终寄存器一定减少；加上 async 也不等于吞吐一定提高。搬运和计算仍会占用硬件资源，wait 可能真的等待，shared 占用也可能限制驻留 block 数。不要把一次 PASS 当成同步证明，也不要把一次计时当成性能结论。

trial_v4_2 是“先填满、算完再补入”的环形流水线实验，曾用三 stage，后来改为双 stage 对照；它没有替换这里的正式 v4。回来复习先看本文件的四个操作和循环顺序即可。

进一步查语义：[NVIDIA PTX cp.async 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async)。
