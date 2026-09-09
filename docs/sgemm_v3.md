# v3：提前取下一块，计算当前块

[源码](../src/sgemm_v3.cu) · [上一版 v2](sgemm_v2.md) · [下一版 v4](sgemm_v4.md)

v3 的乘加和线程分工沿用 v2，新增的是普通 global load 预取和两套 shared。它尝试在等下一块数据时，先把当前块算掉。

## 没变的切分，多出的存储

仍是 `C(M,N)=A(M,K)×B(K,N)`，block 为 32×8，C tile 为 96×96，K tile 为 32。每线程算 12×3 个输出，行相隔 8、列相隔 32：

```text
row = blockIdx.y*96 + ty + 8*ri
col = blockIdx.x*96 + tx + 32*rj
```

每线程每轮搬 A/B 各 12 个 float。具体分工与 [v2](sgemm_v2.md) 相同。

新增存储有两类：

| 名字 | 保存什么 |
|---|---|
| prefetchA、prefetchB | 每线程下一块的 24 个输入临时值 |
| tileA[2]、tileB[2] | 两套 shared，共 48 KiB |

`regs[12][3]` 是最终输出的累加器，别和 prefetch 混淆。计算时的 `regA/regB` 又是当前 shared 块读出来的输入。

shared 通过动态 shared 分配，host wrapper 设置允许的动态 shared 大小，并在 kernel launch 第三个参数传入 `sizeof(SharedLayout)`。

## 沿着三块数据走一次

把 K 方向依次叫作 tile 0、1、2；stage 只是存储槽位 0 或 1。

```text
开头：
  tile 0：global → prefetch → shared[0]，barrier

第 1 轮：
  预取 tile 1 → prefetch
  计算 tile 0，读取 shared[0]
  prefetch → shared[1]
  barrier，readStage 换成 1

第 2 轮：
  预取 tile 2 → prefetch
  计算 tile 1，读取 shared[1]
  prefetch → shared[0]
  barrier，readStage 换成 0

结尾：
  计算 tile 2，读取 shared[0]
  写回 C
```

因此主循环的 `kOffset` 指向下一块要搬的数据，当前计算的是上一轮准备好的块。循环外最后那次 compute 不能删除，否则漏算最后一个 K tile。只有一个 K tile 时，主循环不执行，开头准备好后直接由结尾计算。

## 为什么可能重叠

关键源码顺序是：

```text
next LDG → current compute → next STS
```

LDG 是普通 global load，STS 是写 shared。current compute 不依赖 next 的预取值，所以给编译器和硬件留下了把加载与独立计算重叠的机会。这里没有 cp.async，也不需要 commit/wait；普通 load 的结果依赖由正常指令机制处理。

不要改成“load next、立刻 store next、再 compute current”后就认为效果一样。那样更早遇到依赖 load 结果的指令，可能压缩重叠窗口。实际调度仍需看编译结果。

## 为什么一个循环末尾 barrier 够用

此时 current 只读 readStage，next 只写另一个 writeStage。barrier 同时保证：

- next 的 shared 数据都写好了，下一轮才能读。
- current 都读完了，下一轮才能覆盖它的槽位。

`readStage ^ 1` 只是在 0、1 之间切换。最后一次 compute 后不再覆盖 shared，因此不用再加 barrier。

A 的 M/K、B 的 K/N 越界在预取时补零，store helper 只管把值写入 shared；输出仍逐个判断 M/N。

v3 的代价是 shared 翻倍，还有 24 个预取值需要跨过 current compute 保持存活。源码拆出了流水线，但不保证寄存器够用、延迟完全隐藏或一定加速。下一版用 cp.async 换掉这段中转搬运。
