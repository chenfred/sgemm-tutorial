# v2 前置学习材料

当前仓库只保留两个 kernel：`sgemm_v0` 与 `sgemm_v1`。配套报告是：

```text
ncu-rep/sgemm.v0v1.0719.ncu-rep
```

建议按以下顺序阅读和实践：

1. [使用 Nsight Compute GUI 分析 v0 与 v1](./ncu-gui-sgemm-analysis.md)
   - 从零认识 Summary、Details、Source、Raw 与 PM Sampling。
   - 学会区分 DRAM、L1/TEX、shared-memory/MIO、FMA pipeline、调度与 occupancy。
   - 按“证据 → 推断 → 下一实验”分析当前报告，而不是寻找一个万能百分比。
2. [从 v1 到 v2：学习路线与实验顺序](./v1-to-v2-learning-roadmap.md)
   - 解释 v1 的真正收益中，数据复用、指令数下降与 ILP 各占什么角色。
   - 说明为什么下一步优先做二维 register tiling，而不是继续交换循环或立刻上 double buffering。
   - 给出实现下一个版本前应补齐的 benchmark、SASS、shared memory 和寄存器基础。

推荐每读完一节就回到 NCU GUI 完成对应操作，并写一句：

```text
证据：v1 的 shared-load 指令从 167.8 M 降到 67.1 M，MIO Throttle 从 21.94 降到 12.37。
推断：v1 的主要收益来自每线程计算 4 个输出后减少了线程数和 shared-memory 指令压力。
实验：下一版在 M/N 两个方向都做寄存器复用，观察 shared load/FMA 是否继续下降。
```

不要试图一次记住所有指标。先能沿着一条完整证据链做出可验证的下一步，学习效率会更高。
