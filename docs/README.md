# v3 前置学习材料

这组文档用于在实现 `sgemm_v3` 前，完成一次从“看懂 profiler”到“根据证据选择优化”的闭环。

建议按以下顺序阅读和实践：

1. [使用 Nsight Compute GUI 分析 naive、v1 与 v2](./ncu-gui-sgemm-analysis.md)
   - 从零认识 `.ncu-rep`、Summary、Details、Source、Baseline。
   - 使用 `report/sgemm_nv1v2_2.ncu-rep` 完整分析四个 kernel。
   - 学会区分吞吐瓶颈、延迟隐藏不足、资源限制和编译器代码生成差异。
2. [从 v2 到 v3：学习路线与实验顺序](./v2-to-v3-learning-roadmap.md)
   - 解释为什么下一步应优先学习二维寄存器 tiling，而不是立即堆更多 ILP 或直接上 double buffering。
   - 给出 v3 前应完成的测量、SASS、访存、寄存器 tiling 和流水化练习。

推荐的学习方式不是一次记住所有指标，而是每读完一节就在 GUI 中完成对应操作，并写下一句“证据 → 推断 → 下一实验”。例如：

```text
证据：v2 的 MIO Throttle 最大，L1/TEX active throughput 约 98%，DRAM 仅约 5%。
推断：当前主要压力更接近 shared-memory/MIO 路径，而不是显存带宽。
实验：增加 N 方向寄存器复用，观察每次 FMA 对应的 shared load 是否下降。
```

这比单独追求某个指标达到 100%，更接近真实的 CUDA 性能优化过程。
