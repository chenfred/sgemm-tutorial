# SGEMM 学习笔记

很久没看代码时，先从下面五篇找回各版的分工。尺寸按当前源码填写，C tile 的顺序是行×列，block 的顺序是 x×y。

| 版本 | block | C tile | 每线程输出 | K tile | shared/block | 主要变化 |
|---|---|---|---|---|---|---|
| [v0](sgemm_v0.md) | 16×16 | 16×16 | 1 | 16 | 2 KiB | shared 分块复用 |
| [v1](sgemm_v1.md) | 32×8 | 32×32 | 4×1 | 32 | 8 KiB | 一线程多算四行 |
| [v2](sgemm_v2.md) | 32×8 | 96×96 | 12×3 | 32 | 24 KiB | 二维寄存器分块、外积 |
| [v3](sgemm_v3.md) | 32×8 | 96×96 | 12×3 | 32 | 48 KiB | 普通 LDG 预取、双缓冲 |
| [v4](sgemm_v4.md) | 32×8 | 96×96 | 12×3 | 32 | 48 KiB | cp.async 直接搬到 shared |

所有版本都计算行存储的 `C(M,N)=A(M,K)×B(K,N)`，覆盖原 C；没有 alpha/beta 或跨 block 的 K 归约。
每个 block 负责一个 C tile，从头到尾遍历 K。文档中的分块大小是实现参数，实际总寄存器数和性能取决于编译及运行环境。

在仓库根目录运行：

```bash
scripts/build.sh --run
```

当前 main 按 v0 → v1 → v2 → v3 → v4 校验默认尺寸 `M=1024,N=4096,K=1024`。
已有验证覆盖了部分非整除尺寸；自己修改后也应检查尾块。`--dry-run` 跳过正确性校验，不能用它确认 PASS。

## 早期 NCU 材料

下面两篇保留 v0/v1 阶段的报告分析和当时的学习路线，文中的“下一步”属于历史上下文。配套报告是：

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
