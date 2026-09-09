# v0：先把一小块数据搬进 shared

[源码](../src/sgemm_v0.cu) · [版本总览](README.md) · [下一版 v1](sgemm_v1.md)

v0 是后面所有版本的起点：一个线程算 C 的一个元素，一群线程先把需要的数据搬到 shared，再一起使用。

## 矩阵怎么切

统一按行存储：`C(M,N) = A(M,K) × B(K,N)`。M 是输出行数，N 是输出列数，K 是求和长度。

| 项目 | 当前配置 |
|---|---|
| 一个 block 的线程 | x=16、y=16，共 256 个 |
| 一个 block 算的 C | 16 行 × 16 列 |
| 每线程输出 | 1 个 float |
| 每轮沿 K 前进 | 16 |
| shared 中的 A/B | 各 16×16，共 2 KiB |

`grid.x=ceil(N/16)`，`grid.y=ceil(M/16)`。不同 block 负责不同的 C 块，不需要互相同步，也不把 K 拆给不同 block。

线程 `(tx,ty)` 的输出位置是：

```text
row = blockIdx.y * 16 + ty
col = blockIdx.x * 16 + tx
```

## 一轮具体做什么

对每个 `kk=0,16,32,...`：

1. 每个线程搬一个 A 元素和一个 B 元素：
   `A[row][kk+tx] → tileA[ty][tx]`；
   `B[kk+ty][col] → tileB[ty][tx]`。
2. 第一次 `__syncthreads()`，等大家搬完。
3. 从 shared 取 A 的一行和 B 的一列，做 16 次乘加，累加到自己的 `elemSum`。
4. 第二次 `__syncthreads()`，等大家读完，才能用下一轮的数据覆盖 shared。

`elemSum` 在整个 K 循环中持续累加，全部算完才写一次 C。一个 A 元素会被这一块中的多个输出列使用，一个 B 元素会被多个输出行使用，避免每个输出都独立重复访问 global。

## 边界和同步

A 加载检查 `row<M && aCol<K`，B 检查 `bRow<K && col<N`；越界位置向 shared 写零。输出只在 `row<M && col<N` 时写回。

例如 K=19，第二轮只搬 3 个有效的 K 位置，其余补零，但计算循环仍跑满 16 次。M/N 的尾块也用同样办法处理。

不要因为某个线程的输出越界就提前 return：它可能还要搬别人需要的数据，而且整个 block 必须一起走到 barrier。

## 回来看哪里

先看 `row/col`，再看两条 shared 赋值，最后看两个 barrier。理解它们后，这个版本就读完了。

v0 的复用主要发生在 block 的 shared 中。下一版让一个线程多算几个输出，把复用进一步带进线程内部。这里 block.x=16，一个 warp 横跨两行；到 v1 的 block.x=32，warp 就对应一整行线程。
