# v5 WMMA：现有 NCU 报告解读

对象：`ncu-rep/sgemm.v1v2v3v4v5.ncu-rep`。只读导入，无新 profiling、无 kernel 修改。报告含 v0～v5，CC12.0、84 SM。

## 主要结论

v5 是 **L2 请求处理吞吐与未隐藏的 global/cache load 延迟受限**，不是 DRAM 带宽打满。它在这份报告里仅快于 v0/v1；v5 为 540.032 us，慢于 v3 的 328.192 us（耗时 1.65 倍）。将计算换成 Tensor Core，同时去掉大 tile 的数据复用，不能视为纯计算单元替换的对照。

## GUI Speed Of Light 对照

L1/TEX 列是 active-cycles 口径，其他 throughput 为 elapsed-cycles；不要加减这些百分比。

| kernel | Duration/us | SM/GHz | DRAM/GHz | Compute/% | Memory/% | L1/TEX/% | L2/% | DRAM/% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 | 703.904 | 2.94 | 14.99 | 96.70 | 96.70 | 97.40 | 30.94 | 4.73 |
| v2 | 339.392 | 2.91 | 14.98 | 70.37 | 70.37 | 75.00 | 22.69 | 8.09 |
| v3 | 328.192 | 2.90 | 14.99 | 73.62 | 73.62 | 78.88 | 23.64 | 7.92 |
| v4 | 412.160 | 2.95 | 14.98 | 73.91 | 73.91 | 79.05 | 18.67 | 6.47 |
| v5 | 540.032 | 2.95 | 14.99 | 56.84 | 90.55 | 81.51 | 90.55 | 5.54 |

频率来源是 GUI 的 `gpc__cycles_elapsed.avg.per_second` / `dram__cycles_elapsed.avg.per_second`。报告警告未固定频率，但约 1.7% 的 SM 频率差异不能解释 65% 的耗时差。

## L2 的哪条路径满了

展开 `Speed Of Light -> Memory Throughput Breakdown`，v5 为：

- `lts__t_tag_requests.avg.pct_of_peak_sustained_elapsed = 90.55%`。
- `lts__xbar2lts_cycles_active.avg.pct_of_peak_sustained_elapsed = 62.61%`。
- `lts__t_sectors.avg.pct_of_peak_sustained_elapsed = 51.46%`。
- `lts__lts2xbar_cycles_active.avg.pct_of_peak_sustained_elapsed = 50.50%`。
- `lts__d_sectors.avg.pct_of_peak_sustained_elapsed = 26.05%`。
- DRAM activity 为 5.54%，实际 DRAM throughput 为 53.11 GB/s；DRAM sector rate 为 4.21%。

因此 L2 聚合值 90.55% 是 tag/request 处理能力的繁忙程度，不能称为 L2 字节带宽占满 90.55%。L1 hit rate=57.11%、L2 hit rate=97.18%，说明 cache 有效，但缓存命中仍需要请求处理、传输和等待。

## 重复加载与 sector 利用率

| 指标 | v2/v3/v4 | v5 |
| --- | ---: | ---: |
| A/B 逻辑读取字节 | 364,904,448（348 MiB） | 2,147,483,648（2 GiB） |
| `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` | 2,850,816 | 16,777,216 |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | 11,403,264 | 100,663,296 |
| sectors/request | 4 | 6 |
| `l1tex__m_xbar2l1tex_read_bytes.sum` | 364,904,448 | 1,380,490,336 |
| `lts__t_sectors_srcunit_tex_op_read.sum` | 11,403,264 | 43,211,781 |

v5 的 L1 load sectors 为 3 GiB：其 2 GiB 逻辑读取因访问粒度利用不足变为 3 GiB，NCU 提示 `33,554,432 excessive sectors`，即三分之一 load sectors；平均每 sector 有效使用 21.33/32 byte。

v5 的 L1 load sectors 是 v3 的 8.83 倍；经 L1 缓存过滤后，L2->L1 read bytes 仍为 3.78 倍。DRAM read bytes 却接近：v2=26,343,680，v3=24,918,784，v4=25,518,336，v5=28,678,912。不能将所有 global load 都描述为 DRAM 读取。

C 逻辑输出为 16,777,216 byte（16 MiB），因此 v5 A/B/C 逻辑总字节为 2,164,260,864。某些 kernel 的计数 DRAM write=0 不代表没写 C，输出可暂留在缓存，计数窗口内 DRAM 字节不是逻辑读写字节。

## 延迟与 occupancy

| 指标 | v2 | v3 | v4 | v5 |
| --- | ---: | ---: | ---: | ---: |
| Warp cycles / instruction | 6.01 | 5.49 | 7.02 | 28.54 |
| Long scoreboard / issue | 0.27 | 0.03 | 0.03 | 22.43 |
| MIO throttle / issue | 0.62 | 0.53 | 2.18 | 0.00 |
| Barrier / issue | 0.33 | 0.29 | 0.42 | 0.09 |
| Active warps/scheduler | 3.71 | 3.73 | 3.69 | 5.83 |
| Eligible warps/scheduler | 1.78 | 2.08 | 1.58 | 0.24 |
| No eligible/% | 38.15 | 32.02 | 47.43 | 79.58 |
| Registers/thread | 96 | 118 | 96 | 40 |
| Achieved occupancy/% | 31.05 | 31.08 | 30.75 | 48.49 |

Long scoreboard 的原始指标为 `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio = 22.43`；分母口径为 `smsp__average_warps_active_per_inst_executed.ratio = 28.54`。两者之比约 78.6%，表示 warp 状态时间分布，不能说整个 GPU 的 78.6% 时间都闲置。

v5 一 warp/block，受到 24 blocks/SM 的硬件上限限制，最多 24 active warps，而 SM 最多 48 warps，因此理论 occupancy 50%；非寄存器限制。v3 的 occupancy 更低却更快，说明不能以 occupancy 的高低直接推断性能。报告没有 local/shared spill 请求。

## Tensor Core 与源码核对

- SASS 有 `HMMA.1684.F32.TF32`，Tensor Core 确实工作。
- Tensor active cycles：elapsed=56.84%，active=58.97%。
- `sm__ops_path_tensor_src_tf32_dst_fp32.sum = 8,589,934,592`，对应矩阵的 `2*M*N*K` 运算量。
- `sm__ops_path_tensor_src_tf32_dst_fp32.sum.pct_of_peak_sustained_elapsed = 12.54%`：分母为该设备此 TF32->FP32 算术路径的 peak-sustained 运算率乘 workload elapsed time。报告 `sum.peak_sustained_elapsed.per_second` 为 126,831,211,187,485.20 op/s（126.83 TOP/s），实际 `sum.per_second` 为 15,906,343,683,337.28 op/s（15.91 TOP/s）；不能将管线活跃 56.84% 读为达到了 56.84% TF32 算术峰值。
- 报告存在 `BAR.SYNC.DEFER_BLOCKING`，与当前源码循环内 `__syncthreads()` 一致。barrier stall 约 0.32%，可清理但非主要瓶颈。
- global->fragment 使用普通 `LD.E`。长 scoreboard 采样集中在首次使用 LD 结果的 TF32 转换指令，如 `FSETP.GEU.AND ... |R6| ...`：这里等待的是 load 的数据依赖，不能据采样行号推断转换算术本身慢。示例 PC `0x901139c80` long-scoreboard samples=7958，前面的 LD 于 `0x901139a90` 产生 R6。

## 下一教学实验

保留最小 v5；新 trial 先做 4 warp/block、2×2 warp tile、32×32 block C tile，整个 block 协作把 A/B 搬入 shared，再从 shared 加载 WMMA fragment。先同步搬运，理解 block tile -> warp tile -> instruction tile 后，再接回 cp.async DB。横向 warp 复用 A、纵向 warp 复用 B，同时绕开一 warp/block 的驻留上限。

验证 duration、L1/L2 sectors、long scoreboard、Tensor TF32 ops throughput 即可；这是新的通用能力组合，无需扩大为参数搜索或逐条 SASS 调度研究。

## 复查命令

```bash
ncu --import ncu-rep/sgemm.v1v2v3v4v5.ncu-rep --page details
ncu --import ncu-rep/sgemm.v1v2v3v4v5.ncu-rep --kernel-name regex:sgemm_v5 --page details --section SpeedOfLight --print-details all --print-metric-name label-name
ncu --import ncu-rep/sgemm.v1v2v3v4v5.ncu-rep --kernel-name regex:sgemm_v5 --page source --print-source sass --metrics smsp__pcsamp_warps_issue_stalled_long_scoreboard
```

## 官方指标说明

- [NCU Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)：throughput 聚合指标、cache requests/sectors 与采样 stall 的定义。
- [NCU Compute Triage Guide](https://docs.nvidia.com/nsight-compute/ComputeTriage/)：区分 DRAM、L2/L1 请求吞吐和访存延迟限制。
- [CUDA Compute Capabilities 表](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)：sm120 的 24 blocks/SM、48 warps/SM 限制。
