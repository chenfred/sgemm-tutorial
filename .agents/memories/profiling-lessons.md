# NCU 与测量经验

## 当前可靠方案

- `full` 指标需要多次 replay。Application Replay 会为每个 pass 重启整个程序，因此应用内 warmup 会自然覆盖所有 pass；当前 NCU 2026.1.1 是 40 个 pass，不要把次数写进代码。
- `--profile-from-start off` 配合 `cudaProfilerStart/Stop` 精确选择正式 kernel，不依赖 kernel 名正则、launch 序号或实现数量。
- `--clock-control none` 不建立时钟锁，也不需要在任务结束后执行 `--clock-control reset`。比较 Duration 前仍必须核对 v0/v1 的 SM 与 DRAM Frequency。
- NCU 下的应用侧 CUDA Event 时间不代表正常单次执行；分析报告使用 NCU Duration，日常 benchmark 使用脱离 NCU 的多次 launch 中位数。

## 已证伪或不采用的方案

### NCU 锁频

- 在 RTX 5080 / sm_120 上测试过 `base`、`boost`、`force-boost`。它们会产生动作，但不同 replay 模式和 `full` pass 下仍不能可靠固定到同一频率；升级到 NCU 2026.1.1 后也没有改变这一结论。
- `force-boost` 没有带来比 `none + Application Replay + warmup` 更可靠的可比性，且不是教学环境需要的默认策略。
- 不要用短指标集的稳定频率推断 `full` 报告也稳定；不同指标集会改变 pass 和功耗行为。

### Kernel Replay 与源码 warmup

- Kernel Replay 只在原始 launch 前执行应用 warmup，后续指标 pass 单独重放 kernel；因此“源码里 warmup 一次”不能保证 `full` 的所有 pass 都处于稳态。
- 曾尝试每个 kernel 手工运行多次并用 launch skip/count 对齐，维护成本高，而且新增 kernel 后需要同步修改约定，已被独立 warmup + Application Replay 取代。
- NCU 没有“自动执行所需次数、前 m 次只 warmup、其余 pass 才采集”的单一参数；`--launch-skip` 只跳过应用本来就会发射的匹配 launch。

### Range Replay

- 用 profiler Start/Stop 定义两个 range、把 H2D/D2H 放在范围外可以正确运行，但不适合当前两个短 kernel 的 `full` 精确比较。
- `none + full` 中第二个 range 曾出现明显降频；交换 v0/v1 顺序后，降频跟随“第二个 range”而不是 kernel，证明存在位置效应。
- Range Replay 的结果名只有 `range`，缺少 `launch__kernel_name`；在 NCU 2026.1.1 中还不能与 `--profile-from-start off` 同用，`--import-source yes` 的组合也被 CLI 拒绝。
- 结论：当前继续使用具名 kernel 的 Application Replay；Range Replay 可留作学习机制，不作为基准报告方案。

## 同步与沙箱

- `cudaProfilerStop()` 不应被当作 device-wide synchronize。默认流 kernel launch 仍是异步 host API；正式 kernel 后显式 `cudaDeviceSynchronize()` 再 Stop。
- 阻塞式 D2H `cudaMemcpy` 和 `cudaFree` 可能产生同步，但代码不应依赖它们来界定 profiler 区间。
- Codex workspace 沙箱内曾出现 `CUDA driver version is insufficient for CUDA runtime version`，同一二进制在沙箱外正常运行。遇到这类错误先怀疑 WSL GPU bridge/沙箱隔离，再判断为 CUDA 安装损坏。
- PM Sampling 中 PM 是 Performance Monitor；它是按间隔采样硬件计数器的时间序列，不是新的执行单元或瓶颈分类。
