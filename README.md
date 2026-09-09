# SGEMM Tutorial

本仓库用于练习开发 CUDA 算子，以单精度矩阵乘法（SGEMM）为主要样例。

各正式版本的分块、线程分工和同步说明见 [版本学习笔记](docs/README.md)。

## 项目结构

```text
include/              公共工具代码，例如 CUDA 错误检查和计时器
include/sgemm_func.h  SGEMM kernel host 侧调用声明
include/sgemm_verify.h CPU golden/verify 声明与 FP32 默认容差
src/main.cpp          测试入口、数据初始化和 GPU 执行流程
src/verify.cpp        多线程 CPU golden 计算与正确性校验
src/warmup.cu         NCU/计时前使用的独立 GPU warmup kernel
src/sgemm_v0.cu       16×16 shared-memory tiling 基线版本
src/sgemm_v1.cu       32×32 shared-memory tiling + 每线程 4 个输出
src/sgemm_v2.cu       96×96 block tile + 每线程 12×3 register tile
src/sgemm_v3.cu       普通 LDG register prefetch + shared double buffering
src/sgemm_v4.cu       cp.async global-to-shared + shared double buffering
scripts/build.sh      CMake 编译脚本
scripts/profile.sh    Nsight Compute profiling 脚本
CMakeLists.txt        CMake 构建配置
```

## 编译

推荐使用脚本编译，脚本可在任意目录执行，构建产物固定生成到项目根目录的 `build/`：

```bash
scripts/build.sh
```

常用参数：

```bash
scripts/build.sh --clean        # 删除 build/ 后重新配置和编译
scripts/build.sh --run          # 编译后自动运行 build/sgemm
scripts/build.sh --clean --run  # 从 0 编译并运行
scripts/build.sh --run --dry-run # 把 --dry-run 传给 build/sgemm
```

`--run` 后面的所有参数都会原样传给程序；因此程序以后新增参数时，无需同步修改脚本解析逻辑。完整用法可运行 `scripts/build.sh --help` 查看。

也可以直接使用 CMake：

```bash
cmake -S . -B build
cmake --build build
```

可执行文件会生成在：

```bash
build/sgemm
```

## 运行

```bash
./build/sgemm
```

程序会对 `M=1024,N=4096,K=1024` 按顺序运行 v0/v1/v2/v3/v4。每个正式 kernel 前都会运行一次独立的长 warmup，再输出正式调用的耗时、GFLOPS 和 `PASS` / `FAIL` 校验结果。

CPU 参考结果由 OpenMP 多线程 `sgemm_golden` 生成一次，`sgemm_verify` 使用 `|out-golden| <= atol + rtol*|golden|` 检查每个实现。FP32 默认采用 [PyTorch `assert_close`](https://docs.pytorch.org/docs/stable/testing.html#torch.testing.assert_close) 的 dtype-specific 容差：`rtol=1.3e-6`、`atol=1e-5`；失败时最多打印前 8 个错误元素。

只想执行数据生成、H2D、kernel 和 D2H，不运行 CPU golden/verify 时，可使用：

```bash
./build/sgemm --dry-run
```

## Profiling

使用 Nsight Compute 生成 profiling 报告：

```bash
scripts/profile.sh -o sgemm.v0v1.0719
```

报告会生成到 `ncu-rep/sgemm.v0v1.0719.ncu-rep`。`ncu-rep/` 目录已加入 `.gitignore`。

脚本使用 `--profile-from-start off` 配合程序内的 `cudaProfilerStart/Stop`，因此报告只包含 `IMPLEMENTATIONS` 中的正式 kernel；不依赖 launch 次数、kernel invocation 编号或逐版本正则。脚本传入 `--dry-run`，避免在 Application Replay 的每个 pass 重复 CPU golden/verify。新增实现到 `IMPLEMENTATIONS` 后，会自动走同一套流程。

脚本还使用 `--replay-mode application`：`full` 集合的每个采集 pass 都会重新启动程序，让每个正式 kernel 在每个 pass 中都重新 warmup。采集约需一分钟，比默认 Kernel Replay 慢，但频率更可比。

`--clock-control none` 明确不修改 GPU 时钟，所以不存在 profiler 结束后忘记解锁的问题。NCU 打印 “Running with unmodified GPU clocks” 是有意选择；仍应在报告中检查待比较 kernel 的 SM/DRAM Frequency 是否接近。

报告阅读方法见 [docs/ncu-gui-sgemm-analysis.md](docs/ncu-gui-sgemm-analysis.md)。

## 开发提示

`CMakeLists.txt` 已开启 `compile_commands.json` 生成，文件位于：

```bash
build/compile_commands.json
```

VS Code / clangd 可使用该文件进行代码补全和跳转。
