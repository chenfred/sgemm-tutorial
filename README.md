# SGEMM Tutorial

本仓库用于练习开发 CUDA 算子，以单精度矩阵乘法（SGEMM）为主要样例。

## 项目结构

```text
include/              公共工具代码，例如 CUDA 错误检查和计时器
include/sgemm_func.h  SGEMM kernel host 侧调用声明
src/main.cpp          测试入口、数据初始化、CPU 参考校验
src/sgemm_naive.cu    naive SGEMM kernel 实现
src/sgemm_v1.cu       SGEMM 优化版本实现
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
```

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

程序会运行默认矩阵尺寸的 SGEMM 测试，输出耗时、GFLOPS 和 `PASS` / `FAIL` 校验结果。

## Profiling

使用 Nsight Compute 生成 profiling 报告：

```bash
scripts/profile.sh -o sgemm_v1
```

报告会生成到 `report/sgemm_v1.ncu-rep`。`report/` 目录已加入 `.gitignore`。

## 开发提示

`CMakeLists.txt` 已开启 `compile_commands.json` 生成，文件位于：

```bash
build/compile_commands.json
```

VS Code / clangd 可使用该文件进行代码补全和跳转。
