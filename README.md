# SGEMM Tutorial

本仓库用于练习开发 CUDA 算子，以单精度矩阵乘法（SGEMM）为主要样例。

## 项目结构

```text
include/            公共工具代码，例如 CUDA 错误检查和计时器
src/main.cpp        测试入口、数据初始化、CPU 参考校验
src/kernel.cu       CUDA SGEMM kernel 实现
CMakeLists.txt      CMake 构建配置
```

## 编译

推荐使用 CMake 在 `build/` 目录中生成构建产物：

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

## 开发提示

`CMakeLists.txt` 已开启 `compile_commands.json` 生成，文件位于：

```bash
build/compile_commands.json
```

VS Code / clangd 可使用该文件进行代码补全和跳转。
