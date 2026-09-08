cmake_minimum_required(VERSION 3.18)

# CMake 分别编译 host C++ 与 CUDA 源文件，再把 CUDA Runtime 和 OpenMP 链接进同一个可执行文件。
# 这不再等价于一条简单的 nvcc 命令：verify.cpp 由 host 编译器开启 OpenMP，.cu 文件则由 nvcc 为 sm_120 编译。

# 启用 CXX 和 CUDA 两种语言：.cpp 由 C++ 编译器处理，.cu 由 CUDA 编译器处理。
project(sgemm_tutorial LANGUAGES CXX CUDA)

# 提供 CUDA Runtime 的头文件、库和 CUDA::cudart imported target。
find_package(CUDAToolkit REQUIRED)

# sgemm_golden/sgemm_verify 使用 OpenMP 并行 CPU 计算和校验；只要求 CXX 组件。
find_package(OpenMP REQUIRED COMPONENTS CXX)

# 单配置生成器未显式选择构建类型时默认使用 Release，避免 CPU golden 停留在 -O0。
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# 让 host C++ 和 CUDA 源文件都使用明确的 C++17 方言。
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# RTX 50 系列游戏显卡是 Blackwell 架构，对应 CUDA compute capability 12.0 / sm_120。
set(CMAKE_CUDA_ARCHITECTURES 86;120)

# 生成 build/compile_commands.json，方便 clangd 等工具做代码补全和跳转。
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 自动收集 src/ 下的 .cpp 和 .cu。CONFIGURE_DEPENDS 让新增源码文件后重新配置构建系统。
file(GLOB SGEMM_SOURCES CONFIGURE_DEPENDS
    src/*.cpp
    src/*.cu
)

add_executable(sgemm ${SGEMM_SOURCES})

target_include_directories(sgemm PRIVATE include)

# cudart 提供 CUDA Runtime/Profiler API；OpenMP::OpenMP_CXX 为 verify.cpp 提供编译和链接选项。
target_link_libraries(sgemm PRIVATE CUDA::cudart OpenMP::OpenMP_CXX)

# 为 Nsight Compute 源码/SASS 对应关系保留行号信息，以及完整调试符号信息，只传给 CUDA 编译单元。
target_compile_options(sgemm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo -g>
)

# 显式指定可执行文件输出到构建目录根部，即 cmake -B build 时生成 build/sgemm。
# 当前简单结构下默认也通常如此，保留它方便以后统一调整输出目录。
set_target_properties(sgemm PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
