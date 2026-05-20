cmake_minimum_required(VERSION 3.18)

# 这个CMake管理的编译流程大致等价于：nvcc -std=c++17 -Iinclude src/*.cpp src/*.cu -o build/sgemm

# 启用 CXX 和 CUDA 两种语言：.cpp 由 C++ 编译器处理，.cu 由 CUDA 编译器处理。
project(sgemm_tutorial LANGUAGES CXX CUDA)

# 提供 CUDA::cudart 等 imported target。main.cpp 通过 common_utils.h 使用 CUDA runtime API，
# 因此需要 cudart 的头文件路径和链接参数。
find_package(CUDAToolkit REQUIRED)

# 让 .cpp 和 .cu 都按 C++17 方言编译；这里不是必须的，只是固定一个明确的语言标准。
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# RTX 50 系列游戏显卡是 Blackwell 架构，对应 CUDA compute capability 12.0 / sm_120。
set(CMAKE_CUDA_ARCHITECTURES 120)

# 生成 build/compile_commands.json，方便 clangd 等工具做代码补全和跳转。
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 自动收集 src/ 下的 .cpp 和 .cu。CONFIGURE_DEPENDS 让新增源码文件后重新配置构建系统。
file(GLOB SGEMM_SOURCES CONFIGURE_DEPENDS
    src/*.cpp
    src/*.cu
)

add_executable(sgemm ${SGEMM_SOURCES})

target_include_directories(sgemm PRIVATE include)
target_link_libraries(sgemm PRIVATE CUDA::cudart)

# 为 Nsight Compute 源码/SASS 对应关系保留行号信息，只传给 CUDA 编译单元。
target_compile_options(sgemm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

# 显式指定可执行文件输出到构建目录根部，即 cmake -B build 时生成 build/sgemm。
# 当前简单结构下默认也通常如此，保留它方便以后统一调整输出目录。
set_target_properties(sgemm PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
