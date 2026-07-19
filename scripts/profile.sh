#!/usr/bin/env bash

set -euo pipefail

# 先定位脚本所在目录，再推导项目根目录。
# 这样无论从哪个目录执行 scripts/profile.sh，ncu-rep/ 和 build/ 都会落在项目根目录下。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

REPORT_DIR="ncu-rep"
TARGET="./build/sgemm"

usage() {
    echo "Usage: $0 -o <name>"
    echo "       $0 --output <name>"
    echo "Generate ${REPORT_DIR}/<name>.ncu-rep with Nsight Compute."
}

output_name=""

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -o|--output)
            if [[ "$#" -lt 2 || "$2" == -* ]]; then
                echo "Missing value for $1" >&2
                usage >&2
                exit 1
            fi
            output_name="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -z "${output_name}" ]]; then
    echo "Missing required -o/--output argument" >&2
    usage >&2
    exit 1
fi

output_name="${output_name%.ncu-rep}"
mkdir -p "${REPORT_DIR}"

ncu_args=(
    # 采集 NCU 提供的完整 section 集合；信息最全，但需要较多 replay pass。
    --set full

    # 把可解析的 CUDA 源码永久嵌入报告，换机器打开时也能查看源码关联。
    --import-source yes

    # 不让 NCU 修改 GPU 时钟；依靠应用内 warmup，并在报告中核对 v0/v1 频率。
    --clock-control none

    # 程序启动时先不采集，只在 cudaProfilerStart/Stop 标记之间启用采集。
    --profile-from-start off

    # 每个指标 pass 都重新启动应用，使每个正式 kernel 在每个 pass 中重新 warmup。
    --replay-mode application

    # 允许覆盖同名报告，方便使用固定实验名重复采样。
    -f

    # 指定报告输出路径；NCU 会自动追加 .ncu-rep 扩展名。
    -o "${REPORT_DIR}/${output_name}"
)

# --dry-run 跳过 CPU golden/verify 和应用侧计时，但仍执行数据生成、H2D、kernel 与 D2H。
ncu "${ncu_args[@]}" "${TARGET}" --dry-run
