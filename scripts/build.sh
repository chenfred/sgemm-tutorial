#!/usr/bin/env bash

set -euo pipefail

# 先定位脚本所在目录，再推导项目根目录。
# 这样无论从哪个目录执行 scripts/build.sh，后面的相对路径都按项目根目录计算。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

BUILD_DIR="build"
TARGET="./${BUILD_DIR}/sgemm"

clean=0
run=0

usage() {
    echo "Usage: $0 [--clean] [--run]"
    echo "  --clean  Remove build/ and configure from scratch before building"
    echo "  --run    Run ${TARGET} after a successful build"
}

for arg in "$@"; do
    case "${arg}" in
        --clean)
            clean=1
            ;;
        --run)
            run=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: ${arg}" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ "${clean}" -eq 1 ]]; then
    rm -rf "${BUILD_DIR}"
fi

cmake -S . -B "${BUILD_DIR}"
cmake --build "${BUILD_DIR}"

if [[ "${run}" -eq 1 ]]; then
    "${TARGET}"
fi
