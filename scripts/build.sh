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
run_args=()

usage() {
    echo "Usage: $0 [--clean] [--run [PROGRAM_ARG...]]"
    echo "  --clean                Remove build/ and configure from scratch before building"
    echo "  --run [PROGRAM_ARG...] Run ${TARGET} after building and pass all remaining arguments to it"
    echo
    echo "Examples:"
    echo "  $0 --run"
    echo "  $0 --clean --run --dry-run"
    echo "  $0 --run --param1 val1 --param2 val2"
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --clean)
            clean=1
            shift
            ;;
        --run)
            run=1
            shift
            run_args=("$@")
            break
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

if [[ "${clean}" -eq 1 ]]; then
    rm -rf "${BUILD_DIR}"
fi

cmake -S . -B "${BUILD_DIR}"
cmake --build "${BUILD_DIR}" -j16

if [[ "${run}" -eq 1 ]]; then
    "${TARGET}" "${run_args[@]}"
fi
