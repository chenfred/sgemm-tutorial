#!/usr/bin/env bash

set -euo pipefail

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
