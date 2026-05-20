#!/usr/bin/env bash

set -euo pipefail

REPORT_DIR="report"
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

ncu --set full \
    --import-source on \
    -f \
    -o "${REPORT_DIR}/${output_name}" \
    "${TARGET}"
