#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN="${PYTHON}"
elif [[ -x "/opt/anaconda3/bin/python3" ]]; then
    PYTHON_BIN="/opt/anaconda3/bin/python3"
else
    PYTHON_BIN="python3"
fi

export PYTHONPATH="${ROOT_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" "${ROOT_DIR}/tests/benchmark_least_squares_n4096.py" "$@"
