#!/usr/bin/env bash
# Reproducible execution of the full study pipeline:
# 1. Unit tests
# 2. Data loading, validation, and quality filtering
# 3. Statistical analysis and hypothesis testing (RQ2 & RQ3)
# 4. Benchmarking verification
set -euo pipefail

if [ -z "${PYTHON:-}" ]; then
    if [ -x ".venv/Scripts/python.exe" ]; then
        PYTHON=".venv/Scripts/python.exe"
    elif [ -x ".venv/bin/python" ]; then
        PYTHON=".venv/bin/python"
    else
        PYTHON="python"
    fi
fi

INPUT_ARG="${1:-}"
if [ "$INPUT_ARG" = "--sample" ]; then
    INTERIM_OUT="data/interim/sample/analysis_set.csv"
    TABLES_OUT="results/sample/tables"
    FIGURES_OUT="results/sample/figures"
    BENCHMARK_OUT="results/sample/benchmark"
else
    INTERIM_OUT="data/interim/analysis_set.csv"
    TABLES_OUT="results/tables"
    FIGURES_OUT="results/figures"
    BENCHMARK_OUT="results/benchmark"
fi

echo "=== [1/4] Running pytest unit tests ==="
"$PYTHON" -m pytest -q

echo "=== [2/4] Running dataset loading and preprocessing ==="
if [ "$INPUT_ARG" = "--sample" ] || [ ! -f "data/synthetic/main_dataset.csv" ]; then
    echo "Using sample dataset (data/sample/sample.csv)..."
    "$PYTHON" src/load_dataset.py --config configs/study.yaml --sample --output "$INTERIM_OUT"
else
    echo "Using full synthetic dataset (data/synthetic/main_dataset.csv)..."
    "$PYTHON" src/load_dataset.py --config configs/study.yaml --output "$INTERIM_OUT"
fi

echo "=== [3/4] Running hypothesis tests (RQ2 & RQ3) ==="
"$PYTHON" src/stats.py --config configs/study.yaml --input "$INTERIM_OUT" --tables-dir "$TABLES_OUT" --figures-dir "$FIGURES_OUT"

echo "=== [4/4] Running benchmark verification ==="
"$PYTHON" src/benchmark.py --config configs/config.yaml --n-runs 120 --out-dir "$BENCHMARK_OUT"

echo "=== Pipeline completed successfully! ==="
