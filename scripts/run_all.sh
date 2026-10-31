#!/usr/bin/env bash
# Reproducible execution of the full study pipeline:
# 1. Unit tests
# 2. Data loading, validation, and quality filtering
# 3. Statistical analysis and hypothesis testing (RQ2 & RQ3)
# 4. Benchmarking verification
set -euo pipefail

INPUT_ARG="${1:-}"

echo "=== [1/4] Running pytest unit tests ==="
python -m pytest -q

echo "=== [2/4] Running dataset loading and preprocessing ==="
if [ "$INPUT_ARG" = "--sample" ] || [ ! -f "data/synthetic/main_dataset.csv" ]; then
    echo "Using sample dataset (data/sample/sample.csv)..."
    python src/load_dataset.py --config configs/study.yaml --sample --output data/interim/analysis_set.csv
else
    echo "Using full synthetic dataset (data/synthetic/main_dataset.csv)..."
    python src/load_dataset.py --config configs/study.yaml --output data/interim/analysis_set.csv
fi

echo "=== [3/4] Running hypothesis tests (RQ2 & RQ3) ==="
python src/stats.py --config configs/study.yaml --input data/interim/analysis_set.csv

echo "=== [4/4] Running benchmark verification ==="
python src/benchmark.py --config configs/config.yaml --n-runs 120 --out-dir data/sample

echo "=== Pipeline completed successfully! ==="
