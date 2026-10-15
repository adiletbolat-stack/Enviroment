#!/usr/bin/env bash
# Full check in one command: unit tests first, then the benchmark of the RQ2 analysis step.
set -euo pipefail
python -m pytest -q
python src/bench_rq2.py --n-runs 1000 --seed 42 --out-dir logs --slo-ms 50
