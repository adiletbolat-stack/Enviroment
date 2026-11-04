"""Independently verify percentile values in metrics_summary.csv from JSONL."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = ROOT / "data/raw/test_run.jsonl"
DEFAULT_SUMMARY = ROOT / "data/processed/metrics_summary.csv"
PERCENTILES = {
    "p50": ("latency_p50_ms", 0.50),
    "p95": ("latency_p95_ms", 0.95),
    "p99": ("latency_p99_ms", 0.99),
}


def verify(log_path: Path, summary_path: Path) -> list[dict[str, float | str | int]]:
    latencies: dict[str, list[float]] = {}
    with log_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{log_path.name}:{line_number}: invalid JSON") from exc
            if (
                record.get("type") == "run"
                and record.get("phase") == "measure"
                and record.get("status") == "ok"
            ):
                if "variant" not in record or "latency_ns" not in record:
                    raise ValueError(
                        f"{log_path.name}:{line_number}: missing variant or latency_ns"
                    )
                latencies.setdefault(record["variant"], []).append(
                    float(record["latency_ns"]) / 1_000_000
                )

    with summary_path.open(encoding="utf-8", newline="") as source:
        summaries = {
            row["variant"]: row
            for row in csv.DictReader(source)
            if row["rps"] == "pooled"
        }

    results = []
    for variant, values in sorted(latencies.items()):
        if variant not in summaries:
            raise ValueError(f"{summary_path.name}: missing pooled row for {variant}")
        expected = np.quantile(np.asarray(values), [p for _, p in PERCENTILES.values()])
        summary_row = summaries[variant]
        result: dict[str, float | str | int] = {
            "variant": variant,
            "n": len(values),
        }
        for (name, (column, _)), calculated in zip(PERCENTILES.items(), expected):
            recorded = float(summary_row[column])
            if not math.isclose(recorded, float(calculated), rel_tol=0, abs_tol=1e-12):
                raise AssertionError(
                    f"{variant} {name} mismatch: summary={recorded:.12f}, "
                    f"log={calculated:.12f}"
                )
            result[f"{name}_ms"] = float(calculated)
        results.append(result)
    if not results:
        raise ValueError(f"{log_path.name}: no successful measured rows")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    for result in verify(args.log, args.summary):
        print(
            f"{result['variant']}: n={result['n']} "
            f"p50={result['p50_ms']:.6f} p95={result['p95_ms']:.6f} "
            f"p99={result['p99_ms']:.6f} ms — MATCH"
        )


if __name__ == "__main__":
    main()
