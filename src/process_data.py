"""Load, validate, summarize, and retain benchmark JSONL measurements.

The repository benchmark records ``latency_ns`` (converted to milliseconds),
``phase`` (``warmup`` or ``measure``), and ``status``. It does not record
throughput, offered load, or system variants, so this module does not infer them.
Bootstrap intervals use an IID percentile bootstrap (seed 42, 2,000 draws); the
benchmark is an ordered time series, so possible autocorrelation is a limitation.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/raw/test_run_8364c978.jsonl"
DEFAULT_OUTPUT = ROOT / "data/processed"
REQUIRED_RUN_FIELDS = {"type", "phase", "idx", "status"}
BOOTSTRAP_REPS = 2_000
RANDOM_SEED = 42

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read valid JSON objects and report blank, malformed, and non-object lines."""
    if not path.is_file():
        raise FileNotFoundError(f"JSONL source does not exist: {path}")
    records: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            issues.append({"line": line_number, "reason": "blank_line"})
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append({"line": line_number, "reason": "invalid_json", "detail": str(exc)})
            continue
        if not isinstance(item, dict):
            issues.append({"line": line_number, "reason": "json_value_not_object"})
            continue
        item["_source_line"] = line_number
        records.append(item)
    return records, {"n_lines": len(lines), "n_parsed_objects": len(records), "issues": issues}


def validate_records(records: list[dict[str, Any]]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Separate metadata/summary records from valid run rows; flag schema errors."""
    run_rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for row in records:
        if row.get("type") != "run":
            continue
        missing = sorted(REQUIRED_RUN_FIELDS - row.keys())
        if missing:
            issues.append({"line": row.get("_source_line"), "reason": "missing_required_fields", "fields": missing})
            continue
        if row["phase"] not in {"warmup", "measure"}:
            issues.append({"line": row.get("_source_line"), "reason": "unknown_phase", "phase": row["phase"]})
            continue
        if not isinstance(row["status"], str) or not row["status"].strip():
            issues.append({"line": row.get("_source_line"), "reason": "invalid_status"})
            continue
        if not isinstance(row["idx"], int) or isinstance(row["idx"], bool):
            issues.append({"line": row.get("_source_line"), "reason": "invalid_idx"})
            continue
        run_rows.append(row)
    frame = pd.DataFrame(run_rows)
    if frame.empty:
        raise ValueError("No valid run records found; expected type=run JSON objects.")
    duplicate_mask = frame.duplicated(["phase", "idx"], keep=False)
    for _, row in frame.loc[duplicate_mask].iterrows():
        issues.append({"line": int(row["_source_line"]), "reason": "duplicate_phase_idx"})
    return frame, issues


def flag_latency_outliers(measurements: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Flag (do not remove) successful measured latencies using 1.5-IQR bounds."""
    frame = measurements.copy()
    valid = frame.loc[frame["status"].eq("ok") & frame["latency_ms"].notna(), "latency_ms"]
    if valid.empty:
        frame["is_iqr_outlier"] = False
        return frame, {"method": "IQR", "multiplier": 1.5, "n_reference": 0, "n_flagged": 0}
    q1, q3 = valid.quantile([0.25, 0.75])
    spread = q3 - q1
    lower, upper = float(q1 - 1.5 * spread), float(q3 + 1.5 * spread)
    frame["is_iqr_outlier"] = frame["latency_ms"].notna() & ~frame["latency_ms"].between(lower, upper)
    return frame, {"method": "IQR", "multiplier": 1.5, "n_reference": int(valid.size),
                   "q1_ms": float(q1), "q3_ms": float(q3), "lower_bound_ms": lower,
                   "upper_bound_ms": upper, "n_flagged": int(frame["is_iqr_outlier"].sum())}


def _bootstrap_interval(values: np.ndarray, rng: np.random.Generator, statistic: str) -> tuple[float, float] | None:
    if len(values) < 2:
        return None
    draws = rng.choice(values, size=(BOOTSTRAP_REPS, len(values)), replace=True)
    if statistic == "mean":
        estimates = draws.mean(axis=1)
    else:
        quantile = {"p50": 0.50, "p95": 0.95, "p99": 0.99}[statistic]
        estimates = np.quantile(draws, quantile, axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def aggregate(measurements: pd.DataFrame) -> pd.DataFrame:
    """Summarize measured latency and request-level error rate by run id."""
    successes = measurements[measurements["status"].eq("ok")].copy()
    rows: list[dict[str, Any]] = []
    for run_id, group in measurements.groupby("run_id", dropna=False, sort=True):
        successful = group.loc[group["status"].eq("ok") & group["latency_ms"].notna(), "latency_ms"].astype(float)
        values = successful.to_numpy()
        rng = np.random.default_rng(RANDOM_SEED)
        ci = {key: _bootstrap_interval(values, rng, key) for key in ("mean", "p50", "p95", "p99")}
        errors = ~group["status"].eq("ok")
        if "error_type" in group:
            errors |= group["error_type"].notna() & group["error_type"].astype(str).ne("")
        if "error_msg" in group:
            errors |= group["error_msg"].notna() & group["error_msg"].astype(str).ne("")
        row: dict[str, Any] = {
            "run_id": run_id, "phase": "measure", "n_requests": int(len(group)),
            "n_success": int((~errors).sum()), "n_errors": int(errors.sum()),
            "error_rate_pct": float(errors.mean() * 100) if len(group) else np.nan,
            "n_latency_observations": int(len(values)),
            "latency_mean_ms": float(successful.mean()) if len(values) else np.nan,
            "latency_sd_ms": float(successful.std(ddof=1)) if len(values) > 1 else np.nan,
            "latency_p50_ms": float(successful.quantile(0.50)) if len(values) else np.nan,
            "latency_p95_ms": float(successful.quantile(0.95)) if len(values) else np.nan,
            "latency_p99_ms": float(successful.quantile(0.99)) if len(values) else np.nan,
            "bootstrap_reps": BOOTSTRAP_REPS, "bootstrap_seed": RANDOM_SEED,
            "bootstrap_method": "IID percentile; 2.5th and 97.5th percentiles",
            "bootstrap_time_dependence_caveat": "Ordered runs may be autocorrelated; IID CI may understate uncertainty",
        }
        for key, interval in ci.items():
            row[f"latency_{key}_ci95_low_ms"] = interval[0] if interval else np.nan
            row[f"latency_{key}_ci95_high_ms"] = interval[1] if interval else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def process_file(input_path: Path = DEFAULT_INPUT, output_dir: Path = DEFAULT_OUTPUT) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    records, parse_report = load_jsonl(input_path)
    runs, schema_issues = validate_records(records)
    metadata = next((record for record in records if record.get("type") == "meta"), {})
    # run_id and experiment are declared once in the leading meta object, not
    # repeated on individual measurement records.
    for field in ("run_id", "experiment"):
        if field in metadata:
            runs[field] = metadata[field]
    warmups = runs[runs["phase"].eq("warmup")].copy()
    measured = runs[runs["phase"].eq("measure")].copy()
    unknown_phases = sorted({
        str(record["phase"])
        for record in records
        if record.get("type") == "run" and record.get("phase") not in {"warmup", "measure"}
    })
    if measured.empty:
        LOGGER.warning("No phase=measure records found; no performance summary will be written.")
    latency = pd.to_numeric(measured.get("latency_ns", pd.Series(index=measured.index, dtype=float)), errors="coerce")
    measured["latency_ms"] = latency / 1_000_000
    invalid_latency = measured["latency_ns"].notna() & measured["latency_ms"].isna() if "latency_ns" in measured else pd.Series(False, index=measured.index)
    for _, row in measured.loc[invalid_latency].iterrows():
        schema_issues.append({"line": int(row["_source_line"]), "reason": "latency_ns_not_numeric"})
    negative = measured["latency_ms"].lt(0)
    for _, row in measured.loc[negative].iterrows():
        schema_issues.append({"line": int(row["_source_line"]), "reason": "negative_latency"})
    measured.loc[negative, "latency_ms"] = np.nan
    measured, outlier_report = flag_latency_outliers(measured)
    summary = aggregate(measured) if not measured.empty else pd.DataFrame()
    phase_counts = runs["phase"].value_counts().to_dict()
    report = {
        "input": str(input_path), "n_lines": parse_report["n_lines"],
        "n_parsed_objects": parse_report["n_parsed_objects"], "parse_issues": parse_report["issues"],
        "n_parse_errors": len(parse_report["issues"]), "n_run_records_validated": int(len(runs)),
        "n_warmup": int(len(warmups)), "n_measure_before_validation": int(len(measured)),
        "n_measure_after_warmup_filter": int(len(measured)), "phase_counts": phase_counts,
        "unknown_phases": unknown_phases, "schema_issues": schema_issues,
        "n_schema_issues": len(schema_issues),
        "n_missing_latency_measure": int(measured["latency_ms"].isna().sum()),
        "n_failed_measure_requests": int((~measured["status"].eq("ok")).sum()),
        "error_denominator": "all valid phase=measure run records, including records without latency",
        "error_definition": "status != 'ok' or non-empty error_type/error_msg",
        "outliers": outlier_report,
        "unsupported_metrics": ["throughput: no request count/time interval field", "load: not recorded", "system variant: not recorded"],
    }
    if not runs["phase"].eq("warmup").any():
        LOGGER.warning("No warm-up records were found.")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    measured.to_csv(output_dir / "measurements.csv", index=False)
    (output_dir / "quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info("Read %d objects; parse issues=%d, warm-up=%d, measure=%d, execution errors=%d, IQR flags=%d",
                len(records), report["n_parse_errors"], len(warmups), len(measured),
                report["n_failed_measure_requests"], outlier_report["n_flagged"])
    LOGGER.info("Wrote %s and %s", output_dir / "metrics_summary.csv", output_dir / "quality_report.json")
    return summary, measured, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    process_file(args.input, args.output_dir)


if __name__ == "__main__":
    main()
