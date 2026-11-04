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
DEFAULT_INPUT = ROOT / "data/raw/test_run.jsonl"
DEFAULT_OUTPUT = ROOT / "data/processed"
DEFAULT_LOAD_INPUT = ROOT / "data/processed/measurements_load.csv"
DEFAULT_LOAD_SUMMARY = DEFAULT_OUTPUT / "load_test_summary.csv"
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
    duplicate_columns = ["phase", "idx"] + (["variant"] if "variant" in frame else [])
    duplicate_mask = frame.duplicated(duplicate_columns, keep=False)
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


def _bootstrap_interval(values: np.ndarray, rng: np.random.Generator, statistic: str,
                        repetitions: int = BOOTSTRAP_REPS) -> tuple[float, float] | None:
    if len(values) < 2 or repetitions < 1:
        return None
    draws = rng.choice(values, size=(repetitions, len(values)), replace=True)
    if statistic == "mean":
        estimates = draws.mean(axis=1)
    else:
        quantile = {"p50": 0.50, "p95": 0.95, "p99": 0.99}[statistic]
        estimates = np.quantile(draws, quantile, axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def aggregate(measurements: pd.DataFrame, bootstrap_reps: int = BOOTSTRAP_REPS,
              seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Summarize measured latency by implementation and offered load."""
    if bootstrap_reps < 1:
        raise ValueError("bootstrap_reps must be positive")
    data = measurements.copy()
    if "variant" not in data:
        data["variant"] = "current"
    if "rps" not in data:
        data["rps"] = np.nan
    rows: list[dict[str, Any]] = []

    def append_group(variant: Any, rps: Any, group: pd.DataFrame, pooled: bool = False) -> None:
        successful = group.loc[group["status"].eq("ok") & group["latency_ms"].notna(), "latency_ms"].astype(float)
        values = successful.to_numpy()
        rng = np.random.default_rng(seed)
        ci = {key: _bootstrap_interval(values, rng, key, bootstrap_reps)
              for key in ("mean", "p50", "p95", "p99")}
        errors = ~group["status"].eq("ok")
        for field in ("error_type", "error_msg"):
            if field in group:
                errors |= group[field].notna() & group[field].astype(str).ne("")
        row: dict[str, Any] = {
            "run_id": str(group["run_id"].iloc[0]) if "run_id" in group and len(group) else "",
            "variant": variant, "rps": "pooled" if pooled else rps, "phase": "measure",
            "n_requests": int(len(group)), "n_success": int((~errors).sum()),
            "n_errors": int(errors.sum()), "error_rate_pct": float(errors.mean() * 100) if len(group) else np.nan,
            "n_latency_observations": int(len(values)),
            "latency_mean_ms": float(successful.mean()) if len(values) else np.nan,
            "latency_sd_ms": float(successful.std(ddof=1)) if len(values) > 1 else np.nan,
            "latency_p25_ms": float(successful.quantile(.25)) if len(values) else np.nan,
            "latency_p50_ms": float(successful.quantile(.50)) if len(values) else np.nan,
            "latency_p75_ms": float(successful.quantile(.75)) if len(values) else np.nan,
            "latency_p95_ms": float(successful.quantile(.95)) if len(values) else np.nan,
            "latency_p99_ms": float(successful.quantile(.99)) if len(values) else np.nan,
            "bootstrap_reps": bootstrap_reps, "bootstrap_seed": seed,
            "bootstrap_method": "IID percentile; 2.5th and 97.5th percentiles",
            "bootstrap_time_dependence_caveat": "Ordered requests may be autocorrelated; IID CI may understate uncertainty",
        }
        for key, interval in ci.items():
            row[f"latency_{key}_ci95_low_ms"] = interval[0] if interval else np.nan
            row[f"latency_{key}_ci95_high_ms"] = interval[1] if interval else np.nan
        rows.append(row)

    for variant, group in data.groupby("variant", dropna=False, sort=True):
        if "rps" in group and group["rps"].notna().any():
            for rps, load_group in group.loc[group["rps"].notna()].groupby(
                "rps", sort=True
            ):
                append_group(variant, rps, load_group)
        append_group(variant, "pooled", group, pooled=True)

    by_variant = {name: group.loc[group["status"].eq("ok") & group["latency_ms"].notna(), "latency_ms"].astype(float).to_numpy()
                  for name, group in data.groupby("variant", dropna=False, sort=True)}
    if "baseline_naive" in by_variant and "proposed_vectorized" in by_variant:
        if "pair_id" not in data:
            raise ValueError("paired implementation comparison requires pair_id")
        paired_rows = data.loc[
            data["variant"].isin(["baseline_naive", "proposed_vectorized"])
            & data["status"].eq("ok")
            & data["latency_ms"].notna()
        ]
        if paired_rows.duplicated(["pair_id", "variant"]).any():
            raise ValueError("paired comparison contains duplicate pair_id/variant rows")
        paired = paired_rows.pivot(
            index="pair_id", columns="variant", values="latency_ms"
        )
        if {"baseline_naive", "proposed_vectorized"}.issubset(paired.columns):
            paired = paired.dropna(subset=["baseline_naive", "proposed_vectorized"])
        if len(paired):
            baseline = paired["baseline_naive"].to_numpy(dtype=float)
            proposed = paired["proposed_vectorized"].to_numpy(dtype=float)
            rng = np.random.default_rng(seed)
            draws = rng.integers(0, len(paired), size=(bootstrap_reps, len(paired)))
            b_draw = baseline[draws].mean(axis=1)
            p_draw = proposed[draws].mean(axis=1)
            ratios = b_draw / np.maximum(p_draw, np.finfo(float).tiny)
            low, high = np.quantile(ratios, [.025, .975])
            for row in rows:
                if row["variant"] == "proposed_vectorized" and row["rps"] == "pooled":
                    row["speedup_vs_baseline"] = float(
                        baseline.mean() / proposed.mean()
                    )
                    row["speedup_ci95_low"] = float(low)
                    row["speedup_ci95_high"] = float(high)
    return pd.DataFrame(rows)


def load_load_test(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the tracked open-loop load-test measurements and sidecar provenance."""
    if not path.is_file():
        raise FileNotFoundError(f"Tracked load-test measurements do not exist: {path}")
    frame = pd.read_csv(path)
    required = {"phase", "idx", "t_utc", "rps", "latency_ms", "status", "run_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name}: load-test CSV missing columns: {missing}")
    frame = frame.loc[frame["phase"].eq("measure")].copy()
    frame["latency_ms"] = pd.to_numeric(frame["latency_ms"], errors="coerce")
    frame["rps"] = pd.to_numeric(frame["rps"], errors="raise")
    frame["variant"] = "load_test"
    if "error_type" not in frame:
        frame["error_type"] = None
    if "error_msg" not in frame:
        frame["error_msg"] = None
    if "pair_id" not in frame:
        frame["pair_id"] = pd.NA

    sidecar = path.with_suffix(".meta.json")
    metadata = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
    return frame, metadata


def _error_mask(frame: pd.DataFrame) -> pd.Series:
    errors = ~frame["status"].eq("ok")
    for field in ("error_type", "error_msg"):
        if field in frame:
            errors |= frame[field].notna() & frame[field].astype(str).ne("")
    return errors


def process_file(input_path: Path = DEFAULT_INPUT, output_dir: Path = DEFAULT_OUTPUT,
                 bootstrap_reps: int = BOOTSTRAP_REPS, seed: int = RANDOM_SEED,
                 load_input: Path | None = None,
                 max_error_rate_pct: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
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
    comparison_summary = aggregate(measured, bootstrap_reps, seed) if not measured.empty else pd.DataFrame()
    load_measurements = pd.DataFrame()
    load_summary = pd.DataFrame()
    load_metadata: dict[str, Any] = {}
    if load_input is not None:
        load_measurements, load_metadata = load_load_test(load_input)
        load_summary = aggregate(load_measurements, bootstrap_reps, seed)
    summary_parts = [part for part in (comparison_summary, load_summary) if not part.empty]
    summary = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()
    phase_counts = runs["phase"].value_counts().to_dict()
    measured_errors = _error_mask(measured)
    error_rate_pct = (
        float(measured_errors.mean() * 100) if len(measured) else None
    )
    if not 0 <= max_error_rate_pct <= 100:
        raise ValueError("max_error_rate_pct must be between 0 and 100")
    report = {
        "input": input_path.name, "n_lines": parse_report["n_lines"],
        "n_parsed_objects": parse_report["n_parsed_objects"], "parse_issues": parse_report["issues"],
        "n_parse_errors": len(parse_report["issues"]), "n_run_records_validated": int(len(runs)),
        "n_warmup": int(len(warmups)), "n_measure_before_validation": int(len(measured)),
        "n_measure_after_warmup_filter": int(len(measured)), "phase_counts": phase_counts,
        "unknown_phases": unknown_phases, "schema_issues": schema_issues,
        "n_schema_issues": len(schema_issues),
        "n_missing_latency_measure": int(measured["latency_ms"].isna().sum()),
        "n_failed_measure_requests": int(measured_errors.sum()),
        "error_rate_pct": error_rate_pct,
        "max_error_rate_pct": max_error_rate_pct,
        "error_guardrail_passed": (
            error_rate_pct is not None and error_rate_pct <= max_error_rate_pct
        ),
        "bootstrap_reps": bootstrap_reps, "bootstrap_seed": seed,
        "environment": metadata.get("environment_label", metadata.get("os", {}).get("platform")),
        "environment_metadata": {key: metadata.get(key) for key in ("cpu", "ram", "container", "python")},
        "comparison": metadata.get("comparison"),
        "load_test_source": None if load_input is None else load_input.name,
        "load_test_environment": {
            key: load_metadata.get(key) for key in ("os", "cpu", "ram", "python")
        },
        "n_load_test_requests": int(len(load_measurements)),
        "n_load_test_errors": int(
            _error_mask(load_measurements).sum()
        ) if not load_measurements.empty else 0,
        "load_test_error_rate_pct": (
            float(_error_mask(load_measurements).mean() * 100)
            if not load_measurements.empty
            else None
        ),
        "error_denominator": "all valid phase=measure run records, including records without latency",
        "error_definition": "status != 'ok' or non-empty error_type/error_msg; latency summaries use successful observations only",
        "outliers": outlier_report,
        "unsupported_metrics": ["throughput: wall-clock request rate is the offered-load variable"],
    }
    if not runs["phase"].eq("warmup").any():
        LOGGER.warning("No warm-up records were found.")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    if not load_summary.empty:
        load_summary.to_csv(output_dir / "load_test_summary.csv", index=False)
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
    parser.add_argument("--load-input", type=Path, default=DEFAULT_LOAD_INPUT)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/config.yaml")
    args = parser.parse_args()
    try:
        import yaml
        config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        LOGGER.warning("Config not found; using built-in bootstrap defaults")
        config = {}
    statistics_cfg = config.get("statistics", {})
    quality_cfg = config.get("quality", {})
    process_file(args.input, args.output_dir,
                 bootstrap_reps=int(statistics_cfg.get("bootstrap_reps", BOOTSTRAP_REPS)),
                 seed=int(statistics_cfg.get("bootstrap_seed", config.get("benchmark", {}).get("seed", RANDOM_SEED))),
                 load_input=args.load_input,
                 max_error_rate_pct=float(quality_cfg.get("max_error_rate_pct", 1.0)))


if __name__ == "__main__":
    main()
