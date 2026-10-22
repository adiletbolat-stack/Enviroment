"""Validate benchmark JSONL logs and add non-destructive data-quality flags."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_RUN_FIELDS = {
    "type",
    "phase",
    "idx",
    "t_utc",
    "t_mono_ns",
    "latency_ns",
    "status",
    "error_type",
    "error_msg",
    "in_final_metrics",
}
HTTP_ERROR_CODE = re.compile(r"(?<!\d)(429|500|504)(?!\d)")


def read_jsonl(path: Path) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Read one benchmark log and validate its meta, run, and summary records."""
    meta: dict[str, Any] | None = None
    summary: dict[str, Any] | None = None
    runs: list[dict[str, Any]] = []

    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc

            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL record must be an object")

            record_type = record.get("type")
            if record_type == "meta":
                if meta is not None:
                    raise ValueError(f"{path}:{line_number}: duplicate meta record")
                meta = record
            elif record_type == "run":
                missing = REQUIRED_RUN_FIELDS - record.keys()
                if missing:
                    raise ValueError(
                        f"{path}:{line_number}: run record missing fields: {sorted(missing)}"
                    )
                runs.append(record)
            elif record_type == "summary":
                if summary is not None:
                    raise ValueError(f"{path}:{line_number}: duplicate summary record")
                summary = record
            else:
                raise ValueError(
                    f"{path}:{line_number}: unsupported record type {record_type!r}"
                )

    if meta is None:
        raise ValueError(f"{path}: missing meta record")
    if summary is None:
        raise ValueError(f"{path}: missing summary record")
    if not runs:
        raise ValueError(f"{path}: no run records")

    frame = pd.DataFrame.from_records(runs)
    if frame["idx"].isna().any():
        raise ValueError(f"{path}: run index cannot be null")
    if frame.duplicated(["phase", "idx"]).any():
        raise ValueError(f"{path}: duplicate (phase, idx) values")
    if not frame["phase"].isin(["warmup", "measure"]).all():
        raise ValueError(f"{path}: unexpected phase value")
    if not frame["status"].isin(["ok", "error"]).all():
        raise ValueError(f"{path}: unexpected status value")
    expected_final_metrics = frame["phase"].eq("measure")
    if not pd.api.types.is_bool_dtype(frame["in_final_metrics"]) or not frame[
        "in_final_metrics"
    ].eq(expected_final_metrics).all():
        raise ValueError(f"{path}: in_final_metrics does not match phase")
    if not frame["t_mono_ns"].map(_is_nonnegative_number).all():
        raise ValueError(f"{path}: t_mono_ns must be a non-negative number")
    if not frame["t_utc"].map(_is_iso_timestamp).all():
        raise ValueError(f"{path}: invalid or missing ISO-8601 timestamp")
    if not frame["latency_ns"].map(_is_nonnegative_number).all():
        raise ValueError(f"{path}: latency_ns must be a non-negative number")

    return meta, frame, summary


def _is_iso_timestamp(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return False
    return not pd.isna(timestamp) and timestamp.tzinfo is not None


def _is_nonnegative_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def flag_outliers_iqr(
    frame: pd.DataFrame, multiplier: float = 1.5
) -> tuple[pd.DataFrame, dict[str, float | int | None]]:
    """Add IQR flags to successful measured rows; never discard outliers or errors."""
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("IQR multiplier must be finite and greater than zero")

    result = frame.copy()
    result["latency_ms"] = pd.to_numeric(result["latency_ns"]) / 1_000_000
    measured_ok = result["phase"].eq("measure") & result["status"].eq("ok")
    values = result.loc[measured_ok, "latency_ms"]
    result["is_iqr_outlier"] = pd.Series(pd.NA, index=result.index, dtype="boolean")

    if len(values) < 4:
        return result, {
            "method": "IQR",
            "multiplier": multiplier,
            "n_reference": int(len(values)),
            "q1_ms": None,
            "q3_ms": None,
            "lower_bound_ms": None,
            "upper_bound_ms": None,
            "n_flagged": 0,
        }

    q1 = float(values.quantile(0.25, interpolation="linear"))
    q3 = float(values.quantile(0.75, interpolation="linear"))
    spread = q3 - q1
    lower, upper = q1 - multiplier * spread, q3 + multiplier * spread
    result.loc[measured_ok, "is_iqr_outlier"] = values.lt(lower) | values.gt(upper)
    return result, {
        "method": "IQR",
        "multiplier": multiplier,
        "n_reference": int(len(values)),
        "q1_ms": q1,
        "q3_ms": q3,
        "lower_bound_ms": lower,
        "upper_bound_ms": upper,
        "n_flagged": int(result["is_iqr_outlier"].fillna(False).sum()),
    }


def build_quality_report(
    meta: dict[str, Any],
    frame: pd.DataFrame,
    summary: dict[str, Any],
    max_error_rate: float,
    outlier_info: dict[str, float | int | None],
) -> dict[str, Any]:
    if not 0 <= max_error_rate <= 1:
        raise ValueError("max_error_rate must be between 0 and 1")

    measured = frame.loc[frame["phase"].eq("measure")]
    error_count = int(measured["status"].eq("error").sum())
    measured_count = int(len(measured))
    error_rate = error_count / measured_count if measured_count else None

    return {
        "type": "quality_report",
        "run_id": meta.get("run_id"),
        "experiment": meta.get("experiment"),
        "source_status": meta.get("status"),
        "warmup_converged": meta.get("warmup_converged"),
        "n_records": int(len(frame)),
        "n_warmup": int(frame["phase"].eq("warmup").sum()),
        "n_measured": measured_count,
        "n_errors": error_count,
        "error_rate": error_rate,
        "max_error_rate": max_error_rate,
        "guardrail_passed": error_rate is not None and error_rate <= max_error_rate,
        "error_types": {
            str(error_type): int(count)
            for error_type, count in (
                measured.loc[measured["status"].eq("error"), "error_type"]
                .fillna("unspecified")
                .value_counts()
                .items()
            )
        },
        "http_error_codes": _http_error_counts(measured),
        "outliers": outlier_info,
        "source_summary": summary,
    }


def _http_error_counts(measured: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    messages = measured.loc[measured["status"].eq("error"), "error_msg"].dropna()
    for message in messages:
        for code in set(HTTP_ERROR_CODE.findall(str(message))):
            counts[code] = counts.get(code, 0) + 1
    return counts


def process_log(
    source: Path,
    output_csv: Path,
    report_json: Path,
    max_error_rate: float = 0.01,
) -> dict[str, Any]:
    if not 0 <= max_error_rate <= 1:
        raise ValueError("max_error_rate must be between 0 and 1")
    if source.resolve() in {output_csv.resolve(), report_json.resolve()}:
        raise ValueError("output paths must not overwrite the source JSONL")
    if output_csv.resolve() == report_json.resolve():
        raise ValueError("CSV output and report output must be different paths")

    meta, frame, summary = read_jsonl(source)
    flagged, outlier_info = flag_outliers_iqr(frame)
    report = build_quality_report(meta, flagged, summary, max_error_rate, outlier_info)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.exists() or report_json.exists():
        raise FileExistsError("refusing to overwrite an existing output file")
    with output_csv.open("x", encoding="utf-8", newline="") as csv_file:
        flagged.to_csv(csv_file, index=False)
    with report_json.open("x", encoding="utf-8") as report_file:
        json.dump(report, report_file, ensure_ascii=False, indent=2, allow_nan=False)
        report_file.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Source benchmark JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Output run-level CSV")
    parser.add_argument("--report", type=Path, required=True, help="Output quality report JSON")
    parser.add_argument(
        "--max-error-rate",
        type=float,
        default=0.01,
        help="Error-rate guardrail as a fraction (default: 0.01, i.e. 1%%)",
    )
    args = parser.parse_args()

    report = process_log(args.input, args.output, args.report, args.max_error_rate)
    print(
        f"processed={report['n_records']} errors={report['n_errors']}/"
        f"{report['n_measured']} guardrail_passed={report['guardrail_passed']} "
        f"iqr_flagged={report['outliers']['n_flagged']}"
    )


if __name__ == "__main__":
    main()
