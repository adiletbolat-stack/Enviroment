import json

import pandas as pd
import pytest

from clean_benchmark import (
    build_quality_report,
    flag_outliers_iqr,
    read_jsonl,
)


def make_run(
    idx, latency_ms, phase="measure", status="ok", error_type=None, error_msg=None
):
    return {
        "type": "run",
        "phase": phase,
        "idx": idx,
        "t_utc": f"2026-10-15T15:53:{idx % 60:02d}.123456+00:00",
        "t_mono_ns": idx * 1_000_000,
        "latency_ns": int(latency_ms * 1_000_000),
        "status": status,
        "error_type": error_type,
        "error_msg": error_msg,
        "in_final_metrics": phase == "measure",
    }


def test_iqr_flags_outliers_without_dropping_runs():
    frame = pd.DataFrame(
        [
            make_run(i, latency)
            for i, latency in enumerate([1, 2, 3, 4, 5, 6, 100])
        ]
        + [make_run(0, 0.5, phase="warmup")]
    )

    flagged, details = flag_outliers_iqr(frame)

    assert len(flagged) == len(frame)
    assert flagged.loc[flagged["phase"].eq("measure"), "is_iqr_outlier"].sum() == 1
    assert bool(flagged.loc[flagged["idx"].eq(6), "is_iqr_outlier"].iloc[0])
    assert pd.isna(flagged.loc[flagged["phase"].eq("warmup"), "is_iqr_outlier"].iloc[0])
    assert details["n_flagged"] == 1


def test_quality_report_counts_errors_as_guardrail_not_as_dropped_rows():
    frame = pd.DataFrame(
        [
            make_run(i, latency, status="error" if i == 9 else "ok",
                     error_type="TimeoutError" if i == 9 else None,
                     error_msg="HTTP 504 gateway timeout" if i == 9 else None)
            for i, latency in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        ]
    )
    _, outlier_details = flag_outliers_iqr(frame)

    report = build_quality_report(
        {"run_id": "test", "experiment": "unit", "status": "completed"},
        frame,
        {"type": "summary"},
        max_error_rate=0.01,
        outlier_info=outlier_details,
    )

    assert report["n_measured"] == 10
    assert report["n_errors"] == 1
    assert report["error_rate"] == 0.1
    assert report["guardrail_passed"] is False
    assert report["error_types"] == {"TimeoutError": 1}
    assert report["http_error_codes"] == {"504": 1}


def test_read_jsonl_rejects_malformed_records(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"type":"meta"}\nnot-json\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"bad.jsonl:2: invalid JSON"):
        read_jsonl(path)


def test_read_jsonl_validates_required_run_schema(tmp_path):
    path = tmp_path / "missing-field.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"type": "meta"}),
                json.dumps({"type": "run", "phase": "measure"}),
                json.dumps({"type": "summary"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="run record missing fields"):
        read_jsonl(path)
