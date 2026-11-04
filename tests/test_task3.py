import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import bench_rq2
from benchmark_comparison import vectorized_author_level
from process_data import (
    DEFAULT_INPUT,
    DEFAULT_LOAD_INPUT,
    aggregate,
    process_file,
)
from scripts.verify_task3_summary import verify
from visualize import generate_figures, load_inputs

ROOT = Path(__file__).parents[1]


def _run_record(idx, phase, latency_ns, variant="current", status="ok", **extra):
    return {
        "type": "run",
        "phase": phase,
        "idx": idx,
        "pair_id": idx,
        "variant": variant,
        "rps": None,
        "t_utc": f"2026-11-04T10:00:{idx % 60:02d}.123456+00:00",
        "t_mono_ns": idx * 1_000_000,
        "latency_ns": latency_ns,
        "status": status,
        "error_type": None,
        "error_msg": None,
        "in_final_metrics": phase == "measure",
        **extra,
    }


def test_vectorized_author_aggregation_matches_rowwise_baseline():
    rows = bench_rq2.make_input(seed=42, n_authors=30)

    import metrics

    assert vectorized_author_level(rows, min_words=100) == metrics.author_level(
        rows, min_words=100
    )


def test_warmup_is_filtered_and_failures_remain_in_error_rate(tmp_path):
    source = tmp_path / "input.jsonl"
    records = [
        {"type": "meta", "run_id": "test", "experiment": "unit"},
        _run_record(0, "warmup", 9_000_000_000),
        _run_record(0, "measure", 1_000_000),
        _run_record(1, "measure", 2_000_000),
        _run_record(
            2,
            "measure",
            5_000_000,
            status="error",
            error_type="TimeoutError",
        ),
        {"type": "summary", "n_measured": 3},
    ]
    source.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    summary, measurements, report = process_file(
        source,
        tmp_path / "processed",
        bootstrap_reps=100,
        seed=42,
    )

    assert len(measurements) == 3
    assert report["n_warmup"] == 1
    assert report["n_measure_after_warmup_filter"] == 3
    assert report["n_failed_measure_requests"] == 1
    assert report["error_rate_pct"] == pytest.approx(100 / 3)
    assert report["error_guardrail_passed"] is False
    pooled = summary.loc[summary["rps"].astype(str).eq("pooled")].iloc[0]
    assert pooled["n_requests"] == 3
    assert pooled["n_errors"] == 1
    assert pooled["latency_p50_ms"] == pytest.approx(1.5)


def test_bootstrap_intervals_are_reproducible_for_paired_comparison():
    rows = []
    baseline = [3.0, 4.0, 5.0, 6.0]
    proposed = [1.5, 2.0, 2.5, 3.0]
    for pair_id, (base_ms, proposed_ms) in enumerate(zip(baseline, proposed)):
        rows.append(
            {
                "pair_id": pair_id,
                "variant": "baseline_naive",
                "phase": "measure",
                "status": "ok",
                "latency_ms": base_ms,
            }
        )
        rows.append(
            {
                "pair_id": pair_id,
                "variant": "proposed_vectorized",
                "phase": "measure",
                "status": "ok",
                "latency_ms": proposed_ms,
            }
        )

    first = aggregate(pd.DataFrame(rows), bootstrap_reps=500, seed=42)
    second = aggregate(pd.DataFrame(rows), bootstrap_reps=500, seed=42)
    proposed_row = first.loc[
        first["variant"].eq("proposed_vectorized")
        & first["rps"].astype(str).eq("pooled")
    ].iloc[0]
    repeated_row = second.loc[
        second["variant"].eq("proposed_vectorized")
        & second["rps"].astype(str).eq("pooled")
    ].iloc[0]

    assert proposed_row["speedup_vs_baseline"] == 2.0
    assert proposed_row["speedup_ci95_low"] == repeated_row["speedup_ci95_low"]
    assert proposed_row["speedup_ci95_high"] == repeated_row["speedup_ci95_high"]


def test_checked_in_summary_and_load_summary_match_tracked_sources(tmp_path):
    assert DEFAULT_INPUT.is_file()
    assert DEFAULT_LOAD_INPUT.is_file()
    assert len(verify(DEFAULT_INPUT, ROOT / "data/processed/metrics_summary.csv")) == 2
    generated, _, _ = process_file(
        DEFAULT_INPUT,
        tmp_path / "processed",
        bootstrap_reps=2000,
        seed=42,
        load_input=DEFAULT_LOAD_INPUT,
    )
    committed = pd.read_csv(ROOT / "data/processed/metrics_summary.csv")
    committed_load = pd.read_csv(ROOT / "data/processed/load_test_summary.csv")
    assert len(committed_load.loc[committed_load["variant"].eq("load_test")]) == 9

    records = [
        json.loads(line)
        for line in DEFAULT_INPUT.read_text(encoding="utf-8").splitlines()
    ]
    measured = pd.DataFrame(
        row
        for row in records
        if row.get("type") == "run"
        and row.get("phase") == "measure"
        and row.get("status") == "ok"
    )
    measured["latency_ms"] = measured["latency_ns"] / 1_000_000
    for variant in ("baseline_naive", "proposed_vectorized"):
        expected = measured.loc[measured["variant"].eq(variant), "latency_ms"]
        expected_quantiles = np.quantile(expected, [0.50, 0.95, 0.99], method="linear")
        for summary in (generated, committed):
            row = summary.loc[
                summary["variant"].eq(variant)
                & summary["rps"].astype(str).eq("pooled")
            ].iloc[0]
            actual = [
                row["latency_p50_ms"],
                row["latency_p95_ms"],
                row["latency_p99_ms"],
            ]
            np.testing.assert_allclose(actual, expected_quantiles, rtol=0, atol=1e-12)


def test_figures_exist_as_pdf_and_300_dpi_png(tmp_path):
    summary, measurements = load_inputs()
    paths = generate_figures(summary, measurements, tmp_path)

    assert {path.suffix for path in paths} == {".pdf", ".png"}
    for path in paths:
        assert path.is_file() and path.stat().st_size > 0
    for path in tmp_path.glob("*.png"):
        with Image.open(path) as image:
            assert image.width >= 1000 and image.height >= 1000
            assert min(image.info["dpi"]) >= 300


def test_notebook_is_executed_without_absolute_paths_or_errors():
    notebook = json.loads(
        (ROOT / "notebooks/01_data_analysis.ipynb").read_text(encoding="utf-8")
    )
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    assert [cell["execution_count"] for cell in code_cells] == list(
        range(1, len(code_cells) + 1)
    )
    assert all(cell["outputs"] for cell in code_cells)
    assert not any(
        output.get("output_type") == "error"
        for cell in code_cells
        for output in cell["outputs"]
    )
    serialized = json.dumps(notebook, ensure_ascii=False)
    assert re.search(r"[A-Za-z]:\\\\|/home/|/Users/", serialized) is None


def test_readme_comparison_rows_are_derived_from_summary_csv():
    summary = pd.read_csv(ROOT / "data/processed/metrics_summary.csv")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    def comparison_row(variant, label, speedup=False):
        row = summary.loc[
            summary["variant"].eq(variant)
            & summary["rps"].astype(str).eq("pooled")
        ].iloc[0]
        speedup_text = "reference"
        if speedup:
            speedup_text = (
                f"{row['speedup_vs_baseline']:.3f}× "
                f"[{row['speedup_ci95_low']:.3f}, "
                f"{row['speedup_ci95_high']:.3f}]"
            )
        return (
            f"| {label} | {int(row['n_requests'])} | "
            f"{int(row['n_errors'])} ({row['error_rate_pct']:.1f}%) | "
            f"{row['latency_p50_ms']:.3f} "
            f"({row['latency_p25_ms']:.3f}–{row['latency_p75_ms']:.3f}) | "
            f"[{row['latency_p50_ci95_low_ms']:.3f}, "
            f"{row['latency_p50_ci95_high_ms']:.3f}] | "
            f"{row['latency_mean_ms']:.3f} ± {row['latency_sd_ms']:.3f} | "
            f"{row['latency_p95_ms']:.3f} "
            f"[{row['latency_p95_ci95_low_ms']:.3f}, "
            f"{row['latency_p95_ci95_high_ms']:.3f}] | "
            f"{row['latency_p99_ms']:.3f} "
            f"[{row['latency_p99_ci95_low_ms']:.3f}, "
            f"{row['latency_p99_ci95_high_ms']:.3f}] | "
            f"{speedup_text} |"
        )

    assert comparison_row("baseline_naive", "Baseline (naive row-wise)") in readme
    assert comparison_row(
        "proposed_vectorized", "Proposed (vectorized)", speedup=True
    ) in readme
