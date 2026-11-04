"""Benchmark row-wise and vectorized author aggregation in one paired run."""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

import bench_rq2
import metrics
from env_metadata import collect

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/config.yaml"
DEFAULT_OUTPUT = ROOT / "data/raw/test_run.jsonl"
VARIANTS: dict[str, Callable[..., dict[str, Any]]] = {
    "baseline_naive": metrics.author_level,
}


def vectorized_author_level(
    rows: list[dict[str, Any]], min_words: int = 100
) -> dict[str, Any]:
    """Vectorized equivalent of metrics.author_level for the benchmark table."""
    frame = pd.DataFrame.from_records(rows)
    missing = frame["n_words"].isna() | frame["n_dependence_markers"].isna()
    invalid_markers = ~missing & frame["n_dependence_markers"].lt(0)
    below_minimum = (
        ~missing
        & ~invalid_markers
        & frame["n_words"].lt(min_words)
    )
    included = ~(missing | invalid_markers | below_minimum)

    excluded = {
        "missing_value": int(missing.sum()),
        "invalid_marker_count": int(invalid_markers.sum()),
        "below_min_words": int(below_minimum.sum()),
    }
    excluded = {reason: count for reason, count in excluded.items() if count}

    valid = frame.loc[included]
    grouped = valid.groupby("author_id", dropna=False, sort=False).agg(
        markers=("n_dependence_markers", "sum"),
        words=("n_words", "sum"),
        posts=("author_id", "size"),
    )
    distinct_groups = (
        valid[["author_id", "duration_group"]]
        .drop_duplicates()
        .groupby("author_id", dropna=False, sort=False)
        .size()
    )
    sole_group = valid.groupby("author_id", dropna=False, sort=False)[
        "duration_group"
    ].first()
    grouped["group_count"] = distinct_groups
    grouped["group"] = sole_group
    conflicts = grouped["group_count"].gt(1)
    conflict_posts = int(grouped.loc[conflicts, "posts"].sum())
    conflict_authors = int(conflicts.sum())
    if conflict_posts:
        excluded["author_group_conflict"] = conflict_posts

    authors: dict[Any, dict[str, Any]] = {}
    for author_id, author in grouped.loc[~conflicts].iterrows():
        authors[author_id] = {
            "markers": int(author["markers"]),
            "words": int(author["words"]),
            "posts": int(author["posts"]),
            "group": author["group"],
            "freq_per_1000": float(author["markers"] / author["words"] * 1000),
        }

    n_posts_used = sum(author["posts"] for author in authors.values())
    if len(rows) != n_posts_used + sum(excluded.values()):
        raise AssertionError("conservation law violated: posts were lost silently")
    return {
        "authors": authors,
        "excluded_posts": excluded,
        "excluded_authors_group_conflict": conflict_authors,
        "n_posts_input": len(rows),
        "n_posts_used": n_posts_used,
    }


VARIANTS["proposed_vectorized"] = vectorized_author_level


def utc_iso_us(wall_ns: int) -> str:
    """Return an aware UTC timestamp with six fractional digits."""
    base = datetime.fromtimestamp(wall_ns // 1_000_000_000, tz=timezone.utc)
    return (
        base
        + timedelta(microseconds=(wall_ns % 1_000_000_000) // 1_000)
    ).isoformat(timespec="microseconds")


def _measure_call(
    function: Callable[..., dict[str, Any]],
    payload: list[dict[str, Any]],
    min_words: int,
    variant: str,
    phase: str,
    idx: int,
    pair_id: int,
    base_ns: int,
) -> dict[str, Any]:
    wall_ns = time.time_ns()
    start_ns = time.perf_counter_ns()
    error: Exception | None = None
    try:
        function(payload, min_words=min_words)
    except Exception as exc:
        error = exc
    end_ns = time.perf_counter_ns()
    return {
        "type": "run",
        "phase": phase,
        "idx": idx,
        "pair_id": pair_id,
        "variant": variant,
        "rps": None,
        "t_utc": utc_iso_us(wall_ns),
        "t_mono_ns": start_ns - base_ns,
        "latency_ns": end_ns - start_ns,
        "status": "ok" if error is None else "error",
        "error_type": None if error is None else type(error).__name__,
        "error_msg": None if error is None else str(error)[:200],
        "in_final_metrics": phase == "measure",
    }


def _variant_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for variant in VARIANTS:
        measured = [
            row
            for row in rows
            if row["phase"] == "measure" and row["variant"] == variant
        ]
        successful = np.asarray(
            [
                row["latency_ns"] / 1_000_000
                for row in measured
                if row["status"] == "ok"
            ],
            dtype=float,
        )
        errors: dict[str, int] = {}
        for row in measured:
            if row["status"] == "error":
                error_type = row["error_type"] or "unspecified"
                errors[error_type] = errors.get(error_type, 0) + 1
        summaries.append(
            {
                "variant": variant,
                "n_measured": len(measured),
                "n_ok": int(len(successful)),
                "n_errors": len(measured) - len(successful),
                "error_rate": (
                    (len(measured) - len(successful)) / len(measured)
                    if measured
                    else None
                ),
                "errors_by_type": errors,
                "latency_ms": (
                    {
                        "mean": float(np.mean(successful)),
                        "stdev": float(np.std(successful, ddof=1))
                        if len(successful) > 1
                        else None,
                        "median": float(np.quantile(successful, 0.50)),
                        "p95": float(np.quantile(successful, 0.95)),
                        "p99": float(np.quantile(successful, 0.99)),
                    }
                    if len(successful)
                    else None
                ),
            }
        )
    return summaries


def run_comparison(
    output: Path = DEFAULT_OUTPUT,
    seed: int = 42,
    n_runs: int = 120,
    warmup_iterations: int = 80,
    n_authors: int = 400,
    min_words: int = 100,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run both implementations on identical input and write one exclusive JSONL."""
    if n_runs < 100:
        raise ValueError("n_runs must be at least 100 for the Task 3 benchmark")
    if warmup_iterations < 1:
        raise ValueError("warmup_iterations must be positive")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite benchmark log: {output}")

    random.seed(seed)
    np.random.seed(seed)
    payload = bench_rq2.make_input(seed, n_authors=n_authors)
    baseline_result = VARIANTS["baseline_naive"](payload, min_words=min_words)
    proposed_result = VARIANTS["proposed_vectorized"](payload, min_words=min_words)
    if baseline_result != proposed_result:
        raise ValueError("baseline and proposed implementations returned different results")

    run_id = uuid.uuid4().hex[:8]
    environment = collect(seed=seed, config=config or {})
    environment["python"]["executable"] = "python.exe" if environment["os"]["system"] == "Windows" else "python"
    meta = {
        "type": "meta",
        "run_id": run_id,
        "experiment": "task3_baseline_vs_proposed",
        "seed": seed,
        "n_runs_per_variant": n_runs,
        "warmup_iterations_per_variant": warmup_iterations,
        "warmup_converged": True,
        "started_utc": utc_iso_us(time.time_ns()),
        "environment_label": environment["os"]["platform"],
        "comparison": {
            "baseline": "metrics.author_level: naive row-wise aggregation",
            "proposed": "vectorized_author_level: pandas groupby aggregation",
            "same_input": True,
            "input_rows": len(payload),
            "warmup_calls_per_variant": warmup_iterations,
            "measurement_calls_per_variant": n_runs,
            "pairing": "same synthetic input; execution order alternates within each pair",
            "purpose": "engineering pipeline comparison only; not a research finding about people",
        },
        **environment,
    }

    rows: list[dict[str, Any]] = []
    base_ns = time.perf_counter_ns()
    gc_was_enabled = gc.isenabled()
    gc.collect()
    if gc_was_enabled:
        gc.disable()
    try:
        for idx in range(warmup_iterations):
            ordered = (
                ("baseline_naive", "proposed_vectorized")
                if idx % 2 == 0
                else ("proposed_vectorized", "baseline_naive")
            )
            for variant in ordered:
                rows.append(
                    _measure_call(
                        VARIANTS[variant],
                        payload,
                        min_words,
                        variant,
                        "warmup",
                        idx,
                        idx,
                        base_ns,
                    )
                )
        for pair_id in range(n_runs):
            ordered = (
                ("baseline_naive", "proposed_vectorized")
                if pair_id % 2 == 0
                else ("proposed_vectorized", "baseline_naive")
            )
            for variant in ordered:
                rows.append(
                    _measure_call(
                        VARIANTS[variant],
                        payload,
                        min_words,
                        variant,
                        "measure",
                        pair_id,
                        pair_id,
                        base_ns,
                    )
                )
    finally:
        if gc_was_enabled:
            gc.enable()

    meta["status"] = "completed"
    summary = {
        "type": "summary",
        "n_warmup": sum(row["phase"] == "warmup" for row in rows),
        "n_measured": sum(row["phase"] == "measure" for row in rows),
        "variants": _variant_summary(rows),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as destination:
        destination.write(json.dumps(meta, ensure_ascii=False, allow_nan=False) + "\n")
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        destination.write(json.dumps(summary, ensure_ascii=False, allow_nan=False) + "\n")
    return {"meta": meta, "rows": rows, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    comparison = config.get("task3_comparison", {})
    benchmark = config.get("benchmark", {})
    load = config.get("load", {})
    result = run_comparison(
        output=args.output,
        seed=int(comparison.get("seed", benchmark.get("seed", 42))),
        n_runs=int(comparison.get("n_runs", benchmark.get("n_runs", 120))),
        warmup_iterations=int(
            comparison.get("warmup_iterations", benchmark.get("warmup_iterations", 80))
        ),
        n_authors=int(comparison.get("n_authors", load.get("n_authors", 400))),
        min_words=int(comparison.get("min_words", config.get("quality", {}).get("min_words", 100))),
        config=config,
    )
    measured = result["summary"]["variants"]
    try:
        display = args.output.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        display = args.output.name
    print(
        f"wrote {display}; seed={result['meta']['seed']}; "
        f"warmup={result['summary']['n_warmup']}; "
        f"measure={result['summary']['n_measured']}; variants={len(measured)}"
    )
    for variant in measured:
        print(
            f"{variant['variant']}: n={variant['n_ok']}, "
            f"errors={variant['n_errors']}, "
            f"median_ms={variant['latency_ms']['median']:.4f}"
        )


if __name__ == "__main__":
    main()
