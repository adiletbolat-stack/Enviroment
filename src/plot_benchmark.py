"""Create publication-ready latency figures from a cleaned benchmark CSV."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REQUIRED_COLUMNS = {"phase", "idx", "latency_ms", "status", "is_iqr_outlier"}


def load_measurements(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns: {sorted(missing)}")

    measurements = frame.loc[frame["phase"].eq("measure")].copy()
    if measurements.empty:
        raise ValueError(f"{path}: no measurement-phase rows")
    if not measurements["status"].isin(["ok", "error"]).all():
        raise ValueError(f"{path}: unexpected status value")

    measurements["latency_ms"] = pd.to_numeric(
        measurements["latency_ms"], errors="raise"
    )
    if measurements["latency_ms"].isna().any() or (
        measurements["latency_ms"] < 0
    ).any():
        raise ValueError(f"{path}: measured latency must be present and non-negative")

    def parse_flag(value: Any) -> bool:
        if pd.isna(value):
            return False
        if isinstance(value, bool):
            return value
        normalized = str(value).strip().casefold()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
        raise ValueError(f"{path}: invalid is_iqr_outlier value {value!r}")

    measurements["is_iqr_outlier"] = measurements["is_iqr_outlier"].map(parse_flag)
    measurements["idx"] = pd.to_numeric(measurements["idx"], errors="raise")
    if measurements["idx"].isna().any() or measurements["idx"].duplicated().any():
        raise ValueError(f"{path}: measurement indices must be present and unique")
    return measurements.sort_values("idx")


def create_plots(
    measurements: pd.DataFrame,
    report: dict[str, Any],
    output_dir: Path,
    prefix: str,
) -> list[Path]:
    successful = measurements.loc[measurements["status"].eq("ok")]
    if successful.empty:
        raise ValueError("cannot plot a latency distribution without successful runs")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", prefix):
        raise ValueError("prefix may contain only letters, digits, underscores, and hyphens")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / f"{prefix}_latency_ecdf.png",
        output_dir / f"{prefix}_latency_ecdf.pdf",
        output_dir / f"{prefix}_run_order.png",
        output_dir / f"{prefix}_run_order.pdf",
    ]
    if any(path.exists() for path in paths):
        raise FileExistsError("refusing to overwrite existing figure files")

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    sns.set_palette("colorblind")
    colors = sns.color_palette("colorblind")
    error_count = int(measurements["status"].eq("error").sum())
    error_rate = error_count / len(measurements)
    guardrail = report.get("guardrail_passed")
    max_error_rate = report.get("max_error_rate")
    guardrail_text = (
        "PASS" if guardrail is True else "FAIL" if guardrail is False else "UNKNOWN"
    )
    annotation = (
        f"errors: {error_count}/{len(measurements)} ({error_rate:.1%}); "
        f"guardrail: {guardrail_text}"
    )
    if isinstance(max_error_rate, (int, float)):
        annotation += f" (limit {max_error_rate:.1%})"

    percentile_values = {
        percentile: float(
            successful["latency_ms"].quantile(
                percentile / 100, interpolation="linear"
            )
        )
        for percentile in (50, 95, 99)
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    sns.ecdfplot(data=successful, x="latency_ms", ax=ax, color=colors[0], linewidth=2)
    for percentile, value, color in zip(
        (50, 95, 99), percentile_values.values(), colors[1:4]
    ):
        ax.axvline(
            value,
            color=color,
            linestyle="--",
            linewidth=1.2,
            label=f"p{percentile} = {value:.2f} ms",
        )
    ax.set(
        title=(
            "Successful measurement latency distribution\n"
            f"n = {len(successful)}; {annotation}"
        ),
        xlabel="Latency (ms)",
        ylabel="Empirical cumulative probability",
        ylim=(0, 1.02),
    )
    ax.legend(loc="lower right", frameon=True)
    fig.savefig(paths[0], dpi=300, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    ax.plot(
        measurements["idx"],
        measurements["latency_ms"],
        color=colors[0],
        linewidth=0.8,
        alpha=0.35,
        zorder=1,
    )
    regular = measurements.loc[
        measurements["status"].eq("ok") & ~measurements["is_iqr_outlier"]
    ]
    flagged = measurements.loc[
        measurements["status"].eq("ok") & measurements["is_iqr_outlier"]
    ]
    errors = measurements.loc[measurements["status"].eq("error")]
    ax.scatter(
        regular["idx"],
        regular["latency_ms"],
        s=22,
        color=colors[0],
        label="Successful run",
        zorder=2,
    )
    if not flagged.empty:
        ax.scatter(
            flagged["idx"],
            flagged["latency_ms"],
            s=54,
            marker="^",
            color=colors[3],
            edgecolor="black",
            linewidth=0.4,
            label="IQR-flagged (retained)",
            zorder=3,
        )
    if not errors.empty:
        ax.scatter(
            errors["idx"],
            errors["latency_ms"],
            s=50,
            marker="x",
            color=colors[2],
            label="Failed run (retained)",
            zorder=4,
        )
    ax.set(
        title=f"Measurement latency by run order\n{annotation}",
        xlabel="Measurement index",
        ylabel="Latency / time to failure (ms)",
    )
    ax.legend(loc="best", frameon=True)
    fig.savefig(paths[2], dpi=300, bbox_inches="tight")
    fig.savefig(paths[3], bbox_inches="tight")
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Cleaned run-level CSV")
    parser.add_argument("--report", type=Path, required=True, help="Quality report JSON")
    parser.add_argument("--out-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--prefix", default=None, help="Figure filename prefix")
    args = parser.parse_args()

    measurements = load_measurements(args.input)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    prefix = args.prefix or args.input.stem.replace("_cleaned", "")
    paths = create_plots(measurements, report, args.out_dir, prefix)
    print(
        f"measurements={len(measurements)} successful="
        f"{int(measurements['status'].eq('ok').sum())} "
        f"errors={int(measurements['status'].eq('error').sum())}"
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
