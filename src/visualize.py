"""Scientific plots for summaries and retained benchmark measurements."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "data/processed/metrics_summary.csv"
DEFAULT_MEASUREMENTS = ROOT / "data/processed/measurements.csv"
DEFAULT_OUTPUT = ROOT / "results/figures"
LOGGER = logging.getLogger(__name__)


def load_summary(path: Path = DEFAULT_SUMMARY) -> pd.DataFrame:
    """Load the aggregate CSV with a useful error when it is unavailable."""
    if not path.is_file():
        raise FileNotFoundError(f"Summary CSV not found: {path}. Run src/process_data.py first.")
    frame = pd.read_csv(path)
    validate_summary(frame)
    return frame


def validate_summary(frame: pd.DataFrame) -> None:
    """Check columns used by the summary and distribution plots."""
    required = {"run_id", "n_latency_observations", "latency_mean_ms", "latency_mean_ci95_low_ms",
                "latency_mean_ci95_high_ms"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Summary is missing columns required for latency plots: {missing}")
    if frame.empty or frame["latency_mean_ms"].notna().sum() == 0:
        raise ValueError("Summary has no valid latency values to plot.")


def plot_latency_summary(summary: pd.DataFrame, output_path: Path) -> Path:
    """Plot mean latency and its precomputed bootstrap 95% CI."""
    validate_summary(summary)
    sns.set_theme(style="whitegrid", font_scale=1.2, palette="colorblind")
    frame = summary.dropna(subset=["latency_mean_ms"]).copy()
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    x = range(len(frame))
    y = frame["latency_mean_ms"].to_numpy()
    lower = y - frame["latency_mean_ci95_low_ms"].to_numpy()
    upper = frame["latency_mean_ci95_high_ms"].to_numpy() - y
    has_ci = pd.notna(lower).all() and pd.notna(upper).all()
    ax.errorbar(x, y, yerr=[lower, upper] if has_ci else None, fmt="o", color="#0072B2",
                ecolor="#333333", capsize=5, linewidth=1.5, label="Mean latency (95% bootstrap CI)" if has_ci else "Mean latency")
    ax.set_xticks(list(x), frame["run_id"].astype(str))
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Benchmark run")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Measured latency with 95% bootstrap confidence interval")
    ax.legend(frameon=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_latency_distribution(measurements_path: Path, output_path: Path) -> Path:
    """Plot an ECDF from individual successful measurements, when available."""
    if not measurements_path.is_file():
        raise FileNotFoundError(f"Individual measurements not found: {measurements_path}")
    frame = pd.read_csv(measurements_path)
    required = {"phase", "status", "latency_ms"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Measurements are missing columns required for a distribution plot: {missing}")
    values = pd.to_numeric(frame.loc[frame["phase"].eq("measure") & frame["status"].eq("ok"), "latency_ms"], errors="coerce").dropna()
    if values.empty:
        raise ValueError("No successful measured latency observations are available for a distribution plot.")
    sns.set_theme(style="whitegrid", font_scale=1.2, palette="colorblind")
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    sns.ecdfplot(x=values, color="#0072B2", linewidth=2, ax=ax)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Empirical cumulative probability")
    ax.set_title(f"Measured latency distribution (n = {len(values)})")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--measurements", type=Path, default=DEFAULT_MEASUREMENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = load_summary(args.summary)
    outputs = [plot_latency_summary(summary, args.output_dir / "metrics_summary_latency_mean.png")]
    try:
        outputs.append(plot_latency_distribution(args.measurements, args.output_dir / "metrics_summary_latency_ecdf.png"))
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.warning("Distribution plot unavailable: %s", exc)
    for path in outputs:
        LOGGER.info("Wrote %s", path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
