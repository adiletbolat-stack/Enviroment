"""Render the Task 3 figures from canonical processed measurements and summaries."""

from __future__ import annotations

import argparse
import math
import struct
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "data/processed/metrics_summary.csv"
DEFAULT_MEASUREMENTS = ROOT / "data/processed/measurements.csv"
DEFAULT_OUTPUT = ROOT / "reports/figures"
SEED = 42


def load_inputs(summary_path: Path = DEFAULT_SUMMARY,
                measurements_path: Path = DEFAULT_MEASUREMENTS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the products of process_data.py and validate required columns."""
    summary = pd.read_csv(summary_path)
    measurements = pd.read_csv(measurements_path)
    required_summary = {"variant", "rps", "n_latency_observations", "latency_p50_ms",
                        "latency_p50_ci95_low_ms", "latency_p50_ci95_high_ms"}
    missing = sorted(required_summary - set(summary.columns))
    if missing:
        raise ValueError(f"Summary missing columns: {missing}")
    required_measurements = {"phase", "status", "latency_ms", "variant", "rps"}
    missing = sorted(required_measurements - set(measurements.columns))
    if missing:
        raise ValueError(f"Measurements missing columns: {missing}")
    return summary, measurements


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path, png_path = output_dir / f"{stem}.pdf", output_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    _ensure_png_resolution(png_path, 300)
    plt.close(fig)
    print(f"Saved {pdf_path.as_posix()} and {png_path.as_posix()} (PNG dpi=300)")
    return pdf_path, png_path


def _ensure_png_resolution(path: Path, minimum_dpi: int) -> None:
    """Round PNG pixels-per-metre up so metadata is not just below 300 DPI."""
    data = path.read_bytes()
    offset = 8
    target_ppm = math.ceil(minimum_dpi / 0.0254)
    while offset < len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        end = offset + 12 + length
        if chunk_type == b"pHYs":
            payload = struct.pack(">IIB", target_ppm, target_ppm, 1)
            crc = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
            data = (
                data[:offset]
                + struct.pack(">I", len(payload))
                + chunk_type
                + payload
                + struct.pack(">I", crc)
                + data[end:]
            )
            path.write_bytes(data)
            with Image.open(path) as image:
                dpi = image.info.get("dpi")
                if not dpi or min(dpi) < minimum_dpi:
                    raise ValueError(f"{path}: PNG resolution is below {minimum_dpi} DPI")
            return
        offset = end
    raise ValueError(f"{path}: PNG does not contain a pHYs resolution chunk")


def plot_latency_vs_load(summary: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    """Plot p50/p95/p99 by offered load with the processed 95% bootstrap intervals."""
    data = summary[(summary["rps"].astype(str) != "pooled") & summary["rps"].notna()].copy()
    data["rps"] = pd.to_numeric(data["rps"], errors="coerce")
    data = data.dropna(subset=["rps"])
    variants = list(data["variant"].dropna().unique())
    if len(data) == 0 or not variants:
        raise ValueError("No offered-load summaries are available to plot")
    sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.0)
    palette = sns.color_palette("colorblind", n_colors=max(2, len(variants)))
    percentile_style = {"p50": ("o", "-"), "p95": ("s", "--"), "p99": ("^", ":")}
    fig, ax = plt.subplots(figsize=(8.2, 5.4), constrained_layout=True)
    for variant_index, variant in enumerate(variants):
        for percentile, (marker, line_style) in percentile_style.items():
            values = data[data["variant"] == variant].sort_values("rps")
            y = values[f"latency_{percentile}_ms"].to_numpy(float)
            low = values[f"latency_{percentile}_ci95_low_ms"].to_numpy(float)
            high = values[f"latency_{percentile}_ci95_high_ms"].to_numpy(float)
            color = palette[variant_index]
            ax.fill_between(values["rps"].to_numpy(float), low, high, color=color, alpha=.10,
                            linewidth=0, zorder=1)
            ax.plot(values["rps"], y, marker=marker, linestyle=line_style, color=color,
                    linewidth=1.8, markersize=5, label=f"{variant} {percentile}", zorder=2)
    all_values = data[["latency_p50_ms", "latency_p95_ms", "latency_p99_ms"]].to_numpy(float)
    positive = all_values[np.isfinite(all_values) & (all_values > 0)]
    if positive.size and positive.max() / positive.min() > 20:
        ax.set_yscale("log")
        ax.set_ylabel("Latency (ms, log scale)")
    else:
        ax.set_ylabel("Latency (ms)")
        ax.set_ylim(bottom=0)
    ax.set_xlabel("Offered load (RPS)")
    ax.set_title(
        "Synthetic local load test: latency percentiles by offered load\n"
        "Shaded bands are 95% bootstrap confidence intervals"
    )
    ax.legend(title="Implementation / percentile", fontsize=8, ncols=2)
    ax.grid(True, which="both", linestyle=":", alpha=.45)
    return _save(fig, output_dir, "fig1_latency_vs_load")


def plot_distribution(summary: pd.DataFrame, measurements: pd.DataFrame,
                      output_dir: Path) -> tuple[Path, Path]:
    """Show both implementation distributions and their processed median CIs."""
    pooled = summary[summary["rps"].astype(str).eq("pooled")].copy()
    pooled = pooled[pooled["variant"].isin(["baseline_naive", "proposed_vectorized"])]
    data = measurements[(measurements["phase"] == "measure") & (measurements["status"] == "ok")].copy()
    errors = measurements[
        (measurements["phase"] == "measure") & (measurements["status"] != "ok")
    ].copy()
    data["latency_ms"] = pd.to_numeric(data["latency_ms"], errors="coerce")
    data = data.dropna(subset=["latency_ms"])
    errors["latency_ms"] = pd.to_numeric(errors["latency_ms"], errors="coerce")
    errors = errors.dropna(subset=["latency_ms"])
    variants = list(pooled["variant"].dropna().unique())
    if not variants:
        raise ValueError("No pooled summaries are available for distribution plot")
    sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.0)
    palette = sns.color_palette("colorblind", n_colors=max(2, len(variants)))
    fig, ax = plt.subplots(figsize=(8.0, 5.4), constrained_layout=True)
    box_data, positions = [], []
    markers = ["o", "s", "^", "P"]
    rng = np.random.default_rng(SEED)
    for i, variant in enumerate(variants):
        values = data.loc[data["variant"] == variant, "latency_ms"].to_numpy(float)
        row = pooled[pooled["variant"] == variant].iloc[0]
        if not len(values):
            continue
        position = i + 1
        positions.append(position)
        box_data.append(values)
        jitter = rng.uniform(-.12, .12, size=len(values))
        ax.scatter(position + jitter, values, color=palette[i], alpha=.35, s=12,
                   marker=markers[i % len(markers)], edgecolors="none", zorder=2)
        median = float(row["latency_p50_ms"])
        low = float(row["latency_p50_ci95_low_ms"])
        high = float(row["latency_p50_ci95_high_ms"])
        ax.errorbar(position, median, yerr=[[median - low], [high - median]], fmt="D",
                    color="black", ecolor="black", capsize=5, markersize=6,
                    label=f"{variant} median (95% bootstrap CI)", zorder=4)
        print(f"{variant}: n={len(values)}, median={median:.4f} ms, 95% bootstrap CI="
              f"[{low:.4f}, {high:.4f}] ms")
    artists = ax.boxplot(
        box_data, positions=positions, widths=.42, showfliers=False, patch_artist=True,
        boxprops={"edgecolor": "black"},
               medianprops={"color": "black", "linewidth": 1.5},
               whiskerprops={"color": "black"}, capprops={"color": "black"})
    for box, position in zip(artists["boxes"], positions):
        box.set_facecolor(palette[position - 1])
        box.set_alpha(.22)
    for variant_index, variant in enumerate(variants):
        failed = errors.loc[errors["variant"] == variant]
        if not failed.empty:
            ax.scatter(
                np.full(len(failed), variant_index + 1),
                failed["latency_ms"],
                marker="X",
                color=palette[variant_index],
                edgecolors="black",
                linewidths=.4,
                s=48,
                label=f"{variant} failed request",
                zorder=5,
            )
    ax.set_xticks(positions, variants)
    ax.set_xlabel("Implementation")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(
        "Synthetic engineering benchmark: measured implementation latencies\n"
        "Diamonds show median 95% bootstrap confidence intervals"
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=.45)
    return _save(fig, output_dir, "fig2_distribution_boxplot")


def generate_figures(summary: pd.DataFrame, measurements: pd.DataFrame,
                     output_dir: Path = DEFAULT_OUTPUT) -> list[Path]:
    fig1 = plot_latency_vs_load(summary, output_dir)
    fig2 = plot_distribution(summary, measurements, output_dir)
    return [*fig1, *fig2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--measurements", type=Path, default=DEFAULT_MEASUREMENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary, measurements = load_inputs(args.summary, args.measurements)
    generate_figures(summary, measurements, args.output_dir)


if __name__ == "__main__":
    main()
