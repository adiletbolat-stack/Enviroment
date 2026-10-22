#!/usr/bin/env python3
"""
Step 4: publication figures from the processed measurements (data/processed/measurements.csv).

Honest by design:
  * Figure 2 (distribution / outliers) is built from the INDIVIDUAL latency observations of one variant.
  * Figure 1 (latency vs load) is built ONLY if the data contain a real load column (RPS). Otherwise the script says
    so and writes no file: record number is never used as a stand-in for load, nothing is invented.
  * Outliers are flagged by the 1.5*IQR rule and drawn with their own marker; no observation is removed.

Run from the repository root:
    python src/make_figures.py
    python src/make_figures.py --measurements data/processed/measurements.csv --out-dir reports/figures
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                                   # no window needed, works in any terminal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

LATENCY_COLUMNS = [("latency_ms", 1.0), ("latency", 1.0), ("latency_ns", 1e-6), ("latency_s", 1e3)]   # (name, factor to ms)
LOAD_COLUMNS = ["rps", "load_rps", "request_rate", "requests_per_second", "target_rps"]
MEASURE_PHASES = {"benchmark", "measure"}
OK_STATUSES = {"ok", "200", "success"}
SEED = 42


# ---------------------------------------------------------------- data
def load_measurements(path):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    print(f"File: {path} | rows: {len(df)} | columns: {df.columns.tolist()}")

    lat_col = factor = None
    for name, f in LATENCY_COLUMNS:
        if name in cols:
            lat_col, factor = cols[name], f
            break
    if lat_col is None:
        sys.exit(f"ERROR: no latency column found. Expected one of {[n for n, _ in LATENCY_COLUMNS]}.")
    unit_note = "converted from nanoseconds" if factor == 1e-6 else "assumed to be milliseconds"
    print(f"Latency column: '{lat_col}' ({unit_note})")

    if "phase" in cols:                                  # keep measured phase only, report what was removed
        n0 = len(df)
        df = df[df[cols["phase"]].isin(MEASURE_PHASES)]
        print(f"Phase filter {sorted(MEASURE_PHASES)}: kept {len(df)} of {n0} rows")

    if "status" in cols:                                 # errors are counted and reported, not hidden
        ok = df[cols["status"]].astype(str).str.lower().isin(OK_STATUSES)
        print(f"Errors: {(~ok).sum()} of {len(df)} rows ({(~ok).mean():.1%}); excluded from the latency plots only")
        df = df[ok]

    df = df.assign(latency_ms=pd.to_numeric(df[lat_col], errors="coerce") * factor)
    n_bad = df["latency_ms"].isna().sum()
    if n_bad:
        print(f"WARNING: {n_bad} rows have non-numeric latency and are skipped")
    df = df.dropna(subset=["latency_ms"]).reset_index(drop=True)
    return df, cols


def iqr_flags(x, k=1.5):
    q1, q3 = np.percentile(x, [25, 75])
    lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
    return (x < lo) | (x > hi), (lo, hi)


# ---------------------------------------------------------------- style and saving
def set_style():
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,          # embed real fonts: text stays vector/selectable
                         "savefig.dpi": 300, "axes.titlesize": 14})
    return sns.color_palette("colorblind")


def save(fig, out_dir, stem):
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf, png = out_dir / f"{stem}.pdf", out_dir / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")                # vector
    fig.savefig(png, dpi=300, bbox_inches="tight")       # raster, 300 DPI
    plt.close(fig)
    print(f"saved: {pdf} and {png}")


# ---------------------------------------------------------------- figure 2
def figure2_distribution(df, cols, out_dir, label):
    pal = set_style()
    x = df["latency_ms"].to_numpy()
    computed, (lo, hi) = iqr_flags(x)
    q1, med, q3 = np.percentile(x, [25, 50, 75])

    # If process_data.py already flagged outliers, use ITS flags (one source of truth) and cross-check them.
    flag_col = next((cols[c] for c in ("is_iqr_outlier", "is_outlier", "outlier") if c in cols), None)
    if flag_col is not None:
        flag = df[flag_col].astype(str).str.lower().isin(["true", "1"]).to_numpy()
        same = np.array_equal(flag, computed)
        print(f"\nOutlier flags from table column '{flag_col}': {int(flag.sum())} | "
              f"1.5xIQR recomputed here: {int(computed.sum())} | identical observations: {same}")
        if not same:
            print("NOTE: the table flags and the 1.5xIQR rule used for the box whiskers differ; "
                  "markers follow the table, whiskers follow 1.5xIQR. Check which rule process_data.py uses.")
    else:
        flag = computed
    print(f"Figure 2: n={len(x)} | median={med:.3f} ms | p25-p75={q1:.3f}-{q3:.3f} | "
          f"1.5xIQR fences=[{lo:.3f}, {hi:.3f}] | flagged={int(flag.sum())} (kept in the data)")

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.boxplot(x, positions=[0], widths=0.45, whis=1.5, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="#e8eef7", edgecolor="black"), medianprops=dict(color="black", linewidth=2),
               whiskerprops=dict(color="black"), capprops=dict(color="black"))
    jitter = np.random.default_rng(SEED).uniform(-0.12, 0.12, size=len(x))
    ax.scatter(jitter[~flag], x[~flag], s=22, marker="o", color=pal[0], alpha=0.55, edgecolor="none",
               label=f"Within 1.5×IQR (n={int((~flag).sum())})", zorder=3)
    ax.scatter(jitter[flag], x[flag], s=70, marker="X", color=pal[3], edgecolor="black", linewidth=0.7,
               label=f"Flagged as IQR outlier (n={int(flag.sum())})", zorder=4)
    ax.set_xticks([0])
    ax.set_xticklabels([f"{label}\n(n = {len(x)})"])
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(0, x.max() * 1.3)                        # starts at 0 (no truncation); headroom keeps the legend off the data
    ax.set_ylabel("Latency, ms")
    ax.set_title("Individual latency measurements", fontsize=13)
    ax.legend(loc="upper left", fontsize=10, frameon=True)
    ax.text(0.97, 0.03, f"median = {med:.2f} ms\np25–p75 = {q1:.2f}–{q3:.2f} ms", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10, bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.6"))
    save(fig, out_dir, "fig2_distribution_boxplot")


# ---------------------------------------------------------------- figure 1
def boot_ci(x, q, n_boot=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    stats = np.percentile(rng.choice(x, size=(n_boot, len(x))), q, axis=1)
    return np.percentile(stats, [2.5, 97.5])


def figure1_latency_vs_load(df, cols, out_dir, min_obs=30):
    load_col = next((cols[c] for c in LOAD_COLUMNS if c in cols), None)
    if load_col is None:
        print("\nFigure 1: BLOCKED. The data contain no load column (looked for: "
              f"{LOAD_COLUMNS}). Latency vs load cannot be drawn honestly; no file written.\n"
              "          Needed: measurements at several request rates (RPS), >= 3 levels, >= "
              f"{min_obs} observations each.")
        return False
    levels = sorted(df[load_col].dropna().unique())
    enough = [lv for lv in levels if (df[load_col] == lv).sum() >= min_obs]
    if len(enough) < 3:
        print(f"\nFigure 1: BLOCKED. Need >= 3 load levels with >= {min_obs} observations; found {len(enough)}.")
        return False

    pal = set_style()
    rows = []
    for lv in enough:
        x = df.loc[df[load_col] == lv, "latency_ms"].to_numpy()
        for q in (50, 95, 99):
            lo, hi = boot_ci(x, q)
            rows.append({"load": lv, "q": q, "value": np.percentile(x, q), "lo": lo, "hi": hi, "n": len(x)})
    res = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7, 4.8))
    style = {50: ("p50", "o", "-", pal[0]), 95: ("p95", "s", "--", pal[1]), 99: ("p99", "^", ":", pal[2])}
    for q, (name, marker, ls, color) in style.items():
        d = res[res["q"] == q]
        ax.fill_between(d["load"], d["lo"], d["hi"], color=color, alpha=0.2, linewidth=0)
        ax.plot(d["load"], d["value"], marker=marker, linestyle=ls, color=color, linewidth=2, markersize=7, label=name)
    ax.set_xlabel("Request rate, req/s (RPS)")
    ax.set_ylabel("Latency, ms")
    if res["value"].max() / max(res["value"].min(), 1e-9) > 20:       # values differ by orders of magnitude
        ax.set_yscale("log")
        ax.set_ylabel("Latency, ms (log scale)")
    else:
        ax.set_ylim(bottom=0)
    ax.set_title("Latency percentiles vs. load (shaded: 95% bootstrap CI)")
    ax.legend(title="Percentile", loc="upper left")
    save(fig, out_dir, "fig1_latency_vs_load")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measurements", default="data/processed/measurements.csv")
    ap.add_argument("--out-dir", default="reports/figures")
    ap.add_argument("--variant-label", default="Current implementation")
    ap.add_argument("--only", choices=["1", "2"], default=None, help="build only figure 1 or only figure 2")
    a = ap.parse_args()
    path = Path(a.measurements)
    if not path.exists():
        sys.exit(f"ERROR: {path.resolve()} not found. Run from the repository root or pass --measurements.")
    df, cols = load_measurements(path)
    if len(df) < 5:
        sys.exit("ERROR: fewer than 5 usable observations, a distribution plot would be meaningless.")
    out = Path(a.out_dir)
    if a.only in (None, "1"):
        figure1_latency_vs_load(df, cols, out)
    if a.only in (None, "2"):
        figure2_distribution(df, cols, out, a.variant_label)


if __name__ == "__main__":
    main()