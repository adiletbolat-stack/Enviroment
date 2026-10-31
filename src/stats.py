#!/usr/bin/env python3
"""
src/stats.py — Hypothesis testing and statistical analysis for RQ2 and RQ3.

Hypotheses:
- RQ2: Emotional dependence marker frequency (F_i) differs between duration groups.
  H0: F(<1m) = F(>3m). H1: F(>3m) > F(<1m).
  Secondary: monotonic trend across <1m, 1-3m, >3m (Jonckheere-Terpstra).
- RQ3: Duration / intensity associated with isolation-avoidance index (I_i).
  H0: rho = 0. H1: rho > 0.
  Controlled OLS regression: platform, sessions, duration.

Outputs all tables and figures with the explicit banner:
"SYNTHETIC DATA — pipeline test only".
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, norm, spearmanr
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("stats")

SYNTHETIC_NOTE = "SYNTHETIC DATA -- pipeline test only"


def load_yaml_config(path: Path | str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def cliffs_delta(x: np.ndarray | list[float], y: np.ndarray | list[float]) -> float:
    """Cliff's delta effect size."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if len(x_arr) == 0 or len(y_arr) == 0:
        return float("nan")
    gt = (x_arr[:, None] > y_arr[None, :]).sum()
    lt = (x_arr[:, None] < y_arr[None, :]).sum()
    return float((gt - lt) / (len(x_arr) * len(y_arr)))


def cliffs_delta_bootstrap_ci(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    reps: int = 2000,
    seed: int = 42,
    alpha: float = 0.05
) -> tuple[float, float]:
    """Bootstrap confidence interval for Cliff's delta."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if len(x_arr) < 2 or len(y_arr) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    deltas = np.empty(reps, dtype=float)
    for i in range(reps):
        xb = rng.choice(x_arr, size=len(x_arr), replace=True)
        yb = rng.choice(y_arr, size=len(y_arr), replace=True)
        gt = (xb[:, None] > yb[None, :]).sum()
        lt = (xb[:, None] < yb[None, :]).sum()
        deltas[i] = (gt - lt) / (len(xb) * len(yb))
    ci_low = float(np.percentile(deltas, 100 * (alpha / 2)))
    ci_high = float(np.percentile(deltas, 100 * (1 - alpha / 2)))
    return ci_low, ci_high


def jonckheere_terpstra(*ordered_groups: np.ndarray | list[float]) -> tuple[float, float, float]:
    """
    Jonckheere-Terpstra non-parametric test for ordered alternatives.
    Returns (J_statistic, Z_score, one_sided_p_value).
    """
    groups = [np.asarray(g, dtype=float) for g in ordered_groups if len(g) > 0]
    k = len(groups)
    if k < 2:
        return float("nan"), float("nan"), float("nan")

    sizes = [len(g) for g in groups]
    n_total = sum(sizes)
    j_stat = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            g1, g2 = groups[i], groups[j]
            gt = (g2[:, None] > g1[None, :]).sum()
            eq = (g2[:, None] == g1[None, :]).sum()
            j_stat += gt + 0.5 * eq

    mean_j = (n_total**2 - sum(n**2 for n in sizes)) / 4.0
    var_j = (n_total**2 * (2 * n_total + 3) - sum(n**2 * (2 * n + 3) for n in sizes)) / 72.0
    if var_j <= 0:
        return float(j_stat), 0.0, 1.0

    z_score = (j_stat - mean_j) / np.sqrt(var_j)
    p_val = float(1.0 - norm.cdf(z_score))
    return float(j_stat), float(z_score), p_val


def write_table_with_synthetic_banner(df: pd.DataFrame, out_path: Path | str) -> None:
    """Save CSV table with explicit synthetic data banner."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {SYNTHETIC_NOTE}\n")
        df.to_csv(f, index=False)
    logger.info("Saved table: %s", path)


def run_rq2_analysis(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    tables_dir: Path
) -> dict[str, Any]:
    """Execute RQ2 group comparisons, effect sizes, and trend test."""
    logger.info("Running RQ2 analysis...")
    seed = cfg.get("seed", 42)
    group_defs = cfg.get("duration_groups", {})
    novice_lbl = group_defs.get("novice", "<1 month")
    inter_lbl = group_defs.get("intermediate", "1-3 months")
    exp_lbl = group_defs.get("experienced", ">3 months")

    group_col = cfg.get("columns", {}).get("duration_group", "duration_group")
    dep_rate_col = "dependence_per_1000"

    # 1. Descriptives by group
    groups_data: dict[str, np.ndarray] = {}
    summary_rows = []
    for lbl in [novice_lbl, inter_lbl, exp_lbl]:
        vals = df.loc[df[group_col] == lbl, dep_rate_col].dropna().values
        groups_data[lbl] = vals
        q25, q75 = (np.percentile(vals, 25), np.percentile(vals, 75)) if len(vals) > 0 else (np.nan, np.nan)
        summary_rows.append({
            "duration_group": lbl,
            "n": len(vals),
            "median": float(np.median(vals)) if len(vals) > 0 else np.nan,
            "iqr": float(q75 - q25) if len(vals) > 0 else np.nan,
            "q25": float(q25),
            "q75": float(q75),
            "mean": float(np.mean(vals)) if len(vals) > 0 else np.nan,
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
        })
    df_summary = pd.DataFrame(summary_rows)
    write_table_with_synthetic_banner(df_summary, tables_dir / "synthetic_rq2_group_summary.csv")

    # 2. Statistical Tests
    test_rows = []
    nov_vals = groups_data.get(novice_lbl, np.array([]))
    exp_vals = groups_data.get(exp_lbl, np.array([]))
    inter_vals = groups_data.get(inter_lbl, np.array([]))

    # Mann-Whitney U: directional H1: experienced > novice
    if len(nov_vals) >= 2 and len(exp_vals) >= 2:
        mwu = mannwhitneyu(exp_vals, nov_vals, alternative="greater")
        delta = cliffs_delta(exp_vals, nov_vals)
        ci_low, ci_high = cliffs_delta_bootstrap_ci(exp_vals, nov_vals, reps=2000, seed=seed)
        test_rows.append({
            "test": "Mann-Whitney U (experienced > novice)",
            "contrast": f"{exp_lbl} vs {novice_lbl}",
            "statistic": float(mwu.statistic),
            "p_value": float(mwu.pvalue),
            "effect_size_name": "Cliff's delta",
            "effect_size": delta,
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
            "note": "one-sided alternative"
        })

    # Kruskal-Wallis across 3 groups
    if all(len(v) >= 2 for v in [nov_vals, inter_vals, exp_vals]):
        kw = kruskal(nov_vals, inter_vals, exp_vals)
        test_rows.append({
            "test": "Kruskal-Wallis H",
            "contrast": "all 3 duration groups",
            "statistic": float(kw.statistic),
            "p_value": float(kw.pvalue),
            "effect_size_name": "eta_squared_H",
            "effect_size": float((kw.statistic - 3 + 1) / (len(df) - 3)),
            "ci_95_low": np.nan,
            "ci_95_high": np.nan,
            "note": "omnibus test"
        })

    # Jonckheere-Terpstra trend test
    if all(len(v) >= 2 for v in [nov_vals, inter_vals, exp_vals]):
        j_stat, z_val, p_val = jonckheere_terpstra(nov_vals, inter_vals, exp_vals)
        test_rows.append({
            "test": "Jonckheere-Terpstra Trend Test",
            "contrast": f"{novice_lbl} < {inter_lbl} < {exp_lbl}",
            "statistic": j_stat,
            "p_value": p_val,
            "effect_size_name": "Z-score",
            "effect_size": z_val,
            "ci_95_low": np.nan,
            "ci_95_high": np.nan,
            "note": "one-sided monotonic trend"
        })

    df_tests = pd.DataFrame(test_rows)
    write_table_with_synthetic_banner(df_tests, tables_dir / "synthetic_rq2_tests.csv")
    return {"summary": df_summary, "tests": df_tests}


def run_rq3_analysis(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    tables_dir: Path
) -> dict[str, Any]:
    """Execute RQ3 correlation and controlled regression for isolation index."""
    logger.info("Running RQ3 analysis...")
    cols = cfg.get("columns", {})
    dur_col = cols.get("duration_months", "duration_months")
    sess_col = cols.get("sessions_per_week", "sessions_per_week")
    iso_rate_col = "isolation_per_1000"

    corr_rows = []
    # 1. Spearman correlations
    for var_name, var_col in [("duration_months", dur_col), ("sessions_per_week", sess_col)]:
        valid = df[[var_col, iso_rate_col]].dropna()
        if len(valid) >= 3:
            rho, pval = spearmanr(valid[var_col], valid[iso_rate_col], alternative="greater")
            corr_rows.append({
                "variable": var_name,
                "target": "isolation_per_1000",
                "spearman_rho": float(rho),
                "p_value": float(pval),
                "n": len(valid),
                "note": "one-sided test (H1: rho > 0)"
            })
    df_corr = pd.DataFrame(corr_rows)
    write_table_with_synthetic_banner(df_corr, tables_dir / "synthetic_rq3_correlation.csv")

    # 2. Multiple Linear Regression: isolation_per_1000 ~ duration + sessions
    reg_rows = []
    try:
        import statsmodels.api as sm
        reg_df = df[[dur_col, sess_col, iso_rate_col]].dropna()
        if len(reg_df) >= 10:
            X = sm.add_constant(reg_df[[dur_col, sess_col]])
            y = reg_df[iso_rate_col]
            model = sm.OLS(y, X).fit()
            for param_name in model.params.index:
                reg_rows.append({
                    "term": param_name,
                    "coef": float(model.params[param_name]),
                    "std_err": float(model.bse[param_name]),
                    "t_value": float(model.tvalues[param_name]),
                    "p_value": float(model.pvalues[param_name]),
                    "r_squared": float(model.rsquared),
                    "n": int(model.nobs)
                })
    except ImportError:
        logger.warning("statsmodels not installed; calculating basic OLS via numpy.")
        reg_df = df[[dur_col, sess_col, iso_rate_col]].dropna()
        if len(reg_df) >= 10:
            X = np.column_stack([np.ones(len(reg_df)), reg_df[dur_col].values, reg_df[sess_col].values])
            y = reg_df[iso_rate_col].values
            beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
            reg_rows.append({"term": "intercept", "coef": float(beta[0]), "r_squared": np.nan, "n": len(reg_df)})
            reg_rows.append({"term": dur_col, "coef": float(beta[1]), "r_squared": np.nan, "n": len(reg_df)})
            reg_rows.append({"term": sess_col, "coef": float(beta[2]), "r_squared": np.nan, "n": len(reg_df)})

    df_reg = pd.DataFrame(reg_rows)
    write_table_with_synthetic_banner(df_reg, tables_dir / "synthetic_rq3_regression.csv")
    return {"correlation": df_corr, "regression": df_reg}


def generate_corpus_descriptives(df: pd.DataFrame, cfg: dict[str, Any], tables_dir: Path) -> pd.DataFrame:
    """Generate overall descriptive statistics for sample corpus."""
    cols = cfg.get("columns", {})
    group_col = cols.get("duration_group", "duration_group")
    word_col = cols.get("word_count", "word_count")

    rows = []
    for grp, grp_df in df.groupby(group_col):
        rows.append({
            "duration_group": grp,
            "n_authors": len(grp_df),
            "total_words": int(grp_df[word_col].sum()),
            "median_words": float(grp_df[word_col].median()),
            "mean_words": float(grp_df[word_col].mean()),
            "total_dependence_markers": int(grp_df["dependence_count"].sum()),
            "total_isolation_markers": int(grp_df["isolation_count"].sum()),
        })
    # Total row
    rows.append({
        "duration_group": "TOTAL",
        "n_authors": len(df),
        "total_words": int(df[word_col].sum()),
        "median_words": float(df[word_col].median()),
        "mean_words": float(df[word_col].mean()),
        "total_dependence_markers": int(df["dependence_count"].sum()),
        "total_isolation_markers": int(df["isolation_count"].sum()),
    })
    df_desc = pd.DataFrame(rows)
    write_table_with_synthetic_banner(df_desc, tables_dir / "synthetic_corpus_descriptives.csv")
    return df_desc


def generate_figures(df: pd.DataFrame, cfg: dict[str, Any], fig_dir: Path) -> None:
    """Generate and save publication-grade plots with synthetic watermark."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    group_col = cfg.get("columns", {}).get("duration_group", "duration_group")
    dur_groups = ["<1 month", "1-3 months", ">3 months"]

    # 1. RQ2 Boxplot
    plt.figure(figsize=(7, 4.5), dpi=150)
    data_to_plot = [df.loc[df[group_col] == g, "dependence_per_1000"].dropna().values for g in dur_groups]
    plt.boxplot(data_to_plot, labels=dur_groups, patch_artist=True,
                boxprops=dict(facecolor="#cce5ff", color="#1f77b4"),
                medianprops=dict(color="#d62728", linewidth=2))
    plt.title(f"{SYNTHETIC_NOTE}\nRQ2: Emotional Dependence Markers per 1,000 Words by Duration Group", fontsize=10, pad=10)
    plt.xlabel("Duration of AI Companion Use")
    plt.ylabel("Markers per 1,000 words")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_dir / "synthetic_rq2_distribution.png")
    plt.close()

    # 2. RQ3 Scatter
    plt.figure(figsize=(7, 4.5), dpi=150)
    plt.scatter(df["sessions_per_week"], df["isolation_per_1000"], alpha=0.7, color="#2ca02c", edgecolors="none")
    # Linear trendline
    if len(df.dropna(subset=["sessions_per_week", "isolation_per_1000"])) >= 5:
        sub = df.dropna(subset=["sessions_per_week", "isolation_per_1000"])
        z = np.polyfit(sub["sessions_per_week"], sub["isolation_per_1000"], 1)
        p = np.poly1d(z)
        xs = np.linspace(sub["sessions_per_week"].min(), sub["sessions_per_week"].max(), 50)
        plt.plot(xs, p(xs), "r--", linewidth=1.5, label="Trend")
        plt.legend()
    plt.title(f"{SYNTHETIC_NOTE}\nRQ3: Sessions per Week vs Isolation Index", fontsize=10, pad=10)
    plt.xlabel("Sessions per week")
    plt.ylabel("Isolation Index (per 1,000 words)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_dir / "synthetic_rq3_scatter.png")
    plt.close()
    logger.info("Saved analysis figures to %s", fig_dir)


def main():
    parser = argparse.ArgumentParser(description="Run statistical testing on study dataset.")
    parser.add_argument("--config", default="configs/study.yaml", help="Path to study.yaml")
    parser.add_argument("--input", default="data/interim/analysis_set.csv", help="Path to input interim CSV")
    parser.add_argument("--tables-dir", default="results/tables", help="Directory for output tables")
    parser.add_argument("--figures-dir", default="results/figures", help="Directory for output figures")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    df = pd.read_csv(args.input)
    tables_dir = Path(args.tables_dir)
    figures_dir = Path(args.figures_dir)

    generate_corpus_descriptives(df, cfg, tables_dir)
    run_rq2_analysis(df, cfg, tables_dir)
    run_rq3_analysis(df, cfg, tables_dir)
    generate_figures(df, cfg, figures_dir)
    logger.info("Statistical pipeline completed successfully.")


if __name__ == "__main__":
    main()
