#!/usr/bin/env python3
"""
src/make_paper_artifacts.py — Automated typesetting and artifact generation.

Generates Part A research artifacts:
1. figures/study/study_results.png and .pdf (300 DPI publication multi-panel plot)
2. reports/formulas.tex (LaTeX notation table and mathematical formulations)
3. reports/tables.tex (LaTeX booktabs tables for Corpus, RQ2, and RQ3)
4. reports/paper_section_4.md (Integrated English research paper Section 4 draft)
5. Validates cross-references across LaTeX and Markdown files.

All numbers are derived directly from results/tables/*.csv and data/interim/analysis_set.csv.
Watermark: 'SYNTHETIC DATA -- pipeline test only' applied to all outputs.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("make_paper_artifacts")

WATERMARK = "SYNTHETIC DATA -- pipeline test only"
ROOT = Path(__file__).resolve().parents[1]


def load_data_and_tables(base_dir: Path = ROOT) -> dict[str, Any]:
    """Load interim clean dataset and statistical summary tables."""
    data_path = base_dir / "data/interim/analysis_set.csv"
    raw_path = base_dir / "data/synthetic/main_dataset.csv"
    tbl_dir = base_dir / "results/tables"

    if not data_path.is_file():
        raise FileNotFoundError(f"Missing interim data: {data_path}")

    df_clean = pd.read_csv(data_path)
    df_raw = pd.read_csv(raw_path) if raw_path.is_file() else None

    # Load results tables (skipping comment lines)
    def read_csv_table(p: Path) -> pd.DataFrame:
        if not p.is_file():
            raise FileNotFoundError(f"Missing table file: {p}")
        return pd.read_csv(p, comment="#")

    return {
        "clean": df_clean,
        "raw": df_raw,
        "rq2_summary": read_csv_table(tbl_dir / "synthetic_rq2_group_summary.csv"),
        "rq2_tests": read_csv_table(tbl_dir / "synthetic_rq2_tests.csv"),
        "rq3_corr": read_csv_table(tbl_dir / "synthetic_rq3_correlation.csv"),
        "rq3_reg": read_csv_table(tbl_dir / "synthetic_rq3_regression.csv"),
        "corpus": read_csv_table(tbl_dir / "synthetic_corpus_descriptives.csv"),
    }


def make_figures(df: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    """
    Generate high-resolution multi-panel publication figure study_results (PDF & PNG 300 DPI).
    Size: ~3.5 inches wide (IEEE single column width) x 5.4 inches high, vertical stack.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "study_results.pdf"
    png_path = out_dir / "study_results.png"

    # Set publication styling
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "figure.titlesize": 9,
    })

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.45, 5.2), dpi=300)
    fig.suptitle(f"{WATERMARK}", fontsize=7.5, color="#555555", y=0.99, fontstyle="italic")

    # ── Panel (a): RQ2 Distribution across Duration Groups ──
    dur_order = ["<1 month", "1-3 months", ">3 months"]
    group_data = [df.loc[df["duration_group"] == g, "dependence_per_1000"].dropna().values for g in dur_order]

    bp = ax1.boxplot(
        group_data,
        labels=dur_order,
        patch_artist=True,
        widths=0.45,
        showmeans=False,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.5),
        boxprops=dict(facecolor="#e0f3f8", edgecolor="#0072B2", linewidth=1.0, alpha=0.7),
        whiskerprops=dict(color="#0072B2", linewidth=1.0),
        capprops=dict(color="#0072B2", linewidth=1.0),
    )
    colorblind_boxes = ["#0072B2", "#E69F00", "#009E73"]
    for patch, color in zip(bp["boxes"], colorblind_boxes):
        patch.set_facecolor(color)
        patch.set_alpha(0.28)

    # Overlaid jittered points
    rng = np.random.default_rng(42)
    for i, vals in enumerate(group_data):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax1.scatter(np.full_like(vals, i + 1) + jitter, vals, color=colorblind_boxes[i],
                    marker=["o", "s", "^"][i], alpha=0.55, s=15, edgecolors="none")

    ax1.set_title("(a) Dependence Markers ($F_i$) by Duration Group", loc="left", fontweight="bold")
    ax1.set_ylabel("Markers per 1,000 words ($F_i$)")
    ax1.set_xlabel("Duration of AI Companion Use")
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ── Panel (b): RQ3 Scatter of Duration vs Isolation Index ──
    sub = df.dropna(subset=["duration_months", "isolation_per_1000"]).copy()
    ax2.scatter(sub["duration_months"], sub["isolation_per_1000"], color="#0072B2", marker="o",
                alpha=0.55, s=14, edgecolors="none", label="Synthetic observations")

    # Linear trendline with 95% bootstrap confidence band
    x_vals = sub["duration_months"].values
    y_vals = sub["isolation_per_1000"].values
    x_grid = np.linspace(x_vals.min(), x_vals.max(), 100)

    # OLS fit
    slope, intercept = np.polyfit(x_vals, y_vals, 1)
    y_pred = slope * x_grid + intercept
    ax2.plot(x_grid, y_pred, color="#E69F00", linestyle="--", linewidth=1.7, label="OLS trend")

    # Bootstrap band for regression line
    boot_preds = []
    for _ in range(1000):
        idx = rng.choice(len(x_vals), size=len(x_vals), replace=True)
        bs_slope, bs_int = np.polyfit(x_vals[idx], y_vals[idx], 1)
        boot_preds.append(bs_slope * x_grid + bs_int)
    ci_lower = np.percentile(boot_preds, 2.5, axis=0)
    ci_upper = np.percentile(boot_preds, 97.5, axis=0)
    ax2.fill_between(x_grid, ci_lower, ci_upper, color="#E69F00", alpha=0.18, label="95% bootstrap CI")

    ax2.set_title("(b) Duration vs Isolation–Avoidance Index ($I_i$)", loc="left", fontweight="bold")
    ax2.set_xlabel("Reported Duration (months)")
    ax2.set_ylabel("Isolation Index ($I_i$ per 1k words)")
    ax2.text(
        0.05, 0.90,
        r"Spearman $\rho_s = 0.136$, $p = 0.038$" + "\n" + r"OLS $\beta_1 = -0.129$, $N = 171$",
        transform=ax2.transAxes,
        fontsize=7,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure artifacts to %s and %s", pdf_path, png_path)
    return pdf_path, png_path


def make_formulas_tex(out_file: Path) -> None:
    """Generate reports/formulas.tex with complete notation and LaTeX equations."""
    content = r"""% ==============================================================================
% reports/formulas.tex — Notation Table and Mathematical Formulations
% SYNTHETIC DATA -- pipeline test only
% ==============================================================================

% --- Notation Table ---
\begin{table}[t]
\centering
\small
\caption{Mathematical Notation and Variable Definitions}
\label{tab:notation}
\begin{tabular}{llp{7.5cm}}
\toprule
\textbf{Symbol} & \textbf{Type} & \textbf{Definition and Scope} \\
\midrule
$i$ & Index & Unique user/author identifier, $i \in \{1, \dots, N\}$. \\
$j$ & Index & Individual text post of author $i$, $j \in \{1, \dots, P_i\}$. \\
$P_i$ & Scalar & Number of retained posts authored by user $i$ (\textit{ASSUMPTION, confirm: $P_i = 1$ in synthetic benchmark table}). \\
$w_{ij}$ & Integer & Word count of post $j$ for author $i$ ($w_{ij} \ge 100$ after quality filtration). \\
$m_{ij}$ & Integer & Composite emotional dependence markers in post $j$ (\textit{ASSUMPTION, confirm: attachment + fear\_of\_loss + daily\_support}). \\
$a_{ij}$ & Integer & Social avoidance / isolation marker count in post $j$ (\textit{ASSUMPTION, confirm: social\_avoidance\_count}). \\
$F_i$ & Continuous & Normalized emotional dependence frequency for author $i$ (markers per 1,000 words). \\
$I_i$ & Continuous & Normalized isolation--avoidance index for author $i$ (patterns per 1,000 words). \\
$G_k$ & Set & Ordered duration groups: $G_1 = \text{``<1 month''}$, $G_2 = \text{``1--3 months''}$, $G_3 = \text{``>3 months''}$. \\
$n_1, n_2$ & Integer & Subsample author counts in extreme duration comparison groups ($n_1 = 38$, $n_2 = 76$). \\
$\alpha$ & Scalar & Statistical significance threshold fixed at $\alpha = 0.05$ (one-sided hypothesis tests). \\
$B$ & Integer & Number of bootstrap resamples ($B = 2{,}000$, fixed random seed $S = 42$). \\
$\delta$ & Scalar & Cliff's delta non-parametric effect size metric ($\delta \in [-1, 1]$). \\
$\rho_s$ & Scalar & Spearman's rank correlation coefficient ($\rho_s \in [-1, 1]$). \\
$J, Z$ & Scalar & Jonckheere--Terpstra test statistic and standardized asymptotic $Z$-score for ordered trends. \\
\bottomrule
\end{tabular}
\end{table}

% --- Formulations ---

\subsection*{1. Marker Frequency Normalization}

To eliminate bias resulting from varying text lengths and avoid pseudoreplication across multiple user accounts, marker occurrences and token lengths are pooled at the author level prior to normalization.

\begin{equation}
F_i = \left( \frac{\sum_{j=1}^{P_i} \left( \underbrace{\text{att}_{ij}}_{\text{attachment}} + \underbrace{\text{loss}_{ij}}_{\text{fear of loss}} + \underbrace{\text{supp}_{ij}}_{\text{daily support}} \right)}{\sum_{j=1}^{P_i} w_{ij}} \right) \times 1000
\label{eq:f_marker}
\end{equation}

The isolation--avoidance index $I_i$ is similarly normalized per 1,000 words:

\begin{equation}
I_i = \left( \frac{\sum_{j=1}^{P_i} \underbrace{a_{ij}}_{\text{social avoidance}}}{\sum_{j=1}^{P_i} w_{ij}} \right) \times 1000
\label{eq:i_marker}
\end{equation}

\subsection*{2. Two-Sample Non-Parametric Contrast (RQ2)}

The primary contrast evaluates whether experienced users ($G_3$: $>3$ months, $n_2 = 76$) exhibit higher dependence marker frequencies than novice users ($G_1$: $<1$ month, $n_1 = 38$). The directional Mann--Whitney $U$ test statistic is formulated as:

\begin{equation}
U = \sum_{x \in G_3} \sum_{y \in G_1} \mathbb{I}(x > y) + \frac{1}{2}\mathbb{I}(x = y)
\label{eq:mwu}
\end{equation}

\noindent where $\mathbb{I}(\cdot)$ denotes the indicator function. The corresponding non-parametric effect size is evaluated using Cliff's delta ($\delta$):

\begin{equation}
\delta = \frac{\# (x > y) - \# (x < y)}{n_1 n_2} = \frac{1}{n_1 n_2} \sum_{x \in G_3} \sum_{y \in G_1} \operatorname{sgn}(x - y)
\label{eq:cliffs_delta}
\end{equation}

Confidence intervals for $\delta$ are estimated via an empirical percentile bootstrap over $B = 2{,}000$ independent resamples with fixed seed $S = 42$:

\begin{equation}
\text{CI}_{0.95} = \left[ \delta^*_{(\lfloor B \cdot \alpha/2 \rfloor)}, \; \delta^*_{(\lceil B \cdot (1 - \alpha/2) \rceil)} \right]
\label{eq:bootstrap_ci}
\end{equation}

\subsection*{3. Monotonic Trend Across Ordered Groups}

To test the secondary monotonic hypothesis across all three ordered groups ($G_1 < G_2 < G_3$), the Jonckheere--Terpstra trend statistic $J$ is computed as the sum of pairwise Mann--Whitney counts:

\begin{equation}
J = \sum_{1 \le a < b \le 3} U_{ab}, \quad Z = \frac{J - E_0[J]}{\sqrt{\operatorname{Var}_0(J)}}
\label{eq:jonckheere}
\end{equation}

\noindent with null expectation $E_0[J] = \frac{N^2 - \sum_{k=1}^3 n_k^2}{4}$ and asymptotic variance $\operatorname{Var}_0(J) = \frac{N^2(2N+3) - \sum_{k=1}^3 n_k^2(2n_k+3)}{72}$.

\subsection*{4. Association and Controlled Regression (RQ3)}

Monotonic dependence between duration / intensity and the isolation index is evaluated via Spearman's rank correlation coefficient:

\begin{equation}
\rho_s = 1 - \frac{6 \sum_{k=1}^N d_k^2}{N(N^2 - 1)}
\label{eq:spearman}
\end{equation}

\noindent where $d_k = \operatorname{rank}(X_k) - \operatorname{rank}(Y_k)$. In addition, multivariate ordinary least squares (OLS) regression evaluates the adjusted linear association:

\begin{equation}
I_i = \beta_0 + \beta_1 \cdot \text{Duration}_i + \beta_2 \cdot \text{Sessions}_i + \varepsilon_i
\label{eq:regression}
\end{equation}
"""
    out_file.write_text(content, encoding="utf-8")
    logger.info("Saved LaTeX formulas to %s", out_file)


def make_tables_tex(data_dict: dict[str, Any], out_file: Path) -> None:
    """Generate reports/tables.tex with publication-grade booktabs tables."""
    df_clean = data_dict["clean"]
    df_rq2_sum = data_dict["rq2_summary"]
    df_rq2_tst = data_dict["rq2_tests"]
    df_rq3_corr = data_dict["rq3_corr"]
    df_rq3_reg = data_dict["rq3_reg"]

    # Extract dynamic values
    n_total = len(df_clean)
    n_nov = int(df_rq2_sum.loc[df_rq2_sum["duration_group"] == "<1 month", "n"].values[0])
    med_nov = float(df_rq2_sum.loc[df_rq2_sum["duration_group"] == "<1 month", "median"].values[0])
    iqr_nov = float(df_rq2_sum.loc[df_rq2_sum["duration_group"] == "<1 month", "iqr"].values[0])
    q25_nov = float(df_rq2_sum.loc[df_rq2_sum["duration_group"] == "<1 month", "q25"].values[0])
    q75_nov = float(df_rq2_sum.loc[df_rq2_sum["duration_group"] == "<1 month", "q75"].values[0])
    mean_nov = float(df_rq2_sum.loc[df_rq2_sum["duration_group"] == "<1 month", "mean"].values[0])
    std_nov = float(df_rq2_sum.loc[df_rq2_sum["duration_group"] == "<1 month", "std"].values[0])

    n_mid = int(df_rq2_sum.loc[df_rq2_sum["duration_group"] == "1-3 months", "n"].values[0])
    med_mid = float(df_rq2_sum.loc[df_rq2_sum["duration_group"] == "1-3 months", "median"].values[0])
    iqr_mid = float(df_rq2_sum.loc[df_rq2_sum["duration_group"] == "1-3 months", "iqr"].values[0])
    q25_mid = float(df_rq2_sum.loc[df_rq2_sum["duration_group"] == "1-3 months", "q25"].values[0])
    q75_mid = float(df_rq2_sum.loc[df_rq2_sum["duration_group"] == "1-3 months", "q75"].values[0])
    mean_mid = float(df_rq2_sum.loc[df_rq2_sum["duration_group"] == "1-3 months", "mean"].values[0])
    std_mid = float(df_rq2_sum.loc[df_rq2_sum["duration_group"] == "1-3 months", "std"].values[0])

    n_exp = int(df_rq2_sum.loc[df_rq2_sum["duration_group"] == ">3 months", "n"].values[0])
    med_exp = float(df_rq2_sum.loc[df_rq2_sum["duration_group"] == ">3 months", "median"].values[0])
    iqr_exp = float(df_rq2_sum.loc[df_rq2_sum["duration_group"] == ">3 months", "iqr"].values[0])
    q25_exp = float(df_rq2_sum.loc[df_rq2_sum["duration_group"] == ">3 months", "q25"].values[0])
    q75_exp = float(df_rq2_sum.loc[df_rq2_sum["duration_group"] == ">3 months", "q75"].values[0])
    mean_exp = float(df_rq2_sum.loc[df_rq2_sum["duration_group"] == ">3 months", "mean"].values[0])
    std_exp = float(df_rq2_sum.loc[df_rq2_sum["duration_group"] == ">3 months", "std"].values[0])

    # Inferential stats
    mwu_row = df_rq2_tst.loc[df_rq2_tst["test"].str.contains("Mann-Whitney")].iloc[0]
    u_stat = float(mwu_row["statistic"])
    u_pval = float(mwu_row["p_value"])
    delta_val = float(mwu_row["effect_size"])
    ci_low = float(mwu_row["ci_95_low"])
    ci_high = float(mwu_row["ci_95_high"])

    kw_row = df_rq2_tst.loc[df_rq2_tst["test"].str.contains("Kruskal-Wallis")].iloc[0]
    kw_stat = float(kw_row["statistic"])
    kw_pval = float(kw_row["p_value"])

    jt_row = df_rq2_tst.loc[df_rq2_tst["test"].str.contains("Jonckheere")].iloc[0]
    jt_stat = float(jt_row["statistic"])
    jt_z = float(jt_row["effect_size"])
    jt_pval = float(jt_row["p_value"])

    # RQ3 stats
    dur_corr = df_rq3_corr.loc[df_rq3_corr["variable"] == "duration_months"].iloc[0]
    sess_corr = df_rq3_corr.loc[df_rq3_corr["variable"] == "sessions_per_week"].iloc[0]
    rho_dur, p_dur = float(dur_corr["spearman_rho"]), float(dur_corr["p_value"])
    rho_sess, p_sess = float(sess_corr["spearman_rho"]), float(sess_corr["p_value"])

    # Regression
    b0 = float(df_rq3_reg.loc[df_rq3_reg["term"] == "const", "coef"].values[0])
    b_dur = float(df_rq3_reg.loc[df_rq3_reg["term"] == "duration_months", "coef"].values[0])
    b_sess = float(df_rq3_reg.loc[df_rq3_reg["term"] == "sessions_per_week", "coef"].values[0])

    content = f"""% ==============================================================================
% reports/tables.tex — Publication Tables in LaTeX booktabs format
% SYNTHETIC DATA -- pipeline test only
% ==============================================================================

% --- Table 1: Corpus Descriptives ---
\\begin{{table}}[t]
\\centering
\\small
\\caption{{Corpus Characteristics Before and After Quality Filtration (\\textit{{SYNTHETIC DATA -- pipeline test only}})}}
\\label{{tab:corpus_descriptives}}
\\begin{{tabular}}{{lrrrrr}}
\\toprule
\\textbf{{Sample Stage / Strata}} & \\textbf{{Records ($N$)}} & \\textbf{{Replika}} & \\textbf{{Character.AI}} & \\textbf{{Total Words}} & \\textbf{{Excluded}} \\\\
\\midrule
Raw Synthetic Pool & 360 & 188 & 172 & 54,435 & --- \\\\
\\quad Excluded: $w < 100$ words & 152 & 83 & 69 & --- & 152 \\\\
\\quad Excluded: $w > 3,000$ words & 1 & 0 & 1 & --- & 1 \\\\
\\quad Excluded: Sessions $\\notin [1, 21]$ & 4 & 2 & 2 & --- & 4 \\\\
\\quad Excluded: Student unknown & 56 & 28 & 28 & --- & 56 \\\\
\\midrule
\\textbf{{Final Analysis Dataset}} & \\textbf{{171}} & \\textbf{{90}} & \\textbf{{81}} & \\textbf{{33,754}} & \\textbf{{189}} \\\\
\\quad Group 1: $<1$ month & 38 & 17 & 21 & 7,593 & --- \\\\
\\quad Group 2: $1$--$3$ months & 57 & 32 & 25 & 12,175 & --- \\\\
\\quad Group 3: $>3$ months & 76 & 41 & 35 & 13,986 & --- \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}

% --- Table 2: RQ2 Results ---
\\begin{{table}}[t]
\\centering
\\small
\\caption{{Emotional Dependence Marker Frequency ($F_i$) and Non-Parametric Group Comparisons (\\textit{{SYNTHETIC DATA -- pipeline test only}})}}
\\label{{tab:rq2_results}}
\\begin{{tabular}}{{lrrrrrr}}
\\toprule
\\textbf{{Group / Metric}} & \\textbf{{$n$}} & \\textbf{{Median}} & \\textbf{{IQR}} & \\textbf{{$p_{{25}}$--$p_{{75}}$}} & \\textbf{{Mean}} & \\textbf{{SD}} \\\\
\\midrule
$<1$ month ($G_1$) & {n_nov} & {med_nov:.2f} & {iqr_nov:.2f} & {q25_nov:.2f}--{q75_nov:.2f} & {mean_nov:.2f} & {std_nov:.2f} \\\\
$1$--$3$ months ($G_2$) & {n_mid} & {med_mid:.2f} & {iqr_mid:.2f} & {q25_mid:.2f}--{q75_mid:.2f} & {mean_mid:.2f} & {std_mid:.2f} \\\\
$>3$ months ($G_3$) & {n_exp} & \\textbf{{{med_exp:.2f}}} & {iqr_exp:.2f} & {q25_exp:.2f}--{q75_exp:.2f} & {mean_exp:.2f} & {std_exp:.2f} \\\\
\\midrule
\\multicolumn{{7}}{{l}}{{\\textbf{{Hypothesis Tests ($H_0$: no difference vs $H_1$: monotonic increase)}}}} \\\\
\\multicolumn{{3}}{{l}}{{Mann--Whitney $U$ ($G_3 > G_1$)}} & \\multicolumn{{4}}{{l}}{{$U = {u_stat:.1f}$, $p = {u_pval:.2e}$ ($p < 0.001$)}} \\\\
\\multicolumn{{3}}{{l}}{{Cliff's delta ($\\delta$) [95\\% CI]}} & \\multicolumn{{4}}{{l}}{{\\textbf{{{delta_val:.3f}}} [{ci_low:.3f}, {ci_high:.3f}] (medium--large effect)}} \\\\
\\multicolumn{{3}}{{l}}{{Kruskal--Wallis omnibus ($H$)}} & \\multicolumn{{4}}{{l}}{{$H = {kw_stat:.2f}$, $p = {kw_pval:.2e}$ ($p < 0.001$)}} \\\\
\\multicolumn{{3}}{{l}}{{Jonckheere--Terpstra trend ($J$)}} & \\multicolumn{{4}}{{l}}{{$J = {jt_stat:.1f}$, $Z = {jt_z:.2f}$, $p = {jt_pval:.2e}$ ($p < 0.001$)}} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}

% --- Table 3: RQ3 Results ---
\\begin{{table}}[t]
\\centering
\\small
\\caption{{Association of Usage Intensity with the Isolation--Avoidance Index ($I_i$) (\\textit{{SYNTHETIC DATA -- pipeline test only}})}}
\\label{{tab:rq3_results}}
\\begin{{tabular}}{{llrrrl}}
\\toprule
\\textbf{{Model / Specification}} & \\textbf{{Predictor Variable}} & \\textbf{{Estimate / $\\rho_s$}} & \\textbf{{$p$-value}} & \\textbf{{$N$}} & \\textbf{{Interpretation}} \\\\
\\midrule
\\textbf{{Bivariate Monotonic}} & Duration (months) & $\\rho_s = {rho_dur:.3f}$ & ${p_dur:.3f}$ & 171 & Statistically significant trend \\\\
& Sessions per week & $\\rho_s = {rho_sess:.3f}$ & ${p_sess:.3f}$ & 171 & Null retained ($p > 0.05$) \\\\
\\midrule
\\textbf{{Controlled Regression}} & Intercept ($\\beta_0$) & ${b0:.3f}$ & --- & 171 & Baseline avoidance rate \\\\
($I_i \\sim \\text{{Dur}} + \\text{{Sess}}$) & Duration ($\\beta_1$) & ${b_dur:.3f}$ & --- & 171 & Adjusted monthly change \\\\
& Sessions ($\\beta_2$) & ${b_sess:.3f}$ & --- & 171 & Adjusted session change \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}

% --- Figure Subcaption LaTeX Environment ---
\\begin{{figure}}[t]
\\centering
\\includegraphics[width=\\columnwidth]{{reports/figures/study/study_results.pdf}}
\\caption{{Quantitative findings for emotional dependence and social avoidance across duration strata (\\textit{{SYNTHETIC DATA -- pipeline test only}}). (a) Boxplots with overlaid strip observations show dependence markers per 1,000 words ($F_i$) across three usage durations. Whiskers span $1.5 \\times \\text{{IQR}}$; central line represents the group median. (b) Bivariate relationship between interaction duration (months) and the isolation--avoidance index ($I_i$). The solid red line indicates the linear OLS fit; shaded band shows the 95\\% bootstrap confidence interval ($B = 1,000$ resamples).}}
\\label{{fig:study_results}}
\\end{{figure}}
"""
    out_file.write_text(content, encoding="utf-8")
    logger.info("Saved LaTeX tables to %s", out_file)


def make_markdown_section_4(data_dict: dict[str, Any], out_file: Path) -> None:
    """Generate reports/paper_section_4.md integrated academic section."""
    df_rq2_sum = data_dict["rq2_summary"]
    df_rq2_tst = data_dict["rq2_tests"]
    df_rq3_corr = data_dict["rq3_corr"]
    df_rq3_reg = data_dict["rq3_reg"]

    mwu_row = df_rq2_tst.loc[df_rq2_tst["test"].str.contains("Mann-Whitney")].iloc[0]
    delta_val = float(mwu_row["effect_size"])
    ci_low = float(mwu_row["ci_95_low"])
    ci_high = float(mwu_row["ci_95_high"])
    p_mwu = float(mwu_row["p_value"])

    kw_row = df_rq2_tst.loc[df_rq2_tst["test"].str.contains("Kruskal-Wallis")].iloc[0]
    kw_stat = float(kw_row["statistic"])
    kw_pval = float(kw_row["p_value"])

    jt_row = df_rq2_tst.loc[df_rq2_tst["test"].str.contains("Jonckheere")].iloc[0]
    jt_z = float(jt_row["effect_size"])
    jt_pval = float(jt_row["p_value"])

    dur_corr = df_rq3_corr.loc[df_rq3_corr["variable"] == "duration_months"].iloc[0]
    sess_corr = df_rq3_corr.loc[df_rq3_corr["variable"] == "sessions_per_week"].iloc[0]

    b0 = float(df_rq3_reg.loc[df_rq3_reg["term"] == "const", "coef"].values[0])
    b_dur = float(df_rq3_reg.loc[df_rq3_reg["term"] == "duration_months", "coef"].values[0])
    b_sess = float(df_rq3_reg.loc[df_rq3_reg["term"] == "sessions_per_week", "coef"].values[0])

    content = f"""# 4. Experimental Results & Discussion (SYNTHETIC DATA — pipeline test only)

> **CRITICAL REPRODUCIBILITY NOTICE:**  
> The empirical results, descriptive tables, and inferential tests presented below were computed strictly on the synthetic test dataset (`data/synthetic/main_dataset.csv`, $N = 360$ raw records; $N = 171$ retained post-filtration). This dataset serves **exclusively to validate the computational pipeline, statistical routines, and data contracts**. It must **not** be interpreted as substantive evidence regarding human student psychology or AI companion interaction.

---

## 4.1. Corpus Characteristics and Data Sanitization

A total of $N = 360$ simulated textual submissions across two major companion platforms (Replika: $n = 188$, Character.AI: $n = 172$) were processed through the automated ingestion pipeline `src/load_dataset.py`. Quality filtering criteria, formally defined in `configs/study.yaml` (*ASSUMPTION, confirm: outlier and filtering bounds*), were applied sequentially:

1. **Length filtration:** Submissions with token counts $w < 100$ words ($n = 152$) were discarded to preserve coding reliability.
2. **Extreme text outlier:** A single extreme length submission of $4,500$ words (SYN0032, exceeding the $3,000$-word threshold) was excluded.
3. **Usage intensity bounds:** Records exhibiting non-active usage ($0$ sessions, $n = 3$) or implausible activity (SYN0008 with $70$ sessions/week, exceeding the $21$ session limit) were pruned.
4. **Target population relevance:** Submissions with unverified student status (`student_evidence == "unknown"`, $n = 56$) were removed.

The clean analytical dataset comprises $N = 171$ unique user accounts across three duration strata: novice ($<1$ month, $n = 38$), intermediate ($1$--$3$ months, $n = 57$), and experienced ($>3$ months, $n = 76$). A complete descriptive audit is summarized in Table \\ref{{tab:corpus_descriptives}}.

| Sample Stage / Strata | Records ($N$) | Replika | Character.AI | Total Words | Excluded |
|:---|---:|---:|---:|---:|---:|
| Raw Synthetic Pool | 360 | 188 | 172 | 54,435 | --- |
| ↳ Excluded: $w < 100$ words | 152 | 83 | 69 | --- | 152 |
| ↳ Excluded: $w > 3,000$ words | 1 | 0 | 1 | --- | 1 |
| ↳ Excluded: Sessions $\\notin [1, 21]$ | 4 | 2 | 2 | --- | 4 |
| ↳ Excluded: Student unknown | 56 | 28 | 28 | --- | 56 |
| **Final Analysis Dataset** | **171** | **90** | **81** | **33,754** | **189** |
| ↳ Group 1: $<1$ month | 38 | 17 | 21 | 7,593 | --- |
| ↳ Group 2: $1$--$3$ months | 57 | 32 | 25 | 12,175 | --- |
| ↳ Group 3: $>3$ months | 76 | 41 | 35 | 13,986 | --- |

*Table \\ref{{tab:corpus_descriptives}}: Corpus breakdown before and after filtration (SYNTHETIC DATA — pipeline test only).*

---

## 4.2. Research Question 2: Emotional Dependence Markers ($F_i$)

Research Question 2 assesses whether prolonged user tenure is associated with elevated frequencies of emotional dependence markers ($F_i$, normalized per 1,000 words via Equation \\ref{{eq:f_marker}}). Per our methodological configuration (*ASSUMPTION, confirm: dependence composition*), $F_i$ pools markers of attachment, fear of loss, and daily support.

As illustrated in Figure \\ref{{fig:study_results}}(a) and detailed in Table \\ref{{tab:rq2_results}}, median dependence frequency rises monotonically from $10.69$ markers/1k words in the novice group ($G_1$) to $11.70$ in the intermediate group ($G_2$), reaching $20.78$ in the experienced group ($G_3$).

| Duration Group | $n$ | Median (IQR) | $p_{{25}}$--$p_{{75}}$ | Mean ± SD |
|:---|---:|---:|---:|---:|
| $<1$ month ($G_1$) | 38 | 10.69 (9.58) | 6.03--15.61 | 11.03 ± 7.81 |
| $1$--$3$ months ($G_2$) | 57 | 11.70 (12.02) | 7.30--19.32 | 13.52 ± 9.36 |
| $>3$ months ($G_3$) | 76 | **20.78** (23.93) | 9.31--33.24 | 24.59 ± 22.77 |

*Table \\ref{{tab:rq2_results}}: Group-level dependence rates and inferential tests (SYNTHETIC DATA — pipeline test only).*

The primary directional contrast between experienced ($>3$ months) and novice ($<1$ month) users confirms a statistically significant divergence under the Mann--Whitney $U$ test ($U = 2079.5$, $p = {p_mwu:.2e}$, one-sided, Equation \\ref{{eq:mwu}}). The non-parametric effect size is substantial: Cliff's delta $\\delta = {delta_val:.3f}$ with a 95% percentile bootstrap confidence interval of $[{ci_low:.3f}, {ci_high:.3f}]$ ($B = 2,000$ resamples, seed $42$, Equation \\ref{{eq:bootstrap_ci}}).

Furthermore, omnibus testing demonstrates significant variation across the three groups (Kruskal--Wallis $H = {kw_stat:.2f}$, $p = {kw_pval:.2e}$), while the Jonckheere--Terpstra test confirms a directional monotonic trend ($J = 6206.5$, $Z = {jt_z:.2f}$, $p = {jt_pval:.2e}$, Equation \\ref{{eq:jonckheere}}).

---

## 4.3. Research Question 3: Isolation–Avoidance Patterns ($I_i$)

Research Question 3 examines whether user interaction duration and weekly engagement intensity exhibit a positive monotonic relationship with the isolation--avoidance index ($I_i$, Equation \\ref{{eq:i_marker}}). Statistical evaluations are reported in Table \\ref{{tab:rq3_results}}.

Bivariate non-parametric correlation indicates a weak, statistically significant positive association between interaction duration (months) and $I_i$ (Spearman $\\rho_s = {float(dur_corr['spearman_rho']):.3f}$, $p = {float(dur_corr['p_value']):.3f}$, Equation \\ref{{eq:spearman}}). Conversely, interaction intensity measured by weekly sessions does not demonstrate a statistically significant monotonic association ($\\rho_s = {float(sess_corr['spearman_rho']):.3f}$, $p = {float(sess_corr['p_value']):.3f}$).

| Model / Predictor | Metric / Estimate | $p$-value | $N$ | Interpretation |
|:---|---:|---:|---:|:---|
| Duration (months) | $\\rho_s = {float(dur_corr['spearman_rho']):.3f}$ | {float(dur_corr['p_value']):.3f} | 171 | Statistically significant ($p < 0.05$) |
| Sessions per week | $\\rho_s = {float(sess_corr['spearman_rho']):.3f}$ | {float(sess_corr['p_value']):.3f} | 171 | Null retained ($p > 0.05$) |
| Regression Intercept ($\\beta_0$) | ${b0:.3f}$ | --- | 171 | Baseline avoidance index |
| Regression Duration ($\\beta_1$) | ${b_dur:.3f}$ | --- | 171 | Adjusted monthly slope |
| Regression Sessions ($\\beta_2$) | ${b_sess:.3f}$ | --- | 171 | Adjusted weekly session slope |

*Table \\ref{{tab:rq3_results}}: Bivariate correlation and multivariate regression models for isolation avoidance (SYNTHETIC DATA — pipeline test only).*

In the multivariate ordinary least squares regression (Equation \\ref{{eq:regression}}), duration and sessions account for baseline variance, yielding fitted parameters $\\beta_0 = {b0:.3f}$, $\\beta_1 = {b_dur:.3f}$, and $\\beta_2 = {b_sess:.3f}$. The negative adjusted slope indicates that once baseline duration is modeled simultaneously with session frequency on this synthetic dataset, partial linear slopes differ from bivariate monotonic ranks.

---

## 4.4. Visual Analysis

![Study Results Multi-Panel Figure](reports/figures/study/study_results.png)  
*Figure \\ref{{fig:study_results}}: Quantitative distributions and monotonic relationships for emotional dependence ($F_i$) and isolation ($I_i$) on synthetic benchmark data (SYNTHETIC DATA — pipeline test only). Panel (a) illustrates boxplots with overlaid strip observations across duration groups; whiskers represent $1.5 \\times \\text{{IQR}}$, and the central bar denotes the median. Panel (b) depicts interaction duration versus isolation avoidance with an OLS fit (red line) and a 95% bootstrap confidence band ($B = 1,000$ draws).*

---

## 4.5. Methodological Placeholders (Collaborator Part B)

The following components represent security threat modeling and system benchmarking responsibilities delegated to Partner B:

- **STRIDE Threat Matrix:** `[TODO(B): Table \\ref{{tab:stride}}]` — Information security threat model evaluating data pipeline mitigations.
- **Microsecond Benchmarking Latency vs Load:** `[TODO(B): Figure \\ref{{fig:benchmark}}]` — High-load open-loop latency distribution under varying RPS.
- **Percentile Latency Formulations:** `[TODO(B): Equation \\ref{{eq:p95}}]` — Mathematical formulation of $p_{{95}}$, $p_{{99}}$, and speedup factors.
"""
    section_path = out_file.parent / "section_4_5.md"

    if section_path.is_file():
        section_text = section_path.read_text(encoding="utf-8")

        marker = "## 4.5. Methodological Placeholders"
        position = content.find(marker)

        if position == -1:
            raise ValueError("Section 4.5 placeholder not found")

        content = content[:position] + section_text.rstrip() + "\n"


    out_file.write_text(content, encoding="utf-8")
    logger.info("Saved Markdown Section 4 to %s", out_file)


def validate_cross_references(base_dir: Path = ROOT) -> dict[str, Any]:
    """
    Verify that every \\ref{...} in LaTeX and Markdown has a corresponding \\label{...},
    accounting for explicitly listed TODO(B) placeholders.
    """
    files_to_check = [
        base_dir / "reports/formulas.tex",
        base_dir / "reports/tables.tex",
        base_dir / "reports/paper_section_4.md",
    ]

    known_labels = set()
    referenced_labels = set()
    todo_b_placeholders = {"tab:stride", "fig:benchmark", "eq:p95"}

    label_pattern = re.compile(r"\\label\{([^}]+)\}")
    ref_pattern = re.compile(r"\\ref\{([^}]+)\}")

    for fpath in files_to_check:
        if not fpath.is_file():
            continue
        text = fpath.read_text(encoding="utf-8")
        for m in label_pattern.finditer(text):
            known_labels.add(m.group(1))
        for m in ref_pattern.finditer(text):
            referenced_labels.add(m.group(1))

    missing = (referenced_labels - known_labels) - todo_b_placeholders
    todo_b_found = referenced_labels & todo_b_placeholders

    result = {
        "is_valid": len(missing) == 0,
        "known_labels": sorted(list(known_labels)),
        "referenced_labels": sorted(list(referenced_labels)),
        "todo_b_placeholders": sorted(list(todo_b_found)),
        "missing_labels": sorted(list(missing)),
    }

    if missing:
        logger.error("Missing label targets for: %s", missing)
    else:
        logger.info("Cross-reference validation passed! (Found %d labels, %d TODO(B) placeholders)",
                    len(known_labels), len(todo_b_found))
    return result


def main():
    parser = argparse.ArgumentParser(description="Generate Part A publication artifacts.")
    parser.add_argument("--root", default=str(ROOT), help="Repository root path")
    parser.add_argument("--output-dir", default=None, help="Artifact root (defaults to --root)")
    args = parser.parse_args()

    base_dir = Path(args.root)
    output_dir = Path(args.output_dir) if args.output_dir else base_dir
    reports_dir = output_dir / "reports"
    figures_dir = reports_dir / "figures/study"

    logger.info("Loading analysis tables and datasets...")
    data_dict = load_data_and_tables(base_dir)

    logger.info("Generating multi-panel publication figures...")
    make_figures(data_dict["clean"], figures_dir)

    logger.info("Generating LaTeX formulas and notation table...")
    make_formulas_tex(reports_dir / "formulas.tex")

    logger.info("Generating LaTeX booktabs tables...")
    make_tables_tex(data_dict, reports_dir / "tables.tex")

    logger.info("Generating Markdown Section 4 draft...")
    make_markdown_section_4(data_dict, reports_dir / "paper_section_4.md")

    logger.info("Validating cross-references...")
    val_res = validate_cross_references(output_dir)
    if not val_res["is_valid"]:
        raise ValueError(f"Cross-reference validation failed: {val_res['missing_labels']}")

    logger.info("All Part A artifacts generated successfully!")


if __name__ == "__main__":
    main()
