# 4. Experimental Results & Discussion (SYNTHETIC DATA — pipeline test only)

> **CRITICAL REPRODUCIBILITY NOTICE:**  
> The empirical results, descriptive tables, and inferential tests presented below were computed strictly on the synthetic test dataset (`data/synthetic/main_dataset.csv`, $N = 360$ raw records; $N = 171$ retained post-filtration). This dataset serves **exclusively to validate the computational pipeline, statistical routines, and data contracts**. It must **not** be interpreted as substantive evidence regarding human student psychology or AI companion interaction.

---

## 4.1. Corpus Characteristics and Data Sanitization

A total of $N = 360$ simulated textual submissions across two major companion platforms (Replika: $n = 188$, Character.AI: $n = 172$) were processed through the automated ingestion pipeline `src/load_dataset.py`. Quality filtering criteria, formally defined in `configs/study.yaml` (*ASSUMPTION, confirm: outlier and filtering bounds*), were applied sequentially:

1. **Length filtration:** Submissions with token counts $w < 100$ words ($n = 152$) were discarded to preserve coding reliability.
2. **Extreme text outlier:** A single extreme length submission of $4,500$ words (SYN0032, exceeding the $3,000$-word threshold) was excluded.
3. **Usage intensity bounds:** Records exhibiting non-active usage ($0$ sessions, $n = 3$) or implausible activity (SYN0008 with $70$ sessions/week, exceeding the $21$ session limit) were pruned.
4. **Target population relevance:** Submissions with unverified student status (`student_evidence == "unknown"`, $n = 56$) were removed.

The clean analytical dataset comprises $N = 171$ unique user accounts across three duration strata: novice ($<1$ month, $n = 38$), intermediate ($1$--$3$ months, $n = 57$), and experienced ($>3$ months, $n = 76$). A complete descriptive audit is summarized in Table \ref{tab:corpus_descriptives}.

| Sample Stage / Strata | Records ($N$) | Replika | Character.AI | Total Words | Excluded |
|:---|---:|---:|---:|---:|---:|
| Raw Synthetic Pool | 360 | 188 | 172 | 54,435 | --- |
| ↳ Excluded: $w < 100$ words | 152 | 83 | 69 | --- | 152 |
| ↳ Excluded: $w > 3,000$ words | 1 | 0 | 1 | --- | 1 |
| ↳ Excluded: Sessions $\notin [1, 21]$ | 4 | 2 | 2 | --- | 4 |
| ↳ Excluded: Student unknown | 56 | 28 | 28 | --- | 56 |
| **Final Analysis Dataset** | **171** | **90** | **81** | **33,754** | **189** |
| ↳ Group 1: $<1$ month | 38 | 17 | 21 | 7,593 | --- |
| ↳ Group 2: $1$--$3$ months | 57 | 32 | 25 | 12,175 | --- |
| ↳ Group 3: $>3$ months | 76 | 41 | 35 | 13,986 | --- |

*Table \ref{tab:corpus_descriptives}: Corpus breakdown before and after filtration (SYNTHETIC DATA — pipeline test only).*

---

## 4.2. Research Question 2: Emotional Dependence Markers ($F_i$)

Research Question 2 assesses whether prolonged user tenure is associated with elevated frequencies of emotional dependence markers ($F_i$, normalized per 1,000 words via Equation \ref{eq:f_marker}). Per our methodological configuration (*ASSUMPTION, confirm: dependence composition*), $F_i$ pools markers of attachment, fear of loss, and daily support.

As illustrated in Figure \ref{fig:study_results}(a) and detailed in Table \ref{tab:rq2_results}, median dependence frequency rises monotonically from $10.69$ markers/1k words in the novice group ($G_1$) to $11.70$ in the intermediate group ($G_2$), reaching $20.78$ in the experienced group ($G_3$).

| Duration Group | $n$ | Median (IQR) | $p_{25}$--$p_{75}$ | Mean ± SD |
|:---|---:|---:|---:|---:|
| $<1$ month ($G_1$) | 38 | 10.69 (9.58) | 6.03--15.61 | 11.03 ± 7.81 |
| $1$--$3$ months ($G_2$) | 57 | 11.70 (12.02) | 7.30--19.32 | 13.52 ± 9.36 |
| $>3$ months ($G_3$) | 76 | **20.78** (23.93) | 9.31--33.24 | 24.59 ± 22.77 |

*Table \ref{tab:rq2_results}: Group-level dependence rates and inferential tests (SYNTHETIC DATA — pipeline test only).*

The primary directional contrast between experienced ($>3$ months) and novice ($<1$ month) users confirms a statistically significant divergence under the Mann--Whitney $U$ test ($U = 2079.5$, $p = 6.73e-05$, one-sided, Equation \ref{eq:mwu}). The non-parametric effect size is substantial: Cliff's delta $\delta = 0.440$ with a 95% percentile bootstrap confidence interval of $[0.250, 0.614]$ ($B = 2,000$ resamples, seed $42$, Equation \ref{eq:bootstrap_ci}).

Furthermore, omnibus testing demonstrates significant variation across the three groups (Kruskal--Wallis $H = 19.15$, $p = 6.95e-05$), while the Jonckheere--Terpstra test confirms a directional monotonic trend ($J = 6206.5$, $Z = 4.35$, $p = 6.67e-06$, Equation \ref{eq:jonckheere}).

---

## 4.3. Research Question 3: Isolation–Avoidance Patterns ($I_i$)

Research Question 3 examines whether user interaction duration and weekly engagement intensity exhibit a positive monotonic relationship with the isolation--avoidance index ($I_i$, Equation \ref{eq:i_marker}). Statistical evaluations are reported in Table \ref{tab:rq3_results}.

Bivariate non-parametric correlation indicates a weak, statistically significant positive association between interaction duration (months) and $I_i$ (Spearman $\rho_s = 0.136$, $p = 0.038$, Equation \ref{eq:spearman}). Conversely, interaction intensity measured by weekly sessions does not demonstrate a statistically significant monotonic association ($\rho_s = 0.082$, $p = 0.144$).

| Model / Predictor | Metric / Estimate | $p$-value | $N$ | Interpretation |
|:---|---:|---:|---:|:---|
| Duration (months) | $\rho_s = 0.136$ | 0.038 | 171 | Statistically significant ($p < 0.05$) |
| Sessions per week | $\rho_s = 0.082$ | 0.144 | 171 | Null retained ($p > 0.05$) |
| Regression Intercept ($\beta_0$) | $7.897$ | --- | 171 | Baseline avoidance index |
| Regression Duration ($\beta_1$) | $-0.129$ | --- | 171 | Adjusted monthly slope |
| Regression Sessions ($\beta_2$) | $-0.619$ | --- | 171 | Adjusted weekly session slope |

*Table \ref{tab:rq3_results}: Bivariate correlation and multivariate regression models for isolation avoidance (SYNTHETIC DATA — pipeline test only).*

In the multivariate ordinary least squares regression (Equation \ref{eq:regression}), duration and sessions account for baseline variance, yielding fitted parameters $\beta_0 = 7.897$, $\beta_1 = -0.129$, and $\beta_2 = -0.619$. The negative adjusted slope indicates that once baseline duration is modeled simultaneously with session frequency on this synthetic dataset, partial linear slopes differ from bivariate monotonic ranks.

---

## 4.4. Visual Analysis

![Study Results Multi-Panel Figure](reports/figures/study/study_results.png)  
*Figure \ref{fig:study_results}: Quantitative distributions and monotonic relationships for emotional dependence ($F_i$) and isolation ($I_i$) on synthetic benchmark data (SYNTHETIC DATA — pipeline test only). Panel (a) illustrates boxplots with overlaid strip observations across duration groups; whiskers represent $1.5 \times \text{IQR}$, and the central bar denotes the median. Panel (b) depicts interaction duration versus isolation avoidance with an OLS fit (red line) and a 95% bootstrap confidence band ($B = 1,000$ draws).*

---

## 4.5. Methodological Placeholders (Collaborator Part B)

The following components represent security threat modeling and system benchmarking responsibilities delegated to Partner B:

- **STRIDE Threat Matrix:** `[TODO(B): Table \ref{tab:stride}]` — Information security threat model evaluating data pipeline mitigations.
- **Microsecond Benchmarking Latency vs Load:** `[TODO(B): Figure \ref{fig:benchmark}]` — High-load open-loop latency distribution under varying RPS.
- **Percentile Latency Formulations:** `[TODO(B): Equation \ref{eq:p95}]` — Mathematical formulation of $p_{95}$, $p_{99}$, and speedup factors.
