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

4.5. Security Threat Modeling and Performance Benchmarking

4.5.1. STRIDE Threat Analysis

The research pipeline processes structured records, statistical outputs, benchmark logs, and generated research artifacts. The STRIDE framework was used to identify potential information security threats and appropriate mitigation measures.

STRIDE Category

Potential Threat

Proposed Mitigation

Spoofing

An unauthorized contributor submits data under another identity.

Repository authentication, restricted access, and verified contribution history.

Tampering

Raw datasets or statistical results are modified without authorization.

File integrity checks, Git version control, and reproducible processing scripts.

Repudiation

A contributor denies modifying datasets or analysis code.

Commit history, timestamps, and experiment metadata.

Information Disclosure

Sensitive conversation records or participant information are exposed.

Data minimization, removal of direct identifiers, restricted access, and exclusion of private raw data from public repositories.

Denial of Service

Excessive requests overload the processing pipeline.

Resource limits, timeouts, load testing, and monitoring.

Elevation of Privilege

An unauthorized process gains access to protected project files.

Least-privilege access, isolated environments, and dependency management.

Table: STRIDE threat matrix for the research data-processing pipeline. The mitigations are proposed controls and are not all independently verified as implemented.

4.5.2. Latency vs. Load

A separate load-testing experiment was conducted on October 22, 2026, using eight configured request rates: 10, 20, 30, 40, 50, 75, 100, and 150 requests per second (RPS).

The experiment recorded 7,125 requests, including 7,123 successful requests and two errors. Both errors occurred at the 150 RPS level, corresponding to an overall observed error rate of approximately 0.028%.

The benchmark configuration specified 15 seconds per load level, 64 workers, a timeout of 10 seconds, and a random seed of 42. Measurements were collected on Windows 10 with an Intel Core i5-10300H processor and Python 3.13.

Latency analysis excludes unsuccessful requests while retaining valid high-latency observations. The resulting figure presents median (p50), p95, and p99 latency against configured request rate, with percentile bootstrap confidence intervals.



Figure: Latency percentiles versus configured request rate. Shaded regions indicate 95% percentile-bootstrap confidence intervals.

The latency results indicate a substantial increase in response times at higher offered loads. In particular, the previously calculated median latency increased from approximately 5.54 ms at 40 RPS to 177.82 ms at 50 RPS, suggesting possible resource saturation under the tested configuration.

These findings describe the tested implementation and environment only. Configured request rate should not be interpreted as independently verified achieved throughput. Additionally, the bootstrap intervals assume independent observations and may not fully capture temporal autocorrelation.

4.5.3. Latency Distribution

A separate benchmark dataset containing 120 successful measurements was used to examine latency variability and outliers.



Figure: Distribution of 120 successful latency measurements, including observations identified using the 1.5 × IQR rule.

The median latency was 3.185 ms, with the first and third quartiles equal to 2.633 ms and 3.912 ms, respectively. Two observations were identified as potential outliers using the 1.5 × IQR criterion. These observations were retained rather than removed.

The distribution benchmark and the load-testing benchmark represent separate experiments and should not be combined into a single statistical sample.


4.5.4. Percentile Latency and Speedup

For a sample of successful latency measurements $t_1,\ldots,t_n$, percentile latency is defined using the empirical quantile function:

$$
p_q = Q_q(t_1,\ldots,t_n), \qquad q \in {0.50,0.95,0.99}.
$$

Here, $p_{95}$ denotes the latency threshold at or below which approximately 95% of observed successful requests fall, while $p_{99}$ describes the corresponding 99th percentile.

When two comparable system implementations are benchmarked under equivalent conditions, speedup may be calculated as:

$$
S = \frac{T_{\mathrm{baseline}}}{T_{\mathrm{optimized}}}.
$$

No speedup factor is reported in this study because a matched baseline-versus-optimized comparison was not established.
4.5.5. Limitations

The security threat matrix represents a structured risk assessment rather than evidence that all proposed controls have been deployed. The benchmarking results were obtained on one local hardware and software configuration and may not generalize to other environments.

The research-question analyses use synthetic data exclusively and are intended to demonstrate reproducibility and validate the analysis pipeline. Their statistical associations must not be interpreted as empirical findings about real students or AI-companion users.
