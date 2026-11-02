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