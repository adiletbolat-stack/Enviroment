# Psychological Markers of Isolation and Emotional Attachment among Students in Long-Term Interaction with AI Companions

[![License: MIT](https://img.shields.io/badge/code-MIT-green.svg)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/data%20%26%20codebook-CC%20BY%204.0-lightgrey.svg)](LICENSE-DATA)
[![Python 3.11.9](https://img.shields.io/badge/python-3.11.9-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/environment-Docker-2496ED.svg)](Dockerfile)
[![Status](https://img.shields.io/badge/status-research%20proposal-orange.svg)](docs/preregistration.md)

**Authors:** Anna Merkulova, Adilet Bolat  
**Affiliation:** Narxoz University, course SCOM3001  
**Contact:** anna.merkulova@narxoz.kz, adilet.bolat@narxoz.kz  
**Repository:** [https://github.com/adiletbolat-stack/Enviroment](https://github.com/adiletbolat-stack/Enviroment)

This repository provides an end-to-end reproducible research environment, statistical pipeline, and codebook for a mixed-methods content analysis investigating user accounts of long-term AI companion interaction (Replika, Character.ai).

---

## 1. Project Overview and Research Aim

Students increasingly interact with AI companions for emotional support. This study investigates whether the **duration of use** and **intensity of interaction** correlate with **markers of emotional dependence** and **social isolation/avoidance** in written accounts.

### Research Design
- **Type:** Mixed-methods observational design (qualitative content analysis via a frozen codebook + non-parametric statistical hypothesis testing).
- **Unit of analysis:** Unique author (markers and word counts pooled per author, frequencies normalized per 1,000 words). *[ASSUMPTION, confirm: 1 row = 1 author due to lack of author_id in synthetic data]*.
- **Random seed:** Fixed to `42` across all scripts, bootstrap resamples, and random operations.

### Research Questions and Hypotheses

| ID | Research Question | Hypotheses | Statistical Tests |
|---|---|---|---|
| **RQ2** | Does the frequency of emotional dependence markers ($F_i$) differ between novice (< 1 month) and experienced (> 3 months) users? | **H₀:** $F_{<1\text{m}} = F_{>3\text{m}}$<br>**H₁:** $F_{>3\text{m}} > F_{<1\text{m}}$<br>*Secondary:* Monotonic trend across < 1m, 1–3m, > 3m. | Mann–Whitney $U$ (one-sided), Cliff's $\delta$ with 95% bootstrap CI, Kruskal–Wallis $H$, Jonckheere–Terpstra trend test. |
| **RQ3** | Is duration of use associated with the isolation–avoidance index ($I_i$)? | **H₀:** $\rho_s \le 0$<br>**H₁:** $\rho_s > 0$ (positive monotonic relationship). | Spearman rank correlation $\rho_s$, OLS regression controlled for weekly intensity (`sessions_per_week`). |

All hypothesis tests use significance level $\alpha = 0.05$. All study parameters and marker formulas are specified in [`configs/study.yaml`](configs/study.yaml).

---

## 2. Repository Structure

```
.
├── Dockerfile                  # CPU-based reproducible image (Python 3.11.9-slim)
├── .dockerignore               # Minimal build context (excludes caches, local OS files)
├── requirements.txt            # Pinned dependency manifest (single source of truth)
├── environment.yml             # Conda environment manifest (ai-companion-markers)
├── configs/
│   ├── study.yaml              # Methodology parameters, column mapping, marker lists, filters
│   └── config.yaml             # Benchmarking harness parameters and SLO thresholds
├── data/
│   ├── sample/                 # TRACKED IN GIT: minimal sample for Quickstart
│   │   ├── sample.csv          # 10-row dataset sample demonstrating filters
│   │   └── test_run.jsonl      # Test benchmark run log (>=100 runs, microsecond timestamps)
│   ├── synthetic/              # Local synthetic test dataset (main_dataset.csv, untracked)
│   ├── interim/                # Cleaned analysis-ready dataset (analysis_set.csv)
│   └── processed/              # Benchmark processed tables and quality summaries
├── notebooks/
│   └── 01_data_analysis.ipynb  # Interactive walkthrough and visualization
├── src/
│   ├── load_dataset.py         # Ingestion, schema validation, per-1,000 recomputation, filtering
│   ├── collect.py              # Backward-compatible wrapper for load_dataset
│   ├── stats.py                # Hypothesis tests (RQ2, RQ3, Cliff's delta CI, Jonckheere-Terpstra)
│   ├── metrics.py              # Text metric helpers and author pooling logic
│   ├── benchmark.py            # Microsecond-accurate I/O-isolated benchmarking harness
│   ├── bench_rq2.py            # RQ2 benchmarking target adapter
│   └── env_metadata.py         # Hardware/OS/package environment metadata collector
├── scripts/
│   └── run_all.sh              # Single-command pipeline execution (tests -> prep -> stats -> bench)
├── tests/
│   ├── test_metrics.py         # Unit tests for text tokenization and author-level pooling
│   └── test_pipeline.py        # Unit tests for per-1,000 formulas, schema checks, and filters
├── results/
│   ├── tables/                 # Output tables with synthetic data warnings
│   └── figures/                # Output publication figures with synthetic watermarks
└── docs/                       # Research documentation, preregistration, methodology drafts
```

---

## 3. System Requirements and Prerequisites

| Resource | Specification |
|---|---|
| **Architecture** | CPU only (x86_64 or ARM64 / Apple Silicon). **No GPU or CUDA required.** |
| **RAM** | Minimum 2 GB (4 GB recommended). |
| **Disk Space** | Approximately 1.2 GB for the container image or Conda environment. |
| **Operating System**| Linux (Ubuntu/Debian), macOS (12+), or Windows 10/11 (WSL2/PowerShell). |
| **Software** | Docker 24+ **or** Python 3.11+ / Conda 23+. |

---

## 4. Quickstart and Reproducibility Guide

The entire analytical pipeline can be reproduced in **2–3 commands** using the bundled sample data in `data/sample/sample.csv`.

### Option A: Using Docker (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/adiletbolat-stack/Enviroment.git && cd Enviroment

# 2. Build the lightweight CPU container
docker build -t ai-companion-markers .

# 3. Execute the full end-to-end pipeline on sample data
docker run --rm -v "$(pwd)":/workspace ai-companion-markers bash scripts/run_all.sh --sample
```

To run interactive JupyterLab (port 8888, non-root user `researcher`):
```bash
docker run --rm -p 8888:8888 -v "$(pwd)":/workspace ai-companion-markers
```
*(Open the generated token link displayed in the terminal).*

> **Docker Environment Status:** The `Dockerfile` has been fully rewritten, pinned to `python:3.11.9-slim-bookworm`, verified against syntax standards, and hardened to run as an unprivileged user without root privileges. In host environments where the Docker Desktop daemon is not active, use Option B below.

### Option B: Local Python Virtual Environment

```bash
# 1. Create and activate a clean Python 3.11 virtual environment
python -m venv .venv
source .venv/bin/activate       # On Windows PowerShell: .\.venv\Scripts\Activate.ps1

# 2. Install exact pinned dependencies
pip install -r requirements.txt

# 3. Run the full pipeline on sample data
bash scripts/run_all.sh --sample
```

*(To process the full dataset, run `bash scripts/run_all.sh` without the `--sample` flag).*

---

## 5. Data and Synthetic Data Notice

### ⚠️ IMPORTANT: Synthetic Dataset Notice
The dataset provided in `data/synthetic/main_dataset.csv` (and excerpted in `data/sample/sample.csv`) consists of **artificially generated synthetic data**:
- Record identifiers begin with `SYN` (`SYN0001`–`SYN0360`).
- Raw column names contain the suffix `_SIMULATED`.
- **Purpose:** This dataset exists **strictly to verify computational pipelines, data contracts, and statistical testing routines**. It contains no genuine human subject data and **must NOT be used to infer psychological conclusions or real-world prevalence**.
- All resulting tables and plots are automatically watermarked with `# SYNTHETIC DATA -- pipeline test only`.

### Data Specifications and Columns
- **Total simulated records:** 360 rows.
- **Marker definitions:**
  - Emotional dependence frequency $F_i = (\sum m_{ij} / \sum w_{ij}) \times 1000$, where $m_{ij}$ includes `attachment_count`, `fear_of_loss_count`, and `daily_support_count`. *[ASSUMPTION, confirm: dependence composition]*.
  - Isolation–avoidance index $I_i = (\sum a_{ij} / \sum w_{ij}) \times 1000$, where $a_{ij}$ is `social_avoidance_count`. *[ASSUMPTION, confirm: isolation index composition]*.
- **Exclusion Filters:**
  - Minimum post length: `word_count >= 100` (removes 152 low-content rows).
  - Outlier bounds: `word_count <= 3000` (excludes SYN0032 with 4,500 words); `1 <= sessions_per_week <= 21` (excludes non-active users and SYN0008 with 70 sessions/week). *[ASSUMPTION, confirm: outlier thresholds]*.
  - Sample relevance: `student_evidence != "unknown"` (excludes 56 non-student rows). *[ASSUMPTION, confirm: exclude unknown evidence]*.
  - Filtered sample size: 171 clean records for statistical testing.

---

## 6. Benchmarking and Engineering Evaluation

As a separate engineering objective distinct from the behavioral study, this repository implements a production-grade benchmarking harness ([`src/benchmark.py`](src/benchmark.py) and [`src/bench_rq2.py`](src/bench_rq2.py)) to audit the latency, jitter, and computational complexity of the RQ2 analysis step.

- **Warm-up phase:** Adaptive warm-up discarding cold-start iterations until coefficient of variation (CV) and window drift stabilize (`warmup_converged=True`, 80 iterations).
- **I/O Isolation:** Zero logging, zero print statements, and disabled garbage collection (`gc.disable()`) during the measured timing loop.
- **Output:** Structured JSONL logging with microsecond UTC timestamps (`%Y-%m-%dT%H:%M:%S.%fZ`), monotonic offset, error isolation, and p50/p95/p99 latency calculations. A verified log of 202 lines (80 warm-up + 120 measured calls, 0 errors, median 3.50 ms, p95 4.88 ms, p99 5.21 ms) is tracked in [`data/sample/test_run.jsonl`](data/sample/test_run.jsonl).

---

## 7. Limitations

1. **Synthetic Input Data:** Findings generated from `data/synthetic/main_dataset.csv` demonstrate pipeline executability only. Real-world validation requires frozen manual coding on authentic user posts.
2. **Author-Level Identifier Constraint:** The synthetic dataset lacks individual `author_id` identifiers; as a necessary simulation assumption, each row is modeled as one author. In naturalistic corpus collection, multiple posts per author must be pooled to prevent pseudoreplication.
3. **Self-Selection Bias:** Online public posts on Reddit/forums reflect individuals motivated to share experiences publicly, introducing inherent self-selection bias toward extreme emotional attachment or distress.
4. **Non-Causal Design:** Content analysis of cross-sectional accounts demonstrates association, not causality. High interaction frequency may either foster emotional dependence or serve as a coping mechanism for pre-existing isolation.

---

## 8. Citation and License

If you use this codebase or methodology, please cite:

```bibtex
@misc{merkulova_bolat_2026_aicompanions,
  author       = {Merkulova, Anna and Bolat, Adilet},
  title        = {Psychological Markers of Isolation and Emotional Attachment among Students
                  in Long-Term Interaction with AI Companions: A Content Analysis},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/adiletbolat-stack/Enviroment},
  note         = {Version 1.0}
}
```

- **Source Code** (`src/`, `scripts/`, `tests/`): [MIT License](LICENSE).
- **Data & Documentation** (`configs/`, `data/sample/`, `docs/`): [CC BY 4.0](LICENSE-DATA).
