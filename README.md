# Enviroment
# Psychological Markers of Isolation and Attachment among Students in Long-Term Interaction with AI Companions

[![License: MIT](https://img.shields.io/badge/code-MIT-green.svg)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/data%20%26%20codebook-CC%20BY%204.0-lightgrey.svg)](LICENSE-DATA)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/environment-Docker-2496ED.svg)](Dockerfile)
[![Status](https://img.shields.io/badge/status-research%20proposal-orange.svg)](docs/preregistration.md)

**Authors:** Anna Merkulova, Adilet Bolat
**Affiliation:** Narxoz University, course SCOM3001
**Contact:** anna.merkulova@narxoz.kz, adilet.bolat@narxoz.kz, https://github.com/adiletbolat-stack/Enviroment

This repository contains the code, codebook, coded data and analysis notebooks for a mixed-methods content analysis of public user posts about long-term use of AI companions (Replika, Character.ai).

---

## 1. Project Overview and Research Aim

Students increasingly use AI companions for emotional support. This project studies whether the **duration and intensity of use** are associated with **markers of emotional dependence** and **isolation/avoidance** in students' own written accounts.

**Design.** Cross-sectional, non-experimental content analysis of public posts (Reddit and thematic communities). Texts are coded manually with a frozen codebook; frequencies are normalized per 1,000 words and aggregated at the **author level**.

### Research questions and hypotheses

| ID | Research question | Hypotheses |
|---|---|---|
| **RQ2** | Does the frequency of emotional dependence markers differ between users with different durations of AI companion use? | **H₀:** no difference between < 1 month and > 3 months. **H₁:** users with > 3 months show a significantly higher marker frequency. Secondary: monotonic trend across < 1, 1–3 and > 3 months. |
| **RQ3** | Are duration and intensity of use associated with isolation–avoidance patterns? | **H₀:** ρ = 0. **H₁:** ρ > 0 (positive monotonic association). |

All tests use α = 0.05. Hypotheses, tests and α are fixed in [`docs/preregistration.md`](docs/preregistration.md) before data inspection.

---

## 2. Repository Structure

```
.
├── README.md
├── LICENSE                 # MIT (code)
├── LICENSE-DATA            # CC BY 4.0 (codebook and coded data)
├── CITATION.cff
├── Dockerfile
├── environment.yml         # fallback: Conda environment
├── requirements.txt        # pinned Python dependencies
├── .devcontainer/          # open the project in a container from VS Code
├── configs/study.yaml      # alpha, random seed, minimum post length, duration groups
├── codebook/               # frozen codebook (v1.0) and change log
├── data/
│   ├── raw/                # NOT in git: local Reddit export
│   ├── interim/            # NOT in git: cleaned texts
│   ├── ids/post_ids.csv    # post IDs for re-collection
│   └── coded/              # manual codes without post texts; author-level table
├── notebooks/              # 01 collection log, 02 preprocessing, 03 kappa, 04 RQ2, 05 RQ3
├── src/                    # collect.py, preprocess.py, metrics.py, stats.py
├── scripts/run_all.sh      # full analysis in one command
├── tests/                  # unit tests for metric formulas
├── results/                # tables/ and figures/
└── docs/                   # preregistration, data management, limitations
```

---

## 3. System Requirements and Prerequisites

| Item | Requirement |
|---|---|
| OS | Windows 10/11, macOS or Linux |
| Hardware | CPU only; 4 GB RAM; about 2 GB free disk space. **No GPU or CUDA needed** |
| Software | Docker 24+ (recommended) **or** Conda 23+ |
| Editor (optional) | VS Code with the *Dev Containers* and *Jupyter* extensions |
| Python | 3.11 (fixed inside the container) |
| Network | Needed only for re-collecting data from Reddit (requires your own API credentials) |

---

## 4. Quickstart and Reproducibility Guide

The analysis can be reproduced **without the raw post texts**, using the coded data in `data/coded/`.

```bash
# 1. Get the code
git clone https://github.com/<username>/ai-companion-markers.git && cd ai-companion-markers

# 2. Build the environment (fixes OS, Python and library versions)
docker build -t ai-companion-markers .

# 3. Reproduce all results (tables and figures appear in results/)
docker run --rm -v "$(pwd)":/workspace ai-companion-markers bash scripts/run_all.sh
```

**Interactive work in Jupyter Lab** (port 8888):

```bash
docker run --rm -p 8888:8888 -v "$(pwd)":/workspace ai-companion-markers
```

Open the printed `http://127.0.0.1:8888/...` link, then run notebooks `03` to `05` in order.

**Without Docker (Conda fallback):**

```bash
conda env create -f environment.yml && conda activate ai-companion-markers
bash scripts/run_all.sh
```

**Randomness.** The random seed (default `42`) and all analysis parameters are set in `configs/study.yaml` and used by every script, including the bootstrap and the random subsample for the second coder.

**Re-collecting the data (optional).** Put your Reddit API credentials in a local `.env` file (never committed), then run `notebooks/01_collection_log.ipynb`. It reads `data/ids/post_ids.csv` and downloads the same posts. Deleted posts cannot be recovered; their number is logged.

---

## 5. Expected Results

Running `scripts/run_all.sh` regenerates the tables below in `results/tables/`. Values are filled in after coding is complete.

| Metric | Research question | Output file | Value |
|---|---|---|---|
| Emotional dependence frequency per 1,000 words, by group (median, IQR) | RQ2 | `rq2_group_summary.csv` | *to be filled* |
| Mann–Whitney U, p-value (< 1 vs > 3 months) | RQ2 | `rq2_tests.csv` | *to be filled* |
| Effect size, Cliff's δ with 95% bootstrap CI | RQ2 | `rq2_tests.csv` | *to be filled* |
| Trend test (Jonckheere–Terpstra), p-value | RQ2 | `rq2_tests.csv` | *to be filled* |
| Spearman's ρ, duration and isolation–avoidance index | RQ3 | `rq3_correlation.csv` | *to be filled* |
| Cohen's κ (second coder, 10–20% subsample) | Validity | `kappa.csv` | *to be filled* (threshold: κ ≥ 0.6) |
| Authors and posts per group, total words | Corpus | `corpus_descriptives.csv` | *to be filled* |

---

## 6. Data and Ethics

- Only **public** posts are analyzed. Raw texts are **not** redistributed (privacy of authors and platform terms).
- The repository contains post IDs, manual codes and derived counts only. Usernames are replaced with anonymous author IDs.
- No direct quotes that could identify an author are published.
- See [`docs/data_management.md`](docs/data_management.md) and [`docs/limitations.md`](docs/limitations.md).

---

## 7. Citation and License

If you use this work, please cite:

```bibtex
@misc{merkulova_bolat_2026_aicompanions,
  author       = {Merkulova, Anna and Bolat, Adilet},
  title        = {Psychological Markers of Isolation and Attachment among Students
                  in Long-Term Interaction with AI Companions: A Content Analysis},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/adiletbolat-stack/Enviroment},
  note         = {Version 1.0}
}
```

**Licenses.**
- Source code (`src/`, `scripts/`, `notebooks/`, `tests/`): [MIT License](LICENSE).
- Codebook and coded data (`codebook/`, `data/coded/`, `results/`): [CC BY 4.0](LICENSE-DATA).
