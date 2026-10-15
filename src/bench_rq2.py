#!/usr/bin/env python3
"""
Benchmark of the RQ2 analysis step (author-level aggregation + Mann-Whitney + Cliff's delta)
on a SYNTHETIC coded table. It plugs the project function into the generic harness in benchmark.py.

    python src/bench_rq2.py --n-runs 1000 --seed 42 --out-dir logs
"""
import sys

import numpy as np

import benchmark as b
import metrics

GROUP_RATE = {"<1m": 4.0, "1-3m": 6.0, ">3m": 8.0}      # true markers per 1,000 words in the simulation


def make_input(seed, n_authors=400):
    """Synthetic coded table. Heavy-tailed posts per author, lognormal lengths, ~3% edge-case rows."""
    rng = np.random.default_rng(seed)
    rows, pid = [], 0
    for a in range(n_authors):
        group = str(rng.choice(list(GROUP_RATE)))
        for _ in range(int(min(1 + rng.pareto(1.2), 200))):
            words = int(rng.lognormal(np.log(150), 0.7))
            markers = int(rng.poisson(GROUP_RATE[group] * words / 1000))
            rows.append({"author_id": f"a{a}", "post_id": f"p{pid}", "duration_group": group,
                         "n_words": words, "n_dependence_markers": markers})
            pid += 1
    for r in rows:                                          # inject edge cases with known counts
        u = rng.random()
        if u < 0.01:
            r["n_words"] = None
        elif u < 0.02:
            r["n_words"] = 0
        elif u < 0.03:
            r["n_dependence_markers"] = None
    return rows


def target(rows):
    return metrics.rq2_compare(rows, min_words=100)


if __name__ == "__main__":
    b.make_input, b.target = make_input, target
    if "--experiment" not in sys.argv:
        sys.argv += ["--experiment", "rq2_pipeline"]
    b.main()
