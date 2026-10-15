"""
Metrics for the content analysis of AI-companion posts.

Data flow: raw post texts stay on the local machine (never in git). For every post a coded table row exists:
    author_id, post_id, duration_group, n_words (counted automatically), n_dependence_markers (coded BY HAND)
Everything below works on that table, so no post text is needed to reproduce the analysis.
"""
import math
import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import mannwhitneyu

_TOKEN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)
_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\ufeff]")
PLACEHOLDERS = {"[deleted]", "[removed]"}


def count_words(text):
    """Words in a post. None for missing text; 0 for platform placeholders and empty text."""
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return None
    text = unicodedata.normalize("NFKC", str(text)).replace("\x00", " ")
    text = _ZERO_WIDTH.sub("", text)
    if text.strip().lower() in PLACEHOLDERS:
        return 0
    return len(_TOKEN.findall(text))


def _missing(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


def author_level(rows, min_words=100, marker_field="n_dependence_markers"):
    """
    Pool markers and words per author, THEN normalise per 1,000 words (avoids pseudoreplication).
    Nothing is dropped silently: every excluded post gets a reason, and the counts must add up.
    """
    excluded, per, groups = Counter(), defaultdict(lambda: {"markers": 0, "words": 0, "posts": 0}), defaultdict(set)
    for r in rows:
        w, m = r.get("n_words"), r.get(marker_field)
        if _missing(w) or _missing(m):
            excluded["missing_value"] += 1
        elif m < 0:
            excluded["invalid_marker_count"] += 1
        elif w < min_words:
            excluded["below_min_words"] += 1
        else:
            a = per[r["author_id"]]
            a["markers"] += m
            a["words"] += w
            a["posts"] += 1
            groups[r["author_id"]].add(r["duration_group"])

    authors, conflict_authors = {}, 0
    for aid, a in per.items():
        if len(groups[aid]) > 1:                      # duration group changed between posts: ambiguous, excluded
            excluded["author_group_conflict"] += a["posts"]
            conflict_authors += 1
            continue
        authors[aid] = {**a, "group": next(iter(groups[aid])), "freq_per_1000": a["markers"] / a["words"] * 1000}

    used = sum(a["posts"] for a in authors.values())
    n_in = len(rows)
    assert n_in == used + sum(excluded.values()), "conservation law violated: posts were lost silently"
    return {"authors": authors, "excluded_posts": dict(excluded), "excluded_authors_group_conflict": conflict_authors,
            "n_posts_input": n_in, "n_posts_used": used}


def cliffs_delta(x, y):
    """delta = (#(x>y) - #(x<y)) / (n1*n2); +1 means every x is larger than every y."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    gt = (x[:, None] > y[None, :]).sum()
    lt = (x[:, None] < y[None, :]).sum()
    return float((gt - lt) / (len(x) * len(y)))


def rq2_compare(rows, min_words=100, novice="<1m", experienced=">3m"):
    """H1 (directional): marker frequency of experienced users > novices. Mann-Whitney U, one-sided."""
    res = author_level(rows, min_words)
    by_group = defaultdict(list)
    for a in res["authors"].values():
        by_group[a["group"]].append(a["freq_per_1000"])
    nov, exp_ = np.array(by_group[novice]), np.array(by_group[experienced])
    out = {"n_posts_input": res["n_posts_input"], "n_posts_used": res["n_posts_used"],
           "excluded_posts": res["excluded_posts"], "n_authors": {g: len(v) for g, v in by_group.items()}}
    if len(nov) < 2 or len(exp_) < 2:
        return {**out, "status": "insufficient_data"}
    u = mannwhitneyu(exp_, nov, alternative="greater")
    return {**out, "status": "ok", "median_novice": float(np.median(nov)), "median_experienced": float(np.median(exp_)),
            "U": float(u.statistic), "p_value": float(u.pvalue), "cliffs_delta": cliffs_delta(exp_, nov)}
