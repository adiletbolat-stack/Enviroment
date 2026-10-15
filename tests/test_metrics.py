import pytest

import metrics


def row(author, group, words, markers, pid="p"):
    return {"author_id": author, "post_id": pid, "duration_group": group,
            "n_words": words, "n_dependence_markers": markers}


def test_count_words_edge_cases():
    assert metrics.count_words("") == 0
    assert metrics.count_words("   \n\t ") == 0
    assert metrics.count_words(None) is None
    assert metrics.count_words(float("nan")) is None
    assert metrics.count_words("[deleted]") == 0
    assert metrics.count_words("[removed]") == 0
    assert metrics.count_words("I can't stop") == 3                 # apostrophe stays inside the word
    assert metrics.count_words("one\u00a0two\u200bthree") == 2      # NBSP splits, zero-width does not
    assert metrics.count_words("😢😢 word 😢") == 1                  # emoji are not words
    assert metrics.count_words("a" * 1_000_000) == 1                # huge single token


@pytest.mark.parametrize("words, included", [(99, False), (100, True), (101, True)])
def test_min_words_boundary(words, included):
    res = metrics.author_level([row("a1", "<1m", words, 1)], min_words=100)
    assert (len(res["authors"]) == 1) is included


def test_pooling_not_averaging():
    rows = [row("a1", "<1m", 100, 10, "p1"), row("a1", "<1m", 900, 0, "p2")]   # 10 / 1000 words = 10.0 (a mean of per-post rates would give 50)
    assert metrics.author_level(rows)["authors"]["a1"]["freq_per_1000"] == pytest.approx(10.0)


def test_conservation_law_and_reasons():
    rows = [row("a1", "<1m", 200, 2), row("a2", ">3m", 50, 1), row("a3", ">3m", None, 1),
            row("a4", ">3m", 200, None), row("a5", ">3m", 200, -1),
            row("a6", "<1m", 200, 1, "p1"), row("a6", ">3m", 200, 1, "p2")]       # a6: group conflict
    res = metrics.author_level(rows)
    assert res["n_posts_input"] == 7
    assert res["n_posts_input"] == res["n_posts_used"] + sum(res["excluded_posts"].values())
    assert res["excluded_posts"]["below_min_words"] == 1
    assert res["excluded_posts"]["missing_value"] == 2
    assert res["excluded_posts"]["invalid_marker_count"] == 1
    assert res["excluded_posts"]["author_group_conflict"] == 2


def test_cliffs_delta_known_values():
    assert metrics.cliffs_delta([5, 6, 7], [1, 2, 3]) == 1.0
    assert metrics.cliffs_delta([1, 2, 3], [5, 6, 7]) == -1.0
    assert metrics.cliffs_delta([1, 2, 3], [1, 2, 3]) == 0.0


def test_rq2_insufficient_data_is_reported_not_crashed():
    assert metrics.rq2_compare([row("a1", "<1m", 200, 2)])["status"] == "insufficient_data"


def test_rq2_detects_a_real_difference():
    rows = []
    for i in range(30):
        rows.append(row(f"n{i}", "<1m", 200, 1 + i % 2, f"pn{i}"))
        rows.append(row(f"e{i}", ">3m", 200, 5 + i % 3, f"pe{i}"))
    res = metrics.rq2_compare(rows)
    assert res["status"] == "ok" and res["p_value"] < 0.001 and res["cliffs_delta"] > 0.9
