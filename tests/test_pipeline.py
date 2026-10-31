"""
tests/test_pipeline.py — Unit tests for data loading, schema validation,
per-1,000 word frequency calculation, and quality filters.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from load_dataset import (
    apply_quality_filters,
    compute_derived_metrics,
    strip_column_suffixes,
    validate_schema_and_types,
)
from stats import cliffs_delta, cliffs_delta_bootstrap_ci, jonckheere_terpstra


@pytest.fixture
def base_config():
    return {
        "columns": {
            "id": "record_id",
            "platform": "platform",
            "student_evidence": "student_evidence",
            "duration_group": "duration_group",
            "duration_months": "duration_months",
            "sessions_per_week": "sessions_per_week",
            "word_count": "word_count",
        },
        "markers": {
            "dependence": [
                "attachment_count",
                "fear_of_loss_count",
                "daily_support_count",
            ],
            "isolation": [
                "social_avoidance_count",
            ],
            "other": [
                "comfort_count",
                "loneliness_count",
                "social_confidence_count",
            ],
        },
        "filters": {
            "min_words": 100,
            "max_words": 3000,
            "min_sessions_per_week": 1,
            "max_sessions_per_week": 21,
            "excluded_student_evidence": ["unknown"],
        },
    }


def test_strip_column_suffixes():
    df = pd.DataFrame({"record_id": ["S1"], "word_count_SIMULATED": [120]})
    cleaned = strip_column_suffixes(df, "_SIMULATED")
    assert "word_count" in cleaned.columns
    assert "word_count_SIMULATED" not in cleaned.columns


def test_per_1000_words_formula(base_config):
    # Case 1: 200 words, 2 attachment, 1 fear_of_loss, 1 daily_support -> 4 markers total -> 20.0 per 1000
    df = pd.DataFrame([{
        "record_id": "SYN1",
        "word_count": 200,
        "attachment_count": 2,
        "fear_of_loss_count": 1,
        "daily_support_count": 1,
        "social_avoidance_count": 3,  # 3 / 200 * 1000 = 15.0 per 1000
    }])
    res = compute_derived_metrics(df, base_config)
    assert res["dependence_count"].iloc[0] == 4
    assert res["dependence_per_1000"].iloc[0] == pytest.approx(20.0)
    assert res["isolation_count"].iloc[0] == 3
    assert res["isolation_per_1000"].iloc[0] == pytest.approx(15.0)


def test_per_1000_words_zero_markers(base_config):
    df = pd.DataFrame([{
        "record_id": "SYN2",
        "word_count": 500,
        "attachment_count": 0,
        "fear_of_loss_count": 0,
        "daily_support_count": 0,
        "social_avoidance_count": 0,
    }])
    res = compute_derived_metrics(df, base_config)
    assert res["dependence_per_1000"].iloc[0] == pytest.approx(0.0)
    assert res["isolation_per_1000"].iloc[0] == pytest.approx(0.0)


def test_per_1000_words_zero_words_no_crash(base_config):
    df = pd.DataFrame([{
        "record_id": "SYN3",
        "word_count": 0,
        "attachment_count": 2,
        "fear_of_loss_count": 0,
        "daily_support_count": 0,
        "social_avoidance_count": 1,
    }])
    res = compute_derived_metrics(df, base_config)
    assert math.isnan(res["dependence_per_1000"].iloc[0])
    assert math.isnan(res["isolation_per_1000"].iloc[0])


def test_schema_validation_valid(base_config):
    df = pd.DataFrame([{
        "record_id": "SYN1",
        "platform": "Replika",
        "student_evidence": "explicit",
        "duration_group": "<1 month",
        "sessions_per_week": 5,
        "word_count": 150,
        "attachment_count": 1,
        "fear_of_loss_count": 0,
        "daily_support_count": 0,
        "social_avoidance_count": 1,
    }])
    is_valid, errors = validate_schema_and_types(df, base_config)
    assert is_valid is True
    assert len(errors) == 0


def test_schema_validation_missing_columns(base_config):
    df = pd.DataFrame([{"record_id": "SYN1"}])
    is_valid, errors = validate_schema_and_types(df, base_config)
    assert is_valid is False
    assert any("word_count" in err for err in errors)


def test_schema_validation_negative_counts(base_config):
    df = pd.DataFrame([{
        "record_id": "SYN1",
        "platform": "Replika",
        "student_evidence": "explicit",
        "duration_group": "<1 month",
        "sessions_per_week": 5,
        "word_count": -50,
        "attachment_count": 1,
        "fear_of_loss_count": 0,
        "daily_support_count": 0,
        "social_avoidance_count": 1,
    }])
    is_valid, errors = validate_schema_and_types(df, base_config)
    assert is_valid is False
    assert any("negative" in err for err in errors)


def test_quality_filters_bounds(base_config):
    rows = [
        # Valid row
        {"record_id": "R1", "word_count": 150, "sessions_per_week": 5, "student_evidence": "explicit"},
        # Below min_words (<100)
        {"record_id": "R2", "word_count": 99, "sessions_per_week": 5, "student_evidence": "explicit"},
        # Above max_words (>3000)
        {"record_id": "R3", "word_count": 4500, "sessions_per_week": 5, "student_evidence": "explicit"},
        # Below min_sessions (<1)
        {"record_id": "R4", "word_count": 200, "sessions_per_week": 0, "student_evidence": "explicit"},
        # Above max_sessions (>21)
        {"record_id": "R5", "word_count": 200, "sessions_per_week": 70, "student_evidence": "explicit"},
        # Unknown student evidence
        {"record_id": "R6", "word_count": 200, "sessions_per_week": 5, "student_evidence": "unknown"},
    ]
    df = pd.DataFrame(rows)
    filtered, summary = apply_quality_filters(df, base_config)
    assert summary["n_initial"] == 6
    assert summary["n_final"] == 1
    assert filtered["record_id"].tolist() == ["R1"]
    assert summary["filters"]["below_min_words"] == 1
    assert summary["filters"]["above_max_words"] == 1
    assert summary["filters"]["below_min_sessions"] == 1
    assert summary["filters"]["above_max_sessions"] == 1
    assert summary["filters"]["excluded_student_evidence"] == 1


def test_cliffs_delta_and_bootstrap_ci():
    x = [10, 11, 12, 13]
    y = [1, 2, 3, 4]
    d = cliffs_delta(x, y)
    assert d == 1.0
    ci_low, ci_high = cliffs_delta_bootstrap_ci(x, y, reps=500, seed=42)
    assert ci_low == 1.0
    assert ci_high == 1.0


def test_jonckheere_terpstra_trend():
    g1 = [1, 2, 2]
    g2 = [3, 4, 4]
    g3 = [5, 6, 7]
    j, z, p = jonckheere_terpstra(g1, g2, g3)
    assert z > 0
    assert p < 0.05
