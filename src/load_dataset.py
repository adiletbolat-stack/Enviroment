#!/usr/bin/env python3
"""
src/load_dataset.py — Ingestion, schema validation, and preprocessing pipeline.

Replaces Reddit API collection for synthetic data workflow:
1. Reads CSV input (sample or full dataset).
2. Strips column suffix (e.g. '_SIMULATED').
3. Validates required schema and data types with explicit error messages.
4. Recalculates per-1,000 word frequencies from raw counts (never trusts stored values):
   - F_i = (sum m_ij / sum w_ij) * 1000  (emotional dependence markers)
   - I_i = (sum a_ij / sum w_ij) * 1000  (isolation-avoidance patterns)
5. Applies quality filters (min_words, max_words, session limits, student evidence).
6. Logs filtered rows per criterion.
7. Saves clean dataset to data/interim/analysis_set.csv.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("load_dataset")


def load_yaml_config(config_path: Path | str) -> dict[str, Any]:
    """Load YAML study configuration."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def strip_column_suffixes(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Strip simulator suffix from all column names."""
    if not suffix:
        return df.copy()
    rename_map = {col: col.replace(suffix, "") for col in df.columns if col.endswith(suffix)}
    return df.rename(columns=rename_map)


def validate_schema_and_types(
    df: pd.DataFrame,
    cfg: dict[str, Any]
) -> tuple[bool, list[str]]:
    """
    Validate that required columns exist and have acceptable data types.
    Returns (is_valid, list_of_errors).
    """
    errors: list[str] = []
    cols = cfg.get("columns", {})
    markers = cfg.get("markers", {})

    # 1. Check required structural columns
    required_cols = [
        cols.get("id", "record_id"),
        cols.get("platform", "platform"),
        cols.get("student_evidence", "student_evidence"),
        cols.get("duration_group", "duration_group"),
        cols.get("sessions_per_week", "sessions_per_week"),
        cols.get("word_count", "word_count"),
    ]
    for col in required_cols:
        if col not in df.columns:
            errors.append(f"Missing required column: '{col}'")

    # 2. Check marker columns
    dep_cols = markers.get("dependence", [])
    iso_cols = markers.get("isolation", [])
    if not dep_cols:
        errors.append("No dependence markers specified in config.")
    for m in dep_cols:
        if m not in df.columns:
            errors.append(f"Missing dependence marker column: '{m}'")
    for m in iso_cols:
        if m not in df.columns:
            errors.append(f"Missing isolation marker column: '{m}'")

    if errors:
        return False, errors

    # 3. Validate types and value ranges
    word_col = cols.get("word_count", "word_count")
    if not pd.api.types.is_numeric_dtype(df[word_col]):
        errors.append(f"Column '{word_col}' must be numeric, got {df[word_col].dtype}")
    elif (df[word_col] < 0).any():
        errors.append(f"Column '{word_col}' contains negative values")

    sess_col = cols.get("sessions_per_week", "sessions_per_week")
    if not pd.api.types.is_numeric_dtype(df[sess_col]):
        errors.append(f"Column '{sess_col}' must be numeric, got {df[sess_col].dtype}")

    for m in dep_cols + iso_cols:
        if not pd.api.types.is_numeric_dtype(df[m]):
            errors.append(f"Marker column '{m}' must be numeric, got {df[m].dtype}")
        elif (df[m] < 0).any():
            errors.append(f"Marker column '{m}' contains negative counts")

    return len(errors) == 0, errors


def compute_derived_metrics(
    df: pd.DataFrame,
    cfg: dict[str, Any]
) -> pd.DataFrame:
    """
    Recalculate per-1,000 word metrics from raw counts:
    - dependence_markers_count = sum(m_ij)
    - dependence_per_1000 = (sum(m_ij) / word_count) * 1000
    - isolation_markers_count = sum(a_ij)
    - isolation_per_1000 = (sum(a_ij) / word_count) * 1000
    """
    out = df.copy()
    cols = cfg.get("columns", {})
    markers = cfg.get("markers", {})
    word_col = cols.get("word_count", "word_count")

    dep_cols = markers.get("dependence", [])
    iso_cols = markers.get("isolation", [])

    # Total marker counts
    out["dependence_count"] = out[dep_cols].sum(axis=1) if dep_cols else 0
    out["isolation_count"] = out[iso_cols].sum(axis=1) if iso_cols else 0

    # Avoid division by zero or NaN: where word_count > 0, compute rate, else NaN
    valid_words = out[word_col] > 0
    out["dependence_per_1000"] = float("nan")
    out.loc[valid_words, "dependence_per_1000"] = (
        out.loc[valid_words, "dependence_count"] / out.loc[valid_words, word_col]
    ) * 1000.0

    out["isolation_per_1000"] = float("nan")
    out.loc[valid_words, "isolation_per_1000"] = (
        out.loc[valid_words, "isolation_count"] / out.loc[valid_words, word_col]
    ) * 1000.0

    return out


def apply_quality_filters(
    df: pd.DataFrame,
    cfg: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Apply methodological exclusion filters with detailed logging.
    Returns (filtered_dataframe, filter_summary_statistics).
    """
    cols = cfg.get("columns", {})
    filters = cfg.get("filters", {})

    word_col = cols.get("word_count", "word_count")
    sess_col = cols.get("sessions_per_week", "sessions_per_week")
    stud_col = cols.get("student_evidence", "student_evidence")

    min_words = filters.get("min_words", 100)
    max_words = filters.get("max_words", 3000)
    min_sessions = filters.get("min_sessions_per_week", 1)
    max_sessions = filters.get("max_sessions_per_week", 21)
    excluded_evidence = set(filters.get("excluded_student_evidence", ["unknown"]))

    n_initial = len(df)
    drop_log: dict[str, int] = {}

    # 1. Word count filter
    mask_min_words = df[word_col] >= min_words
    drop_min_words = int((~mask_min_words).sum())
    drop_log["below_min_words"] = drop_min_words
    logger.info("Filter min_words >= %d: removed %d rows", min_words, drop_min_words)

    # 2. Extreme word count outlier
    mask_max_words = df[word_col] <= max_words
    drop_max_words = int((~mask_max_words).sum())
    drop_log["above_max_words"] = drop_max_words
    logger.info("Filter max_words <= %d: removed %d rows", max_words, drop_max_words)

    # 3. Sessions per week limits
    mask_min_sess = df[sess_col] >= min_sessions
    drop_min_sess = int((~mask_min_sess).sum())
    drop_log["below_min_sessions"] = drop_min_sess
    logger.info("Filter sessions >= %d: removed %d rows", min_sessions, drop_min_sess)

    mask_max_sess = df[sess_col] <= max_sessions
    drop_max_sess = int((~mask_max_sess).sum())
    drop_log["above_max_sessions"] = drop_max_sess
    logger.info("Filter sessions <= %d: removed %d rows", max_sessions, drop_max_sess)

    # 4. Student evidence exclusion
    mask_evidence = ~df[stud_col].isin(excluded_evidence)
    drop_evidence = int((~mask_evidence).sum())
    drop_log["excluded_student_evidence"] = drop_evidence
    logger.info("Filter student_evidence not in %s: removed %d rows", list(excluded_evidence), drop_evidence)

    # Combined pass mask
    combined_mask = (
        mask_min_words & mask_max_words & mask_min_sess & mask_max_sess & mask_evidence
    )
    df_clean = df.loc[combined_mask].copy()
    n_final = len(df_clean)

    summary = {
        "n_initial": n_initial,
        "n_final": n_final,
        "n_excluded_total": n_initial - n_final,
        "filters": drop_log,
    }
    logger.info(
        "Quality filtering complete: %d initial rows -> %d retained (%d excluded total)",
        n_initial,
        n_final,
        n_initial - n_final,
    )
    return df_clean, summary


def process_dataset(
    input_path: Path | str,
    output_path: Path | str,
    config_path: Path | str = "configs/study.yaml"
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Complete dataset loading, validation, metric recomputation, and filtering."""
    cfg = load_yaml_config(config_path)
    in_file = Path(input_path)
    out_file = Path(output_path)

    if not in_file.is_file():
        raise FileNotFoundError(f"Dataset file not found: {in_file}")

    logger.info("Loading dataset from %s", in_file)
    raw_df = pd.read_csv(in_file)

    suffix = cfg.get("input", {}).get("strip_suffix", "_SIMULATED")
    df_clean_cols = strip_column_suffixes(raw_df, suffix)

    is_valid, errors = validate_schema_and_types(df_clean_cols, cfg)
    if not is_valid:
        error_msg = "\n  - " + "\n  - ".join(errors)
        raise ValueError(f"Schema validation failed for {in_file}:{error_msg}")

    df_metrics = compute_derived_metrics(df_clean_cols, cfg)
    df_filtered, filter_summary = apply_quality_filters(df_metrics, cfg)

    # Add pipeline metadata note
    if cfg.get("input", {}).get("synthetic", True):
        df_filtered.attrs["note"] = "SYNTHETIC DATA -- pipeline test only"

    out_file.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_csv(out_file, index=False)
    logger.info("Saved analysis-ready dataset to %s (%d rows)", out_file, len(df_filtered))

    return df_filtered, filter_summary


def main():
    parser = argparse.ArgumentParser(description="Ingest and preprocess study dataset.")
    parser.add_argument("--config", default="configs/study.yaml", help="Path to study.yaml")
    parser.add_argument("--input", default=None, help="Path to input CSV (default from config)")
    parser.add_argument("--output", default="data/interim/analysis_set.csv", help="Path to save interim CSV")
    parser.add_argument("--sample", action="store_true", help="Process data/sample/sample.csv instead of full data")

    args = parser.parse_args()
    cfg = load_yaml_config(args.config)

    if args.input:
        in_path = args.input
    elif args.sample:
        in_path = cfg.get("input", {}).get("sample_path", "data/sample/sample.csv")
    else:
        in_path = cfg.get("input", {}).get("path", "data/synthetic/main_dataset.csv")

    process_dataset(in_path, args.output, args.config)


if __name__ == "__main__":
    main()
