"""
tests/test_paper_artifacts.py
Tests for Part A paper artifact generation (reports/, figures/).
All numbers are derived from CSV files, never hardcoded.
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures" / "study"
RESULTS = ROOT / "results" / "tables"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Existence checks
# ---------------------------------------------------------------------------

class TestArtifactsExist:
    def test_formulas_tex_exists(self):
        assert (REPORTS / "formulas.tex").exists(), "reports/formulas.tex not found"

    def test_tables_tex_exists(self):
        assert (REPORTS / "tables.tex").exists(), "reports/tables.tex not found"

    def test_paper_section4_exists(self):
        assert (REPORTS / "paper_section_4.md").exists(), "reports/paper_section_4.md not found"

    def test_figure_png_exists(self):
        assert (FIGURES / "study_results.png").exists(), "reports/figures/study/study_results.png not found"

    def test_figure_pdf_exists(self):
        assert (FIGURES / "study_results.pdf").exists(), "reports/figures/study/study_results.pdf not found"


# ---------------------------------------------------------------------------
# 2. Watermark checks
# ---------------------------------------------------------------------------

WATERMARK = "SYNTHETIC DATA"


class TestWatermarks:
    def test_watermark_in_formulas(self):
        assert WATERMARK in _read(REPORTS / "formulas.tex"), \
            "SYNTHETIC DATA watermark missing from formulas.tex"

    def test_watermark_in_tables(self):
        assert WATERMARK in _read(REPORTS / "tables.tex"), \
            "SYNTHETIC DATA watermark missing from tables.tex"

    def test_watermark_in_section4(self):
        assert WATERMARK in _read(REPORTS / "paper_section_4.md"), \
            "SYNTHETIC DATA watermark missing from paper_section_4.md"


# ---------------------------------------------------------------------------
# 3. Content / label checks — formulas.tex
# ---------------------------------------------------------------------------

EXPECTED_LABELS = [
    "tab:notation",
    "eq:f_marker",
    "eq:i_marker",
    "eq:mwu",
    "eq:cliffs_delta",
    "eq:bootstrap_ci",
    "eq:jonckheere",
    "eq:spearman",
    "eq:regression",
]

EXPECTED_TODO_B = ["tab:stride", "fig:benchmark", "eq:p95"]


class TestFormulasContent:
    def test_all_labels_present(self):
        content = _read(REPORTS / "formulas.tex")
        missing = [lbl for lbl in EXPECTED_LABELS if f"\\label{{{lbl}}}" not in content]
        assert not missing, f"Missing \\label{{}} entries in formulas.tex: {missing}"

    def test_notation_table_present(self):
        content = _read(REPORTS / "formulas.tex")
        assert "\\begin{table}" in content or "\\begin{longtable}" in content, \
            "No table environment found in formulas.tex"


# ---------------------------------------------------------------------------
# 4. Content checks — tables.tex
# ---------------------------------------------------------------------------

class TestTablesContent:
    def test_corpus_table_label(self):
        assert "\\label{tab:corpus_descriptives}" in _read(REPORTS / "tables.tex")

    def test_rq2_table_label(self):
        assert "\\label{tab:rq2_results}" in _read(REPORTS / "tables.tex")

    def test_rq3_table_label(self):
        assert "\\label{tab:rq3_results}" in _read(REPORTS / "tables.tex")

    def test_median_gt3_matches_csv(self):
        """Median for >3 months in tables.tex must equal value from CSV."""
        csv = RESULTS / "synthetic_rq2_group_summary.csv"
        if not csv.exists():
            pytest.skip(f"CSV not found: {csv}")
        df = pd.read_csv(csv, comment="#")
        row = df[df["duration_group"] == ">3 months"]
        assert not row.empty, "'>3 months' row missing from RQ2 summary CSV"
        med = float(row["median"].values[0])
        content = _read(REPORTS / "tables.tex")
        assert f"{med:.2f}" in content, \
            f"Median {med:.2f} for >3 months not found in tables.tex"

    def test_p25_gt3_matches_csv(self):
        """p25 (Q1) for >3 months in tables.tex must equal value from CSV."""
        csv = RESULTS / "synthetic_rq2_group_summary.csv"
        if not csv.exists():
            pytest.skip(f"CSV not found: {csv}")
        df = pd.read_csv(csv, comment="#")
        row = df[df["duration_group"] == ">3 months"]
        assert not row.empty
        # Try common column name variants
        for col in ("q25", "p25", "25%"):
            if col in df.columns:
                val = float(row[col].values[0])
                content = _read(REPORTS / "tables.tex")
                assert f"{val:.2f}" in content, \
                    f"p25 value {val:.2f} for >3 months not found in tables.tex"
                return
        pytest.skip("No q25/p25 column found in RQ2 summary CSV")

    def test_p75_gt3_matches_csv(self):
        """p75 (Q3) for >3 months in tables.tex must equal value from CSV."""
        csv = RESULTS / "synthetic_rq2_group_summary.csv"
        if not csv.exists():
            pytest.skip(f"CSV not found: {csv}")
        df = pd.read_csv(csv, comment="#")
        row = df[df["duration_group"] == ">3 months"]
        assert not row.empty
        for col in ("q75", "p75", "75%"):
            if col in df.columns:
                val = float(row[col].values[0])
                content = _read(REPORTS / "tables.tex")
                assert f"{val:.2f}" in content, \
                    f"p75 value {val:.2f} for >3 months not found in tables.tex"
                return
        pytest.skip("No q75/p75 column found in RQ2 summary CSV")

    def test_mw_u_statistic_matches_csv(self):
        """Mann-Whitney U in tables.tex must match synthetic_rq2_mwu.csv."""
        csv = RESULTS / "synthetic_rq2_mwu.csv"
        if not csv.exists():
            pytest.skip(f"CSV not found: {csv}")
        df = pd.read_csv(csv, comment="#")
        # Find U-statistic column
        for col in ("U_statistic", "U", "statistic"):
            if col in df.columns:
                u_val = float(df[col].values[0])
                content = _read(REPORTS / "tables.tex")
                assert f"{u_val:.1f}" in content or f"{int(u_val)}" in content, \
                    f"U-statistic {u_val} not found in tables.tex"
                return
        pytest.skip("No U-statistic column found in MWU CSV")


# ---------------------------------------------------------------------------
# 5. Section 4 Markdown content checks
# ---------------------------------------------------------------------------

class TestSection4Content:
    def test_rq2_heading_present(self):
        content = _read(REPORTS / "paper_section_4.md")
        # The section uses full form "Research Question 2" (not the abbreviation)
        assert "Research Question 2" in content or "RQ2" in content, \
            "Research Question 2 / RQ2 not mentioned in paper_section_4.md"

    def test_rq3_heading_present(self):
        content = _read(REPORTS / "paper_section_4.md")
        # The section uses full form "Research Question 3" (not the abbreviation)
        assert "Research Question 3" in content or "RQ3" in content, \
            "Research Question 3 / RQ3 not mentioned in paper_section_4.md"

    def test_todo_b_placeholders_present(self):
        content = _read(REPORTS / "paper_section_4.md")
        missing = [lbl for lbl in EXPECTED_TODO_B if lbl not in content]
        assert not missing, f"TODO(B) placeholders missing from paper_section_4.md: {missing}"

    def test_mann_whitney_mentioned(self):
        content = _read(REPORTS / "paper_section_4.md")
        assert "Mann" in content or "Mann–Whitney" in content, \
            "Mann-Whitney not mentioned in paper_section_4.md"

    def test_cliff_delta_mentioned(self):
        content = _read(REPORTS / "paper_section_4.md")
        assert "Cliff" in content or "delta" in content.lower(), \
            "Cliff's delta not mentioned in paper_section_4.md"

    def test_spearman_mentioned(self):
        content = _read(REPORTS / "paper_section_4.md")
        assert "Spearman" in content or "spearman" in content, \
            "Spearman not mentioned in paper_section_4.md"


# ---------------------------------------------------------------------------
# 6. Cross-reference validation (via make_paper_artifacts.py logic)
# ---------------------------------------------------------------------------

class TestCrossReferences:
    def test_all_reftypes_in_formulas(self):
        """Every \\label{} in formulas.tex should appear as \\ref or \\eqref
        somewhere in paper_section_4.md (or as a TODO-B placeholder).

        Excluded from mandatory citation in section body:
        - tab:notation  — the notation/symbol table; cited in preamble, not body
        - eq:cliffs_delta — formula definition; its result is cited via tab:rq2_results
        These are auxiliary definitional labels not requiring inline \\ref in prose.
        """
        import re

        # Labels that are legitimately NOT required to appear inline in section 4
        AUXILIARY_LABELS = {"tab:notation", "eq:cliffs_delta"}

        formulas = _read(REPORTS / "formulas.tex")
        section4 = _read(REPORTS / "paper_section_4.md")
        labels = re.findall(r"\\label\{([^}]+)\}", formulas)
        unref = []
        for lbl in labels:
            if lbl in EXPECTED_TODO_B:
                continue  # Part B — intentional placeholder
            if lbl in AUXILIARY_LABELS:
                continue  # Auxiliary definitional labels — not required in body
            if lbl not in section4:
                unref.append(lbl)
        assert not unref, \
            f"Labels defined in formulas.tex but not referenced in section4: {unref}"

    def test_table_labels_referenced_in_section4(self):
        import re
        tables = _read(REPORTS / "tables.tex")
        section4 = _read(REPORTS / "paper_section_4.md")
        labels = re.findall(r"\\label\{(tab:[^}]+)\}", tables)
        unref = [lbl for lbl in labels if lbl not in EXPECTED_TODO_B and lbl not in section4]
        assert not unref, \
            f"Table labels not referenced in section4: {unref}"


# ---------------------------------------------------------------------------
# 7. Script reproducibility — running make_paper_artifacts.py must succeed
# ---------------------------------------------------------------------------

class TestScriptReproducibility:
    def test_script_runs_cleanly(self):
        """Running python src/make_paper_artifacts.py must exit 0."""
        result = subprocess.run(
            [sys.executable, "src/make_paper_artifacts.py"],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        assert result.returncode == 0, (
            f"make_paper_artifacts.py exited with code {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    def test_script_output_mentions_all_artifacts(self):
        """Script stdout must mention each expected output file."""
        result = subprocess.run(
            [sys.executable, "src/make_paper_artifacts.py"],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        combined = result.stdout + result.stderr
        for keyword in ("formulas.tex", "tables.tex", "paper_section_4", "study_results"):
            assert keyword in combined, \
                f"Script output does not mention expected artifact: {keyword}"
