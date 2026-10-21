"""Tests for the paper-number extraction layer and the result validator.

The document generators used to hard-code every metric; they now read from
``scripts/paper_numbers.py``. These tests make sure that layer degrades
gracefully (missing artifact -> "n/a", never an invented number) and that
the tracked artifacts stay internally consistent.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import paper_numbers  # noqa: E402


def test_tracked_artifacts_all_exist():
    results = ROOT / "results"
    for name in (
        "significance.json",
        "significance_5seed.json",
        "drift_discriminator_robustness.json",
        "drift_discriminator_decomposition.json",
        "drift_standardized.json",
        "early_detection_multiseed.json",
        "ablations.json",
        "ablations_10pct.json",
    ):
        assert (results / name).is_file(), name


def test_validator_passes_on_the_tracked_artifacts(monkeypatch):
    import validate_results

    monkeypatch.setattr(
        sys, "argv", ["validate_results.py", "--results-dir", str(ROOT / "results")]
    )
    validate_results.main()


def test_main_results_table_reads_the_artifacts():
    rows = paper_numbers.main_results_table("gossipcop")
    assert rows, "expected main results rows"
    names = [row[0] for row in rows]
    assert "Hybrid" in names
    for row in rows:
        assert len(row) == 5
        for cell in row:
            assert cell and cell != "None"


def test_drift_table_reports_both_pooled_and_controlled_statistics():
    rows = paper_numbers.drift_table()
    assert {row[0] for row in rows} == {"gossipcop", "politifact"}
    for row in rows:
        # dataset, real, fake, pooled, hop-std, cascade-level, t, p
        assert len(row) == 8
        assert row[3] != paper_numbers.NA, "pooled ratio missing"
        assert row[4] != paper_numbers.NA, "hop-standardised ratio missing"
        assert row[5] != paper_numbers.NA, "cascade-level ratio missing"


def test_discriminator_table_includes_the_size_control():
    rows = paper_numbers.discriminator_table()
    assert rows
    header_shapes = {len(row) for row in rows}
    assert header_shapes == {7}, header_shapes


def test_missing_artifact_yields_na_not_a_fabricated_number(tmp_path, monkeypatch):
    monkeypatch.setattr(paper_numbers, "RESULTS", str(tmp_path))
    rows = paper_numbers.drift_table()
    assert rows == [] or all(
        cell == paper_numbers.NA
        for row in rows
        for cell in row
        if cell not in {row[0]}
    )


def test_artifact_index_reports_presence():
    rows = paper_numbers.artifact_index()
    assert rows
    for path, _description, status in rows:
        assert status in {"present", "MISSING"}
        if status == "MISSING":
            assert not (ROOT / path).exists()


def test_formatting_helpers():
    assert paper_numbers._fmt(None) == paper_numbers.NA
    assert paper_numbers._fmt(float("nan")) == paper_numbers.NA
    assert paper_numbers._fmt(0.5) == "0.5000"
    assert paper_numbers._fmt_pm(0.5, 0.1) == "0.5000 ± 0.1000"
    assert paper_numbers._fmt_pm(0.5, None) == "0.5000"
    assert paper_numbers._p(0.0) == "< 1e-16"
    assert paper_numbers._p(1e-70) == "1.00e-70"


def test_manuscript_numbers_match_the_artifacts(monkeypatch):
    """The manuscript's quoted numbers must match the tracked artifacts."""
    import check_paper_numbers

    monkeypatch.setattr(
        sys, "argv", ["check_paper_numbers.py"]
    )
    check_paper_numbers.main()
