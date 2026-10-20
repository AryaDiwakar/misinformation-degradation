"""Single source of truth for the numbers quoted in generated documents.

The DOCX and PPTX generators previously hard-coded every metric, so they
silently went stale whenever an experiment was re-run and they cited result
files that did not exist. This module reads the tracked artifacts in
``results/`` and returns formatted tables, so the documents cannot drift from
the data.

Each accessor degrades gracefully: if an artifact is missing, the table cell
reads ``n/a`` instead of inventing a number.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RESULTS = os.path.join(ROOT, "results")

NA = "n/a"


def _load(name: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def _fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return NA
    try:
        if value != value:  # NaN
            return NA
    except TypeError:
        return NA
    return f"{value:.{digits}f}"


def _fmt_pm(mean: Optional[float], std: Optional[float], digits: int = 4) -> str:
    if mean is None:
        return NA
    if std is None:
        return _fmt(mean, digits)
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _p(value: Optional[float]) -> str:
    if value is None:
        return NA
    try:
        if value != value:
            return NA
    except TypeError:
        return NA
    if value == 0.0:
        return "< 1e-16"
    if value < 1e-4:
        return f"{value:.2e}"
    return f"{value:.4g}"


# --------------------------------------------------------------------- models


def main_results_table(subset: str = "gossipcop") -> List[List[str]]:
    """Rows for the main classification table."""
    combined = _load("multiseed_all.json") or {}
    block = combined.get(subset, {}) if isinstance(combined, dict) else {}
    if not block:
        block = _load("multiseed_all.json") or {}

    extra = _load("multiseed_driftpool.json") or {}
    if "DriftPool" in extra and "DriftPool" not in block:
        block = {**block, "DriftPool": extra["DriftPool"]}

    order = [
        "GCN",
        "GAT",
        "BiGCN",
        "DegradGNN",
        "BaselineLSTM",
        "Hybrid",
        "DriftPool",
    ]
    rows: List[List[str]] = []
    for name in order:
        rec = block.get(name)
        if not rec:
            continue
        rows.append(
            [
                name,
                _fmt_pm(rec.get("acc_mean"), rec.get("acc_std")),
                _fmt(rec.get("f1_mean")),
                _fmt(rec.get("auc_mean")),
                _fmt(rec.get("aucpr_mean")),
            ]
        )
    return rows


# ---------------------------------------------------------------------- drift


def drift_table() -> List[List[str]]:
    """Rows for the drift-phenomenon table, pooled *and* hop-standardised."""
    standardized = _load("drift_standardized.json") or {}
    significance = _load("significance.json") or {}

    rows: List[List[str]] = []
    for subset in ("gossipcop", "politifact"):
        block = standardized.get(subset)
        legacy = significance.get(f"drift_ratio_{subset}", {})
        if block:
            pooled = block["pooled"]
            std = block["hop_standardized"]
            cascade = block["cascade_level"]
            rows.append(
                [
                    subset,
                    _fmt(pooled["mean_a"], 2),
                    _fmt(pooled["mean_b"], 2),
                    _fmt(pooled["ratio_b_over_a"], 3),
                    _fmt(std["ratio_fake_over_real"], 3),
                    _fmt(cascade["ratio_b_over_a"], 3),
                    f"{cascade['t']:.2f}",
                    _p(cascade["p"]),
                ]
            )
        elif legacy:
            rows.append(
                [
                    subset,
                    _fmt(legacy.get("real_mean"), 2),
                    _fmt(legacy.get("fake_mean"), 2),
                    _fmt(legacy.get("ratio"), 3),
                    NA,
                    NA,
                    NA,
                    _p(legacy.get("p")),
                ]
            )
    return rows


def discriminator_table() -> List[List[str]]:
    """Rows for the drift-only discriminator table (decomposed)."""
    decomposition = _load("drift_discriminator_decomposition.json") or {}
    robustness = _load("drift_discriminator_robustness.json") or {}

    rows: List[List[str]] = []
    for subset in ("gossipcop", "politifact"):
        block = decomposition.get(subset)
        if block:
            full = block["full"]
            trunc = block["truncated_10pct"]
            content = full["content"]["stratified_5fold_cv"]
            size = full["size"]["stratified_5fold_cv"]
            all_full = full["all"]["stratified_5fold_cv"]
            all_trunc = trunc["all"]["stratified_5fold_cv"]
            excluding = trunc.get("all", {}).get("excluding_root_only")
            if excluding and "stratified_5fold_cv" in excluding:
                excluding = excluding["stratified_5fold_cv"]
            rows.append(
                [
                    subset,
                    "full cascade",
                    _fmt(all_full["acc_mean"]),
                    _fmt(all_full["auc_mean"]),
                    _fmt(content["auc_mean"]),
                    _fmt(size["auc_mean"]),
                    NA,
                ]
            )
            rows.append(
                [
                    subset,
                    "first 10% of cascade",
                    _fmt(all_trunc["acc_mean"]),
                    _fmt(all_trunc["auc_mean"]),
                    _fmt(trunc["content"]["stratified_5fold_cv"]["auc_mean"]),
                    _fmt(trunc["size"]["stratified_5fold_cv"]["auc_mean"]),
                    _fmt(excluding["auc_mean"]) if excluding else NA,
                ]
            )
        elif subset in robustness:
            full = robustness[subset]["full"]
            trunc = robustness[subset]["truncated_10pct"]
            rows.append(
                [
                    subset,
                    "full cascade",
                    _fmt(full["acc_mean"]),
                    _fmt(full["auc_mean"]),
                    NA,
                    NA,
                    NA,
                ]
            )
            rows.append(
                [
                    subset,
                    "first 10% of cascade",
                    _fmt(trunc["acc_mean"]),
                    _fmt(trunc["auc_mean"]),
                    NA,
                    NA,
                    NA,
                ]
            )
    return rows


def early_detection_table() -> List[List[str]]:
    """Rows for the partial-cascade table."""
    early = _load("early_detection_multiseed.json") or {}
    order = ["BaselineLSTM", "DegradGNN", "Hybrid", "DriftPool", "DriftPoolOnly"]
    rows: List[List[str]] = []
    for name in order:
        rec = early.get(name)
        if not rec:
            continue
        cells = [name]
        for pct in (10, 25, 50, 100):
            mean = rec.get(f"acc_{pct}pct_mean")
            std = rec.get(f"acc_{pct}pct_std")
            cells.append(_fmt_pm(mean, std, 4) if mean is not None else NA)
        rows.append(cells)
    return rows


def ablation_table() -> List[List[str]]:
    """Rows for the 100%-cascade ablation table."""
    ablations = _load("ablations.json") or {}
    rows: List[List[str]] = []
    for name, rec in ablations.items():
        if not isinstance(rec, dict) or "accuracy" not in rec:
            continue
        rows.append(
            [
                name,
                _fmt(rec.get("accuracy")),
                _fmt(rec.get("auc_roc")),
            ]
        )
    return rows


def ablation_10pct_table() -> List[List[str]]:
    """Rows for the 10%-truncation ablation table."""
    ablations = _load("ablations_10pct.json") or {}
    rows: List[List[str]] = []
    for name, rec in ablations.items():
        if not isinstance(rec, dict) or "full" not in rec:
            continue
        rows.append(
            [
                name,
                _fmt(rec["full"].get("accuracy")),
                _fmt(rec["truncated_10pct"].get("accuracy")),
            ]
        )
    return rows


def significance_table() -> List[List[str]]:
    """Rows for the paired comparison table."""
    significance = _load("significance.json") or {}
    labels = {
        "hybrid_vs_lstm_gossipcop_acc": "Hybrid vs BaselineLSTM (accuracy)",
        "hybrid_vs_lstm_gossipcop_auc": "Hybrid vs BaselineLSTM (AUC)",
        "hybrid_vs_degradgnn_gossipcop_acc": "Hybrid vs DegradGNN (accuracy)",
    }
    rows: List[List[str]] = []
    for key, label in labels.items():
        rec = significance.get(key)
        if not rec:
            continue
        rows.append(
            [
                label,
                _fmt(rec.get("mean_diff")),
                f"{rec.get('t', float('nan')):.2f}",
                _p(rec.get("p")),
                _fmt(rec.get("cohen_d"), 2),
            ]
        )
    return rows


def artifact_index() -> List[List[str]]:
    """The reproducibility index, listing only artifacts that actually exist."""
    entries = [
        ("results/multiseed_all.json", "3-seed main results (both datasets)"),
        ("results/multiseed_driftpool.json", "3-seed DriftPool results"),
        ("results/significance.json", "paired t-tests, Cohen's d, drift ratios"),
        ("results/significance_5seed.json", "n=3 / n=5 paired statistics"),
        ("results/drift_standardized.json", "hop-standardised and cascade-level drift statistics"),
        ("results/drift_discriminator_robustness.json", "drift-only RF, full vs 10%"),
        ("results/drift_discriminator_decomposition.json", "content vs structural vs size decomposition"),
        ("results/ablations.json", "100% component ablation"),
        ("results/ablations_10pct.json", "10% truncation ablation"),
        ("results/early_detection_multiseed.json", "multi-seed truncation curves"),
    ]
    rows: List[List[str]] = []
    for path, description in entries:
        exists = os.path.exists(os.path.join(ROOT, path))
        rows.append([path, description, "present" if exists else "MISSING"])
    return rows
