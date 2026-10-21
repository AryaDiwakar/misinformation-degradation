"""Cross-check the manuscript's numbers against the tracked result artifacts.

The paper, the README, the claims document and the generated decks all quote
the same handful of statistics. A typo or a stale copy in any of them would
undermine the whole point of shipping the artifacts. This script parses the
numbers back out of the manuscript text and compares them with the JSON.

Usage:
    python3 scripts/check_paper_numbers.py
"""

import argparse
import json
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RESULTS = os.path.join(ROOT, "results")


def load(name):
    with open(os.path.join(RESULTS, name)) as handle:
        return json.load(handle)


def close(a, b, tol=5e-4):
    return abs(float(a) - float(b)) <= tol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", default=os.path.join(ROOT, "paper", "report.md"))
    parser.add_argument("--tex", default=os.path.join(ROOT, "paper", "paper.tex"))
    parser.add_argument("--readme", default=os.path.join(ROOT, "README.md"))
    args = parser.parse_args()

    with open(args.markdown) as handle:
        report = handle.read()
    with open(args.tex) as handle:
        tex = handle.read()
    with open(args.readme) as handle:
        readme = handle.read()

    standardized = load("drift_standardized.json")
    decomposition = load("drift_discriminator_decomposition.json")
    significance = load("significance.json")
    early = load("early_detection_multiseed.json")
    ablations = load("ablations.json")

    failures = []

    def expect(label, needle, documents):
        for name, text in documents:
            if needle in text:
                continue
            # LaTeX tables write .787 where Markdown writes 0.787.
            alt = needle.replace("0.", ".", 1) if needle.startswith("0.") else None
            if alt and alt in text:
                continue
            failures.append(f"{label}: '{needle}' missing from {name}")

    docs = [("report.md", report), ("paper.tex", tex)]

    # --- drift phenomenon ------------------------------------------------
    gc_std = standardized["gossipcop"]["hop_standardized"]
    gc_casc = standardized["gossipcop"]["cascade_level"]
    expect(
        "GossipCop hop-standardised ratio",
        f"{gc_std['ratio_fake_over_real']:.3f}",
        docs,
    )
    expect(
        "GossipCop cascade-level ratio",
        f"{gc_casc['ratio_b_over_a']:.3f}",
        docs,
    )
    expect(
        "GossipCop cascade-level p-value",
        f"{gc_casc['p']:.3f}".rstrip("0").rstrip("."),
        docs,
    )
    pf_casc = standardized["politifact"]["cascade_level"]
    expect(
        "PolitiFact cascade-level ratio",
        f"{pf_casc['ratio_b_over_a']:.3f}",
        docs,
    )

    # --- discriminator decomposition -------------------------------------
    content_auc = decomposition["gossipcop"]["full"]["content"]["stratified_5fold_cv"]["auc_mean"]
    size_auc = decomposition["gossipcop"]["full"]["size"]["stratified_5fold_cv"]["auc_mean"]
    all_auc = decomposition["gossipcop"]["full"]["all"]["stratified_5fold_cv"]["auc_mean"]
    trunc_auc = decomposition["gossipcop"]["truncated_10pct"]["all"]["stratified_5fold_cv"]["auc_mean"]
    excl = decomposition["gossipcop"]["truncated_10pct"]["all"]["excluding_root_only"][
        "stratified_5fold_cv"
    ]["auc_mean"]
    expect("content-only AUC", f"{content_auc:.3f}", docs)
    expect("size-only AUC", f"{size_auc:.3f}", docs)
    expect("all-features AUC", f"{all_auc:.3f}", docs)
    expect("10% truncated AUC", f"{trunc_auc:.3f}", docs)
    expect("10% AUC excluding root-only", f"{excl:.3f}", docs)

    # --- incremental value over cascade shape ------------------------------
    incremental = load("incremental_value_gossipcop.json")
    content_vs_size = incremental["cmp__content_vs__size"]
    size_plus_content = incremental["cmp__size+content_vs__size"]
    content_vs_structural = incremental["cmp__size+content_vs__size+structural"]
    expect(
        "content vs size AUC gap",
        f"{content_vs_size['diff']:.3f}".lstrip("0"),
        docs,
    )
    expect(
        "size+content vs size AUC gap",
        f"{size_plus_content['diff']:.3f}".lstrip("0"),
        docs,
    )
    expect(
        "content vs structural AUC gap",
        f"{content_vs_structural['diff']:.3f}".lstrip("0"),
        docs,
    )
    for label, stat in (
        ("content vs size", content_vs_size),
        ("size+content vs size", size_plus_content),
        ("content vs structural", content_vs_structural),
    ):
        if stat["p_two_sided"] >= 0.05:
            failures.append(
                f"{label}: paired bootstrap p={stat['p_two_sided']:.3f} is not "
                "significant, but the manuscript reports it as significant"
            )

    # --- model comparison -------------------------------------------------
    hybrid_vs_lstm = significance["hybrid_vs_lstm_gossipcop_acc"]
    expect("Hybrid vs LSTM p-value", f"{hybrid_vs_lstm['p']:.3f}".rstrip("0").rstrip("."), docs)
    expect("Hybrid vs DegradGNN p-value", "0.020", docs)

    # --- early detection --------------------------------------------------
    lstm10 = early["BaselineLSTM"]["acc_10pct_mean"]
    expect("LSTM 10% accuracy", f"{lstm10:.3f}", docs)

    # --- ablation ---------------------------------------------------------
    full_acc = ablations["Full (drift+hop+tree)"]["accuracy"]
    expect("ablation Full accuracy", f"{full_acc:.4f}", docs)

    # --- the corrected framing must be present everywhere ------------------
    for name, text in [("report.md", report), ("paper.tex", tex), ("README.md", readme)]:
        if "11%" in text and "not ~11%" not in text and "~11% less" not in text:
            failures.append(f"{name}: still claims an 11% effect without the correction note")

    if failures:
        print("Paper-number check FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)

    print("Paper numbers match the tracked result artifacts.")


if __name__ == "__main__":
    main()
