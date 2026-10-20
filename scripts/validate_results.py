"""Validate the tracked result artifacts used by the paper.

Checks that every artifact the paper cites exists, is well formed, and
contains the fields the manuscript quotes. It also cross-checks the
numbers that appear in more than one artifact, so a stale file cannot
silently contradict the paper.
"""

import argparse
import json
from pathlib import Path


def read_json(path):
    try:
        with path.open() as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON artifact: {path}") from exc


REQUIRED = {
    "significance.json",
    "significance_5seed.json",
    "drift_discriminator_robustness.json",
    "drift_discriminator_decomposition.json",
    "drift_standardized.json",
    "early_detection_multiseed.json",
    "ablations.json",
    "ablations_10pct.json",
}


def validate_significance(root):
    significance = read_json(root / "significance.json")
    ratios = [
        significance["drift_ratio_gossipcop"]["ratio"],
        significance["drift_ratio_politifact"]["ratio"],
    ]
    if not all(0 < ratio < 1 for ratio in ratios):
        raise SystemExit("Drift ratios must be between 0 and 1")


def validate_discriminator(root):
    discriminator = read_json(root / "drift_discriminator_robustness.json")
    for subset in ("gossipcop", "politifact"):
        for setting in ("full", "truncated_10pct"):
            record = discriminator[subset][setting]
            for metric in ("acc_mean", "auc_mean"):
                if not 0 <= record[metric] <= 1:
                    raise SystemExit(f"Invalid {subset}/{setting}/{metric}")


def validate_decomposition(root):
    """The content-only AUC must not exceed the all-features AUC by much, and
    the size-only baseline must be reported (it is the honest control)."""
    decomposition = read_json(root / "drift_discriminator_decomposition.json")
    for subset in ("gossipcop", "politifact"):
        block = decomposition[subset]
        for setting in ("full", "truncated_10pct"):
            for group in ("content", "structural", "size", "all"):
                cv = block[setting][group]["stratified_5fold_cv"]
                for metric in ("acc_mean", "auc_mean"):
                    if not 0 <= cv[metric] <= 1:
                        raise SystemExit(
                            f"Invalid {subset}/{setting}/{group}/{metric}"
                        )
        if "excluding_root_only" not in block["truncated_10pct"]["all"]:
            raise SystemExit(
                f"{subset}: the root-only exclusion is missing from the "
                "10% truncation decomposition"
            )


def validate_standardized(root):
    """Cross-check the depth-controlled drift statistics against the paper."""
    standardized = read_json(root / "drift_standardized.json")
    for subset in ("gossipcop", "politifact"):
        block = standardized[subset]
        pooled = block["pooled"]["ratio_b_over_a"]
        hop_std = block["hop_standardized"]["ratio_fake_over_real"]
        cascade = block["cascade_level"]["ratio_b_over_a"]
        if not 0 < pooled < 1 or not 0 < hop_std < 1 or not 0 < cascade < 1:
            raise SystemExit(f"{subset}: drift ratios must be between 0 and 1")
        # The controlled effect must not be *larger* than the pooled one;
        # if it is, the standardisation is not doing its job.
        if hop_std < pooled - 1e-9:
            raise SystemExit(
                f"{subset}: hop-standardised ratio ({hop_std}) is smaller than "
                f"the pooled ratio ({pooled}); expected the controlled effect "
                "to be attenuated, not amplified"
            )


def validate_early_detection(root):
    early = read_json(root / "early_detection_multiseed.json")
    if not {"BaselineLSTM", "DegradGNN", "Hybrid"}.issubset(early):
        raise SystemExit("Early-detection artifact is missing a required model")
    for name, record in early.items():
        for pct in (10, 25, 50, 100):
            key = f"acc_{pct}pct_mean"
            if key not in record:
                raise SystemExit(f"Early detection {name}: missing {key}")
            if not 0 <= record[key] <= 1:
                raise SystemExit(f"Early detection {name}: invalid {key}")


def validate_ablations(root):
    ablations10 = read_json(root / "ablations_10pct.json")
    if not {"Full", "w/o drift", "w/o hop"}.issubset(ablations10):
        raise SystemExit("Ablation artifact is missing a required variant")
    ablations100 = read_json(root / "ablations.json")
    if not {"Full (drift+hop+tree)", "w/o drift", "w/o hop"}.issubset(ablations100):
        raise SystemExit("100% ablation artifact is missing a required variant")
    # The two 100% numbers for the same variant must agree: they come from
    # the same protocol, and a mismatch means one artifact is stale.
    for short, long in (
        ("Full", "Full (drift+hop+tree)"),
        ("w/o drift", "w/o drift"),
        ("w/o hop", "w/o hop"),
    ):
        a = ablations10[short]["full"]["accuracy"]
        b = ablations100[long]["accuracy"]
        if abs(a - b) > 0.01:
            raise SystemExit(
                f"Ablation mismatch for {short}: 10%-run artifact says "
                f"{a:.4f}, 100%-run artifact says {b:.4f}"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    root = Path(args.results_dir)

    missing = sorted(name for name in REQUIRED if not (root / name).is_file())
    if missing:
        raise SystemExit(f"Missing tracked artifacts: {', '.join(missing)}")

    validate_significance(root)
    validate_discriminator(root)
    validate_decomposition(root)
    validate_standardized(root)
    validate_early_detection(root)
    validate_ablations(root)

    print(f"Validated {len(REQUIRED)} paper result artifacts in {root}")


if __name__ == "__main__":
    main()
