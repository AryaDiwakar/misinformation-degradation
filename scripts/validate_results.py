"""Validate the tracked result artifacts used by the paper."""

import argparse
import json
from pathlib import Path


def read_json(path):
    try:
        with path.open() as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON artifact: {path}") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    root = Path(args.results_dir)
    required = {
        "significance.json",
        "drift_discriminator_robustness.json",
        "early_detection_multiseed.json",
        "ablations_10pct.json",
    }
    missing = sorted(name for name in required if not (root / name).is_file())
    if missing:
        raise SystemExit(f"Missing tracked artifacts: {', '.join(missing)}")

    significance = read_json(root / "significance.json")
    ratios = [
        significance["drift_ratio_gossipcop"]["ratio"],
        significance["drift_ratio_politifact"]["ratio"],
    ]
    if not all(0 < ratio < 1 for ratio in ratios):
        raise SystemExit("Drift ratios must be between 0 and 1")

    discriminator = read_json(root / "drift_discriminator_robustness.json")
    for subset in ("gossipcop", "politifact"):
        for setting in ("full", "truncated_10pct"):
            record = discriminator[subset][setting]
            for metric in ("acc_mean", "auc_mean"):
                if not 0 <= record[metric] <= 1:
                    raise SystemExit(f"Invalid {subset}/{setting}/{metric}")

    early = read_json(root / "early_detection_multiseed.json")
    if not {"BaselineLSTM", "DegradGNN", "Hybrid"}.issubset(early):
        raise SystemExit("Early-detection artifact is missing a required model")
    ablations = read_json(root / "ablations_10pct.json")
    if not {"Full", "w/o drift", "w/o hop"}.issubset(ablations):
        raise SystemExit("Ablation artifact is missing a required variant")
    print(f"Validated {len(required)} paper result artifacts in {root}")


if __name__ == "__main__":
    main()
