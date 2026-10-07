"""Run the canonical paper-reproduction stages in a documented order."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command, dry_run=False):
    print("$", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--subset", choices=["gossipcop", "politifact", "both"], default="both")
    parser.add_argument("--stages", default="validate,benchmark,early,discriminator,ablation")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    stages = {stage.strip() for stage in args.stages.split(",") if stage.strip()}
    python = sys.executable

    if "validate" in stages:
        run([python, "scripts/prepare_data.py"], args.dry_run)

    if "benchmark" in stages:
        subsets = [args.subset] if args.subset != "both" else ["gossipcop", "politifact"]
        for subset in subsets:
            run(
                [
                    python,
                    "multiseed.py",
                    "--subset",
                    subset,
                    "--seeds",
                    args.seeds,
                    "--models",
                    "GCN,GAT,BiGCN,BaselineLSTM,DegradGNN,Hybrid",
                    "--out",
                    f"results/reproduction_{subset}.json",
                ],
                args.dry_run,
            )

    if "early" in stages:
        run(
            [
                python,
                "scripts/run_early_detection.py",
                "--seeds",
                args.seeds,
                "--models",
                "BaselineLSTM,DegradGNN,Hybrid",
                "--out",
                "results/reproduction_early_detection.json",
            ],
            args.dry_run,
        )

    if "discriminator" in stages:
        run([python, "scripts/run_discriminator_robustness.py"], args.dry_run)

    if "ablation" in stages:
        run(
            [
                python,
                "scripts/run_ablation_10pct.py",
                "--out",
                "results/reproduction_ablations_10pct.json",
            ],
            args.dry_run,
        )

    print("Reproduction stages complete.")


if __name__ == "__main__":
    main()
