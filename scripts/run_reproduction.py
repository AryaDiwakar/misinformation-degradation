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
    parser.add_argument("--out-dir", default="results/rerun_2026-10-24")
    parser.add_argument("--checkpoint-dir", default="checkpoints/rerun_2026-10-24")
    parser.add_argument(
        "--stages",
        default=(
            "validate,benchmark,early,discriminator,decomposition,incremental,"
            "mechanism,standardized,ablation"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
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
                    "GCN,GAT,BiGCN,BaselineLSTM,DegradGNN,Hybrid,DriftPool,DriftPoolOnly",
                    "--out",
                    f"{args.out_dir}/multiseed_{subset}.json",
                    "--checkpoint-dir",
                    args.checkpoint_dir,
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
                "BaselineLSTM,DegradGNN,Hybrid,DriftPool",
                "--out",
                f"{args.out_dir}/early_detection.json",
            ],
            args.dry_run,
        )

    if "discriminator" in stages:
        # Write to a separate file so the tracked paper artifact is not
        # overwritten by a reproduction run.
        run(
            [
                python,
                "scripts/run_discriminator_robustness.py",
                "--out",
                f"{args.out_dir}/drift_discriminator.json",
            ],
            args.dry_run,
        )

    if "decomposition" in stages:
        run(
            [
                python,
                "scripts/analysis_discriminator_decomposition.py",
                "--out",
                    f"{args.out_dir}/discriminator_decomposition.json",
            ],
            args.dry_run,
        )

    if "standardized" in stages:
        run(
            [
                python,
                "scripts/analysis_drift_standardized.py",
                "--out",
                f"{args.out_dir}/drift_standardized.json",
            ],
            args.dry_run,
        )

    if "incremental" in stages:
        run(
            [
                python,
                "scripts/analysis_incremental_value.py",
                "gossipcop",
                "--out",
                f"{args.out_dir}/incremental_value_gossipcop.json",
            ],
            args.dry_run,
        )

    if "mechanism" in stages:
        run(
            [
                python,
                "scripts/analysis_mechanism_probe.py",
                "gossipcop",
                "--out",
                f"{args.out_dir}/mechanism_probe_gossipcop.json",
            ],
            args.dry_run,
        )
        run(
            [
                python,
                "scripts/analysis_mechanism_linkage.py",
                "gossipcop",
                "--out",
                f"{args.out_dir}/mechanism_linkage_gossipcop.json",
            ],
            args.dry_run,
        )

    if "ablation" in stages:
        run(
            [
                python,
                "scripts/run_ablation_10pct.py",
                "--out",
                f"{args.out_dir}/ablations_10pct.json",
            ],
            args.dry_run,
        )
        run(
            [
                python,
                "scripts/run_ablation_100pct.py",
                "--out",
                f"{args.out_dir}/ablations_100pct.json",
            ],
            args.dry_run,
        )

    print("Reproduction stages complete.")


if __name__ == "__main__":
    main()
