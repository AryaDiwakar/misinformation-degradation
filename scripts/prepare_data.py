"""Prepare or validate the local UPFD dataset.

The upstream UPFD archive must be downloaded manually because it is large and
subject to upstream platform and licensing restrictions. This command validates
the extracted directory before any experiment is started.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.data_validation import validate_subsets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/upfd/raw")
    parser.add_argument("--subsets", default="gossipcop,politifact")
    parser.add_argument("--feature-type", default="bert")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        result = validate_subsets(
            args.data_root,
            [s.strip() for s in args.subsets.split(",") if s.strip()],
            args.feature_type,
        )
    except ValueError as exc:
        print(f"Dataset is not ready: {exc}", file=sys.stderr)
        print(
            "Download UPFD from https://github.com/safe-graph/GNN-FakeNews "
            f"and extract the selected subsets under {args.data_root}.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for summary in result.values():
            print(
                f"Ready: {summary['subset']} | graphs={summary['num_graphs']} "
                f"nodes={summary['num_nodes']} features={summary['feature_dim']}"
            )


if __name__ == "__main__":
    main()
