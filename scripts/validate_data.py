"""Validate the local UPFD dataset before training."""

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
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    summaries = validate_subsets(
        args.data_root,
        [name.strip() for name in args.subsets.split(",") if name.strip()],
        args.feature_type,
    )
    if args.json:
        print(json.dumps(summaries, indent=2))
        return
    for summary in summaries.values():
        print(
            f"{summary['subset']}: {summary['num_graphs']} graphs, "
            f"{summary['num_nodes']} nodes, {summary['feature_dim']}-dim features, "
            f"splits={summary['splits']}"
        )


if __name__ == "__main__":
    main()
