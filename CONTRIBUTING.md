# Contributing

This repository is currently a private research project.

Before opening a change:

1. Install development dependencies with `pip install -r requirements-dev.txt`.
2. Run `python3 -m pytest -q`.
3. Run `python3 -m compileall -q pipeline data features models scripts train.py evaluate.py`.
4. Keep datasets, checkpoints, logs, and generated artifacts out of Git.
5. Update the README or paper when a result protocol changes.

Research claims must include the dataset, split, seed count, and evaluation protocol that produced them.
