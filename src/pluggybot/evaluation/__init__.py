"""Measurement (M14, docs/Evaluation.md): how we find out whether the mind is
doing anything.

A new domain rather than a corner of `mind/` or `economy/` (CLAUDE.md,
"src/ is divided by DOMAIN"): nothing in here decides, scores or moves the
robot. `record.py` says what one run is worth writing down and writes it,
`rollup.py` aggregates runs and refuses to aggregate across a data-file
edit, and `run.py` is the child process `scripts/experiment.py` spawns per
run.
"""
