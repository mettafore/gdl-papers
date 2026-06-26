---
description: Evaluate a saved run of a paper via the run-paper skill
---

Use the `run-paper` skill to EVALUATE a paper in this repo.

Arguments: `$ARGUMENTS` — the paper folder name and the run to score, e.g.
`00-template --run 20260626-104704`.

Steps:
1. Read `papers/<paper>/README.md` `## Run` for the evaluate command.
2. Run `uv run python papers/<paper>/evaluate.py --run <run_id>`.
3. Report the metric.
