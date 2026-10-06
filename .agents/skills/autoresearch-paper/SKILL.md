---
name: autoresearch-paper
description: Use when a user asks to autoresearch, run an autonomous experiment campaign or overnight sweep, or repeatedly optimize a gdl-papers reproduction against a validation metric.
---

# Autoresearch a GDL Paper

Keep validation and held-out evaluation outside candidate control.

## Approve the contract

Read `AGENTS.md`, the paper README, and `papers/<paper>/autoresearch.md`. Require:

- validation metric/direction, baseline, admission threshold, ranking/tie rule;
- literal train, protected-validation, and held-out-test commands;
- baseline config, candidate-mutable and protected paths, allowed ranges;
- dataset/split identity, sizes, seed, and fingerprint;
- launch/retry limits, time/compute budget, resource envelope, retryable
  failures, and target;
- one-seed screening, three-seed final-winner stability check, and final test
  gate.

Run the README's literal `uv run python` commands; do not depend on `run-paper`
or invent a runner. Require a protected validation scorer that accepts the run
ID and scores only validation. Treat
`uv run python papers/<paper>/src/evaluate.py --run <run_id>` as held-out test
evaluation unless the README documents validation mode. If none exists, propose
a validation scorer and stop for approval.

If the contract is absent/incomplete, propose it and stop; write it only after
explicit user approval. Require authorization for paid compute, dependencies,
or long runs.

Authorize only named paths. A contract cannot override learning rules:
`src/model.py`, `src/data.py`, `src/train.py`, `src/evaluate.py`, and math paths
require explicit approval. Freeze evaluator and split before the baseline.

## Protect the campaign

Require a clean, dedicated branch and record baseline `HEAD`; otherwise stop.
Never clean user work. Use worktrees only for approved parallel writers.

Keep contract, split, evaluator, tests, lockfile, validation/test data, and
evaluator I/O immutable. Candidates train and write run artifacts; the protected
evaluator scores. Forbid candidate-worker access to protected paths/data/output,
self-scoring, test inspection, pushes, and PRs.

Before each code-changing candidate commit:

1. Reject changes outside the candidate-mutable allowlist.
2. Confirm datasets, checkpoints, plots, raw output, caches, and temporary files
   are untracked and ignored.
3. Run repository-required formatting, linting, typing, tests, and reviews.
4. Commit one coherent candidate locally and record its SHA.

Identify config-only candidates by code SHA and canonical config digest; never
create empty commits. Never reset/checkout work or amend/rebase history.

## Record the campaign

Track under `papers/<paper>/autoresearch/<tag>/`:

- `research_log.md`: hypothesis, change, outcome, interpretation, lesson;
- `results.tsv`: baseline/candidate/revert SHAs, config digest, run ID, command,
  split fingerprint, seed, metric, runtime, environment, status, failure;
- `final_report.md`: winner, failures, and reproduction steps.

Commit records separately from candidate code. Record the untouched baseline.

## Run the experiment loop

For each iteration:

1. Review the best result and failed hypotheses.
2. State one falsifiable hypothesis and make one coherent change.
3. Run gates, commit code changes, then train.
4. Score with the protected validation evaluator.
5. Apply the predeclared admission and ranking rules.
6. If accepted, log `keep` and commit the record.
7. If rejected or crashed, revert any code commit, then log `discard` or `crash`
   with candidate/revert SHAs and commit the record.

Screen each candidate once. Do not aggregate seeds during search. Give only the
highest-ranked admissible candidate the three-seed check; report instability.

Count every launch/retry. Retry only defined mechanical failures within budget.
Record OOM as `crash`; resource changes create a new candidate. Stop at the
target, any budget, or an integrity/compute blocker.

## Finish

Freeze the winning SHA/configuration and write `final_report.md`. Run the
held-out test once after explicit human approval; report it without resuming
tuning or broadening the campaign.
