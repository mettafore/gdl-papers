# Homework — Spec, Delegate, Verify, Refine

*This homework turns Part II into a real delivery workflow. You are no longer just asking the agent for help with a feature. You are learning to express work as durable artefacts, hand it off in a controlled way, and improve the surrounding setup until the feature becomes reliably completable agentically.*

## Topic

**Deliver one meaningful feature through specification-driven agentic development.**

## Main Goal

Extend the same app from Part I with **one bounded feature** and use it to practise real **spec-driven development**. The goal is to learn how to turn a feature into a durable brief, a workable plan, a reviewable implementation pass, and a verification loop that the agent can actually follow. The feature should be small enough to merge in one sitting, but large enough that it needs a real spec and touches more than one file.

Good examples include due dates and overdue filtering in a todo app, tags and tag filtering in a bookmarks app, status tracking in a reading list, favourites in a recipe box, or a streak display in a habit tracker. Avoid redesigns, vague clean-up work, or anything so broad that the agent has to guess where the edges are.

## Core Exercise

Choose one feature and create a **small set of durable artefacts** that define and control the work. The exact structure is up to you. You might keep everything under `specs/<feature>/`, put it in `docs/`, or use a simpler layout that fits your project. Suggested files such as `spec.md`, `plan.md`, `tasks.md`, and `review.md` are useful because they keep intent, execution, decomposition, and reflection separate, but they are suggestions rather than a mandatory template.

What matters is that your workflow includes the following:

- a written **specification** covering the **intent**, **acceptance criteria**, **constraints**, **non-goals**, **relevant context**, **verification**, and clear **always / ask first / never** boundaries;
- a **plan** for how the work will be approached, including likely files or components, execution order, checkpoints, and likely risks or ambiguities;
- a **task breakdown** into a few reviewable chunks;
- a short **review or retrospective note** recording whether the first pass was mergeable or close to it, what changed after the first run, what was underspecified, and which checks actually proved completion.

These can live in separate files, in one folder, or as clearly separated sections in fewer files. The important thing is that the work is durable, reviewable, and usable as a handoff.

Then hand the work to the agent using those artefacts as the brief. A sensible pattern is to ask the agent to review the spec and plan first, surface missing risks, refine the spec once if needed, implement against the written brief, pause at checkpoints, and then review and verify the result properly. For the core assignment, your evidence should include one diff review, one pass against the acceptance criteria, one test or smoke-check result, and one short note on what had to change before the task became reliably completable by the agent.

The pedagogical point is precise: if the first run is weak, do not quietly finish the hard parts by hand and congratulate yourself. Improve the artefacts or the harness until the feature can actually be delivered through the engineered workflow.

## Stretch Goal

Turn the single feature into a **small spec-driven MVP backlog**. Define an MVP scope for the app, break it into **3-5 mergeable slices**, and give each slice its own written brief, plan, task breakdown, and review trail. You may choose to keep these in separate files, per-feature folders, or another structure that makes the slices easy to execute and review independently. It is also useful to keep one higher-level MVP note describing the overall goal, slice order, dependencies, and what counts as “done enough”.

The goal of this stretch is decomposition. Instead of one large ambiguous brief, you should end up with a sequence of smaller features that can be implemented, reviewed, verified, and merged independently.

## Advanced Stretch

For the advanced version, push the workflow rather than merely increasing feature count. You might add a dedicated verifier or reviewer flow, move repeated failures into the harness by improving `AGENTS.md` or a canonical smoke test, compare tighter steering with broader delegation on similar slices, or measure first-pass yield across several specs. Another strong direction is to make the work more production-shaped with checkpoint commits, draft PRs, and CI evidence around the slices.

Choose one or two directions and treat them as engineering experiments. The interesting question is not whether the agent can produce more output. It is whether you can shape the surrounding system so the output becomes clearer, easier to review, and closer to mergeable on the first serious pass.

