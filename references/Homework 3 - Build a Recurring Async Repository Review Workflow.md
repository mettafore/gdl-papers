# Homework — Build a Recurring Async Repository Review Workflow

*This homework turns Part III into a small continuous-AI system. The aim is not to create flashy autonomy for its own sake. The aim is to take the app and workflow you built in Parts I and II and operationalise them as a remote, recurring, reviewable automation with explicit guardrails.*

## Topic

**Create a report-first remote workflow that reviews your repository on a schedule.**

## Main Goal

Build a **remote, async, recurring AI workflow** that runs outside your local IDE session, reviews the repository, and returns a reviewable artefact with useful suggestions for improving the app. For the core assignment, keep it **report-first**: the workflow should not push code or merge anything. It should surface a small number of structured recommendations that a human can inspect and act on.

Choose one execution mode:

- a **hosted remote agent**; or
- a **CLI-capable agent running in GitHub Actions** or similar CI automation.

Either is acceptable. Hosted systems usually win on convenience; CLI-in-CI usually wins on environment control, permissions design, and observability.

## Core Exercise

Create a durable workflow brief such as `automation/repo-review.md` and define the system using the same Week 3 pattern from the reader: **Trigger**, **Environment**, **Context**, **Specification**, **Execution**, and **Output**. Treat that file as the handoff for unattended work, not as decorative documentation.

The workflow should review the repository through at least **four lenses** such as code quality, tests and verification gaps, documentation drift, dependency hygiene, UX/accessibility, or architecture and boundary issues. Even so, the output must stay bounded: return only the **top 3-5 prioritised suggestions**. For each suggestion, include a short title, the supporting evidence or file/path basis, why it matters, an effort or risk estimate, and the recommended next step such as ignore, backlog item, candidate spec, or low-risk automation candidate.

Use the context stack you already built. The workflow should refer to your `AGENTS.md` or equivalent rules, project Skills, reference docs, and relevant Week 2 specs and plans where appropriate. Add both a **manual trigger** for testing and a **recurring trigger** such as a nightly or weekly schedule. Keep the output safe and reviewable by returning one of the following: an issue, a PR comment, a structured Markdown report artefact, or another inspectable status surface.

Finally, define basic guardrails in the workflow brief or notes: an **iteration cap**, a **permission model**, an **idempotency rule**, one **approval boundary**, and two **success signals** you will watch. Then run the workflow at least once through the remote path, inspect the result, and tighten one thing based on what you learned. A good minimum repository shape would include the workflow definition, the durable brief, and short notes on guardrails and observed behaviour.

## Stretch Goal

Take one narrow review lane and turn it into an **effectful but bounded** async workflow that opens a **draft PR**. Good candidates include documentation refresh, low-risk dependency hygiene, changelog or README updates, generated repo-doc refreshes, or a narrow test-maintenance task.

Keep the lane low-risk, give it a clear trigger, require a short PR summary, and make the idempotency rule explicit so reruns do not create chaos. The strongest version uses a two-phase pattern: the core review workflow identifies a candidate, and a second workflow turns that candidate into a bounded draft PR.

## Advanced Stretch

For deeper experimentation, make the system **clearer, safer, more scalable, or more measurable**. You could chain several workflows together with explicit state surfaces, harden the security posture with least-privilege credentials and staged approvals, compare the same workflow in hosted and CLI-in-CI forms, add measurement such as acceptance rate or cost per useful run, split the work into several narrow remote lanes, or add a verifier workflow that judges the output of the effectful lane.

Choose one or two directions and treat the result like a small operating system for unattended work. The interesting question is not “can I automate more?” but “can I make recurring automation useful, reviewable, and trustworthy enough that I would actually leave it switched on?”

