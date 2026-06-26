# Validation — fresh-session checks

Purpose: prove the **AI operating surface works** — that an agent with *no* build
context can operate this repo using only its files (AGENTS.md, skills, MCP, docs).
This is distinct from `docs/spec.md`'s Verification (which tests that the *code*
runs). Here we test whether the *context stack* onboards a cold agent.

Run this in a **fresh Claude Code session** (skills only load at session start, and
a fresh agent can't "cheat" from build memory).

---

## Handoff prompt — paste this into the new session

> You're in the `gdl-papers` repo. I'm validating its AI operating surface — answer
> each question **using only the repo's files** (AGENTS.md, `.agents/skills/`, `docs/`),
> not prior knowledge or guesses. Cite where each answer comes from.
>
> 1. Which package manager and run command does this repo use?
> 2. Where do the durable docs live (scope/contracts, doc index)?
> 3. Which skill runs a paper's train/eval, and how is it invoked?
> 4. Which MCP server is available, and what for?
> 5. What is out of bounds (the "never" rules)?
> 6. How do I add a new paper?
>
> Then run one real workflow:
> - Say "train 00-template" — does the **`run-paper`** skill activate and run training?
> - Ask "how is this repo laid out / how do I add a paper?" — does the **`gdl-repo-map`**
>   skill activate and answer (vs the model answering generically without the skill)?
> - Confirm the **context7** MCP is reachable (a quick library-doc lookup).
>
> Record the results in `docs/validation.md` (the tables below). If any answer
> wobbles or a skill doesn't fire, note it, fix the surface once, and record what
> changed.

---

## Results (fill in during the fresh session)

### Surface questions
| Question | Agent's answer | From where | Correct? |
|---|---|---|---|
| Package manager + run cmd | | | |
| Durable docs location | | | |
| Skill for train/eval | | | |
| MCP available | | | |
| Out of bounds ("never") | | | |
| How to add a paper | | | |

### Real workflow
| Step | Result |
|---|---|
| "train 00-template" → **run-paper** skill activates + runs? | ☐ yes ☐ no — |
| "how to add a paper?" → **gdl-repo-map** skill activates + answers? | ☐ yes ☐ no — |
| context7 MCP reachable? | ☐ yes ☐ no — |

### Wobbles + fixes
- (none) — or: what failed, what was changed, what improved.

---

*Run date: ___  ·  Session: fresh*
