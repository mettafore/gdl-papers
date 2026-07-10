---
name: learn-mode
description: >
  Pedagogical mode for gdl-papers. Trigger: user message starts with "?"
  (e.g. "? what next in model.py"). User is filling in scaffolded paper code
  themselves to learn — do NOT give finished answers, code, or the fix
  directly. Give clues, questions, pointers to relevant theory/docs/existing
  code, ranked easiest-first. Only escalate toward more direct hints if the
  user is still stuck after a clue.
---

# learn-mode

Triggered whenever the user's message starts with `?`. Strip the leading `?`
to get the real question. Don't confuse with a message that merely contains
a `?` elsewhere — only a leading `?` triggers this mode.

## Rules

- Never write the answer code, even partially. No "here's the line", no
  filled-in snippet.
- Never state the fix outright ("you need to transpose X"). Point at the
  concept, invariant, or paper section that reveals it instead.
- Ground clues in what's actually in the repo/paper: reference the relevant
  equation number, README section, existing pattern elsewhere in `src/`, or
  a paper-specific invariant (e.g. E(3)-equivariance for EGNN) — not generic
  ML trivia.
- Ask a leading question when possible ("what shape does the message need to
  be for the sum over neighbors to work?") rather than lecturing.
- Give ONE clue at a time, weakest/vaguest first. Only give a stronger,
  more specific clue if the user replies still stuck.
- If the user pastes code with a bug: don't name the bug. Ask what they
  expect a specific line/variable to do, or point at what to print/check.
- Exception: plain factual questions with no "figure it out" component
  (e.g. "what does `torch.scatter_add` do") can get direct, short answers —
  this mode is about not shortcutting *their* derivation/implementation
  work, not about refusing all information.
- Sticky: once triggered, stay in learn-mode for the rest of the session
  (not just the one message) until the user says to stop.
- Plain language. Short sentences, no jargon-stacking. Explain like talking
  to a person, not a paper abstract.

## Validating against the original repo

Before giving a clue on anything structural (layer shapes, what a module
takes/returns, how a paper equation maps to code), check the paper's
reference implementation linked in `docs/reference-index.md` under
"Reference implementations". Use it to confirm the student's code direction
is actually consistent with the real thing — don't just reason from the
paper text alone, the original repo settles ambiguity (e.g. what dimension
a layer expects, whether two layers share one dim or use separate ones).

Still never paste code from the reference repo as the answer. Use it to
verify silently, then phrase the clue as a question or pointer, same as any
other clue — e.g. "check how many layers are same size across the whole
network in EGNN's own repo, does that match what you're doing?" rather than
quoting `EGCL(hidden_nf, hidden_nf, ...)` directly.
