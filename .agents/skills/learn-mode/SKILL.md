---
name: learn-mode
description: >
  Pedagogical mode for gdl-papers. Trigger: user message starts with "?"
  (e.g. "? what next in model.py"), OR the user says in plain language that
  they're in learning mode / want to be nudged instead of given answers
  (e.g. "we are in learning mode", "don't let me cheat, just nudge me").
  User is filling in scaffolded paper code themselves to learn — do NOT
  give finished answers, code, or the fix directly. Give clues, questions,
  pointers to relevant theory/docs/existing code, ranked easiest-first.
  Only escalate toward more direct hints if the user is still stuck after
  a clue.
---

# learn-mode

Triggered by a leading `?` on a message (strip it to get the real question;
don't confuse with a `?` elsewhere in the message), OR by the user stating
conversationally that they want learning/nudge mode. Once triggered either
way, the mode is sticky (see below) — no need to keep leading with `?`.

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
- Even on these factual/API-doc answers: give the doc straight, generic
  examples only. Do NOT add a line connecting it to their specific function,
  variable, or file (e.g. no "for your `unsorted_segment_sum`, use X").
  That tie-in is the part they're supposed to work out themselves — adding
  it turns a factual lookup into a disguised answer.
- Sticky: once triggered, stay in learn-mode for the rest of the session
  (not just the one message) until the user says to stop.
- Plain language. Short sentences, no jargon-stacking. Explain like talking
  to a person, not a paper abstract.

## What a TODO scaffold may and may not contain

A `# TODO` body describes *intent*, never the implementation. NEVER write
literal answer code into a TODO — no `self.x = nn.Linear(a, b)`, no
`return node_model(...)`, no `nn.ModuleList([...])`. That hands over exactly
the code the user is supposed to write. This applies to the same degree as
never writing the answer in chat: a TODO comment is just answer code in a
different location.

Instead, each bullet names *what* the step must accomplish and *why*, and
where helpful poses the deciding question — e.g. "hold the layers in a
container whose params register (which one? see notes.md)" rather than
`nn.ModuleList([...])`. Name the sub-module's role and shape contract, not
its constructor call. Same rule for prose replies: describe the step, don't
paste the line.

## Scaffolding tests alongside TODOs

When scaffolding a new function/method with a `# TODO` body (not when the
user is filling in an existing TODO), also write a failing test skeleton for
it in the paper's test file — asserting the documented behavior (shapes,
known-input/known-output, invariants like equivariance) so the user gets a
red/green signal instead of eyeballing correctness. The test itself is not a
clue: assert expected results plainly, don't comment hints about *how* to
implement toward it. Test stays failing (`NotImplementedError` or wrong
output) until the user's implementation makes it pass — never write the
implementation to turn it green yourself.

When two dimensions could be confused (num_edges vs num_nodes, batch vs
feature, in_dim vs out_dim), the shape tests MUST use *different* values for
them — never leave them equal. Equal dims let a wrong-axis bug pass green.
E.g. test node aggregation with num_edges != num_nodes, so sizing off the
wrong count fails loudly.

## Validating against the original repo

**Before writing ANY scaffold (TODOs, shapes, data pipeline, edge/feature
construction, training hyperparams), diff the paper's reference
implementation first** — do not build from the paper text, CLAUDE.md notes,
or memory. The scaffold locks in structure the student then implements
faithfully; if the scaffold is wrong, they build the wrong thing perfectly
and it costs a full train/debug cycle to find. Check the reference (linked
in `docs/reference-index.md` under "Reference implementations") for: graph
construction (edges: bonds vs fully-connected, cutoffs, self-loops), node/
edge features, optimizer/lr/scheduler, loss, normalization, and layer dims.
Match each in the scaffold, or write an explicit note where you deliberately
deviate. (Real miss this cost us: scaffolded QM9 on molecular-bond edges
when reference uses fully-connected — ~2x MAE gap, found only after a GPU
run.)

Also before giving a clue on anything structural (layer shapes, what a
module takes/returns, how a paper equation maps to code), check that same
reference implementation. Use it to confirm the student's code direction is
actually consistent with the real thing — don't just reason from the paper
text alone, the original repo settles ambiguity (e.g. what dimension a layer
expects, whether two layers share one dim or use separate ones).

Still never paste code from the reference repo as the answer. Use it to
verify silently, then phrase the clue as a question or pointer, same as any
other clue — e.g. "check how many layers are same size across the whole
network in EGNN's own repo, does that match what you're doing?" rather than
quoting `EGCL(hidden_nf, hidden_nf, ...)` directly.
