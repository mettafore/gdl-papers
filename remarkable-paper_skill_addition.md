## Route by document class FIRST (before any compile)

Grep the `\documentclass` line and pick the route before injecting geometry.
Skipping this is the #1 time-sink — geometry-injecting a REVTeX paper fails
three different ways before you realize it needed conversion.

| Detected `\documentclass`                          | Route                                   | Why |
|----------------------------------------------------|-----------------------------------------|-----|
| `article`, `scrartcl`, `amsart` (single-column)    | Inject geometry directly                | Cooperates with `geometry`; already reflows |
| `revtex4-1` / `revtex4-2` (`aps`,`prl`,`reprint`)  | **Convert to `article`** + shims        | REVTeX breaks `geometry` (Extra `\endgroup`), old class clashes with the modern LaTeX kernel hook system, and it reasserts its own width so injected size is ignored (title clips off-page) |
| `IEEEtran`, `acmart`, two-column `article`         | Strip to one column, then geometry      | Two columns read badly on e-ink |
| `elsarticle`, `llncs`, other one-column classes    | Inject geometry; shim only if it fights | Usually fine — verify page size took effect |

**REVTeX → article recipe:** swap to `\documentclass[11pt]{article}`; add
`amsmath`,`amssymb` explicitly (REVTeX supplied them via class options);
accumulate repeated `\author`/`\affiliation` into one centered title block,
redefine `\maketitle`, and move it *before* `\begin{abstract}`; add no-op
shims for `\widetext`/`\onecolumngrid`/`\squeezetable`. The REVTeX `.bbl` is
usually self-contained (it `\providecommand`s `\bibinfo`/`\eprint`/`\url`) and
compiles in `article` unchanged.

**Offline note:** only github.com (git clone), pypi, and pythonhosted are
reachable; CTAN and CDNs are blocked. Prefer a shim over a download for a
missing package (e.g. `dsfont` unavailable → `\providecommand{\mathds}{\mathbb}`).
If a class genuinely must be fetched, `git clone` a TeX texmf-tree mirror from
github and point `TEXINPUTS` at it — but for REVTeX, converting beats fetching.

**Apply the narrowing robustness pack upfront, not reactively:** load `xurl`
(break long URLs), `seqsplit` (break long hashes/tokens), `microtype`, and
`\sloppy`; auto-wrap every `tabular` not already in a `\resizebox` with
shrink-to-fit; wrap any over-wide display equation in
`\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{$\displaystyle ... $}`.
