---
name: remarkable-paper
description: >-
  Reformat a LaTeX paper (e.g. an arXiv source) into a reMarkable 2-friendly
  PDF: a 3:4 portrait page that fills the device screen with no panning or
  zooming, single column, slim margins, slightly enlarged text. Use whenever
  the user wants to read a paper comfortably on a reMarkable (or similar
  e-ink tablet). Triggers include: "make this reMarkable friendly",
  "reMarkable 2 PDF", "reformat this paper for my reMarkable", "make this
  readable on e-ink", "remarkabilize this paper".
---

# reMarkable-friendly paper reformatting

## Goal

Take a LaTeX paper source and produce a variant PDF sized for the
**reMarkable 2** screen — 1404 × 1872 px, a **3:4 portrait** aspect ratio
(ratio 1.3333). Matching that ratio means the whole page fills the display
with no panning or pinch-zoom, and a normal 10–11 pt body appears slightly
enlarged once scaled to the device width. The result is a single-column,
slim-margin reading copy.

This requires the **LaTeX source**, not just a PDF. If the user only has a
PDF, point them to the arXiv "Other formats → source" download, or tell them
this skill needs the `.tex`. (Re-laying-out an existing PDF is a different,
lossy job — cropping margins with `pdfcrop`/`k2pdfopt` — not covered here.)

## The narrowing trap (read this first)

Shrinking the page width is the whole point of this skill, but it is also
where it breaks. Anything in the paper whose width was hand-tuned for the
**original wide page** — full-width tables, `\scalebox{0.x}{...}` wrappers,
`\begin{adjustwidth}{-Xcm}{-Xcm}` margin bleeds, wide figures, fixed-width
minipages — does **not** reflow. It keeps its absolute width and runs off the
narrow page, usually clipped at *both* margins (so it is easy to miss if you
only glance at the middle of a page). Tables are the #1 offender. Treat
"did any table get clipped?" as a first-class part of the job, not an
afterthought.

## Workflow

1. **Locate the main `.tex`** and its directory. Note any precompiled
   `main.bbl`, the `graphics/` folder, and custom `.sty`/`.cls` files — the
   build must run in that directory so they resolve.

2. **Run the helper script** to produce a `<stem>_remarkable.tex` next to the
   original, with reMarkable geometry injected and the common pitfalls fixed:

   ```bash
   python3 scripts/remarkabilize.py /path/to/main.tex --compile
   ```

   The script:
   - strips CR / CRLF line endings from `.tex`, `.sty`, `.cls`, `.bbl`
     (old arXiv sources often have them, which breaks `\usepackage` lines);
   - injects a geometry override just before `\begin{document}`:
     ```latex
     \usepackage{geometry}
     \geometry{paperwidth=146mm,paperheight=194.67mm}
     \AtBeginDocument{\newgeometry{top=10mm,bottom=12mm,left=11mm,right=11mm,headsep=5pt,footskip=14pt}}
     ```
     (`\geometry` sets the paper size in the preamble; the `\AtBeginDocument`
     hook runs **after** any class hook — e.g. NeurIPS/NIPS sets its own
     `\newgeometry` at begin-document — so this wins and reclaims the margins.)
   - injects a **build-scoped `adjustwidth` neutralizer**: many papers wrap
     wide tables in `\begin{adjustwidth}{-2.5cm}{-2.5cm}` to bleed into the
     wide page's margins; on the narrow page that runs them off both edges.
     The injected `\renewenvironment{adjustwidth}` makes the bleed a no-op
     **only in the generated file** (it does not touch the shared content
     `.tex` files or the original build). It is guarded, so it does nothing
     if the paper never loads `adjustwidth`/`changepage`.
   - copies `<stem>.bbl` → `<newstem>.bbl` so citations resolve (a precompiled
     `.bbl` must match the jobname);
   - **scans for wide-table risks** (`\scalebox` around tables, `adjustwidth`)
     and prints their file:line locations to inspect;
   - with `--compile`, runs `pdflatex` twice, reports errors, and flags
     `Overfull \hbox > 2pt` warnings (the signature of a table running wide).

   Flags: `--width-mm`, `--margin`, `--top`, `--bottom` to tune the layout.
   Height is always derived as `width × 4/3` to keep the exact 3:4 ratio.

3. **Resolve remaining paper-specific issues** — these need judgment, not the
   script. Iterate `pdflatex` until clean *and* nothing is clipped.

   **Compile errors:**
   - **Missing package** (e.g. `siunitx.sty not found`): if the package is
     loaded but never actually used (`grep` for its commands like `\SI`,
     `\num`, `\si`; or `algorithm`/`algpseudocode` with no `\begin{algorithm}`),
     comment out the `\usepackage` line. If a `siunitx` `S` table column is
     used, change the `\begin{tabular}{ ... S S S }` spec to `r r r`.
   - **`Illegal pream-token (S)`**: same `S`-column issue as above.
   - **Undefined citations / `?` marks**: the `.bbl` jobname didn't match —
     confirm `<newstem>.bbl` exists, or run `bibtex <newstem>` if there's a
     `.bib`.

   **Wide tables / figures clipped at the margins** (do NOT skip — see the
   narrowing trap above). The script already neutralized `adjustwidth`; what
   remains is content with a baked-in width:
   - A fixed `\scalebox{0.6}{ \begin{tabular}...\end{tabular} }` was tuned for
     the old page and still overflows. Replace the wrapper with a
     **shrink-to-fit** `\resizebox`:
     ```latex
     \resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{ ... }
     ```
     This scales any over-wide box down to exactly the text width and leaves
     boxes that already fit untouched. Use `\textwidth` (a fixed document
     dimension), **not** `\linewidth` — inside `adjustwidth`/list contexts
     `\linewidth` can be inflated, so the test silently fails to shrink.
   - If you see a residual `\begin{adjustwidth}{-Xcm}{-Xcm}` you want to fix in
     the source itself, zero the two arguments (`{0pt}{0pt}`).
   - For an over-wide `\includegraphics`, add/clamp `width=\textwidth`.
   - These tables usually live in `\input`-ed files (e.g. `content/5_*.tex`),
     not the main file — `grep -rn "scalebox\|adjustwidth\|tabular" content/`.
   - Editing a shared content file changes the original build too. That is
     normally harmless (shrink-to-fit and zeroed bleed both look fine on the
     wide page). If you must keep the original pristine, prefer the
     preamble-override approach the script uses for `adjustwidth`.

4. **Verify** before delivering — this step is mandatory and the most common
   place this skill goes wrong, so be thorough:
   - **Page size**: `pdfinfo <pdf> | grep -i "page size"` → width:height ≈
     1:1.333 (e.g. `413.86 x 551.82 pts`).
   - **Compile log**: no `Overfull \hbox > 2pt` (the script lists them). A wide
     overfull box almost always means a clipped table or figure.
   - **Render EVERY table and figure page — not a sample.** Find them and
     render each:
     ```bash
     # list pages containing tables/figures
     for p in $(seq 1 $(pdfinfo f.pdf | awk '/Pages/{print $2}')); do
       pdftotext -layout -f $p -l $p f.pdf - 2>/dev/null \
         | grep -qE "Table [0-9]+:|Figure [0-9]+:" && echo "page $p"
     done
     # then: pdftoppm -png -r 110 -f <p> -l <p> f.pdf prev_<p>
     ```
     Open each rendered page and check **both the left and right margins** for
     columns running off the edge (the title row and the last numeric column
     are where clipping shows first). Also confirm single-column flow and that
     figures aren't overflowing. Only when every table/figure page is clean do
     you move on.
   - For a long paper, do this in batches; do not declare done after eyeballing
     three pages.

5. **Deliver** the PDF and the modified `.tex` to the user's folder and
   present them.

## Defaults and tuning

- Default page: **146 mm × 194.67 mm** (exact 3:4), margins 10–12 mm. This
  scales ~1.07× to the 157 mm physical screen width, gently enlarging text.
- Want bigger text? **Narrow** the page (`--width-mm 132`) — the device
  scales the narrower page up more. (Note: narrowing also makes wide tables
  more likely to need the shrink-to-fit `\resizebox` from step 3.)
- Want more annotation room? Raise `--margin` (e.g. `--margin 16`).
- The paper stays single-column if it already is. Two-column classes
  (`\documentclass[twocolumn]` or `IEEEtran`) read badly on e-ink — switch to
  one column (`\documentclass{...}` without `twocolumn`, or `\onecolumn`)
  before applying geometry.
