#!/usr/bin/env python3
"""
remarkabilize.py -- turn a LaTeX paper into a reMarkable 2-friendly variant.

Produces <stem>_remarkable.tex next to the original main file, with:
  * CR / CRLF line endings stripped from .tex/.sty/.cls/.bbl in the source dir
  * a reMarkable geometry override injected before \\begin{document}
  * an \\adjustwidth margin-bleed neutralizer injected (build-scoped) so wide
    tables stop running off both edges of the narrow page
  * the precompiled .bbl (if any) copied to the new jobname
Optionally compiles it with pdflatex (twice), reports the result, and flags
likely table/figure overflow (Overfull \\hbox + wide-table scan).

The reMarkable 2 screen is 1404x1872 px -> 3:4 portrait (ratio 1.3333).
Page height is always derived as width * 4/3 to match that exactly.

Usage:
  python3 remarkabilize.py path/to/main.tex [--compile]
        [--width-mm 146] [--margin 11] [--top 10] [--bottom 12]

Paper-specific compile errors (missing/unused packages, siunitx `S` table
columns, etc.) are NOT auto-fixed -- inspect the log and edit by hand. See
SKILL.md for the common ones. Wide tables that still overflow after the
adjustwidth neutralizer need the shrink-to-fit \\resizebox idiom -- the script
prints a warning pointing you at them; SKILL.md step 3 has the exact fix.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

GEOMETRY_TEMPLATE = r"""
%% ===================== reMarkable 2 friendly layout =====================
%% reMarkable 2 screen: 1404x1872 px, 3:4 portrait (ratio 1.3333).
%% Page matches that aspect ratio with slim margins so it fills the screen
%% with no panning; \AtBeginDocument runs after any class hook so it wins.
\usepackage{geometry}
\geometry{paperwidth=@@WIDTH@@mm,paperheight=@@HEIGHT@@mm}
\AtBeginDocument{\newgeometry{top=@@TOP@@mm,bottom=@@BOTTOM@@mm,left=@@LEFT@@mm,right=@@RIGHT@@mm,headsep=5pt,footskip=14pt}}

%% --- wide-table margin-bleed neutralizer (build-scoped, no content edits) ---
%% Many papers wrap wide tables in \begin{adjustwidth}{-Xcm}{-Xcm} to push
%% them into the (wide-page) margins. On the narrow reMarkable page that bleed
%% runs the table off BOTH edges. Redefine the environment to a no-op shift so
%% the content is still typeset, just not pushed out. Guarded so it is a no-op
%% if the paper never loads adjustwidth/changepage.
\makeatletter
\AtBeginDocument{%
  \@ifundefined{adjustwidth}{}{\renewenvironment{adjustwidth}[2]{}{}}%
}
\makeatother
%% ========================================================================

"""

EXTS_TO_DOS2UNIX = (".tex", ".sty", ".cls", ".bbl", ".bib")

# The shrink-to-fit idiom the agent should apply by hand to any wide table that
# still overflows (printed in the warning for convenience).
SHRINK_IDIOM = (
    r"\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{ ... }"
)


def strip_cr_in_dir(d):
    fixed = []
    for name in os.listdir(d):
        if name.lower().endswith(EXTS_TO_DOS2UNIX):
            p = os.path.join(d, name)
            try:
                with open(p, "rb") as f:
                    data = f.read()
                if b"\r" in data:
                    with open(p, "wb") as f:
                        f.write(data.replace(b"\r\n", b"\n").replace(b"\r", b"\n"))
                    fixed.append(name)
            except OSError:
                pass
    return fixed


def _iter_tex(src_dir):
    for root, _dirs, names in os.walk(src_dir):
        for name in names:
            if name.lower().endswith(".tex"):
                yield os.path.join(root, name)


def scan_wide_tables(src_dir):
    """Report constructs that commonly overflow the narrow page so the agent
    knows exactly which table pages to inspect and fix. Read-only; no edits."""
    # fixed-factor table scaling that was tuned for the wide page
    scalebox_tab = re.compile(r"\\scalebox\{[0-9.]+\}\{")
    adjustwidth = re.compile(r"\\begin\{adjustwidth\}\{\s*-?[0-9.]+\s*[a-z]*\}")
    hits = []
    for path in _iter_tex(src_dir):
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except OSError:
            continue
        for i, line in enumerate(lines, 1):
            stripped = line.lstrip()
            if stripped.startswith("%"):
                continue  # commented-out, ignore
            if scalebox_tab.search(line):
                hits.append((path, i, "scalebox (fixed-scale table)"))
            elif adjustwidth.search(line):
                hits.append((path, i, "adjustwidth (margin bleed)"))
    if hits:
        print("\n*** WIDE-TABLE WATCH ***")
        print("These constructs were sized for the original wide page and are")
        print("the usual cause of tables clipped at the margins. The injected")
        print("adjustwidth neutralizer fixes the bleed; a fixed \\scalebox can")
        print("still overflow. After compiling, RENDER EVERY ONE of these table")
        print("pages and check BOTH the left and right margins for clipped")
        print("columns. If a table is clipped, replace its wrapper with:")
        print(f"    {SHRINK_IDIOM}")
        print("(shrinks an over-wide table to the text width, leaves narrow")
        print("ones untouched). Locations:")
        for path, ln, what in hits:
            rel = os.path.relpath(path, src_dir)
            print(f"    {rel}:{ln}  {what}")
        print("*** end wide-table watch ***\n")
    return hits


def report_overfull(src_dir, new_stem):
    """Surface Overfull \\hbox warnings -- horizontal overflow that the eye
    might miss in a quick skim, and the signature of a table running wide."""
    log = os.path.join(src_dir, new_stem + ".log")
    if not os.path.isfile(log):
        return
    with open(log, encoding="utf-8", errors="replace") as f:
        text = f.read()
    # Only flag meaningfully-wide overflow (>2pt); sub-pt is cosmetic.
    big = []
    for m in re.finditer(r"Overfull \\hbox \(([0-9.]+)pt too wide\)(.*)", text):
        pt = float(m.group(1))
        if pt > 2.0:
            big.append((pt, m.group(2).strip()))
    if big:
        big.sort(reverse=True)
        print(f"\n*** {len(big)} Overfull \\hbox > 2pt -- possible table/figure "
              f"overflow; inspect those pages ***")
        for pt, ctx in big[:12]:
            print(f"    {pt:7.1f}pt  {ctx}")
        print("(If these are tables, apply the shrink-to-fit \\resizebox idiom.)\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("main_tex", help="path to the main .tex file")
    ap.add_argument("--width-mm", type=float, default=146.0)
    ap.add_argument("--margin", type=int, default=11, help="left/right margin mm")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--bottom", type=int, default=12)
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    main_tex = os.path.abspath(args.main_tex)
    if not os.path.isfile(main_tex):
        sys.exit(f"not found: {main_tex}")
    src_dir = os.path.dirname(main_tex)
    stem = os.path.splitext(os.path.basename(main_tex))[0]
    new_stem = f"{stem}_remarkable"
    new_tex = os.path.join(src_dir, new_stem + ".tex")

    width = args.width_mm
    height = width * 4.0 / 3.0

    fixed = strip_cr_in_dir(src_dir)
    if fixed:
        print(f"stripped CR from: {', '.join(fixed)}")

    with open(main_tex, encoding="utf-8", errors="replace") as f:
        src = f.read()

    if r"\begin{document}" not in src:
        sys.exit("no \\begin{document} found -- is this the main file?")

    block = GEOMETRY_TEMPLATE
    for token, value in (
        ("@@WIDTH@@", f"{width:.3f}"),
        ("@@HEIGHT@@", f"{height:.3f}"),
        ("@@TOP@@", str(args.top)),
        ("@@BOTTOM@@", str(args.bottom)),
        ("@@LEFT@@", str(args.margin)),
        ("@@RIGHT@@", str(args.margin)),
    ):
        block = block.replace(token, value)
    out = src.replace(r"\begin{document}", block + r"\begin{document}", 1)
    with open(new_tex, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"wrote {new_tex}  (page {width:.1f} x {height:.1f} mm, 3:4)")

    bbl = os.path.join(src_dir, stem + ".bbl")
    if os.path.isfile(bbl):
        shutil.copyfile(bbl, os.path.join(src_dir, new_stem + ".bbl"))
        print(f"copied {stem}.bbl -> {new_stem}.bbl")

    # Always surface wide-table risks so they get verified, compile or not.
    scan_wide_tables(src_dir)

    if not args.compile:
        print("\nNext: cd into the source dir and run pdflatex twice, e.g.\n"
              f"  cd '{src_dir}' && pdflatex {new_stem} && pdflatex {new_stem}")
        return

    if shutil.which("pdflatex") is None:
        sys.exit("pdflatex not on PATH; compile manually.")
    rc = 0
    for i in range(2):
        rc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", new_stem + ".tex"],
            cwd=src_dir,
        ).returncode
    pdf = os.path.join(src_dir, new_stem + ".pdf")
    if rc == 0 and os.path.isfile(pdf):
        print(f"\nOK -> {pdf}")
        if shutil.which("pdfinfo"):
            subprocess.run(f"pdfinfo '{pdf}' | grep -i 'page size\\|pages:'", shell=True)
        report_overfull(src_dir, new_stem)
        print("VERIFY: render EVERY table/figure page (not just a sample) and "
              "check both margins for clipping. See SKILL.md step 4.")
    else:
        print("\npdflatex reported errors -- see the .log; fix paper-specific "
              "issues (missing/unused packages, siunitx S columns) and rerun.")
        sys.exit(rc or 1)


if __name__ == "__main__":
    main()
