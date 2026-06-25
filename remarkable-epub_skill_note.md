## Note: journal-class LaTeX sources (REVTeX, IEEEtran, etc.)

This skill uses pandoc, which reads `.tex` directly without needing the
document class installed and produces reflowable EPUB with no fixed page — so
the page-geometry problems that affect the PDF (reMarkable-paper) skill do not
apply here.

The one caveat: pandoc's LaTeX reader handles standard macros but silently
drops or garbles class-specific ones. For a REVTeX / two-column journal source,
the front matter is the risk area — `\affiliation`, the repeated `\author{}`
blocks, `\maketitle` ordering, and `\widetext` may come out missing or
scrambled. After conversion, spot-check the title/author/abstract region and,
if mangled, lightly pre-clean the source (flatten the author block to plain
text, drop `\widetext`/`\onecolumngrid`) before re-running pandoc.
