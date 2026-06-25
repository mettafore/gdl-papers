# remarkable-paper skill — update

Two files, ready to drop into your installed `remarkable-paper` skill,
replacing the existing ones:

- `SKILL.md`            → the skill root
- `scripts/remarkabilize.py` → the skill's `scripts/` folder

I can't write to the installed skill from a Cowork session (the skill cache is
read-only here). To install, open **Settings → Capabilities**, find the
`remarkable-paper` skill, and replace its `SKILL.md` and
`scripts/remarkabilize.py` with these copies.

## What changed and why

The skill kept shipping PDFs with tables clipped at the margins. Two root
causes, now addressed:

1. **`adjustwidth` margin bleed.** Papers wrap wide tables in
   `\begin{adjustwidth}{-2.5cm}{-2.5cm}` to push them into the *wide* page's
   margins. On the narrow reMarkable page that runs them off both edges.
   The script now injects a **build-scoped** `\renewenvironment{adjustwidth}`
   that zeroes the bleed — only in the generated `_remarkable.tex`, with no
   edits to shared content files and no effect on the original build. Guarded
   so it's a no-op when the paper doesn't use `adjustwidth`.

2. **Fixed `\scalebox{0.x}` tables.** A scale factor tuned for the old page
   still overflows the narrow one. The script can't safely auto-rewrite these
   (the same wrapper is used for figures), so instead it now **scans and
   reports** every `\scalebox`/`adjustwidth` location, flags `Overfull \hbox`
   after compile, and SKILL.md documents the manual fix: the shrink-to-fit
   `\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{ ... }`
   idiom (use `\textwidth`, not `\linewidth`).

3. **Verification hardened.** SKILL.md step 4 now requires rendering *every*
   table/figure page (with a snippet to find them) and checking **both**
   margins for clipping — not eyeballing three sample pages.
