# Intro theme

A 16.5 s title sequence (1920×1080, 60 fps) with an original synthesised score.
Not part of the research artifact — branding only.

Two looks share one storyboard:

- **classic** (`theme.html`) — gold and pale blue on near-black, restrained.
- **vivid** (`theme-vivid.html`) — iridescent cyan→violet→magenta→gold palette,
  aurora background, motion streaks, lens streak and chromatic aberration, light
  rays on the downbeat; the score gains an arpeggio, kick and hats.

| Time | Picture | Reference |
|---|---|---|
| 0.0 | a single point of light, then a burst into a point cloud | raw data, `x ∈ ℝ³` |
| 3.0 | the cloud condenses into a molecular graph; three rounds of messages travel the bonds | EGNN / message passing, `f(Rx) = R f(x)` |
| 6.0 | the atoms spread over a sphere and flow along geodesics into a data distribution | Riemannian Flow Matching on S² |
| 9.0 | the sphere folds into real spherical harmonics (Y₂⁰ → Y₃²), coloured by sign | irreps / MACE |
| 12.0 | thirteen circles and every line between their centres | symmetry itself (D₆) |
| 13.5 | emblem and title | |

## Files

- `theme.html`, `theme-vivid.html` — the whole animation; every frame is a pure function of time.
  Open `theme.html?play=1` in a browser for a live preview.
  Query options: `name=…`, `tag=…`, `title=0` (no name/tagline).
- `render.mjs` — renders frames with headless Chromium (Playwright).
- `music.py` — synthesises the score (numpy + scipy, no samples). 80 BPM, D lydian.
- `build.sh` — frames + score → the videos in `out/`.
- `fonts/` — Cormorant Garamond (SIL OFL 1.1, licence alongside).

## Build

```bash
./build.sh            # classic; needs node + playwright, python3 + numpy/scipy, ffmpeg
./build.sh vivid
```

Outputs (not versioned): `out/intro-theme.mp4` (titled), `out/intro-theme-clean.mp4` (no name,
for overlaying your own title), `out/intro-theme-sting.mp4` (6 s short cut),
`out/intro-theme.wav` (score only); the vivid build writes the same set as
`out/intro-theme-vivid*`.
