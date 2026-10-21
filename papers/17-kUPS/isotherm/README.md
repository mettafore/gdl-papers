# CO₂ isotherm in UiO-66 with kUPS: run files

The files behind the isotherm in the [kUPS page](../README.md#appendix-co2-isotherm-in-uio-66).

| File | What it is |
|---|---|
| `template.yaml` | kUPS rigid-GCMC config (CO₂ in RUBTAK = UiO-66, 298.15 K, 3×3×3 cells, 1,000 warm-up + 10,000 production cycles), with `PRESSURE_PA` and `OUT_FILE` as blanks |
| `run_isotherm.sh` | Runs one GCMC simulation per pressure (0.01–10 bar), writing a `.h5` result and a `.log` per pressure and wall times to `timings.tsv` |
| `plot_isotherm.py` | Block-averages each run, writes `isotherm.csv` and the plots |
| `isotherm.csv` | Final results: uptake per unit cell and in mmol/g, heat of adsorption, equilibration check, run time |

## Reproduce

1. Clone [kUPS](https://github.com/cusp-ai-oss/kUPS) and check out commit `3af6439`, then run `uv sync`.
2. Make a folder inside the checkout, e.g. `runs/isotherm/`, and copy these files into it, together with kUPS's `examples/adsorbate`, `examples/host` and `examples/lennard_jones` folders.
3. Run `./run_isotherm.sh` (about 3 hours on an M1 CPU), then `uv run python plot_isotherm.py`.

## Notes

- 5 bar was rerun with 5,000 warm-up and 20,000 production cycles (as `p5bar_long.h5`), and at 10 bar the first 3,000 production cycles are discarded, because those runs were still equilibrating. `plot_isotherm.py` applies both through its `FILE` and `TRIM` settings; the rerun itself is not part of `run_isotherm.sh`.
- `plot_isotherm.py` uses a private kUPS function (`_analyze_single_system`), so it may need updating for newer kUPS versions.
- Classical force-field result (TraPPE CO₂, UFF framework Lennard-Jones plus Ewald), not validated against experiment.
