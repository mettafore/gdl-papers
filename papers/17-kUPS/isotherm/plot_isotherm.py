"""Collect the p*bar.h5 runs into isotherm.csv and three figures.

Run from this folder:  uv run python plot_isotherm.py
"""

from pathlib import Path

import h5py
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from kups.application.mcmc.analysis import _analyze_single_system

N_CELLS = 27  # cell_replication: 3 -> 3x3x3 unit cells in the box
CELL_MASS = 1.66403  # kg/mol of one unit cell, C48H28O32Zr6
EV_TO_KJMOL = 96.485
TRIM = {10: 3000}  # production cycles discarded as still equilibrating (bar -> cycles)
FILE = {5: "p5bar_long.h5"}  # pressures re-run with a longer warm-up (bar -> file)
RUNTIME_S = {5: 4966}  # wall time of those re-runs (the long 5 bar run: 82.8 min)
BLUE, INK, MUTED, GRID = "#2a78d6", "#0b0b0b", "#898781", "#e1e0d9"

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
    }
)

here = Path(__file__).parent
timings = dict(line.split() for line in (here / "timings.tsv").read_text().splitlines())
runs = sorted(here.glob("p*bar.h5"), key=lambda p: float(p.stem[1:-3]))
runs = [here / FILE.get(float(p.stem[1:-3]), p.name) for p in runs]


def analyze(path, trim):
    """Block-average loading and heat of adsorption, skipping the first `trim` cycles."""
    with h5py.File(path) as f:
        counts = f["group.per_step/array.particle_count.data"][trim:]
        energy = f["group.per_step/array.systems.data.potential_energy"][trim:, 0]
        temperature = float(f["group.fixed/array.systems.data.temperature"][0])
    return _analyze_single_system(
        energy=jnp.asarray(energy),
        counts=jnp.asarray(counts),
        temperature=temperature,
        n_blocks=None,
    )


rows, traces = [], []
for path in runs:
    bar = float(path.stem.split("bar")[0][1:])
    trim = TRIM.get(bar, 0)
    res = analyze(path, trim)
    n, n_err = float(res.loading.mean[0]), float(res.loading.sem[0])
    with h5py.File(path) as f:
        trace = f["group.per_step/array.particle_count.data"][:, 0]
    kept = trace[trim:]
    half = len(kept) // 2
    rows.append(
        dict(
            bar=bar,
            per_cell=n / N_CELLS,
            per_cell_err=n_err / N_CELLS,
            mmol_g=n / N_CELLS / CELL_MASS,
            mmol_g_err=n_err / N_CELLS / CELL_MASS,
            qst=-float(res.heat_of_adsorption.mean[0]) * EV_TO_KJMOL,
            qst_err=float(res.heat_of_adsorption.sem[0]) * EV_TO_KJMOL,
            first_half=kept[:half].mean() / N_CELLS,
            second_half=kept[half:].mean() / N_CELLS,
            n_blocks=int(res.loading.n_blocks),
            runtime_s=RUNTIME_S.get(bar, int(timings.get(f"{bar:g}", -1))),
        )
    )
    traces.append((bar, trace / N_CELLS))

cols = list(rows[0])
with open(here / "isotherm.csv", "w") as f:
    f.write(",".join(cols) + "\n")
    for r in rows:
        f.write(",".join(f"{r[c]:.4g}" for c in cols) + "\n")

print(
    f"{'bar':>6} {'CO2/cell':>14} {'mmol/g':>14} {'Qst kJ/mol':>12} {'1st/2nd half':>14} {'blocks':>6} {'min':>5}"
)
for r in rows:
    print(
        f"{r['bar']:>6g} {r['per_cell']:>7.3f} ± {r['per_cell_err']:<5.3f}"
        f"{r['mmol_g']:>7.3f} ± {r['mmol_g_err']:<5.3f}{r['qst']:>6.1f} ± {r['qst_err']:<4.1f}"
        f"{r['first_half']:>7.2f}/{r['second_half']:<6.2f}{r['n_blocks']:>6}{r['runtime_s'] / 60:>6.1f}"
    )

P = np.array([r["bar"] for r in rows])
y, yerr = (
    np.array([r["mmol_g"] for r in rows]),
    np.array([r["mmol_g_err"] for r in rows]),
)

# 1. Isotherm: uptake vs pressure, linear and log pressure side by side.
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, log in zip(axes, (False, True), strict=True):
    ax.errorbar(
        P,
        y,
        yerr=yerr,
        color=BLUE,
        lw=2,
        marker="o",
        ms=7,
        capsize=3,
        mec="white",
        mew=1.5,
    )
    ax.set_xlabel("CO₂ pressure (bar)" + (", log scale" if log else ""))
    ax.set_ylabel("Uptake (mmol/g)")
    if log:
        ax.set_xscale("log")
fig.suptitle(
    "Simulated CO₂ isotherm, RUBTAK (UiO-66), 298 K", color=INK, x=0.06, ha="left"
)
fig.text(
    0.06,
    0.005,
    "kUPS rigid GCMC · TraPPE CO₂, UFF framework LJ + Ewald · "
    "1k warm-up + 10k cycles (5 bar: 5k + 20k; 10 bar: first 3k cycles discarded) · error bars = block-average SEM, smaller than markers",
    color=MUTED,
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(here / "isotherm.png", dpi=200)

# 2. Heat of adsorption vs loading (its own chart: different unit, no dual axis).
fig, ax = plt.subplots(figsize=(5.5, 4))
ax.errorbar(
    [r["per_cell"] for r in rows],
    [r["qst"] for r in rows],
    yerr=[r["qst_err"] for r in rows],
    color=BLUE,
    lw=2,
    marker="o",
    ms=7,
    capsize=3,
    mec="white",
    mew=1.5,
)
ax.set_xlabel("Loading (CO₂ per unit cell)")
ax.set_ylabel("Isosteric heat Qst (kJ/mol)")
ax.set_title("Binding strength vs loading", color=INK, loc="left")
fig.tight_layout()
fig.savefig(here / "qst.png", dpi=200)

# 3. Equilibration check: loading per production cycle, one panel per pressure.
fig, axes = plt.subplots(
    len(traces), 1, figsize=(8, 1.6 * len(traces)), sharex=True, squeeze=False
)
axes = axes[:, 0]
for ax, (bar, tr) in zip(axes, traces, strict=True):
    ax.plot(tr, color=BLUE, lw=0.8)
    ax.axhline(tr.mean(), color=INK, lw=1, ls="--")
    ax.set_ylabel(f"{bar:g} bar", rotation=0, ha="right", va="center")
axes[-1].set_xlabel("Production cycle")
fig.suptitle(
    "CO₂ per unit cell during production (flat = equilibrated)",
    color=INK,
    x=0.06,
    ha="left",
)
fig.tight_layout()
fig.savefig(here / "equilibration.png", dpi=200)
print("wrote isotherm.csv, isotherm.png, qst.png, equilibration.png")
