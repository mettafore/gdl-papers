# MACE author-code audit

Audited on 2026-10-06 against ACEsuit/mace v0.2.0, commit
`b3e3b2c6c9ac89dbc375905740e15e1e5b1ad721`. The comparison targets
`ScaleShiftMACE` with first `RealAgnosticInteractionBlock`, second
`RealAgnosticResidualInteractionBlock`, two interactions, correlation 3,
internal ell 3, and hidden angular orders through L=2.

## Corrections applied

| Area | Correction | Reference |
|---|---|---|
| First interaction | Add scalar sender mixing, learned A projection, division by training neighbour average, and receiver-element selector | [blocks.py](https://github.com/ACEsuit/mace/blob/b3e3b2c6c9ac89dbc375905740e15e1e5b1ad721/mace/modules/blocks.py) |
| First radial output | Interpret MLP weights in harmonic-order/channel order, matching the reference tensor product | [irreps_tools.py](https://github.com/ACEsuit/mace/blob/b3e3b2c6c9ac89dbc375905740e15e1e5b1ad721/mace/modules/irreps_tools.py) |
| Initial embedding | Use e3nn Linear, including its normalized normal initialization, rather than PyTorch Linear's uniform initialization | [blocks.py](https://github.com/ACEsuit/mace/blob/b3e3b2c6c9ac89dbc375905740e15e1e5b1ad721/mace/modules/blocks.py) |
| Final interaction/product | Keep the A input through ell=3, but restrict the final skip, contraction output, and readout input to scalar channels | [models.py](https://github.com/ACEsuit/mace/blob/b3e3b2c6c9ac89dbc375905740e15e1e5b1ad721/mace/modules/models.py) |
| Final activation | Use e3nn Activation with normalized SiLU, matching NonLinearReadoutBlock | [blocks.py](https://github.com/ACEsuit/mace/blob/b3e3b2c6c9ac89dbc375905740e15e1e5b1ad721/mace/modules/blocks.py) |
| Precision | Construct model parameters and fixed CG tensors directly in float64 for training/evaluation | [run_train.py](https://github.com/ACEsuit/mace/blob/b3e3b2c6c9ac89dbc375905740e15e1e5b1ad721/scripts/run_train.py) |
| Weight decay | Regularize interaction output projections and all product weights, including their update linears; exclude embedding, sender linears, selectors, and readouts | [run_train.py](https://github.com/ACEsuit/mace/blob/b3e3b2c6c9ac89dbc375905740e15e1e5b1ad721/scripts/run_train.py) |
| Gradient clipping | Apply norm clipping before Adam, with the reference default of 10; allow an explicit override | [train.py](https://github.com/ACEsuit/mace/blob/b3e3b2c6c9ac89dbc375905740e15e1e5b1ad721/mace/tools/train.py) |
| EMA | Preserve initial shadow parameters and use the update-count warm-up of torch_ema before the configured 0.99 decay | [torch_ema](https://github.com/fadel/pytorch_ema/blob/master/torch_ema/ema.py) |
| Graph cutoff | Forward the configured model cutoff to training and evaluation graph construction | Local configuration contract |
| Checkpoints | Save implementation version 2 and the author revision; reject earlier run configurations before loading incompatible weights | Local checkpoint contract |

The additional sender linear can be absorbed into the initial species embedding
mathematically. It is retained to match the reference parameterization, not
because Equation 9 requires two separate linear maps. The A projection performs
the channel-mixing operation in Equation 10. The receiver selector is an additional
species-conditioned transform used by the author code; Equation 10 does not
explicitly display it.

## Verified components

The radial Bessel basis and polynomial envelope, edge direction, component-normalized
spherical harmonics, tensor-product routes, receiver aggregation, second-interaction
residual, product update, atomic baseline, energy shift/scale, and negative energy
gradient were compared with the pinned implementation. All nine generalized-CG
tensors for correlation orders 1, 2, 3 and outputs 0e, 1o, 2e match exactly.

The comparison ran the actual downloaded author modules under the installed e3nn
0.6.0/PyTorch 2.2.2 CPU environment, without installing MACE or modifying its source.
Minimal package bindings supplied only an annotation type and conversion helper.
Numerical layers, radial functions, contractions, and autograd came from the
downloaded author source.

After transferring identical random parameters between models with two channels:

| Check | Synthetic two-graph batch | Two real 27-atom 3BPA frames |
|---|---:|---:|
| Maximum energy difference | 8.33e-17 eV | 1.33e-15 eV |
| Maximum force-component difference | 2.26e-17 eV/angstrom | 1.67e-16 eV/angstrom |
| Maximum checked force-loss parameter-gradient difference | 1.33e-15 | 1.33e-15 |
| Trainable parameter count, each model | 21,846 | 21,846 |

The same comparison also passed with the published 256-channel architecture on
two real 27-atom frames, with 3,769,616 trainable parameters in each model:

| Check | Maximum difference |
|---|---:|
| Energy | 2.22e-15 eV |
| Force component | 1.28e-15 eV/angstrom |
| Checked force-loss parameter gradient | 2.13e-14 |

Mean residual energy per atom, force-component RMS, and average receiver degree
also matched on the real batch. Its statistics comparison used identical synthetic
isolated energies to isolate the statistics algorithm from dataset references.
The saved regression fixture uses author-generated energies and forces with
all trainable parameters set to 0.2. Tests additionally check receiver conditioning,
neighbour normalization, optimizer decay behavior, float64 CG precision, old-run
rejection, and the training/evaluation/resume path.

## Training recipe and limits

The local loss follows paper Equation 15: graph-energy MSE plus force-component
MSE weighted by 1 and 1000. The reference CLI also supports atom-normalized weighted
energy losses and other tasks; these are not substituted for the selected paper
recipe. The local defaults retain the published 450/50 split, batch size 5,
8 radial bases, 5-angstrom cutoff, AMSGrad Adam at 0.01, and plateau scheduling
with factor 0.8 and patience 50. The epoch limit remains configurable at 500.
Periodic systems, stress, multi-head models, SWA, and modern MACE extensions are
outside this 3BPA reproduction.

This audit establishes forward and derivative agreement for the checked reference
configuration. It does not prove identical trajectories across library versions,
the original paper's initialization seeds, CUDA kernels, or convergence at 256
channels. The corrected code has not run the full three-seed benchmark. No paid
training was launched during the audit. Original run files and checkpoints remain
on disk; the revised model requires a fresh run.

Temporary reference source and comparison logs are in
`/private/tmp/mace-author-audit-20261006/`. Reproduce the numerical comparisons
from the repository root with:

```sh
uv run python /private/tmp/mace-author-audit-20261006/compare.py --random --assert
uv run python /private/tmp/mace-author-audit-20261006/compare.py --random --real --assert
uv run python /private/tmp/mace-author-audit-20261006/compare.py --published --random --real --assert
uv run pytest papers/15-MACE/ -q
```
