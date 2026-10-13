# RFM result artifacts

- `fire-comparison.png`: existing qualitative visualization from run
  `20260818-103835`; not a held-out metric.
- `historical-calibration-result.json`: downloaded from Modal volume
  `rfm-fire-results`, path `calibration/latest_result.json`, on 2026-10-13.
  It records NLL −2.473926271923619 from the training/calibration wrapper's
  evaluation. This is not the later reported standalone evaluation of −0.700.

Training and standalone evaluation are separate jobs. The saved checkpoint and
configuration remain in the Modal volume, but the standalone evaluation output
supporting the README's −0.700 result has not been recovered. Do not substitute
the historical calibration result for that reported value. Commit `88f680c`
later corrected sphere divergence; the calibration JSON has no evaluator revision
or solver settings sufficient to establish its numerical comparability.
