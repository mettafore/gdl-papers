# RFM result artifacts

- `fire-comparison.png`: existing qualitative visualization from run
  `20260818-103835`; not a held-out metric.
- `historical-calibration-result.json`: downloaded from Modal volume
  `rfm-fire-results`, path `calibration/latest_result.json`, on 2026-10-13.
  It records NLL −2.473926271923619 from the training/calibration wrapper's
  evaluation. This is not the later reported standalone evaluation of −0.700.

- `local-evaluation-20261013.json`: confirmed local CPU evaluation of the saved
  checkpoint, with NLL −0.7004 across all 1,282 held-out points and exit code 0.
  It records solver settings, evaluator revision, and checkpoint/data hashes.
  The CLI reports four decimal places, so the record does not claim more precision.

Training and standalone evaluation are separate jobs. The original later
standalone output remains unrecovered, but the fresh local evaluation confirms
its reported −0.700 value to three decimal places. The historical calibration
result is retained separately. Commit `88f680c` corrected sphere divergence;
that historical JSON lacks evaluator revision and solver settings.
