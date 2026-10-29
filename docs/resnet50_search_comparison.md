# Controlled grid-versus-random search comparison

This is a measured classifier-tuning comparison, not a new deployment model.
The selected 320px model and its previously reported test results remain unchanged.

## Shared protocol

- Same cached frozen iNaturalist ResNet-50 encoder features: 4,794 training and 1,200 validation images, 2,048 dimensions, deterministic 320px full-image fit-and-pad.
- Original training views only for this controlled study; the selected deployment model used original-plus-flipped training views.
- One tuned parameter: LDA covariance shrinkage in [0.03, 0.95].
- Grid: 8 log-spaced points including the endpoints. Random: 8 log-uniform samples, seed42.
- Same CPU, two threads, fitting function and validation metrics for both methods. Trial order interleaves the methods and reverses after each pair.
- Training features alone fit class means/covariance. Confidence calibration is disabled; validation is used for scoring/selection only. No official test examples are evaluated.
- Best trial: highest validation accuracy, then macro F1, then earlier trial.
- Time is the sum of fitting/scoring durations; it excludes shared loading/extraction, logging, plotting and coefficient saving.

## Measured results

| Method | Trials | Best shrinkage | Validation accuracy | Validation macro F1 | Fit + score time |
|---|---:|---:|---:|---:|---:|
| grid | 8 | 0.049147 | 87.9167% | 87.8934% | 12.80s |
| random | 8 | 0.435032 | 87.9167% | 87.8052% | 11.87s |

| Best head | Train accuracy | Validation macro precision | Validation macro recall |
|---|---:|---:|---:|
| grid | 98.3521% | 89.6247% | 87.9167% |
| random | 95.9950% | 89.4177% | 87.9167% |

Both best heads correctly classify 1,055 of1,200 validation images. Their
errors differ across classes, so equal accuracy does not imply equal macro F1.
Grid has slightly higher F1 here; random reaches the tied best accuracy on its
first trial, while grid reaches it on its second trial. Random also takes
slightly less measured fit/score time. These observations describe this run only.

The random winner uses stronger shrinkage and has lower training accuracy.
A smaller train/validation gap alone is not a reason to replace the selected
model or to declare one search method generally superior.

## Reproduction

See [the exact CLI command](resnet50_training.md#grid-versus-random-search).
Use a new output directory for each rerun. The feature cache and matching source
checkpoint are external inputs, and are not committed to Git. The cache producer
records remain under `outputs/resnet50_inaturalist/resolution_review/`.

Source checkpoint SHA256: `a6ed5697a0a7555292e84ab2219848428cda7a18eb43fe9e02dff43c88eeb221`.

Feature cache SHA256: `b01132aaeb6b902bc284d3dee924093b424de5a44296941509187106099263c3`.

Local result directory: `outputs/resnet50_inaturalist/grid_random_comparison/`.
It contains the saved candidate lists/provenance (`plan.json`), every measured
trial (`trials.csv`), best-trial summaries (`summary.json`), the progress plot
(`comparison.png`) and sixteen small classifier coefficient files (`heads/`).
Each coefficient file references the same source encoder and feature-cache hash.

MLflow experiment: `resnet50_grid_vs_random`. Parent run:
`4097dee21d594694b4876610377e98d1`. All sixteen trials have nested runs, parameters,
metrics and classifier artifacts. Training/validation inputs and the selected
deployment checkpoint were verified unchanged.

## Limits

Only one encoder, one hyperparameter, one random seed and a small trial budget
were studied. The validation split was reused across prior project experiments.
Small metric/time differences should not be generalized to other datasets,
budgets or hardware. No test-based search-method selection or improvement claim
is made. Earlier 64-grid/16-random searches used different protocols and are
historical experiments, not additional trials in this comparison.
