# Solar Filament Segmentation Challenge 2026

This repository contains a reproducible, model-generated instance-segmentation pipeline for the MAGFiLO solar-filament challenge. The old notebook's precomputed RLE payload is not used by the new workflow.

## Recommended Kaggle entry point

Use **[`solar-filament-final-kaggle.ipynb`](solar-filament-final-kaggle.ipynb)**. It clones this branch at runtime, checks the installed PyTorch/CUDA environment, installs only missing non-Torch dependencies, verifies COCO polygons, trains the semantic/boundary model, tunes instance reconstruction on a tuning partition, reports metrics on a separate held-out partition, optionally retrains on all labelled views, and generates `/kaggle/working/submission.csv`.

### Kaggle settings

1. Attach the competition dataset.
2. Set Accelerator to GPU and enable Internet for the public GitHub clone and optional ImageNet encoder weights.
3. Run the notebook from top to bottom. Keep `RUN_MODE = "smoke"` only for a quick test; use `"competition"` for the full training and submission run.
4. Do not pip-install a random version of PyTorch or Torchvision into Kaggle. The notebook intentionally preserves Kaggle's installed Torch/CUDA stack. If Torch itself cannot import, restart the runtime and remove earlier Torch-install cells instead of repeatedly downgrading packages.

## Pipeline module

- `solar_filament/pipeline.py`: COCO parsing, physical-observation grouped splits, compressed instance-mask caching, high-resolution crop sampling, ResNet-34 U-Net-style semantic/boundary prediction, PQ metrics, tiled inference, watershed reconstruction, and RLE export.
- `tests/test_metrics.py`: synthetic metric, string-ID, polygon-rasterization, physical-group-split, and RLE checks.
- `requirements.txt`: optional non-Torch dependencies; Torch/Torchvision are deliberately not pinned.

## Reproducibility and limitations

The notebook saves the Git commit, environment details, training history, checkpoint(s), selected post-processing parameters, held-out metrics, and submission. The local PQ implementation is a reproducible project evaluator; before claiming exact parity, compare it against the organizer's latest [self-evaluation notebook](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook).

No leaderboard score is claimed by this repository without an actual Kaggle submission. Local held-out metrics and public leaderboard scores must be reported separately.
