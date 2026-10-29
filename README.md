# Solar Filament Segmentation Challenge 2026

This repository contains a reproducible, model-generated instance-segmentation pipeline for the MAGFiLO solar-filament challenge. The old notebook's precomputed RLE payload is not used by the new workflow.

## Recommended Kaggle entry point

Use **[`solar-filament-final-kaggle.ipynb`](solar-filament-final-kaggle.ipynb)**. It clones this branch at runtime, checks the installed PyTorch/CUDA environment, installs only missing non-Torch dependencies, verifies COCO polygons, trains a ResNet-34 U-Net semantic/boundary model, and then trains a YOLO11-seg proposal model using the public no-cost pretrained starting weights. Detector-guided fusion is selected only when it improves pooled PQ on the tuning partition; an untouched set of held-out dates is reported separately. If selected, the detector is fine-tuned on all labelled annotation views, and the notebook generates `/kaggle/working/submission.csv` from actual test-image inference.

### Kaggle settings

1. Attach the competition dataset.
2. Set Accelerator to GPU and enable Internet for the public GitHub clone and optional ImageNet encoder weights.
3. Run the notebook from top to bottom. Keep `RUN_MODE = "smoke"` only for a quick test; use `"competition"` for the full training and submission run.
4. Do not pip-install a random version of PyTorch or Torchvision into Kaggle. The notebook intentionally preserves Kaggle's installed Torch/CUDA stack. If Torch itself cannot import, restart the runtime and remove earlier Torch-install cells instead of repeatedly downgrading packages.

## Pipeline module

- `solar_filament/pipeline.py`: COCO parsing, date-grouped splits (keeping each physical JPEG/date and its annotator views together), compressed label caches, corrected-intensity/radius-channel construction, high-resolution crop sampling, ResNet-34 U-Net-style semantic/boundary prediction, pooled PQ metrics, tiled TTA, gap-aware instance reconstruction, and RLE export.
- `solar_filament/disk.py`: solar-limb fitting and limb-darkening correction. Attribution and licensing are documented in `THIRD_PARTY_NOTICES.md`.
- `tests/test_metrics.py`: synthetic metric, string-ID, polygon-rasterization, physical-group-split, and RLE checks.
- `requirements.txt`: optional non-Torch dependencies; Torch/Torchvision are deliberately not pinned.

## Reproducibility and limitations

The notebook saves the Git commit, environment details, training history, checkpoint(s), selected post-processing parameters, held-out metrics, and submission. The local pooled PQ implementation follows the published IoU > 0.5 matching rule and scores each annotator view separately. Still compare it with the organizer's latest [self-evaluation notebook](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook) before claiming exact parity. The notebook has not been run end-to-end on a Kaggle GPU from this development environment; the first actual metrics and runtime must be recorded there.

No leaderboard score is claimed by this repository without an actual Kaggle submission. Local held-out metrics and public leaderboard scores must be reported separately.
