# Solar Filament Segmentation Challenge 2026

This repository contains a competition-compliant, instance-aware MAGFiLO solar
filament segmentation pipeline. The legacy notebook's precomputed submission
payload is not retained. All final masks originate from trained model inference.

## Contents

- `solar-filament-unet-segmentation-0-55.ipynb` — an auditable Kaggle notebook
  with the legacy-pipeline audit and a measured-only workflow.
- `solar_filament/pipeline.py` — reusable COCO parsing, group split, high-
  resolution sampling, ResNet-34 encoder/decoder, semantic/boundary losses,
  PQ, tiled inference, watershed reconstruction, and validated COCO RLE output.
- `tests/test_metrics.py` — synthetic PQ split/merge/perfect-match checks.
- `requirements.txt` — pinned Kaggle-compatible dependencies.

## Usage

Attach the competition input in Kaggle, install the pinned requirements when
needed, and run the notebook from top to bottom. Its first executable cell
finds the repository source from either the current checkout or Kaggle working
storage and adds it to `sys.path` before testing the `solar_filament` import.
It then discovers the files beneath `/kaggle/input`, renders 20 annotation
overlays, prints dataset/annotation statistics and measured class imbalance,
and uses an observation-grouped validation split. It writes only model-derived
artifacts to `/kaggle/working` after a checkpoint has been trained and measured.

The final run uses all training observations (`max_train_images=None`) and has
explicit `RUN_TRAINING`, `RUN_VALIDATION`, and `RUN_TEST_INFERENCE` controls.
`DEBUG_MODE=True` is the only path that uses a small subset or shorter run.

No metric in this repository is a claimed leaderboard result. Validation Dice,
IoU, and PQ are deliberately produced only by the validation workflow on the
available held-out observations.
