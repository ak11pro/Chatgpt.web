# HOTC 2026 Tracker

This repository now contains the reproducible Python foundation for the
**Hyperspectral Object Tracking Challenge 2026**. The pre-existing React scene
is unrelated to HOTC and has intentionally not been altered.

## Status

The first experiment is a non-invasive data audit. It discovers competition and
optional Kaggle Dataset mounts, validates annotation/submission schemas, and
reports whether raw frame cubes are actually available. It never substitutes
annotations for frames, and it does not modify a known-good endpoint baseline.

## Quick start

```bash
python scripts/audit_hotc_data.py --output experiments/exp001_data_audit/artifacts/report.json
pytest -q
```

Outside Kaggle, pass one or more roots explicitly:

```bash
python scripts/audit_hotc_data.py --root /data/hotc --root /kaggle/input
```

## Planned integration

`src/hotc_tracker/data` is the single frame/annotation discovery boundary.
Once legal raw HSI cubes are mounted, add the dataset-specific `FrameSource`
implementation there; tracking modules must consume its `(H, W, C)` arrays.
`src/hotc_tracker/representations.py` provides deterministic RGB projections
for SAM2/SAMURAI experiments while retaining the original cube for a separate
spectral branch. `src/hotc_tracker/evaluation` provides OPE metrics for local
training-only validation.
The intended first model experiment is a stratified training-sequence comparison
of SAMURAI using robust band-selected RGB versus per-sequence PCA RGB. It must
only run after the audit reports accessible training frames.
