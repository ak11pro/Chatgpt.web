# exp001 — Input availability audit

- **Version:** exp001
- **Purpose:** establish whether a legal raw HSI source is mounted before any tracking experiment.
- **Configuration:** dynamic discovery under supplied roots; candidate cube suffixes only.
- **Code commit:** recorded in the repository history.
- **Dataset version:** not available in this non-Kaggle environment.
- **Runtime / GPU:** CPU-only; expected to be seconds on typical Kaggle mounts.
- **Validation score:** not applicable: no visual tracker is run.
- **Failure count:** not applicable.
- **Submission candidate:** none.

Run `python scripts/audit_hotc_data.py --output experiments/exp001_data_audit/artifacts/report.json` before attempting SAM2/SAMURAI inference. A `tracker_ready: false` report is an explicit stop condition, not a failed tracker result.
