from pathlib import Path

from hotc_tracker.data import discover_catalog


def test_catalog_finds_csvs_and_candidate_frames(tmp_path: Path) -> None:
    (tmp_path / "2026training.csv").write_text("ID,x,y,width,height\na_1,0,0,2,2\n")
    (tmp_path / "sample_submisson.csv").write_text("ID,x,y,width,height\na_1,0,0,2,2\n")
    (tmp_path / "frame_001.npy").write_bytes(b"placeholder")
    catalog = discover_catalog([tmp_path])
    assert catalog.training_csv and catalog.sample_submission_csv
    assert catalog.frames_available

