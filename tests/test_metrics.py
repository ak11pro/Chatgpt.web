"""Fast, synthetic checks for the competition-aligned instance metrics."""

from solar_filament.pipeline import run_synthetic_metric_tests


def test_synthetic_panoptic_metrics() -> None:
    run_synthetic_metric_tests()
