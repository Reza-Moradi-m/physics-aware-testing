# tests/staleness_test.py
from src.model import train_and_evaluate, evaluate_with_staleness


def test_staleness_degrades_accuracy():
    baseline = train_and_evaluate()

    # no staleness should be close-ish to baseline
    acc0 = evaluate_with_staleness(delay_ms=0, drift_per_ms=0.01, seed=0)
    assert acc0 >= baseline - 0.02

    # high staleness should drop meaningfully
    acc_high = evaluate_with_staleness(delay_ms=300, drift_per_ms=0.01, seed=0)

    # rule: must drop by at least 5% relative to baseline
    assert acc_high <= baseline - 0.05