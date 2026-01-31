from src.model import evaluate_with_latency_effects


def test_high_latency_effective_accuracy_drops():
    """
    Engineering rule:
    Under high latency that exceeds timeout, effective accuracy should drop below X.
    This proves our latency simulation is meaningful (timeouts count as failures).
    """
    metrics = evaluate_with_latency_effects(
        delay_ms=200,
        timeout_ms=120,
        drop_rate=0.05,
        seed=0,
    )

    # Choose X as a clear degradation threshold.
    # If your results are slightly different, we can tune this.
    assert metrics["effective_accuracy"] < 0.90, (
        f"Expected effective accuracy to drop under high latency. "
        f"Got {metrics['effective_accuracy']:.4f}"
    )