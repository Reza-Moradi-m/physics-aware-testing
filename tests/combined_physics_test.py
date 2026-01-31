from src.model import evaluate_with_noise_and_latency


def test_combined_physics_is_harder_than_cleanish_case():
    """
    Combined constraints should not be easier than mild conditions.
    This sanity check ensures stacking constraints makes the task harder.
    """
    mild = evaluate_with_noise_and_latency(
        noise_std=0.2,
        delay_ms=50,
        timeout_ms=120,
        drop_rate=0.0,
        seed=0,
    )

    harsh = evaluate_with_noise_and_latency(
        noise_std=0.8,
        delay_ms=200,
        timeout_ms=120,
        drop_rate=0.05,
        seed=0,
    )

    assert harsh["effective_accuracy"] <= mild["effective_accuracy"], (
        f"Expected harsh case <= mild case. "
        f"mild={mild['effective_accuracy']:.4f}, harsh={harsh['effective_accuracy']:.4f}"
    )