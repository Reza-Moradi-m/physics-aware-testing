from src.model import train_and_evaluate, evaluate_with_noise


def test_baseline_is_strong():
    """
    Baseline under ideal conditions should be high.
    """
    acc = train_and_evaluate()
    assert acc > 0.90, f"Baseline too low: {acc:.4f}"


def test_low_noise_should_stay_high():
    """
    Small sensor noise should not break the model.
    """
    acc = evaluate_with_noise(std=0.1, seed=0)
    assert acc > 0.95, f"Too sensitive to small noise: {acc:.4f}"


def test_high_noise_should_degrade():
    """
    Under heavy noise, accuracy should drop noticeably compared to baseline.
    This proves the physics-aware test is meaningful.
    """
    baseline = train_and_evaluate()
    noisy = evaluate_with_noise(std=1.0, seed=0)

    # Require at least a 0.03 drop (tune if needed).
    drop = baseline - noisy
    assert drop > 0.03, f"Expected degradation under noise. baseline={baseline:.4f}, noisy={noisy:.4f}, drop={drop:.4f}"


def test_accuracy_monotonic_sanity_check():
    """
    Sanity check: accuracy should generally not improve as noise increases.
    We allow tiny wiggles (tolerance) because ML is stochastic.
    """
    stds = [0.0, 0.3, 0.8, 1.0]
    accs = [evaluate_with_noise(std=s, seed=0) for s in stds]

    # Allow a small tolerance for tiny variations
    tolerance = 0.01
    assert accs[0] + tolerance >= accs[1], f"Unexpected increase: {stds[0]}->{accs[0]:.4f} then {stds[1]}->{accs[1]:.4f}"
    assert accs[1] + tolerance >= accs[2], f"Unexpected increase: {stds[1]}->{accs[1]:.4f} then {stds[2]}->{accs[2]:.4f}"
    assert accs[2] + tolerance >= accs[3], f"Unexpected increase: {stds[2]}->{accs[2]:.4f} then {stds[3]}->{accs[3]:.4f}"