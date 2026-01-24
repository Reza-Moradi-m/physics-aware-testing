from src.model import train_and_evaluate


def test_baseline_accuracy():
    acc = train_and_evaluate()
    assert acc > 0.85, f"Baseline accuracy too low: {acc}"