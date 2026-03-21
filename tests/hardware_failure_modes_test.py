import numpy as np
from src.constraints import apply_sensor_saturation, apply_quantization, simulate_packet_burst_loss

def test_saturation_shape_and_bounds():
    X_train = np.random.randn(100, 5)
    X = np.random.randn(20, 5) * 100
    Xs = apply_sensor_saturation(X, per_feature=True, ref_X=X_train, clip_percentile=99.0)
    assert Xs.shape == X.shape

def test_quantization_rounding():
    X = np.array([[1.234, 9.876]])
    Xq = apply_quantization(X, decimals=1)
    assert float(Xq[0,0]) == 1.2
    assert float(Xq[0,1]) == 9.9

def test_burst_loss_mask_length():
    sim = simulate_packet_burst_loss(100, burst_ms=500, sample_period_ms=10, seed=0)
    assert len(sim["dropped_mask"]) == 100
    assert sim["dropped_mask"].dtype == bool
    assert sim["dropped_mask"].any()