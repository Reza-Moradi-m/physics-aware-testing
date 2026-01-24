import numpy as np


def add_gaussian_noise(X: np.ndarray, std: float, seed: int = 0) -> np.ndarray:
    """
    Simulate sensor noise by adding Gaussian noise to inputs.
    
    Parameters:
        X   : input features
        std : standard deviation of noise (noise severity)
        seed: random seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=std, size=X.shape)
    return X + noise