import numpy as np


def add_gaussian_noise_by_snr(samples: np.ndarray, snr_db: float, seed: int | None = None) -> np.ndarray:
    """Add independent Gaussian noise to each sample at the requested SNR."""
    rng = np.random.default_rng(seed)
    samples = np.asarray(samples, dtype=np.float32)
    signal_power = np.mean(samples ** 2, axis=1, keepdims=True)
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, 1.0, size=samples.shape).astype(np.float32)
    noise_std = np.sqrt(np.maximum(noise_power, 1e-12)).astype(np.float32)
    return samples + noise * noise_std
