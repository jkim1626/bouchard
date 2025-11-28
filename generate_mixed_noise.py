# generate_mixed_noise_signals.py

import os
from pathlib import Path
import numpy as np


def generate_clean_signal(
    length: int = 2048,
    fs: float = 256.0,
    n_components: int = 5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate a synthetic 1D 'clean' signal as a sum of damped sinusoids.

    Returns:
        clean: 1D float32 array of shape (length,)
    """
    if rng is None:
        rng = np.random.default_rng()
    t = np.arange(length, dtype=np.float32) / fs
    signal = np.zeros_like(t, dtype=np.float32)

    n_comp = rng.integers(2, n_components + 1)
    for _ in range(n_comp):
        amp = float(rng.lognormal(mean=0.0, sigma=0.3))     # positive amplitude
        freq = float(rng.uniform(1.0, 40.0))                # Hz
        decay = float(rng.uniform(0.5, 3.0))                # sec
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        signal += (
            amp
            * np.exp(-t / decay, dtype=np.float32)
            * np.cos(2.0 * np.pi * freq * t + phase, dtype=np.float32)
        )

    # Small baseline drift
    slope = float(rng.normal(0.0, 0.02))
    offset = float(rng.normal(0.0, 0.05))
    baseline = offset + slope * (t - t.mean())
    signal += baseline.astype(np.float32)

    return signal.astype(np.float32)


def generate_additive_noise(clean: np.ndarray, target_power: float,
                            rng: np.random.Generator) -> np.ndarray:
    """
    Generate additive white Gaussian noise with a specified power.
    """
    noise = rng.normal(0.0, 1.0, size=clean.shape).astype(np.float32)
    current_power = float(np.mean(noise.astype(np.float64) ** 2) + 1e-12)
    scale = np.sqrt(target_power / current_power)
    return (scale * noise).astype(np.float32)


def generate_multiplicative_noise(clean: np.ndarray, target_power: float,
                                  rng: np.random.Generator) -> np.ndarray:
    """
    Generate multiplicative noise component: n_mult = clean * m_scaled,
    and scale so that E[n_mult^2] ~ target_power.
    """
    m = rng.normal(0.0, 1.0, size=clean.shape).astype(np.float32)
    noise_raw = clean * m  # unscaled multiplicative noise
    current_power = float(np.mean(noise_raw.astype(np.float64) ** 2) + 1e-12)
    scale = np.sqrt(target_power / current_power)
    return (scale * noise_raw).astype(np.float32)


def generate_jump_noise(clean: np.ndarray,
                        target_power: float,
                        rng: np.random.Generator,
                        jump_prob: float = 0.01) -> np.ndarray:
    """
    Generate sparse impulsive (jump) noise: random large spikes at random positions,
    scaled to achieve approximately target_power.
    """
    length = clean.shape[0]
    # Random Bernoulli mask for jump locations
    mask = rng.random(size=length) < jump_prob
    # Start with unit-variance spikes at those locations
    jumps = np.zeros_like(clean, dtype=np.float32)
    n_jumps = int(mask.sum())
    if n_jumps > 0:
        jumps[mask] = rng.normal(0.0, 1.0, size=n_jumps).astype(np.float32)
    current_power = float(np.mean(jumps.astype(np.float64) ** 2) + 1e-12)
    if current_power < 1e-12:
        # No jumps happened; just return zeros
        return jumps
    scale = np.sqrt(target_power / current_power)
    return (scale * jumps).astype(np.float32)


def generate_mixed_noise_sample(
    length: int,
    fs: float,
    snr_total_db: float,
    mixture_weights: tuple[float, float, float],
    rng: np.random.Generator,
) -> dict:
    """
    Generate one sample with mixed additive, multiplicative, and jump noise.

    Args:
        length          : signal length
        fs              : sampling frequency (only used for signal generation)
        snr_total_db    : desired total SNR in dB (signal power / total noise power)
        mixture_weights : (w_add, w_mult, w_jump), sum ~ 1
        rng             : numpy random Generator

    Returns:
        dict with:
          clean, noisy,
          noise_add, noise_mult, noise_jump,
          snr_total_db,
          noise_power_add, noise_power_mult, noise_power_jump,
          noise_ratio_add, noise_ratio_mult, noise_ratio_jump,
          mixture_weights (original)
    """
    clean = generate_clean_signal(length=length, fs=fs, rng=rng)

    # Signal power
    signal_power = float(np.mean(clean.astype(np.float64) ** 2) + 1e-12)

    # Total noise power from total SNR
    snr_linear = 10.0 ** (snr_total_db / 10.0)
    total_noise_power = signal_power / max(snr_linear, 1e-12)

    w_add, w_mult, w_jump = mixture_weights
    w_sum = w_add + w_mult + w_jump + 1e-12
    w_add /= w_sum
    w_mult /= w_sum
    w_jump /= w_sum

    target_power_add = total_noise_power * w_add
    target_power_mult = total_noise_power * w_mult
    target_power_jump = total_noise_power * w_jump

    noise_add = generate_additive_noise(clean, target_power_add, rng)
    noise_mult = generate_multiplicative_noise(clean, target_power_mult, rng)
    noise_jump = generate_jump_noise(clean, target_power_jump, rng)

    noise_total = noise_add + noise_mult + noise_jump
    noisy = clean + noise_total

    # Recompute realized powers
    noise_power_add = float(np.mean(noise_add.astype(np.float64) ** 2))
    noise_power_mult = float(np.mean(noise_mult.astype(np.float64) ** 2))
    noise_power_jump = float(np.mean(noise_jump.astype(np.float64) ** 2))
    noise_power_total = (
        noise_power_add + noise_power_mult + noise_power_jump + 1e-12
    )

    noise_ratio_add = noise_power_add / noise_power_total
    noise_ratio_mult = noise_power_mult / noise_power_total
    noise_ratio_jump = noise_power_jump / noise_power_total

    return {
        "clean": clean.astype(np.float32),
        "noisy": noisy.astype(np.float32),
        "noise_add": noise_add.astype(np.float32),
        "noise_mult": noise_mult.astype(np.float32),
        "noise_jump": noise_jump.astype(np.float32),
        "snr_total_db": float(snr_total_db),
        "noise_power_add": noise_power_add,
        "noise_power_mult": noise_power_mult,
        "noise_power_jump": noise_power_jump,
        "noise_ratio_add": noise_ratio_add,
        "noise_ratio_mult": noise_ratio_mult,
        "noise_ratio_jump": noise_ratio_jump,
        "mixture_weights": np.array(
            [w_add, w_mult, w_jump], dtype=np.float32
        ),
    }


def main():
    # Output directory
    out_dir = Path("./synthetic_mixed_noise")
    out_dir.mkdir(parents=True, exist_ok=True)

    length = 2048
    fs = 256.0

    # Total SNR levels (signal vs ALL noise together)
    snr_list_db = [0.0, 5.0, 10.0, 20.0]

    # Mixture patterns (w_add, w_mult, w_jump)
    mixture_list = [
        (1.0, 0.0, 0.0),         # pure additive
        (0.0, 1.0, 0.0),         # pure multiplicative
        (0.0, 0.0, 1.0),         # pure jump
        (0.5, 0.5, 0.0),         # additive + multiplicative
        (0.5, 0.0, 0.5),         # additive + jump
        (0.0, 0.5, 0.5),         # multiplicative + jump
        (1/3, 1/3, 1/3),         # balanced mixture
    ]

    num_realizations_per_combo = 20

    base_seed = 12345
    rng_global = np.random.default_rng(base_seed)

    total_files = 0

    for mixture_idx, mixture_weights in enumerate(mixture_list):
        for snr_db in snr_list_db:
            for k in range(num_realizations_per_combo):
                rng = np.random.default_rng(int(rng_global.integers(0, 1_000_000)))

                sample = generate_mixed_noise_sample(
                    length=length,
                    fs=fs,
                    snr_total_db=snr_db,
                    mixture_weights=mixture_weights,
                    rng=rng,
                )

                fname = (
                    f"mixed_noise_m{mixture_idx:02d}"
                    f"_snr_{snr_db:+.1f}dB"
                    f"_rep{k:02d}.npz"
                )
                np.savez(out_dir / fname, **sample)
                total_files += 1

    print(f"Generated {total_files} files in {out_dir}")


if __name__ == "__main__":
    main()
