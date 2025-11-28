"""
noise_classifier.py

Heuristic estimator of how much additive, multiplicative, and jump (impulsive)
noise a 1D noisy signal is likely to contain.

This version uses:
  - Global kurtosis / tail statistics (jump detection)
  - Local variance vs mean^2 correlation (multiplicative detection)
  - Log-domain variance behavior (multiplicative vs additive)
  - A more structured way of turning features into soft weights.

Returns:
  {
    "weights": {
        "additive": w_add,
        "multiplicative": w_mult,
        "jump": w_jump,
    },
    "features": { ... },
  }
"""

import numpy as np


# ---------- basic moment helpers ---------- #

def _safe_var(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.var(x) + 1e-12)


def compute_kurtosis(x: np.ndarray) -> float:
    """
    Normalized 4th central moment (Gaussian ≈ 3).
    """
    x = np.asarray(x, dtype=np.float64)
    mu = float(np.mean(x))
    v = _safe_var(x)
    centered = x - mu
    m4 = float(np.mean(centered ** 4))
    return m4 / (v ** 2 + 1e-12)


def compute_skewness(x: np.ndarray) -> float:
    """
    3rd central moment normalized by variance^(3/2).
    """
    x = np.asarray(x, dtype=np.float64)
    mu = float(np.mean(x))
    v = _safe_var(x)
    centered = x - mu
    m3 = float(np.mean(centered ** 3))
    return m3 / ((v ** 1.5) + 1e-12)


def local_mean_var(x: np.ndarray, window: int = 64):
    """
    Compute local mean and variance over non-overlapping windows.
    Returns:
        means: (M,)
        vars : (M,)
    """
    x = np.asarray(x, dtype=np.float64)
    L = len(x)
    if L < window:
        window = max(8, L // 2)
    if window < 4:
        window = L

    means = []
    vars_ = []
    for i in range(0, L, window):
        seg = x[i:i + window]
        if len(seg) < 4:
            break
        means.append(np.mean(seg))
        vars_.append(np.var(seg))
    if not means:
        means = [np.mean(x)]
        vars_ = [np.var(x)]
    return np.asarray(means), np.asarray(vars_)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """
    Simple Pearson correlation between two 1D arrays.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size != b.size or a.size < 2:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    sa = float(np.std(a) + 1e-12)
    sb = float(np.std(b) + 1e-12)
    return float(np.mean((a / sa) * (b / sb)))


# ---------- heuristic confidence estimator ---------- #

def estimate_noise_confidence(y: np.ndarray, window: int = 64):
    """
    Estimate soft confidence weights for additive, multiplicative, and jump noise.

    Args:
        y      : 1D noisy signal (array-like)
        window : window size for local statistics

    Returns:
        {
          "weights": {
              "additive": w_add,
              "multiplicative": w_mult,
              "jump": w_jump,
          },
          "features": {
              "kurtosis": ...,
              "skewness": ...,
              "tail_ratio": ...,
              "tail_mass_6sigma": ...,
              "tail_mass_8sigma": ...,
              "mult_corr": ...,
              "log_var": ...,
              "raw_var": ...,
              "log_var_ratio": ...,
          },
        }
    """
    y = np.asarray(y, dtype=np.float64)
    L = len(y)

    # --- global stats ---
    mu = float(np.mean(y))
    raw_var = _safe_var(y)
    std = float(np.sqrt(raw_var))

    kurt = compute_kurtosis(y)
    skew = compute_skewness(y)

    abs_y = np.abs(y)
    p99 = float(np.percentile(abs_y, 99.0))
    rms = float(np.sqrt(np.mean(y ** 2) + 1e-12))
    tail_ratio = p99 / (rms + 1e-12)

    # Tail mass beyond multiples of std (impulsiveness)
    if std < 1e-8:
        tail_mass_6 = 0.0
        tail_mass_8 = 0.0
    else:
        z = np.abs(y - mu) / std
        tail_mass_6 = float(np.mean(z > 6.0))
        tail_mass_8 = float(np.mean(z > 8.0))

    # --- multiplicative proxy: var ~ mean^2? ---
    local_means, local_vars = local_mean_var(y, window=window)
    if local_means.size >= 3:
        m2 = local_means ** 2
        mult_corr = _corr(m2, local_vars)  # positive -> variance tracks mean^2
    else:
        mult_corr = 0.0

    # --- log-domain behavior (multiplicative vs additive) ---
    log_y = np.log(np.abs(y) + 1e-6)
    log_var = float(np.var(log_y) + 1e-12)
    log_kurt = compute_kurtosis(log_y)

    # ratio of raw variance to log variance
    log_var_ratio = raw_var / (log_var + 1e-12)

    # ---------- convert features into heuristic scores ---------- #

    # 1) Jump score: dominated by heavy tails and tail mass
    # Gaussian kurtosis ~3, small tail_mass_6/8.
    excess_kurt = max(0.0, kurt - 3.0)
    excess_log_kurt = max(0.0, log_kurt - 3.0)

    # Normalize and combine
    jump_from_kurt = (excess_kurt / 6.0) + (excess_log_kurt / 8.0)
    jump_from_tail_ratio = max(0.0, tail_ratio - 3.0) / 3.0
    jump_from_tail_mass = 10.0 * (tail_mass_6 + 2.0 * tail_mass_8)

    score_jump_raw = jump_from_kurt + jump_from_tail_ratio + jump_from_tail_mass
    # Cap to reasonable range
    score_jump_raw = min(score_jump_raw, 10.0)

    # 2) Multiplicative score:
    #   - strong positive mult_corr
    #   - raw variance grows relative to log variance (log_var_ratio > 1)
    score_mult_corr = max(0.0, mult_corr)  # [0,1]
    score_mult_corr = (score_mult_corr ** 2) * 4.0  # emphasize strong corr

    # log_var_ratio:  ~1 for additive-like, larger when amplitude-scaling dominates.
    score_mult_log = max(0.0, log_var_ratio - 1.0)  # 0 when equal
    score_mult_log = min(score_mult_log, 4.0)

    score_mult_raw = score_mult_corr + 0.5 * score_mult_log
    score_mult_raw = min(score_mult_raw, 10.0)

    # 3) Additive score:
    #    We want additive high when both jump and mult are weak.
    #    Also symmetric distributions (low |skew|) are more additive-like.
    skew_penalty = min(abs(skew), 3.0) / 3.0  # in [0,1]
    base_add = 1.5 * (1.0 - 0.5 * skew_penalty)  # in ~[0.75, 1.5]

    # Suppress additive when others are very strong
    suppression = 0.2 * score_jump_raw + 0.15 * score_mult_raw
    score_add_raw = base_add / (1.0 + suppression)
    score_add_raw = max(0.05, score_add_raw)

    # ---------- normalize to get weights ---------- #

    score_add = max(0.0, score_add_raw)
    score_mult = max(0.0, score_mult_raw)
    score_jump = max(0.0, score_jump_raw)

    s = score_add + score_mult + score_jump
    if s <= 1e-8:
        # Degenerate: fall back to mostly additive
        w_add, w_mult, w_jump = 0.9, 0.05, 0.05
    else:
        w_add = score_add / s
        w_mult = score_mult / s
        w_jump = score_jump / s

    weights = {
        "additive": float(w_add),
        "multiplicative": float(w_mult),
        "jump": float(w_jump),
    }

    features = {
        "kurtosis": float(kurt),
        "skewness": float(skew),
        "tail_ratio": float(tail_ratio),
        "tail_mass_6sigma": float(tail_mass_6),
        "tail_mass_8sigma": float(tail_mass_8),
        "mult_corr": float(mult_corr),
        "log_var": float(log_var),
        "raw_var": float(raw_var),
        "log_var_ratio": float(log_var_ratio),
        "log_kurtosis": float(log_kurt),
    }

    return {
        "weights": weights,
        "features": features,
    }
