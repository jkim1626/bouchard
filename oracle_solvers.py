"""
oracle_solvers.py

Minimal set of oracle score functions for:
    1. Additive Gaussian noise (VP SDE)
    2. Multiplicative noise (log-domain OU)
    3. Jump noise (compound Poisson + Gaussian)
    
This file does NOT:
    - generate data
    - run any experiments
    - train any models
    - include a main() section

It ONLY defines the closed-form oracle score functions needed
for synthetic experiments or for training supervision.
"""

import numpy as np


# ============================================================
# 1. ADDITIVE GAUSSIAN ORACLE SCORE
# ============================================================

def additive_forward_marginal(x0, alpha_t, sigma_t, eps):
    """
    Forward SDE marginal:
        x_t = alpha_t * x0 + sigma_t * eps
    (Given to show relationship; not used for now.)
    """
    return alpha_t * x0 + sigma_t * eps


def oracle_score_additive(x_t, x0, sigma_t):
    """
    Oracle score for additive VP SDE:
        ∇_x log p(x_t | x0)
    Closed form:
        s(x_t) = -(x_t - alpha_t * x0) / sigma_t^2
    but more commonly for training:
        s(x_t) = -epsilon / sigma_t

    Inputs:
        x_t      : noisy signal at time t
        x0       : clean signal (oracle-only)
        sigma_t  : scalar or array (same shape as x_t)

    Returns:
        score (same shape as x_t)
    """
    # Equivalent oracle form using (x_t - x0) if alpha=1 at t=1,
    # but we allow general sigma_t.
    return -(x_t - x0) / (sigma_t ** 2 + 1e-12)


# ============================================================
# 2. MULTIPLICATIVE NOISE (LOG-DOMAIN ORACLE SCORE)
# ============================================================

def log_transform(x):
    """ Safe log-transform for multiplicative noise. """
    return np.log(np.abs(x) + 1e-8)


def oracle_score_multiplicative(v_t, v0, sigma_t):
    """
    Oracle score in log-domain where multiplicative noise becomes additive.
    Inputs:
        v_t     : log(|x_t|)
        v0      : log(|x0|)
        sigma_t : diffusion sigma at time t

    Closed-form additive domain score:
        s(v_t) = -(v_t - v0) / sigma_t^2
    """
    return -(v_t - v0) / (sigma_t ** 2 + 1e-12)


def inverse_log_transform(v, sign):
    """
    Map back from log-magnitude to original domain:
        x = sign * exp(v)
    """
    return sign * np.exp(v)


# ============================================================
# 3. JUMP NOISE (ORACLE SCORE FOR JUMP-DIFFUSION)
# ============================================================

def sample_jump_process(x0, g, jump_locs, jump_sizes):
    """
    Produces a jump-diffusion signal (oracle-only).
    Not used actively here — just included for completeness.

    x_t = x0 + g * W_t + sum_{k}( jump_k )
    """
    x = x0.copy()
    for loc, val in zip(jump_locs, jump_sizes):
        x[loc] += val
    return x


def oracle_score_jump(x_t, x0, g, jump_locs, jump_sizes):
    """
    Oracle jump score:
        score = continuous OU score + exact correction at jump sites.

    Inputs:
        x_t        : noisy signal at time t
        x0         : clean signal (oracle-only)
        g          : Gaussian diffusion coefficient
        jump_locs  : indices where jumps occurred
        jump_sizes : jump displacements

    Returns:
        score (same shape as x_t)
    """
    # Continuous Gaussian part:
    # s_cont = -(x_t - x0) / g^2
    score = -(x_t - x0) / (g ** 2 + 1e-12)

    # Jump corrections (oracle-only):
    # Set score equal to exact difference at jump points
    for loc, val in zip(jump_locs, jump_sizes):
        score[loc] = -(x_t[loc] - (x0[loc] + val)) / (g ** 2 + 1e-12)

    return score
