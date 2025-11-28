import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style

# --- CONFIGURATION AND STYLING ---
D = 200      # Dimension of the signal
T = 1.0      # Total time
N = 1000     # Number of time steps
beta_min = 0.1
beta_max = 35.0 

style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (18, 5)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Inter'

# --- SHARED FUNCTIONS AND NOISE SCHEDULE ---
dt = T / N
ts = np.linspace(0, T, N + 1)

# Pre-compute noise schedule (Variance Preserving)
betas_sde = beta_min + 0.5 * (beta_max - beta_min) * (1 - np.cos(np.pi * ts / T))
alphas_sde = np.exp(-0.5 * np.cumsum(betas_sde) * dt)
sigmas_sde = np.sqrt(1 - alphas_sde**2)

# Generate the clean signal
x_axis = np.linspace(0, 10, D)
x0 = np.sin(x_axis) * np.cos(x_axis * 0.5)
x0[int(D * 0.2):int(D * 0.25)] += 0.8
x0[int(D * 0.7):int(D * 0.72)] -= 1.0

def plot_results(ax, title, noisy_x, denoised_x, color):
    """Helper function to plot the results on a given axis."""
    # The clean signal is a solid blue line.
    ax.plot(x0, color='blue', linewidth=2, label='Clean Signal', alpha=0.9, zorder=2)
    # The noisy signal is a thin gray solid line.
    ax.plot(noisy_x, color='gray', linestyle='-', linewidth=1.5, label='Noisy Signal', alpha=0.7, zorder=1)
    # The denoised signal is a thicker, dashed line plotted on top.
    ax.plot(denoised_x, color=color, linewidth=3, linestyle='--', label='Denoised Signal', zorder=3)
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlabel("Signal Dimension")
    ax.set_ylabel("Amplitude")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# --- 1. ADDITIVE GAUSSIAN NOISE ---

def get_noisy_additive(t_idx):
    """Adds additive Gaussian noise for a given time step index."""
    alpha_t = alphas_sde[t_idx]
    sigma_t = sigmas_sde[t_idx]
    noise = np.random.randn(D) * sigma_t
    return alpha_t * x0 + noise

def oracle_score_additive(xt, t_idx):
    """The perfect 'oracle' score function for additive noise."""
    alpha_t = alphas_sde[t_idx]
    sigma_t = sigmas_sde[t_idx]
    if sigma_t < 1e-5:
        return np.zeros_like(xt)
    return -(xt - alpha_t * x0) / (sigma_t**2)

def denoise_additive(noisy_x):
    """
    FIXED: Runs the reverse process using a STABLE ancestral sampler (DDPM-style).
    This robustly prevents the numerical explosions that were causing NaNs.
    """
    xt = np.copy(noisy_x)
    alpha_bars = alphas_sde**2
    
    for i in range(N, 0, -1):
        z = np.random.randn(D) if i > 1 else np.zeros(D)
        
        # Predict the original signal (x0) using the score. This is a key step.
        score = oracle_score_additive(xt, i)
        epsilon_pred = -sigmas_sde[i] * score
        x0_pred = (xt - sigmas_sde[i] * epsilon_pred) / alphas_sde[i]
        x0_pred = np.clip(x0_pred, -2.5, 2.5) # Clip for stability

        # Use the predicted x0 to calculate the posterior mean of x_{t-1}
        alpha_bar_prev = alpha_bars[i-1]
        
        # Posterior mean coefficients
        coeff1 = np.sqrt(alpha_bar_prev) * (1 - alpha_bars[i] / alpha_bar_prev) / (1 - alpha_bars[i])
        coeff2 = np.sqrt(alphas_sde[i]**2 / alpha_bar_prev) * (1 - alpha_bar_prev) / (1 - alpha_bars[i])
        posterior_mean = coeff1 * x0_pred + coeff2 * xt

        # Posterior variance
        posterior_variance = (1 - alpha_bars[i] / alpha_bar_prev) * (1 - alpha_bar_prev) / (1 - alpha_bars[i])

        xt = posterior_mean + np.sqrt(np.maximum(posterior_variance, 1e-9)) * z
        
    return xt

# --- 2. MULTIPLICATIVE (SPECKLE) NOISE ---

def get_noisy_multiplicative(t_idx):
    """Applies multiplicative noise via log-space transformation."""
    log_x0 = np.log(np.abs(x0) + 1e-6)
    Lambda_t = np.sum(betas_sde[:t_idx+1]) * dt
    noise = np.random.randn(D) * np.sqrt(Lambda_t)
    log_xt = log_x0 - 0.5 * Lambda_t + noise
    return np.exp(log_xt) * np.sign(x0) # Noise affects magnitude, not sign

def denoise_multiplicative(noisy_x):
    """
    FIXED: Denoises magnitude in log-space and preserves the original sign.
    """
    # Store the sign of the noisy signal to re-apply it later.
    signs = np.sign(noisy_x)
    
    # Denoise the magnitude in log-space
    yt = np.log(np.abs(noisy_x) + 1e-6)
    log_x0 = np.log(np.abs(x0) + 1e-6)

    # Denoise the Ornstein-Uhlenbeck process in log-space
    for i in range(N, 0, -1):
        z = np.random.randn(D) if i > 1 else np.zeros(D)
        
        Lambda_i = np.sum(betas_sde[:i+1]) * dt
        Lambda_i_minus_1 = np.sum(betas_sde[:i]) * dt

        var_i = Lambda_i
        var_i_minus_1 = Lambda_i_minus_1
        
        # Posterior mean of p(y_{t-1} | y_t, y_0) for the OU process
        mean_pred = (var_i_minus_1 * yt + (var_i - var_i_minus_1) * log_x0) / (var_i + 1e-9)
        # Posterior variance
        variance = (var_i - var_i_minus_1) * var_i_minus_1 / (var_i + 1e-9)
        
        yt = mean_pred + np.sqrt(np.maximum(variance, 1e-9)) * z

    # Transform back to signal space and re-apply the original sign.
    denoised_magnitude = np.exp(yt)
    return denoised_magnitude * signs

# --- 3. IMPULSIVE (JUMP) NOISE ---

jump_locations = []

def get_noisy_jump(t_idx):
    """Adds a combination of Gaussian and impulsive noise."""
    global jump_locations
    sigma_t = sigmas_sde[t_idx] * 0.5  # Less background noise
    noise = np.random.randn(D) * sigma_t
    xt = alphas_sde[t_idx] * x0 + noise
    
    jump_locations = []
    num_jumps = 7 # Increased number of jumps for a better demo
    for _ in range(num_jumps):
        loc = np.random.randint(0, D)
        mag = (np.random.choice([-1, 1])) * (1.5 + np.random.rand() * 2)
        xt[loc] += mag
        jump_locations.append(loc)
    return xt

def denoise_jump(noisy_x):
    """
    FIXED: Uses the stable ancestral sampler and a stable oracle for jumps.
    """
    xt = np.copy(noisy_x)
    alpha_bars = alphas_sde**2
    
    for i in range(N, 0, -1):
        z = np.random.randn(D) if i > 1 else np.zeros(D)
        
        # Standard prediction using additive score
        score = oracle_score_additive(xt, i)
        epsilon_pred = -sigmas_sde[i] * score
        x0_pred = (xt - sigmas_sde[i] * epsilon_pred) / alphas_sde[i]

        # ORACLE HACK for JUMPS:
        # Instead of a huge score, directly correct the x0 prediction at jump locations.
        # This simulates a perfect model that knows how to fill in the gaps.
        for loc in jump_locations:
             x0_pred[loc] = x0[loc]

        x0_pred = np.clip(x0_pred, -2.5, 2.5)

        # Use the corrected x0 to calculate the posterior mean of x_{t-1}
        alpha_bar_prev = alpha_bars[i-1]
        
        coeff1 = np.sqrt(alpha_bar_prev) * (1 - alpha_bars[i] / alpha_bar_prev) / (1 - alpha_bars[i])
        coeff2 = np.sqrt(alphas_sde[i]**2 / alpha_bar_prev) * (1 - alpha_bar_prev) / (1 - alpha_bars[i])
        posterior_mean = coeff1 * x0_pred + coeff2 * xt

        posterior_variance = (1 - alpha_bars[i] / alpha_bar_prev) * (1 - alpha_bar_prev) / (1 - alpha_bars[i])

        xt = posterior_mean + np.sqrt(np.maximum(posterior_variance, 1e-9)) * z
        
    return xt


# --- MAIN EXECUTION AND PLOTTING ---
if __name__ == '__main__':
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle('Corrected & Stabilized SDE Denoising in Python', fontsize=22, fontweight='bold')

    print("Running Additive Denoising...")
    noisy_additive = get_noisy_additive(N)
    denoised_additive = denoise_additive(noisy_additive)
    plot_results(ax1, '1. Additive Gaussian Noise', noisy_additive, denoised_additive, 'green')

    print("Running Multiplicative (Speckle) Denoising...")
    noisy_multiplicative = get_noisy_multiplicative(N)
    denoised_multiplicative = denoise_multiplicative(noisy_multiplicative)
    plot_results(ax2, '2. Multiplicative (Speckle) Noise', noisy_multiplicative, denoised_multiplicative, 'red')

    print("Running Impulsive (Jump) Denoising...")
    noisy_jump = get_noisy_jump(N)
    denoised_jump = denoise_jump(noisy_jump)
    plot_results(ax3, '3. Impulsive (Jump) Noise', noisy_jump, denoised_jump, 'orange')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


