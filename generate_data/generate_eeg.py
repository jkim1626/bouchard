import os
import numpy as np

def generate_clean_eeg(duration_s=10.0, fs=256, n_channels=1, random_state=None):
    rng = np.random.default_rng(random_state)
    n_samples = int(duration_s * fs)
    t = np.arange(n_samples) / fs
    clean = np.zeros((n_channels, n_samples), dtype=np.float32)
    bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
    for ch in range(n_channels):
        signal = np.zeros(n_samples, dtype=np.float32)
        for f_low, f_high in bands:
            freq = rng.uniform(f_low, f_high)
            phase = rng.uniform(0, 2*np.pi)
            amp = 1.0 / max(freq, 1.0)
            signal += amp * np.sin(2*np.pi*freq*t + phase)
        drift_freq = rng.uniform(0.05, 0.2)
        drift_phase = rng.uniform(0, 2*np.pi)
        signal += 0.5 * np.sin(2*np.pi*drift_freq*t + drift_phase)
        clean[ch] = signal.astype(np.float32)
    return clean, fs

def generate_pink_noise(n_samples, n_channels=1, random_state=None):
    rng = np.random.default_rng(random_state)
    pink = np.zeros((n_channels, n_samples), dtype=np.float32)
    freqs = np.fft.rfftfreq(n_samples)
    for ch in range(n_channels):
        re = rng.normal(size=freqs.shape)
        im = rng.normal(size=freqs.shape)
        spectrum = re + 1j * im
        scaling = np.ones_like(freqs)
        nz = freqs > 0
        scaling[nz] = 1.0 / np.sqrt(freqs[nz])
        spectrum *= scaling
        noise = np.fft.irfft(spectrum, n=n_samples)
        noise = noise / np.std(noise)
        pink[ch] = noise.astype(np.float32)
    return pink

def add_noise_at_snr(clean, noise, snr_db):
    signal_power = np.mean(clean**2)
    noise_power_raw = np.mean(noise**2)
    snr_linear = 10.0**(snr_db / 10.0)
    scale = np.sqrt(signal_power / (snr_linear * noise_power_raw))
    return (clean + scale * noise).astype(np.float32)

def main():
    output_dir = "../synthetic_data/EEG_data"
    os.makedirs(output_dir, exist_ok=True)

    duration_s = 10.0
    fs = 256
    n_channels = 4
    num_files_per_snr = 5
    base_seed = 42
    snr_list_db = [-5, 0, 5, 10, 20]

    clean, fs_used = generate_clean_eeg(duration_s, fs, n_channels, random_state=base_seed)

    for snr_idx, snr_db in enumerate(snr_list_db):
        for k in range(num_files_per_snr):
            seed = base_seed + snr_idx * 100 + k
            pink = generate_pink_noise(clean.shape[1], n_channels, random_state=seed)
            noisy = add_noise_at_snr(clean, pink, snr_db)
            fname = f"eeg_pink_snr_{snr_db}dB_sample_{k}.npz"
            np.savez(
                os.path.join(output_dir, fname),
                clean=clean,
                noisy=noisy,
                fs=fs_used,
                snr_db=np.array(snr_db, dtype=np.float32),
            )
            print(f"Generated: {fname}")

    print(f"\nGenerated {len(snr_list_db) * num_files_per_snr} files in {output_dir}/")

if __name__ == "__main__":
    main()
