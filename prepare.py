"""
PA simulation and evaluation harness for autonomous DPD research.
Generates PA behavioral model data and provides fixed evaluation metrics.

Usage:
    python prepare.py                      # generate data with defaults
    python prepare.py --num-samples 200000 # custom training sample count

Data is stored in ~/.cache/autoresearch-dpd/.
"""

import os
import sys
import time
import argparse
import numpy as np
from scipy.signal import welch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

SAMPLE_RATE = 122.88e6          # Sample rate (Hz), standard for 5G NR
SIGNAL_BANDWIDTH = 20e6         # Signal bandwidth (Hz), 20 MHz LTE/NR channel
NUM_TRAIN_SAMPLES = 200_000     # Training samples
NUM_VAL_SAMPLES = 50_000        # Validation samples
TIME_BUDGET = 300               # Time budget in seconds (5 minutes)
RANDOM_SEED = 42                # Fixed seed for reproducibility

# ---------------------------------------------------------------------------
# Cache configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-dpd")

# ---------------------------------------------------------------------------
# PA Behavioral Model: Saleh Model with Memory (Wiener structure)
# ---------------------------------------------------------------------------
# Classic Saleh AM/AM and AM/PM model (A. Saleh, IEEE Trans. Comm., 1981)
# preceded by a linear FIR filter to capture memory effects.
#
# Wiener structure: x -> [FIR memory filter] -> z -> [Saleh NL] -> y
#
# AM/AM: A(r) = alpha_a * r / (1 + beta_a * r^2)
# AM/PM: Phi(r) = alpha_phi * r^2 / (1 + beta_phi * r^2)
# Output: y = A(|z|) * exp(j * (arg(z) + Phi(|z|)))
#
# The memory filter models electrical/thermal memory effects in the PA
# matching network and bias circuitry. Combined with the Saleh nonlinearity,
# this produces a PA that requires both nonlinear compensation and memory
# equalization — a basic memoryless DPD won't fully linearize it.

# Saleh model parameters (solid-state class AB PA)
# AM/AM has moderate compression; AM/PM is mild at average power
# but significant at peaks (~11 degrees at r=0.9).
SALEH_ALPHA_A = 2.0          # AM/AM small-signal gain
SALEH_BETA_A = 0.5           # AM/AM compression factor
SALEH_ALPHA_PHI = 0.3        # AM/PM coefficient (radians)
SALEH_BETA_PHI = 0.3         # AM/PM saturation factor

# Memory filter coefficients (FIR, applied before nonlinearity)
PA_MEMORY_FILTER = np.array([
    1.0000 + 0.0000j,       # main tap
    0.0500 - 0.0200j,       # 1-sample memory
    0.0120 + 0.0080j,       # 2-sample memory
   -0.0040 + 0.0025j,       # 3-sample memory
], dtype=np.complex128)

PA_MEMORY_DEPTH = len(PA_MEMORY_FILTER) - 1

# ---------------------------------------------------------------------------
# PA Model (DO NOT MODIFY — this is the ground truth PA)
# ---------------------------------------------------------------------------

def _saleh_nonlinearity(x):
    """Apply memoryless Saleh AM/AM and AM/PM distortion."""
    r = np.abs(x)
    theta = np.angle(x)

    # AM/AM: A(r) = alpha_a * r / (1 + beta_a * r^2)
    am_am = SALEH_ALPHA_A * r / (1 + SALEH_BETA_A * r ** 2)

    # AM/PM: Phi(r) = alpha_phi * r^2 / (1 + beta_phi * r^2)
    am_pm = SALEH_ALPHA_PHI * r ** 2 / (1 + SALEH_BETA_PHI * r ** 2)

    return am_am * np.exp(1j * (theta + am_pm))


def pa_model(x):
    """
    Apply the PA behavioral model to complex baseband input signal.
    Wiener structure: linear FIR memory filter followed by Saleh nonlinearity.

    Args:
        x: complex numpy array, input signal
    Returns:
        y: complex numpy array, PA output signal (same length as x)
    """
    N = len(x)
    M = PA_MEMORY_DEPTH

    # Step 1: Linear memory filter (FIR convolution)
    x_pad = np.concatenate([np.zeros(M, dtype=np.complex128), x])
    z = np.zeros(N, dtype=np.complex128)
    for m, h in enumerate(PA_MEMORY_FILTER):
        z += h * x_pad[M - m : N + M - m]

    # Step 2: Saleh memoryless nonlinearity
    return _saleh_nonlinearity(z)

# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def generate_ofdm_signal(num_samples, seed):
    """
    Generate OFDM-like complex baseband test signal.
    Uses random 16-QAM modulation on active subcarriers within 20 MHz.
    Subcarrier spacing: 122.88 MHz / 2048 = 60 kHz.
    Active subcarriers: 334 (spanning ~20 MHz).
    Produces realistic PAPR (~8-10 dB) and flat in-band spectrum.
    """
    rng = np.random.default_rng(seed)

    nfft = 2048
    num_active = 334        # 334 * 60 kHz = 20.04 MHz signal bandwidth
    cp_len = 144            # normal cyclic prefix
    symbol_len = nfft + cp_len

    num_symbols = num_samples // symbol_len + 2

    samples = []
    for _ in range(num_symbols):
        # 16-QAM constellation (normalized to unit average power)
        re = rng.integers(0, 4, num_active) * 2 - 3  # {-3, -1, 1, 3}
        im = rng.integers(0, 4, num_active) * 2 - 3
        qam = (re + 1j * im) / np.sqrt(10)  # E[|s|^2] = 1

        # Map to subcarriers (DC null, guard bands at edges)
        freq = np.zeros(nfft, dtype=np.complex128)
        freq[1:num_active // 2 + 1] = qam[:num_active // 2]
        freq[nfft - num_active // 2:] = qam[num_active // 2:]

        # IFFT to time domain
        td = np.fft.ifft(freq) * np.sqrt(nfft)

        # Add cyclic prefix
        symbol = np.concatenate([td[-cp_len:], td])
        samples.append(symbol)

    signal = np.concatenate(samples)[:num_samples]

    # Normalize to target RMS for ~6 dB input back-off
    # With Saleh saturation near |x|~0.9, RMS = 0.3 gives peaks near compression
    target_rms = 0.3
    signal = signal / np.sqrt(np.mean(np.abs(signal) ** 2)) * target_rms

    return signal.astype(np.complex128)

# ---------------------------------------------------------------------------
# Evaluation metrics (DO NOT CHANGE — these are the fixed metrics)
# ---------------------------------------------------------------------------

def compute_nmse_db(y_actual, x_ref):
    """
    Normalized Mean Square Error in dB. Primary optimization metric.

    First aligns y_actual to x_ref via least-squares gain estimation
    (compensating for PA linear gain), then computes NMSE.
    More negative = better linearization.

    Args:
        y_actual: PA output (complex array)
        x_ref: original input signal (complex array)
    Returns:
        NMSE in dB (float, negative)
    """
    # LS gain alignment: alpha = (x^H y) / (x^H x)
    alpha = np.vdot(x_ref, y_actual) / np.vdot(x_ref, x_ref)
    y_ref = alpha * x_ref
    error = y_actual - y_ref
    nmse = np.sum(np.abs(error) ** 2) / np.sum(np.abs(y_ref) ** 2)
    return 10 * np.log10(max(nmse, 1e-100))


def compute_acpr_db(signal, sample_rate=SAMPLE_RATE, signal_bw=SIGNAL_BANDWIDTH):
    """
    Adjacent Channel Power Ratio in dBc.
    Measures spectral regrowth in adjacent channels (worst of upper/lower).
    More negative = better.

    Main channel:    [-BW/2, +BW/2]
    Lower adjacent:  [-3*BW/2, -BW/2]
    Upper adjacent:  [+BW/2, +3*BW/2]
    """
    nperseg = min(4096, len(signal))
    freqs, psd = welch(signal, fs=sample_rate, nperseg=nperseg,
                       return_onesided=False, scaling='density')

    # fftshift to center DC
    freqs = np.fft.fftshift(freqs)
    psd = np.fft.fftshift(psd)

    half_bw = signal_bw / 2
    df = freqs[1] - freqs[0]

    main_mask = np.abs(freqs) <= half_bw
    lower_mask = (freqs >= -3 * half_bw) & (freqs < -half_bw)
    upper_mask = (freqs > half_bw) & (freqs <= 3 * half_bw)

    main_power = np.sum(psd[main_mask]) * df
    lower_power = np.sum(psd[lower_mask]) * df
    upper_power = np.sum(psd[upper_mask]) * df

    adj_power = max(lower_power, upper_power)
    if main_power <= 0:
        return 0.0
    return 10 * np.log10(max(adj_power / main_power, 1e-100))


def compute_evm_percent(y_actual, x_ref):
    """
    Error Vector Magnitude as a percentage.
    Lower = better.
    """
    alpha = np.vdot(x_ref, y_actual) / np.vdot(x_ref, x_ref)
    y_ref = alpha * x_ref
    error = y_actual - y_ref
    evm = np.sqrt(np.mean(np.abs(error) ** 2) / np.mean(np.abs(y_ref) ** 2))
    return evm * 100


def evaluate_dpd(dpd_fn):
    """
    Fixed DPD evaluation (DO NOT CHANGE — this is the ground truth metric).

    Applies DPD to the validation input, passes through the PA model,
    and computes all metrics. Also computes PA-only metrics for comparison.

    Args:
        dpd_fn: callable, takes complex input array -> returns predistorted array
    Returns:
        dict with nmse_db, acpr_before_dbc, acpr_after_dbc, evm_percent
    """
    x_val, _ = load_data("val")

    # PA output with DPD
    x_dpd = dpd_fn(x_val)
    y_with_dpd = pa_model(x_dpd)

    # PA output without DPD (for comparison)
    y_no_dpd = pa_model(x_val)

    return {
        "nmse_db": compute_nmse_db(y_with_dpd, x_val),
        "nmse_no_dpd_db": compute_nmse_db(y_no_dpd, x_val),
        "acpr_before_dbc": compute_acpr_db(y_no_dpd),
        "acpr_after_dbc": compute_acpr_db(y_with_dpd),
        "evm_before_percent": compute_evm_percent(y_no_dpd, x_val),
        "evm_percent": compute_evm_percent(y_with_dpd, x_val),
    }

# ---------------------------------------------------------------------------
# Diagnostic plots (DO NOT CHANGE — fixed visualization)
# ---------------------------------------------------------------------------

def plot_diagnostics(x_input, y_no_dpd, y_with_dpd, filename="diagnostics.png"):
    """
    Generate diagnostic plots comparing PA output with and without DPD.
    Creates a 2x2 figure: PSD (ACPR), AM/AM, AM/PM, Constellation.

    Args:
        x_input: original input signal (complex array)
        y_no_dpd: PA output without DPD (complex array)
        y_with_dpd: PA output with DPD applied (complex array)
        filename: output file path for the figure
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ---- PSD / ACPR ----
    ax = axes[0, 0]
    nperseg = min(4096, len(x_input))

    f_in, psd_in = welch(x_input, fs=SAMPLE_RATE, nperseg=nperseg,
                         return_onesided=False, scaling='density')
    f_no, psd_no = welch(y_no_dpd, fs=SAMPLE_RATE, nperseg=nperseg,
                         return_onesided=False, scaling='density')
    f_wd, psd_wd = welch(y_with_dpd, fs=SAMPLE_RATE, nperseg=nperseg,
                         return_onesided=False, scaling='density')

    # Center DC and convert to dB
    f_in, psd_in = np.fft.fftshift(f_in), np.fft.fftshift(psd_in)
    f_no, psd_no = np.fft.fftshift(f_no), np.fft.fftshift(psd_no)
    f_wd, psd_wd = np.fft.fftshift(f_wd), np.fft.fftshift(psd_wd)

    # Normalize PSD to peak of no-DPD for easier comparison
    peak_psd = np.max(psd_no)
    psd_in_db = 10 * np.log10(np.maximum(psd_in / peak_psd, 1e-20))
    psd_no_db = 10 * np.log10(np.maximum(psd_no / peak_psd, 1e-20))
    psd_wd_db = 10 * np.log10(np.maximum(psd_wd / peak_psd, 1e-20))

    ax.plot(f_in / 1e6, psd_in_db, color='#4a9eed', alpha=0.5,
            linewidth=1, label='Input')
    ax.plot(f_no / 1e6, psd_no_db, color='#ef4444', alpha=0.8,
            linewidth=1.2, label='PA only (no DPD)')
    ax.plot(f_wd / 1e6, psd_wd_db, color='#22c55e', alpha=0.8,
            linewidth=1.2, label='PA + DPD')

    half_bw_mhz = SIGNAL_BANDWIDTH / 2e6
    for bw in [-half_bw_mhz, half_bw_mhz]:
        ax.axvline(bw, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
    ax.axvspan(-half_bw_mhz, half_bw_mhz, alpha=0.05, color='blue')

    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Normalized PSD (dB)')
    ax.set_title('Power Spectral Density (ACPR)')
    ax.legend(fontsize=9, loc='lower center')
    ax.set_xlim([-3.5 * half_bw_mhz, 3.5 * half_bw_mhz])
    ax.set_ylim(bottom=-80)
    ax.grid(True, alpha=0.3)

    # ---- AM/AM ----
    ax = axes[0, 1]
    r_in = np.abs(x_input)
    r_out_no = np.abs(y_no_dpd)
    r_out_wd = np.abs(y_with_dpd)

    # Downsample for scatter plot
    step = max(1, len(r_in) // 3000)
    sort_idx = np.argsort(r_in)
    idx = sort_idx[::step]

    ax.scatter(r_in[idx], r_out_no[idx], s=1, alpha=0.3, c='#ef4444',
               label='PA only (no DPD)', rasterized=True)
    ax.scatter(r_in[idx], r_out_wd[idx], s=1, alpha=0.3, c='#22c55e',
               label='PA + DPD', rasterized=True)

    # Ideal linear reference lines
    gain_no = np.abs(np.vdot(x_input, y_no_dpd) / np.vdot(x_input, x_input))
    gain_wd = np.abs(np.vdot(x_input, y_with_dpd) / np.vdot(x_input, x_input))
    r_max = np.max(r_in) * 1.05
    ax.plot([0, r_max], [0, gain_no * r_max], 'r--', alpha=0.4,
            linewidth=1, label=f'Ideal linear (G={gain_no:.2f})')

    ax.set_xlabel('Input Amplitude |x|')
    ax.set_ylabel('Output Amplitude |y|')
    ax.set_title('AM/AM Characteristic')
    ax.legend(fontsize=9, loc='upper left', markerscale=8)
    ax.set_xlim([0, r_max])
    ax.grid(True, alpha=0.3)

    # ---- AM/PM ----
    ax = axes[1, 0]

    # Phase distortion: angle(y * conj(x)) handles wrapping correctly
    mask = r_in > 0.03 * np.max(r_in)  # ignore near-zero amplitudes
    phase_no = np.degrees(np.angle(y_no_dpd[mask] * np.conj(x_input[mask])))
    phase_wd = np.degrees(np.angle(y_with_dpd[mask] * np.conj(x_input[mask])))
    r_masked = r_in[mask]

    sort_idx2 = np.argsort(r_masked)
    step2 = max(1, len(r_masked) // 3000)
    idx2 = sort_idx2[::step2]

    ax.scatter(r_masked[idx2], phase_no[idx2], s=1, alpha=0.3, c='#ef4444',
               label='PA only (no DPD)', rasterized=True)
    ax.scatter(r_masked[idx2], phase_wd[idx2], s=1, alpha=0.3, c='#22c55e',
               label='PA + DPD', rasterized=True)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)

    ax.set_xlabel('Input Amplitude |x|')
    ax.set_ylabel('Phase Distortion (degrees)')
    ax.set_title('AM/PM Characteristic')
    ax.legend(fontsize=9, loc='upper left', markerscale=8)
    ax.grid(True, alpha=0.3)

    # ---- Constellation (I/Q scatter) ----
    ax = axes[1, 1]

    # Gain-align outputs to input for fair constellation comparison
    alpha_no = np.vdot(x_input, y_no_dpd) / np.vdot(x_input, x_input)
    alpha_wd = np.vdot(x_input, y_with_dpd) / np.vdot(x_input, x_input)
    y_no_aligned = y_no_dpd / alpha_no
    y_wd_aligned = y_with_dpd / alpha_wd

    n_plot = min(5000, len(x_input))
    ax.scatter(np.real(y_no_aligned[:n_plot]), np.imag(y_no_aligned[:n_plot]),
               s=1, alpha=0.15, c='#ef4444', label='PA only (no DPD)',
               rasterized=True)
    ax.scatter(np.real(y_wd_aligned[:n_plot]), np.imag(y_wd_aligned[:n_plot]),
               s=1, alpha=0.15, c='#22c55e', label='PA + DPD',
               rasterized=True)
    ax.scatter(np.real(x_input[:n_plot]), np.imag(x_input[:n_plot]),
               s=1, alpha=0.08, c='#4a9eed', label='Input (reference)',
               rasterized=True)

    ax.set_xlabel('In-Phase (I)')
    ax.set_ylabel('Quadrature (Q)')
    ax.set_title('Constellation (gain-aligned)')
    ax.legend(fontsize=9, loc='upper right', markerscale=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    fig.suptitle('DPD Diagnostics — With vs Without DPD', fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Diagnostics saved to {filename}")

# ---------------------------------------------------------------------------
# Data management
# ---------------------------------------------------------------------------

def load_data(split="train"):
    """
    Load PA input/output data.
    Returns (input_signal, pa_output) as complex numpy arrays.
    """
    assert split in ["train", "val"]
    x = np.load(os.path.join(CACHE_DIR, f"{split}_input.npy"))
    y = np.load(os.path.join(CACHE_DIR, f"{split}_output.npy"))
    return x, y


def generate_and_save_data(num_train=NUM_TRAIN_SAMPLES, num_val=NUM_VAL_SAMPLES):
    """Generate PA simulation data and save to cache directory."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    files = ["train_input.npy", "train_output.npy",
             "val_input.npy", "val_output.npy"]
    if all(os.path.exists(os.path.join(CACHE_DIR, f)) for f in files):
        print(f"Data: already generated at {CACHE_DIR}")
        return

    print("Generating PA simulation data...")
    t0 = time.time()

    # Training data
    print(f"  Training signal ({num_train:,} samples)...")
    x_train = generate_ofdm_signal(num_train, seed=RANDOM_SEED)
    y_train = pa_model(x_train)
    np.save(os.path.join(CACHE_DIR, "train_input.npy"), x_train)
    np.save(os.path.join(CACHE_DIR, "train_output.npy"), y_train)

    # Validation data (different seed for independent evaluation)
    print(f"  Validation signal ({num_val:,} samples)...")
    x_val = generate_ofdm_signal(num_val, seed=RANDOM_SEED + 1)
    y_val = pa_model(x_val)
    np.save(os.path.join(CACHE_DIR, "val_input.npy"), x_val)
    np.save(os.path.join(CACHE_DIR, "val_output.npy"), y_val)

    t1 = time.time()

    # Summary
    nmse_no_dpd = compute_nmse_db(y_val, x_val)
    acpr_no_dpd = compute_acpr_db(y_val)
    papr_db = 10 * np.log10(np.max(np.abs(x_train) ** 2) / np.mean(np.abs(x_train) ** 2))
    print(f"  Signal PAPR: {papr_db:.1f} dB")
    print(f"  PA NMSE (no DPD):  {nmse_no_dpd:.2f} dB")
    print(f"  PA ACPR (no DPD):  {acpr_no_dpd:.2f} dBc")
    print(f"  Generated in {t1 - t0:.1f}s")
    print(f"  Saved to {CACHE_DIR}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate PA simulation data for DPD research")
    parser.add_argument("--num-samples", type=int, default=NUM_TRAIN_SAMPLES,
                        help="Number of training samples")
    parser.add_argument("--regenerate", action="store_true",
                        help="Force regeneration even if data exists")
    args = parser.parse_args()

    print(f"Cache directory: {CACHE_DIR}")
    print()

    if args.regenerate:
        import shutil
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)
            print("Cleared existing data.")

    generate_and_save_data(num_train=args.num_samples)
    print()
    print("Done! Ready to run linearizer.py")
