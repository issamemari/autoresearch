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
# PA Behavioral Model: Memory Polynomial (Parallel Hammerstein)
# ---------------------------------------------------------------------------
# Represents a GaN Doherty PA operating at ~6 dB input back-off.
# Model: y(n) = sum_k sum_m a_{k,m} * x(n-m) * |x(n-m)|^{2k}
#
# Each nonlinear order has a different memory depth, producing
# order-dependent memory effects. This makes the PA harder to linearize
# than a simple memoryless model — a basic memory polynomial DPD won't
# fully capture the structure, leaving room for the agent to improve
# with GMP cross-terms, neural networks, or other advanced techniques.
#
# Polynomial order 2k+1: 1 (linear), 3, 5, 7, 9
# Coefficients are complex-valued (capturing both AM/AM and AM/PM).

PA_COEFFICIENTS = [
    # (k, m, coefficient)  where order = 2k+1, m = memory tap
    # k=0: linear path (4 memory taps — electrical memory)
    (0, 0,  1.0000 + 0.0000j),
    (0, 1,  0.0500 - 0.0200j),
    (0, 2,  0.0120 + 0.0080j),
    (0, 3, -0.0040 + 0.0025j),
    # k=1: 3rd order (3 memory taps)
    (1, 0, -0.4000 + 0.1500j),
    (1, 1, -0.0600 + 0.0250j),
    (1, 2, -0.0180 + 0.0100j),
    # k=2: 5th order (2 memory taps)
    (2, 0,  0.1500 - 0.0800j),
    (2, 1,  0.0300 - 0.0150j),
    # k=3: 7th order (2 memory taps)
    (3, 0, -0.0400 + 0.0250j),
    (3, 1, -0.0100 + 0.0060j),
    # k=4: 9th order (memoryless)
    (4, 0,  0.0080 - 0.0050j),
]

# Derived constants
PA_MAX_K = max(k for k, m, c in PA_COEFFICIENTS)
PA_MAX_MEMORY = max(m for k, m, c in PA_COEFFICIENTS)

# Parse into dict for fast lookup
_pa_coeffs = {}
for k, m, c in PA_COEFFICIENTS:
    _pa_coeffs[(k, m)] = c

# ---------------------------------------------------------------------------
# PA Model (DO NOT MODIFY — this is the ground truth PA)
# ---------------------------------------------------------------------------

def pa_model(x):
    """
    Apply the PA behavioral model to complex baseband input signal.
    Uses the fixed memory polynomial coefficients defined above.

    Args:
        x: complex numpy array, input signal
    Returns:
        y: complex numpy array, PA output signal (same length as x)
    """
    N = len(x)
    M = PA_MAX_MEMORY
    x_pad = np.concatenate([np.zeros(M, dtype=np.complex128), x])
    y = np.zeros(N, dtype=np.complex128)

    for (k, m), coeff in _pa_coeffs.items():
        x_del = x_pad[M - m : N + M - m]
        y += coeff * x_del * np.abs(x_del) ** (2 * k)

    return y

# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def generate_ofdm_signal(num_samples, seed):
    """
    Generate OFDM-like complex baseband test signal.
    Uses random 64-QAM modulation on active subcarriers within 20 MHz.
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
        # 64-QAM constellation (normalized)
        re = rng.integers(0, 8, num_active) * 2 - 7  # {-7,-5,-3,-1,1,3,5,7}
        im = rng.integers(0, 8, num_active) * 2 - 7
        qam = (re + 1j * im) / np.sqrt(42)  # normalized to unit avg power

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
    # With PA Asat ~ 1.0, RMS = 0.3 gives peaks near saturation
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
        "acpr_before_dbc": compute_acpr_db(y_no_dpd),
        "acpr_after_dbc": compute_acpr_db(y_with_dpd),
        "evm_percent": compute_evm_percent(y_with_dpd, x_val),
    }

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
