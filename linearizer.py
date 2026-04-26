"""
DPD linearizer — GMP with ILA, 1 iteration.
Best-performing approach from previous experiments, now with
band-limited input signal that has lower ACPR floor.

Usage: uv run linearizer.py
"""

import time
import numpy as np
from prepare import (
    TIME_BUDGET, SAMPLE_RATE, SIGNAL_BANDWIDTH,
    load_data, pa_model, evaluate_dpd, plot_diagnostics,
    compute_nmse_db, compute_acpr_db,
)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

POLY_ORDER = 7
MEMORY_DEPTH = 3
CROSS_ORDER = 5
CROSS_MEMORY = 2
CROSS_LAG = 2
NUM_ITERATIONS = 1
REGULARIZATION = 1e-6
DPD_FILTER_BW = 0       # 0 = no DPD output filtering

# ---------------------------------------------------------------------------
# GMP Basis Matrix
# ---------------------------------------------------------------------------

def build_gmp_basis_matrix(x, K_a, M_a, K_c, M_c, L_c):
    N = len(x)
    pad = max(M_a, M_c + L_c)
    x_pad = np.concatenate([np.zeros(pad, dtype=np.complex128), x,
                            np.zeros(L_c, dtype=np.complex128)])
    columns = []

    for k in range(K_a):
        for m in range(M_a + 1):
            x_del = x_pad[pad - m : N + pad - m]
            columns.append(x_del * np.abs(x_del) ** (2 * k))

    for k in range(1, K_c + 1):
        for m in range(M_c + 1):
            for l in range(1, L_c + 1):
                x_sig = x_pad[pad - m : N + pad - m]
                x_env = x_pad[pad - m - l : N + pad - m - l]
                columns.append(x_sig * np.abs(x_env) ** (2 * k))

    for k in range(1, K_c + 1):
        for m in range(M_c + 1):
            for l in range(1, L_c + 1):
                x_sig = x_pad[pad - m : N + pad - m]
                x_env = x_pad[pad - m + l : N + pad - m + l]
                columns.append(x_sig * np.abs(x_env) ** (2 * k))

    return np.column_stack(columns)


def _bandlimit_filter(x, bw_mult):
    """Apply a low-pass filter to limit DPD output bandwidth."""
    from scipy.signal import firwin, lfilter
    cutoff = bw_mult * SIGNAL_BANDWIDTH / SAMPLE_RATE
    cutoff = min(cutoff, 0.99)  # stay below Nyquist
    num_taps = 101
    filt = firwin(num_taps, cutoff, window='hamming')
    filtered = lfilter(filt, 1.0, x)
    # Compensate group delay
    delay = (num_taps - 1) // 2
    filtered = np.roll(filtered, -delay)
    return filtered


def apply_dpd(x, coefficients):
    K_a = (POLY_ORDER + 1) // 2
    K_c = (CROSS_ORDER - 1) // 2
    U = build_gmp_basis_matrix(x, K_a, MEMORY_DEPTH, K_c, CROSS_MEMORY, CROSS_LAG)
    y = U @ coefficients
    if DPD_FILTER_BW > 0:
        y = _bandlimit_filter(y, DPD_FILTER_BW)
    return y

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
np.random.seed(42)

x_train, y_train = load_data("train")
print(f"Training samples: {len(x_train):,}")

pa_gain = np.vdot(x_train, y_train) / np.vdot(x_train, x_train)
print(f"PA linear gain: {abs(pa_gain):.4f} (phase: {np.degrees(np.angle(pa_gain)):.2f} deg)")

nmse_no_dpd = compute_nmse_db(y_train, x_train)
print(f"PA NMSE (no DPD): {nmse_no_dpd:.2f} dB")

K_a = (POLY_ORDER + 1) // 2
K_c = (CROSS_ORDER - 1) // 2
n_aligned = K_a * (MEMORY_DEPTH + 1)
n_cross = 2 * K_c * (CROSS_MEMORY + 1) * CROSS_LAG
num_coefficients = n_aligned + n_cross
print(f"GMP: {n_aligned} aligned + {n_cross} cross = {num_coefficients} coefficients")

# ---------------------------------------------------------------------------
# Training: ILA
# ---------------------------------------------------------------------------

print()
print("Training DPD (ILA)...")

coefficients = None

for iteration in range(NUM_ITERATIONS):
    t_iter = time.time()

    if iteration == 0:
        dpd_input = y_train / pa_gain
        dpd_target = x_train
    else:
        x_dpd = apply_dpd(x_train, coefficients)
        y_new = pa_model(x_dpd)
        dpd_input = y_new / pa_gain
        dpd_target = x_train

    U = build_gmp_basis_matrix(dpd_input, K_a, MEMORY_DEPTH,
                                K_c, CROSS_MEMORY, CROSS_LAG)

    A = U.conj().T @ U + REGULARIZATION * np.eye(num_coefficients)
    b = U.conj().T @ dpd_target
    coefficients = np.linalg.solve(A, b)

    x_dpd_check = apply_dpd(x_train, coefficients)
    y_check = pa_model(x_dpd_check)
    train_nmse = compute_nmse_db(y_check, x_train)
    train_acpr = compute_acpr_db(y_check)

    elapsed = time.time() - t_iter
    print(f"  Iter {iteration + 1}/{NUM_ITERATIONS}: "
          f"NMSE={train_nmse:.2f} dB  ACPR={train_acpr:.2f} dBc ({elapsed:.1f}s)")

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

print()
print("Evaluating on validation data...")

def dpd_fn(x):
    return apply_dpd(x, coefficients)

results = evaluate_dpd(dpd_fn)

x_val, _ = load_data("val")
y_val_nodpd = pa_model(x_val)
x_dpd_val = dpd_fn(x_val)
y_val_dpd = pa_model(x_dpd_val)
plot_diagnostics(x_val, y_val_nodpd, y_val_dpd, "diagnostics.png")

t_end = time.time()

nmse_improvement = results['nmse_no_dpd_db'] - results['nmse_db']

print()
print("---")
print(f"nmse_db:          {results['nmse_db']:.2f}")
print(f"nmse_no_dpd_db:   {results['nmse_no_dpd_db']:.2f}")
print(f"nmse_improvement: {nmse_improvement:.2f}")
print(f"acpr_before_dbc:  {results['acpr_before_dbc']:.2f}")
print(f"acpr_after_dbc:   {results['acpr_after_dbc']:.2f}")
print(f"evm_before_pct:   {results['evm_before_percent']:.2f}")
print(f"evm_after_pct:    {results['evm_percent']:.2f}")
print(f"num_coefficients: {num_coefficients}")
print(f"total_seconds:    {t_end - t_start:.1f}")
