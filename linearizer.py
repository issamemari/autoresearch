"""
DPD linearizer — LUT + FIR equalizer (Hammerstein structure).
Matches the inverse of the Wiener PA: first invert the Saleh nonlinearity
via a lookup table estimated from data, then compensate memory with an FIR.

Usage: uv run linearizer.py
"""

import time
import numpy as np
from scipy.interpolate import interp1d
from prepare import (
    TIME_BUDGET, SAMPLE_RATE, SIGNAL_BANDWIDTH,
    load_data, pa_model, evaluate_dpd, plot_diagnostics,
    compute_nmse_db, compute_acpr_db,
)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

LUT_NUM_BINS = 512          # AM/AM and AM/PM lookup table resolution
FIR_LENGTH = 7              # FIR equalizer taps for memory compensation
REGULARIZATION = 1e-8       # ridge regularization for FIR estimation
NUM_REFINE_ITERS = 3        # iterative refinement passes

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

# ---------------------------------------------------------------------------
# Stage 1: Estimate AM/AM and AM/PM from PA I/O data -> build inverse LUT
# ---------------------------------------------------------------------------

print()
print("Stage 1: Building AM/AM and AM/PM lookup tables...")

r_in = np.abs(x_train)
r_out = np.abs(y_train)
phase_diff = np.angle(y_train * np.conj(x_train))

r_max = np.max(r_in) * 1.2
bin_edges = np.linspace(0, r_max, LUT_NUM_BINS + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

am_am_fwd = np.zeros(LUT_NUM_BINS)
am_pm_fwd = np.zeros(LUT_NUM_BINS)

for i in range(LUT_NUM_BINS):
    mask = (r_in >= bin_edges[i]) & (r_in < bin_edges[i + 1])
    count = np.sum(mask)
    if count > 0:
        am_am_fwd[i] = np.mean(r_out[mask])
        am_pm_fwd[i] = np.mean(phase_diff[mask])
    elif i > 0:
        am_am_fwd[i] = am_am_fwd[i - 1]
        am_pm_fwd[i] = am_pm_fwd[i - 1]

# Inverse AM/AM: desired_output_amplitude -> required_input_amplitude
peak_idx = np.argmax(am_am_fwd)
if peak_idx < 2:
    peak_idx = LUT_NUM_BINS - 1

r_in_mono = bin_centers[:peak_idx + 1]
r_out_mono = am_am_fwd[:peak_idx + 1]

# Ensure strictly increasing
mask_inc = np.concatenate([[True], np.diff(r_out_mono) > 0])
r_in_mono = r_in_mono[mask_inc]
r_out_mono = r_out_mono[mask_inc]

am_am_inverse = interp1d(r_out_mono, r_in_mono, kind='linear',
                          fill_value='extrapolate', bounds_error=False)

am_pm_interp = interp1d(bin_centers, am_pm_fwd, kind='linear',
                          fill_value='extrapolate', bounds_error=False)

print(f"  LUT bins: {LUT_NUM_BINS}")
print(f"  Monotonic range: |y| up to {r_out_mono[-1]:.4f}")


def apply_lut_dpd(x):
    """Memoryless LUT DPD: inverse AM/AM + inverse AM/PM."""
    r = np.abs(x)
    theta = np.angle(x)
    desired_r_out = np.abs(pa_gain) * r
    r_dpd = np.maximum(am_am_inverse(desired_r_out), 0)
    phase_correction = am_pm_interp(r_dpd)
    return r_dpd * np.exp(1j * (theta + np.angle(pa_gain) - phase_correction))


x_dpd_lut = apply_lut_dpd(x_train)
y_lut = pa_model(x_dpd_lut)
lut_nmse = compute_nmse_db(y_lut, x_train)
lut_acpr = compute_acpr_db(y_lut)
print(f"  LUT-only: NMSE={lut_nmse:.2f} dB, ACPR={lut_acpr:.2f} dBc")

# ---------------------------------------------------------------------------
# Stage 2: FIR memory equalizer
# ---------------------------------------------------------------------------

print()
print("Stage 2: FIR memory equalizer...")

M = FIR_LENGTH
fir_coeffs = np.zeros(M, dtype=np.complex128)
fir_coeffs[0] = 1.0


def apply_fir(x, h):
    """Apply FIR filter h to signal x."""
    N = len(x)
    Mf = len(h)
    x_pad = np.concatenate([np.zeros(Mf - 1, dtype=np.complex128), x])
    y = np.zeros(N, dtype=np.complex128)
    for m in range(Mf):
        y += h[m] * x_pad[Mf - 1 - m: N + Mf - 1 - m]
    return y


def apply_full_dpd(x):
    """Full DPD: LUT (NL inverse) -> FIR (memory inverse)."""
    z = apply_lut_dpd(x)
    return apply_fir(z, fir_coeffs)


for refine_iter in range(NUM_REFINE_ITERS):
    t_iter = time.time()

    x_dpd = apply_full_dpd(x_train)
    y_actual = pa_model(x_dpd)
    y_desired = pa_gain * x_train

    z_lut = apply_lut_dpd(x_train)

    N = len(z_lut)
    z_pad = np.concatenate([np.zeros(M - 1, dtype=np.complex128), z_lut])
    Z = np.column_stack([z_pad[M - 1 - m: N + M - 1 - m] for m in range(M)])

    # Linearize PA around current operating point
    eps = 1e-7
    x_dpd_cur = apply_fir(z_lut, fir_coeffs)
    y_cur = pa_model(x_dpd_cur)
    pa_jac = (pa_model(x_dpd_cur + eps) - y_cur) / eps

    Z_w = pa_jac[:, None] * Z

    A = Z_w.conj().T @ Z_w + REGULARIZATION * np.eye(M)
    b = Z_w.conj().T @ y_desired
    fir_coeffs = np.linalg.solve(A, b)

    x_dpd_check = apply_full_dpd(x_train)
    y_check = pa_model(x_dpd_check)
    train_nmse = compute_nmse_db(y_check, x_train)
    train_acpr = compute_acpr_db(y_check)

    elapsed = time.time() - t_iter
    print(f"  Refine {refine_iter + 1}/{NUM_REFINE_ITERS}: "
          f"NMSE={train_nmse:.2f} dB  ACPR={train_acpr:.2f} dBc ({elapsed:.1f}s)")

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

print()
print("Evaluating on validation data...")


def dpd_fn(x):
    return apply_full_dpd(x)


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
print(f"lut_bins:         {LUT_NUM_BINS}")
print(f"fir_length:       {FIR_LENGTH}")
print(f"total_seconds:    {t_end - t_start:.1f}")
