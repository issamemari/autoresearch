"""
DPD linearizer script. Single run, outputs performance metrics.
This is the file the agent modifies. Everything is fair game: DPD model
structure, coefficient estimation algorithm, hyperparameters, iterations,
pre/post-processing. The only constraint: use the fixed PA model and
evaluation from prepare.py.

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
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# DPD model structure (GMP: Generalized Memory Polynomial)
POLY_ORDER = 7          # maximum polynomial order (odd orders: 1, 3, 5, 7)
MEMORY_DEPTH = 3        # aligned memory depth
CROSS_ORDER = 5         # cross-term polynomial order (odd: 3, 5)
CROSS_MEMORY = 2        # cross-term memory depth
CROSS_LAG = 2           # max envelope lag for cross-terms

# Estimation
NUM_ITERATIONS = 1      # indirect learning architecture iterations
REGULARIZATION = 1e-6   # ridge regularization for least squares

# ---------------------------------------------------------------------------
# DPD Model: Generalized Memory Polynomial (GMP)
# ---------------------------------------------------------------------------
# Aligned terms:  x(n-m) * |x(n-m)|^{2k}
# Lagging cross:  x(n-m) * |x(n-m-l)|^{2k}  (envelope from past samples)
# Leading cross:  x(n-m) * |x(n-m+l)|^{2k}  (envelope from future samples)
#
# The cross-terms capture signal-envelope interactions that a standard
# memory polynomial misses — critical for Wiener-type PAs.

def build_gmp_basis_matrix(x, K_a, M_a, K_c, M_c, L_c):
    """
    Build the GMP basis matrix with aligned + lagging + leading cross-terms.

    Args:
        x: complex input signal (length N)
        K_a: aligned polynomial terms (orders 1, 3, ..., 2*K_a-1)
        M_a: aligned memory depth
        K_c: cross-term polynomial terms (orders 3, 5, ..., 2*K_c+1)
        M_c: cross-term memory depth
        L_c: max cross-term envelope lag/lead
    Returns:
        U: basis matrix
    """
    N = len(x)
    pad = max(M_a, M_c + L_c)
    x_pad = np.concatenate([np.zeros(pad, dtype=np.complex128), x,
                            np.zeros(L_c, dtype=np.complex128)])

    columns = []

    # Aligned terms: x(n-m) * |x(n-m)|^{2k}
    for k in range(K_a):
        for m in range(M_a + 1):
            x_del = x_pad[pad - m : N + pad - m]
            columns.append(x_del * np.abs(x_del) ** (2 * k))

    # Lagging cross-terms: x(n-m) * |x(n-m-l)|^{2k}  for l >= 1
    for k in range(1, K_c + 1):
        for m in range(M_c + 1):
            for l in range(1, L_c + 1):
                x_sig = x_pad[pad - m : N + pad - m]
                x_env = x_pad[pad - m - l : N + pad - m - l]
                columns.append(x_sig * np.abs(x_env) ** (2 * k))

    # Leading cross-terms: x(n-m) * |x(n-m+l)|^{2k}  for l >= 1
    for k in range(1, K_c + 1):
        for m in range(M_c + 1):
            for l in range(1, L_c + 1):
                x_sig = x_pad[pad - m : N + pad - m]
                x_env = x_pad[pad - m + l : N + pad - m + l]
                columns.append(x_sig * np.abs(x_env) ** (2 * k))

    return np.column_stack(columns)


def apply_dpd(x, coefficients):
    """Apply GMP DPD with the given coefficients to input signal x."""
    K_a = (POLY_ORDER + 1) // 2
    K_c = (CROSS_ORDER - 1) // 2
    U = build_gmp_basis_matrix(x, K_a, MEMORY_DEPTH, K_c, CROSS_MEMORY, CROSS_LAG)
    return U @ coefficients

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
np.random.seed(42)

# Load PA training data
x_train, y_train = load_data("train")
print(f"Training samples: {len(x_train):,}")

# Estimate PA linear gain for indirect learning normalization
pa_gain = np.vdot(x_train, y_train) / np.vdot(x_train, x_train)
print(f"PA linear gain: {abs(pa_gain):.4f} (phase: {np.degrees(np.angle(pa_gain)):.2f} deg)")

# PA-only performance (baseline without DPD)
nmse_no_dpd = compute_nmse_db(y_train, x_train)
print(f"PA NMSE (no DPD): {nmse_no_dpd:.2f} dB")

K_a = (POLY_ORDER + 1) // 2
K_c = (CROSS_ORDER - 1) // 2
n_aligned = K_a * (MEMORY_DEPTH + 1)
n_cross = 2 * K_c * (CROSS_MEMORY + 1) * CROSS_LAG  # lagging + leading
num_coefficients = n_aligned + n_cross
print(f"GMP model: aligned order={POLY_ORDER} mem={MEMORY_DEPTH} ({n_aligned} terms)")
print(f"           cross order={CROSS_ORDER} mem={CROSS_MEMORY} lag={CROSS_LAG} ({n_cross} terms)")
print(f"           total coefficients: {num_coefficients}")

# ---------------------------------------------------------------------------
# Training: Indirect Learning Architecture (ILA)
# ---------------------------------------------------------------------------

print()
print("Training DPD (Indirect Learning Architecture)...")

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

    # Build GMP basis matrix
    U = build_gmp_basis_matrix(dpd_input, K_a, MEMORY_DEPTH,
                               K_c, CROSS_MEMORY, CROSS_LAG)

    # Regularized least squares
    A = U.conj().T @ U + REGULARIZATION * np.eye(num_coefficients)
    b = U.conj().T @ dpd_target
    coefficients = np.linalg.solve(A, b)

    # Quick training NMSE check
    x_dpd_check = apply_dpd(x_train, coefficients)
    y_check = pa_model(x_dpd_check)
    train_nmse = compute_nmse_db(y_check, x_train)

    elapsed = time.time() - t_iter
    print(f"  Iteration {iteration + 1}/{NUM_ITERATIONS}: "
          f"train_nmse = {train_nmse:.2f} dB ({elapsed:.1f}s)")

# ---------------------------------------------------------------------------
# Evaluation (uses fixed validation data from prepare.py)
# ---------------------------------------------------------------------------

print()
print("Evaluating on validation data...")

def dpd_fn(x):
    return apply_dpd(x, coefficients)

results = evaluate_dpd(dpd_fn)

# Generate diagnostic plots
x_val, _ = load_data("val")
y_val_nodpd = pa_model(x_val)
x_dpd_val = dpd_fn(x_val)
y_val_dpd = pa_model(x_dpd_val)
plot_diagnostics(x_val, y_val_nodpd, y_val_dpd, "diagnostics.png")

t_end = time.time()

# ---------------------------------------------------------------------------
# Output summary
# ---------------------------------------------------------------------------

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
print(f"num_iterations:   {NUM_ITERATIONS}")
print(f"total_seconds:    {t_end - t_start:.1f}")
