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

# DPD model structure
POLY_ORDER = 7          # maximum polynomial order (odd orders: 1, 3, 5, 7)
MEMORY_DEPTH = 3        # number of memory taps (0 = memoryless DPD)

# Estimation
NUM_ITERATIONS = 3      # indirect learning architecture iterations
REGULARIZATION = 1e-6   # ridge regularization for least squares

# ---------------------------------------------------------------------------
# DPD Model: Memory Polynomial
# ---------------------------------------------------------------------------
# y(n) = sum_{k=0}^{K-1} sum_{m=0}^{M} w_{k,m} * x(n-m) * |x(n-m)|^{2k}
#
# K = (POLY_ORDER + 1) / 2 polynomial terms (odd orders only)
# M = MEMORY_DEPTH memory taps
# Total coefficients: K * (M + 1)

def build_basis_matrix(x, K, M):
    """
    Build the memory polynomial basis matrix.

    Each column is one basis function: x(n-m) * |x(n-m)|^{2k}
    for k = 0, ..., K-1 and m = 0, ..., M.

    Args:
        x: complex input signal (length N)
        K: number of polynomial terms
        M: memory depth
    Returns:
        U: basis matrix of shape (N, K*(M+1))
    """
    N = len(x)
    x_pad = np.concatenate([np.zeros(M, dtype=np.complex128), x])

    columns = []
    for k in range(K):
        for m in range(M + 1):
            x_del = x_pad[M - m : N + M - m]
            columns.append(x_del * np.abs(x_del) ** (2 * k))

    return np.column_stack(columns)


def apply_dpd(x, coefficients, K, M):
    """Apply DPD with the given coefficients to input signal x."""
    U = build_basis_matrix(x, K, M)
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

K = (POLY_ORDER + 1) // 2  # number of polynomial terms
num_coefficients = K * (MEMORY_DEPTH + 1)
print(f"DPD model: order={POLY_ORDER}, memory={MEMORY_DEPTH}, "
      f"coefficients={num_coefficients}")

# ---------------------------------------------------------------------------
# Training: Indirect Learning Architecture (ILA)
# ---------------------------------------------------------------------------
# The ILA estimates a post-inverse of the PA, then uses it as a pre-distorter.
#
# Iteration 1: Estimate post-distorter from PA I/O data directly.
#   - Input to basis matrix: PA output (normalized)
#   - Target: PA input
#
# Iterations 2+: Apply current DPD, capture new PA output, re-estimate.
#   - Input to basis matrix: new PA output (normalized)
#   - Target: original input
#
# This iterative refinement converges to a better DPD as the operating
# point of the post-distorter aligns with the pre-distorter usage.

print()
print("Training DPD (Indirect Learning Architecture)...")

coefficients = None

for iteration in range(NUM_ITERATIONS):
    t_iter = time.time()

    if iteration == 0:
        # First iteration: use PA output as post-distorter input
        dpd_input = y_train / pa_gain
        dpd_target = x_train
    else:
        # Subsequent: apply current DPD, capture new PA output, re-estimate
        x_dpd = apply_dpd(x_train, coefficients, K, MEMORY_DEPTH)
        y_new = pa_model(x_dpd)
        dpd_input = y_new / pa_gain
        dpd_target = x_train

    # Build basis matrix
    U = build_basis_matrix(dpd_input, K, MEMORY_DEPTH)

    # Regularized least squares: (U^H U + lambda*I) w = U^H d
    A = U.conj().T @ U + REGULARIZATION * np.eye(num_coefficients)
    b = U.conj().T @ dpd_target
    coefficients = np.linalg.solve(A, b)

    # Quick training NMSE check
    x_dpd_check = apply_dpd(x_train, coefficients, K, MEMORY_DEPTH)
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
    return apply_dpd(x, coefficients, K, MEMORY_DEPTH)

results = evaluate_dpd(dpd_fn)

# Generate diagnostic plots (PSD/ACPR, AM/AM, AM/PM, constellation)
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
print(f"poly_order:       {POLY_ORDER}")
print(f"memory_depth:     {MEMORY_DEPTH}")
print(f"num_coefficients: {num_coefficients}")
print(f"num_iterations:   {NUM_ITERATIONS}")
print(f"total_seconds:    {t_end - t_start:.1f}")
