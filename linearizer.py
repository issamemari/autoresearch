"""
DPD linearizer — Real-valued neural network DPD.
Uses a small feedforward NN operating on [|x|, Re(x), Im(x)] features
with memory taps, trained to minimize PA output error directly.

Usage: uv run linearizer.py
"""

import time
import numpy as np
import torch
import torch.nn as nn
from prepare import (
    TIME_BUDGET, SAMPLE_RATE, SIGNAL_BANDWIDTH,
    load_data, pa_model, evaluate_dpd, plot_diagnostics,
    compute_nmse_db, compute_acpr_db,
)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

MEMORY_TAPS = 5         # number of memory taps (past samples)
HIDDEN_SIZE = 64        # NN hidden layer size
NUM_LAYERS = 3          # NN depth
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
BATCH_SIZE = 4096

# ---------------------------------------------------------------------------
# Neural Network DPD Model
# ---------------------------------------------------------------------------

class DPDNN(nn.Module):
    """Real-valued NN DPD operating on [Re, Im, |x|] features with memory."""

    def __init__(self, memory_taps, hidden_size, num_layers):
        super().__init__()
        # Input: for each of (memory_taps+1) taps: [Re(x), Im(x), |x|]
        input_dim = (memory_taps + 1) * 3
        layers = []
        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(nn.Tanh())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())
        # Output: [Re(y_dpd), Im(y_dpd)]
        layers.append(nn.Linear(hidden_size, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x_features):
        return self.net(x_features)


def build_features(x, memory_taps):
    """Build NN input features: [Re(x(n)), Im(x(n)), |x(n)|, ..., Re(x(n-M)), ...]."""
    N = len(x)
    M = memory_taps
    x_pad = np.concatenate([np.zeros(M, dtype=np.complex128), x])

    feat_list = []
    for m in range(M + 1):
        x_del = x_pad[M - m : N + M - m]
        feat_list.append(np.real(x_del))
        feat_list.append(np.imag(x_del))
        feat_list.append(np.abs(x_del))

    return np.column_stack(feat_list)


def apply_nn_dpd(x, model, memory_taps, device='cpu'):
    """Apply the NN DPD to a signal."""
    features = build_features(x, memory_taps)
    feat_tensor = torch.tensor(features, dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        out = model(feat_tensor).cpu().numpy()

    return out[:, 0] + 1j * out[:, 1]


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
np.random.seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

x_train, y_train = load_data("train")
print(f"Training samples: {len(x_train):,}")

pa_gain = np.vdot(x_train, y_train) / np.vdot(x_train, x_train)
print(f"PA linear gain: {abs(pa_gain):.4f} (phase: {np.degrees(np.angle(pa_gain)):.2f} deg)")

nmse_no_dpd = compute_nmse_db(y_train, x_train)
print(f"PA NMSE (no DPD): {nmse_no_dpd:.2f} dB")

# ---------------------------------------------------------------------------
# Phase 1: Train NN as post-distorter (ILA-style initialization)
# ---------------------------------------------------------------------------
# Train NN to map PA_output/G -> PA_input (the post-inverse).
# Then use it as a pre-distorter.

print()
print("Phase 1: Training NN post-inverse (ILA init)...")

# Build training data for post-inverse
dpd_input = y_train / pa_gain  # normalized PA output
dpd_target = x_train            # original input

features_train = build_features(dpd_input, MEMORY_TAPS)
targets_re = np.real(dpd_target)
targets_im = np.imag(dpd_target)
targets_np = np.column_stack([targets_re, targets_im])

feat_tensor = torch.tensor(features_train, dtype=torch.float32, device=device)
tgt_tensor = torch.tensor(targets_np, dtype=torch.float32, device=device)

model = DPDNN(MEMORY_TAPS, HIDDEN_SIZE, NUM_LAYERS).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

N = len(x_train)
num_batches = (N + BATCH_SIZE - 1) // BATCH_SIZE

for epoch in range(NUM_EPOCHS):
    model.train()
    perm = torch.randperm(N, device=device)
    epoch_loss = 0.0

    for i in range(num_batches):
        idx = perm[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        feat_batch = feat_tensor[idx]
        tgt_batch = tgt_tensor[idx]

        pred = model(feat_batch)
        loss = nn.functional.mse_loss(pred, tgt_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / num_batches

    if (epoch + 1) % 10 == 0 or epoch == 0:
        # Evaluate on training set
        x_dpd_check = apply_nn_dpd(x_train, model, MEMORY_TAPS, device)
        y_check = pa_model(x_dpd_check)
        train_nmse = compute_nmse_db(y_check, x_train)
        train_acpr = compute_acpr_db(y_check)
        print(f"  Epoch {epoch + 1:3d}/{NUM_EPOCHS}: loss={avg_loss:.6f}  "
              f"NMSE={train_nmse:.2f} dB  ACPR={train_acpr:.2f} dBc")

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

print()
print("Evaluating on validation data...")


def dpd_fn(x):
    return apply_nn_dpd(x, model, MEMORY_TAPS, device)


results = evaluate_dpd(dpd_fn)

x_val, _ = load_data("val")
y_val_nodpd = pa_model(x_val)
x_dpd_val = dpd_fn(x_val)
y_val_dpd = pa_model(x_dpd_val)
plot_diagnostics(x_val, y_val_nodpd, y_val_dpd, "diagnostics.png")

t_end = time.time()

n_params = sum(p.numel() for p in model.parameters())
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
print(f"nn_params:        {n_params}")
print(f"total_seconds:    {t_end - t_start:.1f}")
