# autoresearch-dpd

Autonomous AI-driven research for digital predistortion (DPD) of power amplifiers.

Forked from [karpathy/autoresearch](https://github.com/karpathy/autoresearch), which uses AI agents to autonomously optimize LLM pretraining. This fork applies the same idea to a different domain: finding the best digital predistorter for a nonlinear power amplifier.

## How it works

An AI agent is given a PA simulation environment and a baseline DPD implementation. It modifies the DPD code, runs the estimation, checks if the result improved (lower NMSE), keeps or discards, and repeats. You wake up to a log of experiments and a better linearizer.

The repo has three files that matter:

- **`prepare.py`** — fixed PA behavioral model (Saleh with memory), 16-QAM OFDM signal generation, and evaluation metrics (NMSE, ACPR, EVM). Not modified.
- **`linearizer.py`** — the single file the agent edits. Contains the DPD model, coefficient estimation algorithm, and hyperparameters. Everything is fair game: model structure, estimation method, basis functions, neural networks, etc.
- **`program.md`** — instructions for the agent. Edited by the human.

The metric is **nmse_db** (Normalized Mean Square Error in dB) — more negative is better. The PA model and validation data are fixed, so all experiments are directly comparable.

## The PA model

The simulated PA uses the classic **Saleh model** (A. Saleh, IEEE Trans. Comm., 1981) with a Wiener memory structure:

- **AM/AM**: `A(r) = alpha_a * r / (1 + beta_a * r^2)` — compressive gain
- **AM/PM**: `Phi(r) = alpha_phi * r^2 / (1 + beta_phi * r^2)` — phase distortion
- **Memory**: linear FIR pre-filter (4 taps) before the Saleh nonlinearity (Wiener model)
- **Signal**: 16-QAM OFDM, 20 MHz bandwidth, ~6 dB input back-off

The Wiener structure (linear filter -> nonlinearity) means a basic memoryless DPD won't fully linearize the PA — the agent needs to compensate for both the Saleh nonlinearity and the memory effects.

## Quick start

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/). No GPU required (classical DPD runs on CPU; torch is available for neural network DPD).

```bash
# 1. Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Generate PA simulation data (one-time, ~5 seconds)
uv run prepare.py

# 4. Run a single DPD experiment
uv run linearizer.py
```

## Running the agent

Point your Claude/Codex agent at this repo and prompt:

```
Read program.md and let's kick off a new experiment!
```

The agent will set up a branch, establish a baseline, and start iterating autonomously.

## What the agent explores

The DPD search space includes:

- **Model structure**: Memory polynomial, Generalized Memory Polynomial (GMP), Volterra series, neural networks
- **Estimation algorithms**: Least squares, RLS, LMS/NLMS, gradient descent, direct/indirect learning
- **Hyperparameters**: Polynomial order, memory depth, cross-term lags, regularization
- **Advanced techniques**: Basis pruning, hybrid polynomial+NN models, crest factor reduction

## Project structure

```
prepare.py      — PA model, signal generation, evaluation metrics (do not modify)
linearizer.py   — DPD model + estimation (agent modifies this)
program.md      — agent instructions
pyproject.toml  — dependencies
```

## Design choices

- **Single file to modify.** The agent only touches `linearizer.py`. Diffs are reviewable.
- **Fixed PA and evaluation.** The PA model and metrics in `prepare.py` are the ground truth. This ensures all experiments are fairly compared.
- **No GPU required.** Classical DPD (least squares) runs in seconds on CPU. Torch is included for agents who want to try neural network DPD.
- **Simulation-based.** The PA is a behavioral model, not real hardware. This allows hundreds of experiments per night. The best DPD found could then be deployed to a real PA.

## License

MIT
