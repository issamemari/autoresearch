# autoresearch-dpd

This is an experiment to have the LLM do its own research on digital predistortion (DPD) for power amplifier linearization.

## Background

Power amplifiers (PAs) are inherently nonlinear — when driven near saturation for efficiency, they distort the signal (AM/AM compression, AM/PM distortion, memory effects). Digital predistortion applies an inverse nonlinearity before the PA so the cascade is approximately linear. The goal is to find the best DPD model and estimation algorithm to minimize distortion.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr26`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, Saleh PA model with memory, 16-QAM OFDM signal generation, evaluation harness. Do not modify.
   - `linearizer.py` — the file you modify. DPD model, coefficient estimation, hyperparameters.
4. **Verify data exists**: Check that `~/.cache/autoresearch-dpd/` contains the `.npy` data files. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs a single DPD estimation and evaluation. You launch it simply as: `uv run linearizer.py`.

**What you CAN do:**
- Modify `linearizer.py` — this is the only file you edit. Everything is fair game: DPD model structure, polynomial order, memory depth, cross-terms (GMP), estimation algorithm (LS, RLS, LMS, neural networks), number of iterations, regularization, pre/post-processing, basis function selection, pruning, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed PA model, signal generation, and evaluation metrics (NMSE, ACPR, EVM).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml` (numpy, scipy, matplotlib, pandas, torch).
- Modify the evaluation harness. The `evaluate_dpd` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest (most negative) nmse_db.** This is the Normalized Mean Square Error between the linearized PA output and the ideal linear output, measured on a fixed validation set. More negative = better linearization.

**ACPR** (Adjacent Channel Power Ratio) is a secondary metric. It should improve with NMSE but is not the primary optimization target.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing code and getting equal or better results is a win.

**The first run**: Your very first run should always be to establish the baseline, so you will run the linearizer script as is.

## What to explore

The DPD search space is rich. Here are ideas roughly ordered from easy to ambitious:

1. **Hyperparameter tuning**: polynomial order, memory depth, regularization, number of ILA iterations
2. **Generalized Memory Polynomial (GMP)**: add lagging and/or leading cross-terms between the signal and delayed envelope
3. **Basis function pruning**: remove insignificant terms to reduce overfitting
4. **Different estimation algorithms**: recursive least squares (RLS), LMS/NLMS, total least squares
5. **Direct learning architecture**: use the PA model in a closed-loop gradient descent instead of indirect learning
6. **Neural network DPD**: use torch to train a small neural network as the predistorter (real-valued or complex-valued)
7. **Hybrid models**: polynomial + neural network residual correction
8. **Envelope-dependent models**: separate models for different power levels
9. **Crest factor reduction**: pre-process the signal to reduce PAPR before DPD

## Output format

Once the script finishes it prints a summary like this:

```
---
nmse_db:          -38.52
nmse_no_dpd_db:   -20.15
nmse_improvement: 18.37
acpr_before_dbc:  -28.30
acpr_after_dbc:   -48.15
evm_percent:      1.23
poly_order:       7
memory_depth:     3
num_coefficients: 16
num_iterations:   3
total_seconds:    2.5
```

You can extract the key metric from the log file:

```
grep "^nmse_db:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	nmse_db	acpr_after_dbc	status	description
```

1. git commit hash (short, 7 chars)
2. nmse_db achieved (e.g. -38.52) — use 0.00 for crashes
3. acpr_after_dbc (e.g. -48.15) — use 0.00 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	nmse_db	acpr_after_dbc	status	description
a1b2c3d	-35.20	-45.30	keep	baseline MP order=7 memory=3
b2c3d4e	-38.52	-48.15	keep	increase to order=9 memory=5
c3d4e5f	-34.80	-44.90	discard	switch to LMS (worse than LS)
d4e5f6g	0.00	0.00	crash	neural net DPD (shape mismatch)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/apr26`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `linearizer.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run linearizer.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^nmse_db:\|^acpr_after_dbc:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If nmse_db improved (more negative), you "advance" the branch, keeping the git commit
9. If nmse_db is equal or worse (less negative), you git reset back to where you started

**Timeout**: Each experiment should take well under 5 minutes for classical DPD. Neural network DPD might take a few minutes. If a run exceeds 10 minutes, kill it and treat it as a failure.

**Crashes**: If a run crashes (bug, shape mismatch, etc.), use your judgment: If it's something dumb and easy to fix, fix it and re-run. If the idea itself is fundamentally broken, skip it, log "crash", and move on.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human might be asleep. You are autonomous. If you run out of ideas, think harder — try combining previous near-misses, try more radical approaches (neural nets, hybrid models), explore the scipy and torch APIs for useful tools. The loop runs until the human interrupts you, period.
