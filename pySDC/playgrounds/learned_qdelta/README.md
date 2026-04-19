# learned_qdelta playground

Research prototype for a **learned Q_delta-style preconditioner** in pySDC.

The main idea is deliberately narrow:

- pySDC still owns the collocation problem, SDC iteration, and time stepping
- the neural network only proposes **one sweep correction**
- a safety gate checks whether that proposal actually improves the residual enough
- if not, the code falls back to the classical sweeper update

So this playground is **not** a learned time integrator. It is a learned sweep proposal inside a standard pySDC solve.

---

## What this playground contains

### Core scripts

- `data_generation.py`  
  Runs pySDC on a chosen test problem and records one-sweep training samples.

- `dataset.py`  
  Loads saved datasets, validates the dataset contract, normalizes features/targets, and builds train/validation datasets.

- `models.py`  
  Contains the baseline PyTorch model and the `build_model(config)` factory.

- `train.py`  
  Offline training entry point. Supports plain random train/validation split and Dahlquist-specific regime-aware splitting.

- `learned_sweeper.py`  
  Contains:
  - `DataCollectingImplicitSweeper` to export training samples from pySDC sweeps
  - `LearnedQDeltaSweeper` to load a trained model and use it online with fallback

- `hooks.py`  
  Adds extra pySDC statistics for learned-proposal acceptance and residual diagnostics.

- `evaluate.py`  
  Runs a single baseline-vs-learned comparison for one chosen problem setup.

### Dahlquist experiment helpers

- `dahlquist_benchmark.py`  
  Runs a matrix of Dahlquist test cases over several `(lambda, dt)` values and groups results by stiffness regime `|lambda * dt|`.

- `dahlquist_pipeline.py`  
  Convenience driver for the full Dahlquist workflow:
  1. generate data
  2. train model
  3. run benchmark

- `config_dahlquist_baseline.json`  
  Reproducible config template for the full Dahlquist workflow.

### Supporting files

- `environment_micromamba.yml`  
  Minimal micromamba environment suggestion.

- `__init__.py`, `sweeper_utils.py`  
  Small utilities and package structure.

---

## What data is saved?

Each recorded sample corresponds to **one classical SDC sweep** and contains:

- `u0` : state at the left interval boundary
- `U_k` : current stage values before the sweep
- `R_k` : collocation residual-like quantity before the sweep
- `dt` : step size
- `nodes` : collocation nodes
- `qmat` : collocation matrix block used by the sweeper
- `problem_params` : compact vector of problem parameters
- `target_dU` : correction produced by one classical sweep

For the current Dahlquist case, `problem_params[:, 0]` is the scalar `lambda` value.

Datasets also carry metadata:

- `contract_version`
- `problem_name`
- `seed`

The loader in `dataset.py` checks that the required keys exist and that the contract version matches what the code expects.

---

## File-by-file details

## `data_generation.py`

### What it does

This script runs pySDC with the special `DataCollectingImplicitSweeper`. That sweeper behaves like the normal implicit sweeper, but after each classical sweep it records:

- what the solver saw before the sweep
- what correction the classical sweep produced

Those sweep pairs become supervised learning data.

### Why it matters

This is the bridge between pySDC and offline training. If the exported samples are not representative, the model will not learn a useful correction.

### Main outputs

Usually a compressed `.npz` file such as:

- `pySDC/playgrounds/learned_qdelta/data/dahlquist_sweeps.npz`

### Typical usage

```bash
python -m pySDC.playgrounds.learned_qdelta.data_generation \
  --problem dahlquist \
  --output pySDC/playgrounds/learned_qdelta/data/dahlquist_sweeps.npz \
  --num-cases 200 --nsteps 4 --maxiter 4 --num-nodes 3 \
  --lambda-min -25 --lambda-max -1 \
  --dt-min 0.02 --dt-max 0.2 --seed 7
```

### What “good output” looks like

- the file exists
- it contains all required arrays
- the number of samples is comfortably larger than the number of trainable parameters in the model baseline
- the sampled `lambda` and `dt` ranges actually cover the region you want to test later

---

## `dataset.py`

### What it does

This script is not normally run directly. It is used by `train.py` to:

- load `.npz` data
- validate the data contract
- flatten inputs/targets into vectors for the baseline MLP
- compute normalization statistics
- create train/validation PyTorch datasets

### Dahlquist-specific features

For the Dahlquist case it also provides helpers to compute:

- `|lambda * dt|`
- stiffness regime labels from thresholds such as `1.0, 5.0, 15.0`

This is used for regime-aware splits.

### Why it matters

If you want the learned proposal to generalize, random train/validation splits are often too optimistic. The regime-aware split is a more honest test for Dahlquist.

---

## `models.py`

### What it does

Currently this holds a small baseline MLP and the model factory:

- `build_model(config, input_dim, output_dim)`

### Why it matters

The rest of the code only talks to the model through this factory. That means you can later replace the MLP with a neural operator or another architecture without rewriting the sweeper logic.

### Current status

- good for debugging the pipeline
- not intended as the final architecture for more ambitious PDE/operator experiments

---

## `train.py`

### What it does

This is the main offline training script.

It:

1. loads data
2. splits into train/validation sets
3. fits normalizers on the training data
4. builds the model
5. trains using supervised correction loss
6. optionally adds a residual-based penalty
7. writes checkpoints and a manifest

### Important options

- `--split-strategy random`  
  simple random split

- `--split-strategy dahlquist_regime`  
  split according to `|lambda * dt|` stiffness regimes

- `--holdout-regime stiff`  
  keep the stiffest regime entirely out of training and use it for validation

- `--curriculum-thresholds ...`  
  train first on easier samples, then progressively include harder ones

- `--residual-weight`  
  adds a residual-aware term to the loss for the scalar Dahlquist case

### Main outputs

Inside the output directory, typically:

- `best.pt` : best validation checkpoint
- `last.pt` : last checkpoint
- `train_manifest.json` : reproducibility metadata

### What `train_manifest.json` tells you

It records things like:

- dataset path
- split strategy
- regime thresholds
- holdout regime
- curriculum thresholds
- train/validation sizes
- random seed
- git commit hash
- model config

This is the first file to inspect when you want to understand what exactly was trained.

### Typical usage for Dahlquist

```bash
python -m pySDC.playgrounds.learned_qdelta.train \
  --data pySDC/playgrounds/learned_qdelta/data/dahlquist_sweeps.npz \
  --output-dir pySDC/playgrounds/learned_qdelta/checkpoints/dahlquist \
  --split-strategy dahlquist_regime \
  --regime-edges 1.0,5.0,15.0 --holdout-regime stiff \
  --curriculum-thresholds 2.0,8.0,100.0 \
  --residual-weight 0.1 --epochs 60 --batch-size 256
```

### What good training output looks like

During training, you will see lines like:

```text
epoch=   8  train_loss=...  train_mse=...  val_loss=...  val_mse=...  curriculum_thr=...
```

For the Dahlquist test case, promising signs are:

- `train_mse` goes down
- `val_mse` also goes down, not just `train_mse`
- `val_loss` does not diverge when curriculum expands to harder samples
- the best validation checkpoint is not simply the very first epoch

Warning signs are:

- training loss decreases but validation stays flat or gets worse
- validation becomes much worse when entering stiffer regimes
- strong instability from one epoch to the next

---

## `learned_sweeper.py`

### What it does

This file contains the online part of the prototype.

#### `DataCollectingImplicitSweeper`

Used during data generation. It performs a standard classical sweep and records the corresponding training sample.

#### `LearnedQDeltaSweeper`

Used during inference/evaluation. It:

1. builds the model input from current pySDC state
2. predicts a sweep correction
3. forms a trial update
4. computes the old residual and the trial residual
5. accepts the learned proposal only if

```text
trial_residual <= accept_factor * old_residual
```

6. otherwise runs the fallback classical sweep

### Why it matters

This is the safety-critical component. Even if the model is poor, the fallback should preserve robustness.

### What “good” means here

- learned proposals are accepted often enough to matter
- accepted proposals reduce residual reliably
- fallback catches bad proposals cleanly

---

## `hooks.py`

### What it does

Adds extra pySDC statistics after each sweep:

- whether the learned proposal was accepted
- old residual before proposal
- trial residual after learned proposal

### Why it matters

Without this file, you would know only the final step behavior. With it, you can diagnose whether the model is helping at the sweep level.

---

## `evaluate.py`

### What it does

This is the simplest comparison script. It runs one chosen setup twice:

1. classical sweeper only
2. learned sweeper with fallback

and prints summary metrics.

### Typical usage

```bash
python -m pySDC.playgrounds.learned_qdelta.evaluate \
  --checkpoint pySDC/playgrounds/learned_qdelta/checkpoints/best.pt
```

### Example printed output

```text
=== Baseline (classical sweeper) ===
runtime: ...
avg_residual_ratio: ...
avg_niter: ...
acceptance_rate: nan

=== Learned sweeper with fallback ===
runtime: ...
avg_residual_ratio: ...
avg_niter: ...
acceptance_rate: ...
```

### How to read it

- `runtime`  
  Total wall-clock time for the run.

- `avg_residual_ratio`  
  Average ratio of residuals across sweeps. Smaller is better.

- `avg_niter`  
  Average number of SDC iterations per step. Smaller is better.

- `acceptance_rate`  
  Fraction of learned proposals that passed the residual gate. Higher is not automatically better, but a rate near zero means the model is not useful yet.

---

## `dahlquist_benchmark.py`

### What it does

This is the main diagnostic script for the Dahlquist case.

Instead of testing one `(lambda, dt)` pair, it tests a whole matrix and then groups results by stiffness regime based on `|lambda * dt|`.

### Typical usage

Use `=` for negative lambda lists so the CLI parses them correctly:

```bash
python -m pySDC.playgrounds.learned_qdelta.dahlquist_benchmark \
  --checkpoint pySDC/playgrounds/learned_qdelta/checkpoints/dahlquist/best.pt \
  --output pySDC/playgrounds/learned_qdelta/results/dahlquist_matrix.json \
  --lambdas=-2,-5,-10,-20 --dts=0.02,0.05,0.1,0.2 \
  --regime-edges 1.0,5.0,15.0 --maxiter 6 --accept-factor 0.95
```

### Main output

A JSON file such as:

- `pySDC/playgrounds/learned_qdelta/results/dahlquist_matrix.json`

It contains:

- `meta` : benchmark settings
- `rows` : one entry per `(lambda, dt)` pair
- `regime_summary` : averages grouped by regime

### What to inspect in the JSON

Each row includes:

- `lam`
- `dt`
- `lamdt_abs`
- `baseline_avg_niter`
- `learned_avg_niter`
- `baseline_runtime`
- `learned_runtime`
- `baseline_avg_residual_ratio`
- `learned_avg_residual_ratio`
- `learned_acceptance_rate`

### What “good” looks like

For at least some regimes, especially the intended target regime:

- `learned_avg_niter <= baseline_avg_niter`
- `learned_avg_residual_ratio <= baseline_avg_residual_ratio`
- runtime does not become dramatically worse
- acceptance is clearly above zero

Best case:

- learned proposals are accepted frequently in mild/moderate stiffness regimes
- learned method reduces sweeps or time
- fallback keeps stiff regimes from failing badly

---

## `dahlquist_pipeline.py`

### What it does

This script automates the full Dahlquist workflow from a JSON config:

1. generate dataset
2. train model
3. run matrix benchmark

### Why it matters

This is the easiest way to run a reproducible experiment and keep all settings in one place.

### Typical usage

```bash
python -m pySDC.playgrounds.learned_qdelta.dahlquist_pipeline \
  --config pySDC/playgrounds/learned_qdelta/config_dahlquist_baseline.json \
  --output-root pySDC/playgrounds/learned_qdelta/results/pipeline_run
```

### Main outputs

Inside the chosen output directory:

- `dahlquist_sweeps.npz`
- `checkpoints/best.pt`
- `checkpoints/last.pt`
- `checkpoints/train_manifest.json`
- `dahlquist_matrix.json`

If you want a single folder that captures one complete experiment, this is the script to use.

---

## Recommended workflow for the Dahlquist case

### 1. Generate training data

```bash
python -m pySDC.playgrounds.learned_qdelta.data_generation \
  --problem dahlquist \
  --output pySDC/playgrounds/learned_qdelta/data/dahlquist_sweeps.npz \
  --num-cases 200 --nsteps 4 --maxiter 4 --num-nodes 3 \
  --lambda-min -25 --lambda-max -1 \
  --dt-min 0.02 --dt-max 0.2 --seed 7
```

### 2. Train with a regime-aware split

```bash
python -m pySDC.playgrounds.learned_qdelta.train \
  --data pySDC/playgrounds/learned_qdelta/data/dahlquist_sweeps.npz \
  --output-dir pySDC/playgrounds/learned_qdelta/checkpoints/dahlquist \
  --split-strategy dahlquist_regime \
  --regime-edges 1.0,5.0,15.0 --holdout-regime stiff \
  --curriculum-thresholds 2.0,8.0,100.0 \
  --residual-weight 0.1 --epochs 60 --batch-size 256
```

### 3. Sanity-check on one setup

```bash
python -m pySDC.playgrounds.learned_qdelta.evaluate \
  --problem dahlquist \
  --dt 0.1 --lam -8.0 --Tend 1.0 --maxiter 6 \
  --checkpoint pySDC/playgrounds/learned_qdelta/checkpoints/dahlquist/best.pt
```

### 4. Run the benchmark matrix

```bash
python -m pySDC.playgrounds.learned_qdelta.dahlquist_benchmark \
  --checkpoint pySDC/playgrounds/learned_qdelta/checkpoints/dahlquist/best.pt \
  --output pySDC/playgrounds/learned_qdelta/results/dahlquist_matrix.json \
  --lambdas=-2,-5,-10,-20 --dts=0.02,0.05,0.1,0.2 \
  --regime-edges 1.0,5.0,15.0 --maxiter 6 --accept-factor 0.95
```

### 5. Inspect the outputs

- `train_manifest.json`
- benchmark JSON regime summaries
- checkpoint metadata
- printed acceptance and residual behavior

---

## How to decide whether the trained model is any good

For this prototype, there are **two levels of “good”**:

### A. Good as a predictor

Look at training and validation numbers from `train.py`.

The model is promising if:

- validation loss decreases over training
- validation MSE tracks training MSE reasonably well
- the model does not collapse when the curriculum introduces stiffer data
- held-out stiff-regime validation is still finite and improving

The model is weak if:

- validation barely improves
- validation is much worse than training
- the stiff holdout regime remains essentially unsolved

### B. Good as a preconditioner proposal

This is the more important criterion.

Even if the regression loss looks good, the learned proposal is only valuable if it helps pySDC.

Check the evaluation and benchmark outputs:

#### 1. Acceptance rate

- near `0.0` means the gate rejects almost everything
- moderate rates can still be useful if accepted proposals are genuinely strong
- very high rates are only good if iterations/time also improve

#### 2. Residual reduction

Compare:

- `learned_avg_residual_ratio`
- `baseline_avg_residual_ratio`

Lower is better.

#### 3. Iteration counts

Compare:

- `learned_avg_niter`
- `baseline_avg_niter`

If the learned sweeper does not reduce iterations in any meaningful regime, the current model is not yet useful.

#### 4. Runtime

Compare:

- `learned_runtime`
- `baseline_runtime`

Small slowdowns can be acceptable in a prototype if iteration quality improves, but large slowdowns are a warning sign.

#### 5. Regime dependence

Look at which `|lambda*dt|` regimes improve.

That is often more informative than global averages. A useful first result is:

- learned proposals help in mild/moderate regimes
- fallback preserves behavior in very stiff regimes

### A practical rule of thumb

For the Dahlquist prototype, call the training run **successful** if all of the following are true:

1. validation loss improves during training
2. acceptance rate is clearly above zero in at least one target regime
3. learned method matches or improves residual reduction in that regime
4. learned method matches or reduces average SDC iterations in that regime
5. fallback prevents catastrophic degradation outside the training comfort zone

If only item 1 is true, the network may be learning the regression task but not yet helping the solver.

---

## 1D heat equation (MLP) example

The same MLP infrastructure also supports `heatNd_unforced` as a second
proof-of-flexibility problem class.

Generate data:

```bash
python -m pySDC.playgrounds.learned_qdelta.data_generation \
  --problem heat1d \
  --output pySDC/playgrounds/learned_qdelta/data/heat1d_sweeps.npz \
  --num-cases 20 --nsteps 3 --maxiter 4
```

Train:

```bash
python -m pySDC.playgrounds.learned_qdelta.train \
  --data pySDC/playgrounds/learned_qdelta/data/heat1d_sweeps.npz \
  --output-dir pySDC/playgrounds/learned_qdelta/checkpoints/heat1d \
  --epochs 30 --batch-size 64
```

Evaluate:

```bash
python -m pySDC.playgrounds.learned_qdelta.evaluate \
  --problem heat1d \
  --dt 0.01 --nu 0.1 --Tend 0.05 --maxiter 4 \
  --checkpoint pySDC/playgrounds/learned_qdelta/checkpoints/heat1d/best.pt
```

---

## 1D heat equation – FNO pipeline

The **Fourier Neural Operator** (FNO) variant is the recommended approach for
spatial PDEs.  Unlike the MLP, the FNO operates directly on the spatial fields
`(u0, U_k, R_k)` via global FFT + pointwise convolutions, which means:

- it can be **trained on one grid size** (e.g. `nvars=127`) and then
  **applied to any other grid size** (`nvars=255`, `nvars=511`, …) without
  retraining (resolution-transfer test).
- the correction output is a full spatial field `ΔU` — richer than the
  per-node scale-factors produced by the dimension-agnostic MLP.

### Components

| Script | Role |
|---|---|
| `train_fno.py` | Train `FNO1d` on raw spatial-field sweep data |
| `learned_sweeper.py` → `FNOLearnedQDeltaSweeper` | Online inference with fallback gate |
| `heat1d_fno_benchmark.py` | Compare FNO vs classical on multiple grid sizes |
| `heat1d_fno_pipeline.py` | **Convenience driver**: generate → train → benchmark |

### One-command workflow

```bash
python -m pySDC.playgrounds.learned_qdelta.heat1d_fno_pipeline \
    --num-cases 200 --epochs 80 \
    --nvars 127,255 \
    --output-root pySDC/playgrounds/learned_qdelta/results/heat1d_fno
```

This runs all three steps and writes:

```
results/heat1d_fno/
  heat1d_sweeps.npz          ← training data
  checkpoints/best.pt        ← best FNO checkpoint
  checkpoints/last.pt
  checkpoints/train_manifest.json
  benchmark_report.json      ← per-nvars metrics
  plots/
    training_curve.png
    niter_vs_nvars.png
    runtime_vs_nvars.png
    residual_ratio_parity.png
    gate_behavior.png
    composite_score.png
    inference_overhead.png
```

### Smoke test (≈1 min)

```bash
python -m pySDC.playgrounds.learned_qdelta.heat1d_fno_pipeline \
    --num-cases 20 --epochs 5 --nvars 127 \
    --output-root /tmp/heat1d_fno_smoke
```

### Manual step-by-step

#### 1. Generate spatial-field training data

```bash
python -m pySDC.playgrounds.learned_qdelta.data_generation \
  --problem heat1d \
  --output pySDC/playgrounds/learned_qdelta/data/heat1d_sweeps.npz \
  --num-cases 200 --nsteps 4 --maxiter 4
```

All REQUIRED_SAMPLE_KEYS (including raw spatial arrays `u0`, `U_k`, `R_k`,
`target_dU`) are stored — required by the FNO loader.

#### 2. Train the FNO

```bash
python -m pySDC.playgrounds.learned_qdelta.train_fno \
  --data pySDC/playgrounds/learned_qdelta/data/heat1d_sweeps.npz \
  --output-dir pySDC/playgrounds/learned_qdelta/checkpoints/heat1d_fno_v1 \
  --epochs 80 --width 64 --modes 16 --depth 4 --lr 1e-3 \
  --balance-cfl --cfl-edges 0.2,1.0,4.0 \
  --residual-proj-weight 0.2 --cosine-gate-weight 0.05 --checkpoint-metric composite
```

#### 3. Benchmark (resolution transfer included)

```bash
python -m pySDC.playgrounds.learned_qdelta.heat1d_fno_benchmark \
  --checkpoint pySDC/playgrounds/learned_qdelta/checkpoints/heat1d_fno_v1/best.pt \
  --nvars 127,255 \
  --repeats 3 \
  --accept-factor 0.95 --accept-factor-min 0.92 --accept-factor-max 1.01 \
  --accept-factor-slope 0.02 --confidence-ratio-max 4.0 \
  --parity-runtime-factor 1.1 --min-acceptance 0.05 \
  --output pySDC/playgrounds/learned_qdelta/results/heat1d_fno_benchmark.json \
  --plot-dir pySDC/playgrounds/learned_qdelta/results/heat1d_fno_plots
```

### New parity-focused diagnostics

`heat1d_fno_benchmark.py` now writes a stricter report with:

- repeated-run means/stds (`--repeats`) to reduce runtime noise
- parity flags per grid (`niter`, `residual`, `runtime`, `acceptance`)
- a composite score for checkpoint comparison
- gate diagnostics (acceptance, confidence pre-reject fraction, gate reason histogram)
- timing diagnostics (`avg_inference_time`, `avg_trial_eval_time`) for overhead tracking

And plot outputs for fast visual checking:

- `training_curve.png`
- `niter_vs_nvars.png`
- `runtime_vs_nvars.png`
- `residual_ratio_parity.png`
- `gate_behavior.png`
- `composite_score.png`
- `inference_overhead.png`

### What "good" results look like

For the heat1d FNO, call the run **successful** if:

1. FNO validation MSE decreases during training.
2. Acceptance rate is clearly above zero on the training grid (`nvars=127`).
3. FNO average iterations ≤ classical on the training grid.
4. Error does not degrade significantly on the transfer grid (`nvars=255`).
5. Fallback gate prevents any accuracy catastrophe on unseen grid sizes.

The most developed workflow is still the Dahlquist one, but the heat1d FNO
pipeline demonstrates genuine resolution-transfer capability.

---

## Notes and caveats

- First target problem is scalar Dahlquist (`testequation0d`) with real negative `lambda`.
- The same pipeline also supports a 1D heat equation preset (`heatNd_unforced`) to demonstrate problem-class flexibility.
- Tensor layout supports flattened state vectors and can later be adapted to operator-style models.
- FNO / NeuralOperator integration is intentionally left as a future extension point in `build_model`.
- The optional residual regularizer in `train.py` is currently specialized for scalar Dahlquist data.
- Data files include a contract version, and the loader validates required keys.
- Training writes `train_manifest.json` for reproducibility.
- If your environment reports duplicate `libomp` runtime issues on macOS, prefix commands with `KMP_DUPLICATE_LIB_OK=TRUE` as a temporary workaround.


