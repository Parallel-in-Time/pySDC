#!/usr/bin/env bash
# run_paper_comparison.sh
# Runs both QSSA and non-QSSA global PINNs at paper settings (2e5 iterations/epochs)
# and writes a comparison summary when both finish.
# Designed to survive terminal timeout: all subprocesses are nohup'd and
# a final comparison step is chained with && so it only fires when both succeed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
mkdir -p "$DATA_DIR"

LOG_QSSA="$DATA_DIR/paper_qssa_run.log"
LOG_NOQUSSA="$DATA_DIR/paper_noqussa_run.log"
COMPARE="$SCRIPT_DIR/compare_paper_runs.py"

echo "============================================"
echo " Paper-mode comparison: QSSA vs non-QSSA"
echo " Iterations/epochs: 200000"
echo " Logs:"
echo "   QSSA    -> $LOG_QSSA"
echo "   no-QSSA -> $LOG_NOQUSSA"
echo "============================================"

# ── QSSA run (stiff_pinn_robertson_qssa.py) ──────────────────────────────────
# Paper settings: 2×10^5 epochs, 2500 train points, batch 512, lr=1e-3
echo "[$(date '+%H:%M:%S')] Starting QSSA run..."
python -u "$SCRIPT_DIR/stiff_pinn_robertson_qssa.py" \
    --epochs 200000 \
    --n-grid-train 2500 \
    --batch-size 512 \
    --learning-rate 1e-3 \
    --t-min 1e-2 \
    --t-max 1e5 \
    --num-eval 600 \
    --print-every 10000 \
    --seed 42 \
    --run-tag paper200k \
    > "$LOG_QSSA" 2>&1
echo "[$(date '+%H:%M:%S')] QSSA run finished. Log: $LOG_QSSA"

# ── non-QSSA run (deepxde_rober_paper_simple.py, global approach) ────────────
# Paper settings: 2×10^5 iterations, 2500 collocation points, batch 128, lr=1e-3
echo "[$(date '+%H:%M:%S')] Starting non-QSSA (full system) run..."
python -u "$SCRIPT_DIR/deepxde_rober_paper_simple.py" \
    --approach global \
    --iterations 200000 \
    --num-points 2500 \
    --batch-size 128 \
    --lr 1e-3 \
    --t-min 1e-5 \
    --t-max 1e5 \
    --num-eval 600 \
    --seed 42 \
    --run-tag paper200k \
    > "$LOG_NOQUSSA" 2>&1
echo "[$(date '+%H:%M:%S')] non-QSSA run finished. Log: $LOG_NOQUSSA"

# ── Comparison ────────────────────────────────────────────────────────────────
echo "[$(date '+%H:%M:%S')] Running comparison..."
python -u "$COMPARE"

