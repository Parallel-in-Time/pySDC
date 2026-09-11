"""Parameter sweep for Figure-4-style regular PINN ROBER runs.

Runs `deepxde_rober_regular_fig4.py` for a grid of hyperparameters,
collects metrics, and ranks runs by low loss/error while marking early-stopped
or failed cases.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import itertools
import json
import subprocess
import sys
from pathlib import Path


def parse_csv_list(raw: str, cast):
    return [cast(x.strip()) for x in raw.split(",") if x.strip()]


def parse_metrics(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for line in path.read_text(encoding="ascii").splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        data[k.strip()] = v.strip()
    return data


def as_float(metrics: dict[str, str], key: str, default: float = float("inf")) -> float:
    try:
        return float(metrics.get(key, default))
    except Exception:
        return default


def run_one(
    script: Path,
    out_dir: Path,
    sweep_tag: str,
    run_id: int,
    seed: int,
    lr: float,
    batch_size: int,
    num_points: int,
    iterations: int,
    display_every: int,
    max_loss_stop: float,
    max_divergence_loss: float,
    divergence_warmup_iterations: int,
    loss_warmup_iterations: int,
    loss_breach_patience: int,
    min_dynamic_ratio: float,
    force_cpu: bool,
) -> dict:
    tag = f"{sweep_tag}_{run_id:03d}"
    cmd = [
        sys.executable,
        str(script),
        "--seed",
        str(seed),
        "--lr",
        str(lr),
        "--batch-size",
        str(batch_size),
        "--num-points",
        str(num_points),
        "--iterations",
        str(iterations),
        "--display-every",
        str(display_every),
        "--max-loss-stop",
        str(max_loss_stop),
        "--max-divergence-loss",
        str(max_divergence_loss),
        "--divergence-warmup-iterations",
        str(divergence_warmup_iterations),
        "--loss-warmup-iterations",
        str(loss_warmup_iterations),
        "--loss-breach-patience",
        str(loss_breach_patience),
        "--min-dynamic-ratio",
        str(min_dynamic_ratio),
        "--run-tag",
        tag,
    ]
    if not force_cpu:
        cmd.append("--no-force-cpu")

    proc = subprocess.run(cmd, cwd=str(script.parent), capture_output=True, text=True)

    log_path = out_dir / f"deepxde_rober_regular_fig4_{tag}.log"
    log_path.write_text(proc.stdout + "\n\n[stderr]\n" + proc.stderr, encoding="utf-8")

    metrics_path = out_dir / f"deepxde_rober_regular_fig4_{tag}_metrics.txt"
    metrics = parse_metrics(metrics_path)

    early_stopped = metrics.get("early_stopped", "False") == "True"
    diverged_stopped = metrics.get("diverged_stopped", "False") == "True"
    trivial_solution = metrics.get("trivial_solution", "False") == "True"
    final_total_loss = as_float(metrics, "final_total_loss")
    rel_l2_y1 = as_float(metrics, "rel_l2_y1")
    rel_l2_y2 = as_float(metrics, "rel_l2_y2")
    rel_l2_y3 = as_float(metrics, "rel_l2_y3")
    rel_l2_mean = (rel_l2_y1 + rel_l2_y2 + rel_l2_y3) / 3.0
    mass_drift = as_float(metrics, "mass_drift")
    dynamic_ratio_y1 = as_float(metrics, "dynamic_ratio_y1")
    dynamic_ratio_y2 = as_float(metrics, "dynamic_ratio_y2")
    dynamic_ratio_y3 = as_float(metrics, "dynamic_ratio_y3")

    status = "ok"
    if proc.returncode != 0:
        status = "failed"
    elif diverged_stopped:
        status = "diverged"
    elif trivial_solution:
        status = "trivial"
    elif early_stopped:
        status = "early_stopped"

    return {
        "run_id": run_id,
        "tag": tag,
        "status": status,
        "returncode": proc.returncode,
        "seed": seed,
        "lr": lr,
        "batch_size": batch_size,
        "num_points": num_points,
        "iterations": iterations,
        "display_every": display_every,
        "max_loss_stop": max_loss_stop,
        "max_divergence_loss": max_divergence_loss,
        "divergence_warmup_iterations": divergence_warmup_iterations,
        "min_dynamic_ratio": min_dynamic_ratio,
        "diverged_stopped": diverged_stopped,
        "trivial_solution": trivial_solution,
        "final_total_loss": final_total_loss,
        "rel_l2_y1": rel_l2_y1,
        "rel_l2_y2": rel_l2_y2,
        "rel_l2_y3": rel_l2_y3,
        "rel_l2_mean": rel_l2_mean,
        "dynamic_ratio_y1": dynamic_ratio_y1,
        "dynamic_ratio_y2": dynamic_ratio_y2,
        "dynamic_ratio_y3": dynamic_ratio_y3,
        "mass_drift": mass_drift,
        "metrics_path": str(metrics_path),
        "log_path": str(log_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep ROBER regular PINN hyperparameters")
    parser.add_argument("--seeds", type=str, default="1,2,3")
    parser.add_argument("--lrs", type=str, default="1e-5,3e-5,1e-4,3e-4,1e-3")
    parser.add_argument("--batch-sizes", type=str, default="64,128,256")
    parser.add_argument("--num-points", type=str, default="2500")
    parser.add_argument("--iterations", type=int, default=6000)
    parser.add_argument("--display-every", type=int, default=100)
    parser.add_argument("--max-loss-stop", type=float, default=1.0e4)
    parser.add_argument("--max-divergence-loss", type=float, default=1.0e12)
    parser.add_argument("--divergence-warmup-iterations", type=int, default=50)
    parser.add_argument("--loss-warmup-iterations", type=int, default=10000)
    parser.add_argument("--loss-breach-patience", type=int, default=2)
    parser.add_argument("--min-dynamic-ratio", type=float, default=1.0e-2)
    parser.add_argument("--max-runs", type=int, default=0, help="0 means all combinations")
    parser.add_argument("--sweep-tag", type=str, default="", help="Prefix for sweep artifact filenames")
    parser.add_argument("--no-force-cpu", action="store_true")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    script = here / "deepxde_rober_regular_fig4.py"
    out_dir = here / "data"
    out_dir.mkdir(exist_ok=True)

    seeds = parse_csv_list(args.seeds, int)
    lrs = parse_csv_list(args.lrs, float)
    batch_sizes = parse_csv_list(args.batch_sizes, int)
    num_points_list = parse_csv_list(args.num_points, int)

    combos = list(itertools.product(seeds, lrs, batch_sizes, num_points_list))
    if args.max_runs > 0:
        combos = combos[: args.max_runs]
    sweep_tag = args.sweep_tag if args.sweep_tag else f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"Planned runs: {len(combos)}")
    rows = []
    for run_id, (seed, lr, batch_size, num_points) in enumerate(combos, start=1):
        print(
            f"[run {run_id:03d}/{len(combos)}] seed={seed}, lr={lr}, "
            f"batch={batch_size}, points={num_points}"
        )
        row = run_one(
            script=script,
            out_dir=out_dir,
            sweep_tag=sweep_tag,
            run_id=run_id,
            seed=seed,
            lr=lr,
            batch_size=batch_size,
            num_points=num_points,
            iterations=args.iterations,
            display_every=args.display_every,
            max_loss_stop=args.max_loss_stop,
            max_divergence_loss=args.max_divergence_loss,
            divergence_warmup_iterations=args.divergence_warmup_iterations,
            loss_warmup_iterations=args.loss_warmup_iterations,
            loss_breach_patience=args.loss_breach_patience,
            min_dynamic_ratio=args.min_dynamic_ratio,
            force_cpu=not args.no_force_cpu,
        )
        rows.append(row)
        print(
            f"  -> status={row['status']}, final_total_loss={row['final_total_loss']:.3e}, "
            f"mean_rel_l2={row['rel_l2_mean']:.3e}"
        )

    def sort_key(item: dict):
        status_rank = {"ok": 0, "early_stopped": 1, "trivial": 2, "diverged": 3, "failed": 4}.get(item["status"], 9)
        return (status_rank, item["final_total_loss"], item["rel_l2_mean"], item["mass_drift"])

    rows_sorted = sorted(rows, key=sort_key)

    csv_path = out_dir / f"deepxde_rober_regular_fig4_{sweep_tag}_summary.csv"
    with csv_path.open("w", newline="", encoding="ascii") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()) if rows_sorted else [])
        if rows_sorted:
            writer.writeheader()
            writer.writerows(rows_sorted)

    json_path = out_dir / f"deepxde_rober_regular_fig4_{sweep_tag}_summary.json"
    json_path.write_text(json.dumps(rows_sorted, indent=2), encoding="utf-8")

    print(f"Saved summary CSV: {csv_path}")
    print(f"Saved summary JSON: {json_path}")
    if rows_sorted:
        best = rows_sorted[0]
        print("Best run:")
        print(
            f"  tag={best['tag']}, status={best['status']}, lr={best['lr']}, batch={best['batch_size']}, "
            f"seed={best['seed']}, final_total_loss={best['final_total_loss']:.3e}, mean_rel_l2={best['rel_l2_mean']:.3e}"
        )


if __name__ == "__main__":
    main()

