"""Sweep runner for slab IRK Robertson experiments."""

from __future__ import annotations

import argparse
import csv
import itertools
import subprocess
import sys
from pathlib import Path


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def read_metric_value(metrics_path: Path, key: str) -> float:
    with metrics_path.open("r", encoding="ascii") as f:
        for line in f:
            if line.startswith(f"{key}="):
                return float(line.split("=", 1)[1].strip())
    raise ValueError(f"Could not find {key} in {metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep slab IRK2/IRK4 settings for ROBER PINN")
    parser.add_argument("--mode", choices=["smoke", "paper"], default="smoke")
    parser.add_argument("--num-slabs-list", type=str, default="4,8")
    parser.add_argument("--steps-per-slab-list", type=str, default="20,40")
    parser.add_argument("--irk-weight-list", type=str, default="1.0")
    parser.add_argument("--interface-weight-list", type=str, default="0.0")
    parser.add_argument("--lbfgs-iters-list", type=str, default="0")
    parser.add_argument("--seed-list", type=str, default="42")
    parser.add_argument("--sweep-tag", type=str, default="irk_sweep")
    parser.add_argument("--max-runs", type=int, default=0)
    return parser.parse_args()


def build_base_cmd(script: Path, mode: str, seed: int) -> list[str]:
    cmd = [
        sys.executable,
        str(script),
        "--seed",
        str(seed),
        "--approach",
        "slab_irk",
    ]
    if mode == "smoke":
        cmd += ["--iterations", "80", "--num-points", "256", "--num-eval", "300"]
    else:
        cmd += ["--iterations", "1200", "--num-points", "512", "--num-eval", "600"]
    return cmd


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    script = root / "deepxde_rober_paper_simple.py"
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)

    seeds = parse_int_list(args.seed_list)
    slab_values = parse_int_list(args.num_slabs_list)
    step_values = parse_int_list(args.steps_per_slab_list)
    weight_values = parse_float_list(args.irk_weight_list)
    interface_values = parse_float_list(args.interface_weight_list)
    lbfgs_values = parse_int_list(args.lbfgs_iters_list)

    rows: list[dict[str, float | int | str]] = []
    run_counter = 0

    for irk_order, seed, num_slabs, steps_per_slab, irk_weight, interface_weight, lbfgs_iters in itertools.product(
        [2, 4],
        seeds,
        slab_values,
        step_values,
        weight_values,
        interface_values,
        lbfgs_values,
    ):
        if args.max_runs > 0 and run_counter >= args.max_runs:
            break

        run_tag = (
            f"{args.sweep_tag}_{args.mode}_o{irk_order}_s{num_slabs}_k{steps_per_slab}_"
            f"w{irk_weight:g}_iw{interface_weight:g}_lb{lbfgs_iters}_seed{seed}"
        )

        cmd = build_base_cmd(script=script, mode=args.mode, seed=seed)
        cmd += [
            "--irk-order",
            str(irk_order),
            "--num-slabs",
            str(num_slabs),
            "--steps-per-slab",
            str(steps_per_slab),
            "--irk-weight",
            str(irk_weight),
            "--interface-weight",
            str(interface_weight),
            "--lbfgs-iters",
            str(lbfgs_iters),
            "--run-tag",
            run_tag,
        ]

        subprocess.run(cmd, check=True)

        metrics_path = data_dir / f"deepxde_rober_paper_simple_{run_tag}_metrics.txt"
        solution_path = data_dir / f"deepxde_rober_paper_simple_{run_tag}_solution.png"
        y1_path = data_dir / f"deepxde_rober_paper_simple_{run_tag}_y1.png"
        y2_path = data_dir / f"deepxde_rober_paper_simple_{run_tag}_y2.png"
        y3_path = data_dir / f"deepxde_rober_paper_simple_{run_tag}_y3.png"
        if not all(path.exists() for path in (metrics_path, solution_path, y1_path, y2_path, y3_path)):
            raise RuntimeError(f"Missing artifacts for run tag {run_tag}")

        rmse_y1 = read_metric_value(metrics_path, "rmse_y1")
        rmse_y2 = read_metric_value(metrics_path, "rmse_y2")
        rmse_y3 = read_metric_value(metrics_path, "rmse_y3")
        rmse_mean = (rmse_y1 + rmse_y2 + rmse_y3) / 3.0

        rows.append(
            {
                "run_tag": run_tag,
                "irk_order": irk_order,
                "seed": seed,
                "num_slabs": num_slabs,
                "steps_per_slab": steps_per_slab,
                "irk_weight": irk_weight,
                "rmse_y1": rmse_y1,
                "rmse_y2": rmse_y2,
                "rmse_y3": rmse_y3,
                "rmse_mean": rmse_mean,
                "interface_weight": interface_weight,
                "lbfgs_iters": lbfgs_iters,
                "solution_plot": str(solution_path.name),
                "y1_plot": str(y1_path.name),
                "y2_plot": str(y2_path.name),
                "y3_plot": str(y3_path.name),
            }
        )
        run_counter += 1

    rows.sort(key=lambda r: float(r["rmse_mean"]))

    out_csv = data_dir / f"deepxde_rober_paper_simple_{args.sweep_tag}_{args.mode}_summary.csv"
    with out_csv.open("w", newline="", encoding="ascii") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_tag",
                "irk_order",
                "seed",
                "num_slabs",
                "steps_per_slab",
                "irk_weight",
                "interface_weight",
                "lbfgs_iters",
                "rmse_y1",
                "rmse_y2",
                "rmse_y3",
                "rmse_mean",
                "solution_plot",
                "y1_plot",
                "y2_plot",
                "y3_plot",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved sweep summary: {out_csv}")
    if rows:
        best = rows[0]
        print(
            "Best run: {run_tag} | IRK{irk_order} | slabs={num_slabs} | steps={steps_per_slab} | "
            "weight={irk_weight:g} | iface={interface_weight:g} | lbfgs={lbfgs_iters} | rmse_mean={rmse_mean:.6e}".format(
                **best
            )
        )


if __name__ == "__main__":
    main()

