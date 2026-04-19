"""Tiny runner for deepxde_rober_paper_simple.py presets."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def read_metric_value(metrics_path: Path, key: str) -> float:
    with metrics_path.open("r", encoding="ascii") as f:
        for line in f:
            if line.startswith(f"{key}="):
                return float(line.split("=", 1)[1].strip())
    raise ValueError(f"Could not find {key} in {metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simple DeepXDE ROBER paper-style presets")
    parser.add_argument("--mode", choices=["smoke", "paper"], default="smoke")
    parser.add_argument("--approach", choices=["global", "slab_irk"], default="global")
    parser.add_argument("--num-slabs", type=int, default=8)
    parser.add_argument("--irk-order", type=int, choices=[2, 4], default=2)
    parser.add_argument("--compare-irk-orders", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_cmd(args: argparse.Namespace, script: Path, irk_order: int | None = None) -> list[str]:
    cmd = [
        sys.executable,
        str(script),
        "--seed",
        str(args.seed),
        "--approach",
        args.approach,
    ]

    if args.mode == "smoke":
        if args.approach == "slab_irk":
            order = args.irk_order if irk_order is None else irk_order
            cmd += [
                "--iterations",
                "80",
                "--num-points",
                "256",
                "--num-eval",
                "300",
                "--num-slabs",
                str(args.num_slabs),
                "--steps-per-slab",
                "20",
                "--irk-order",
                str(order),
                "--run-tag",
                f"smoke_slab_irk{order}",
            ]
        else:
            cmd += ["--iterations", "200", "--num-points", "512", "--num-eval", "300", "--run-tag", "smoke_global"]
    else:
        if args.approach == "slab_irk":
            order = args.irk_order if irk_order is None else irk_order
            cmd += [
                "--iterations",
                "1200",
                "--num-points",
                "512",
                "--num-eval",
                "600",
                "--num-slabs",
                str(args.num_slabs),
                "--steps-per-slab",
                "40",
                "--irk-order",
                str(order),
                "--run-tag",
                f"paper_slab_irk{order}",
            ]
        else:
            cmd += ["--iterations", "10000", "--num-points", "2500", "--num-eval", "600", "--run-tag", "paper_global"]
    return cmd


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    script = root / "deepxde_rober_paper_simple.py"
    data_dir = root / "data"

    if args.compare_irk_orders:
        if args.approach != "slab_irk":
            raise ValueError("--compare-irk-orders requires --approach slab_irk")
        for order in (2, 4):
            subprocess.run(build_cmd(args=args, script=script, irk_order=order), check=True)

        tag_prefix = "smoke" if args.mode == "smoke" else "paper"
        metrics_2 = data_dir / f"deepxde_rober_paper_simple_{tag_prefix}_slab_irk2_metrics.txt"
        metrics_4 = data_dir / f"deepxde_rober_paper_simple_{tag_prefix}_slab_irk4_metrics.txt"
        rmse2 = [read_metric_value(metrics_2, f"rmse_y{i}") for i in (1, 2, 3)]
        rmse4 = [read_metric_value(metrics_4, f"rmse_y{i}") for i in (1, 2, 3)]
        print(f"IRK2 RMSE: y1={rmse2[0]:.6e}, y2={rmse2[1]:.6e}, y3={rmse2[2]:.6e}")
        print(f"IRK4 RMSE: y1={rmse4[0]:.6e}, y2={rmse4[1]:.6e}, y3={rmse4[2]:.6e}")
        return

    subprocess.run(build_cmd(args=args, script=script), check=True)


if __name__ == "__main__":
    main()

