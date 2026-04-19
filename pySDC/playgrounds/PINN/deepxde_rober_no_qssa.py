"""DeepXDE PINN for the ROBER (Robertson) problem without QSSA.

The full three-species system is

    y1' = -0.04 y1 + 1e4 y2 y3
    y2' =  0.04 y1 - 1e4 y2 y3 - 3e7 y2^2
    y3' =  3e7 y2^2

with initial condition y(0) = (1, 0, 0).

To resolve stiffness over decades in time, this example uses logarithmically
sampled collocation and evaluation points.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# Keep backend explicit but still overridable from the shell.
os.environ.setdefault("DDE_BACKEND", "pytorch")

import deepxde as dde


def rober_rhs(_t: float, y: np.ndarray) -> np.ndarray:
    """Reference RHS for the Robertson kinetics system."""

    y1, y2, y3 = y
    return np.array(
        [
            -0.04 * y1 + 1.0e4 * y2 * y3,
            0.04 * y1 - 1.0e4 * y2 * y3 - 3.0e7 * y2 * y2,
            3.0e7 * y2 * y2,
        ]
    )


def rober_residual(t, y):
    """PINN residual for the full ROBER system (no QSSA)."""

    y1 = y[:, 0:1]
    y2 = y[:, 1:2]
    y3 = y[:, 2:3]

    dy1_dt = dde.grad.jacobian(y, t, i=0, j=0)
    dy2_dt = dde.grad.jacobian(y, t, i=1, j=0)
    dy3_dt = dde.grad.jacobian(y, t, i=2, j=0)

    eq1 = dy1_dt + 0.04 * y1 - 1.0e4 * y2 * y3
    eq2 = dy2_dt - 0.04 * y1 + 1.0e4 * y2 * y3 + 3.0e7 * y2 * y2
    eq3 = dy3_dt - 3.0e7 * y2 * y2

    return [eq1, eq2, eq3]


def run(
    epochs: int = 4000,
    num_collocation: int = 1024,
    num_eval: int = 600,
    t_min: float = 1.0e-5,
    t_max: float = 1.0e5,
) -> None:
    """Train ROBER PINN with log sampling and compare to a stiff reference solve."""

    default_float = "float64"
    if dde.backend.backend_name == "pytorch":
        import torch

        # Apple MPS currently does not support float64 tensors.
        if torch.backends.mps.is_available():
            default_float = "float32"
    dde.config.set_default_float(default_float)

    geom = dde.geometry.TimeDomain(0.0, t_max)

    initial_t = np.array([[0.0]])
    ic1 = dde.icbc.PointSetBC(initial_t, np.array([[1.0]]), component=0)
    ic2 = dde.icbc.PointSetBC(initial_t, np.array([[0.0]]), component=1)
    ic3 = dde.icbc.PointSetBC(initial_t, np.array([[0.0]]), component=2)

    anchors = np.geomspace(t_min, t_max, num_collocation).reshape(-1, 1)

    data = dde.data.PDE(
        geom,
        rober_residual,
        [ic1, ic2, ic3],
        num_domain=0,
        num_boundary=0,
        anchors=anchors,
        num_test=min(2000, num_eval),
    )

    net = dde.nn.FNN([1, 64, 64, 64, 3], "tanh", "Glorot uniform")
    model = dde.Model(data, net)

    model.compile("adam", lr=1.0e-3)
    loss_history, _ = model.train(iterations=epochs, display_every=max(epochs // 10, 1))

    t_eval = np.geomspace(t_min, t_max, num_eval)
    y_pred = model.predict(t_eval.reshape(-1, 1))

    y0 = np.array([1.0, 0.0, 0.0])
    ref = solve_ivp(
        rober_rhs,
        (0.0, t_max),
        y0,
        method="Radau",
        t_eval=t_eval,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    if not ref.success:
        raise RuntimeError(f"Reference solve failed: {ref.message}")
    y_ref = ref.y.T

    rel_l2 = []
    for i in range(3):
        denom = np.linalg.norm(y_ref[:, i])
        denom = max(denom, 1.0e-16)
        rel_l2.append(np.linalg.norm(y_pred[:, i] - y_ref[:, i]) / denom)

    mass_drift = np.max(np.abs(np.sum(y_pred, axis=1) - 1.0))

    print(f"DeepXDE backend: {dde.backend.backend_name}")
    print(f"Relative L2 error y1: {rel_l2[0]:.3e}")
    print(f"Relative L2 error y2: {rel_l2[1]:.3e}")
    print(f"Relative L2 error y3: {rel_l2[2]:.3e}")
    print(f"Max mass-conservation drift |y1+y2+y3-1|: {mass_drift:.3e}")
    print(f"Final training loss: {loss_history.loss_train[-1]}")

    out_dir = Path(__file__).with_name("data")
    out_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    labels = ["y1", "y2", "y3"]
    for i, ax in enumerate(axes):
        ax.semilogx(t_eval, y_ref[:, i], "k-", label=f"reference {labels[i]}")
        ax.semilogx(t_eval, y_pred[:, i], "r--", label=f"PINN {labels[i]}")
        ax.set_ylabel(labels[i])
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best")
    axes[-1].set_xlabel("t")
    fig.suptitle("ROBER without QSSA (log-sampled PINN)")
    fig.tight_layout()
    fig_path = out_dir / "deepxde_rober_no_qssa_solution.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    metrics_path = out_dir / "deepxde_rober_no_qssa_metrics.txt"
    with metrics_path.open("w", encoding="ascii") as f:
        f.write(f"backend={dde.backend.backend_name}\n")
        f.write(f"epochs={epochs}\n")
        f.write(f"t_min={t_min}\n")
        f.write(f"t_max={t_max}\n")
        f.write(f"rel_l2_y1={rel_l2[0]:.16e}\n")
        f.write(f"rel_l2_y2={rel_l2[1]:.16e}\n")
        f.write(f"rel_l2_y3={rel_l2[2]:.16e}\n")
        f.write(f"mass_drift={mass_drift:.16e}\n")

    print(f"Saved plot to: {fig_path}")
    print(f"Saved metrics to: {metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepXDE ROBER PINN (no QSSA) with log sampling")
    parser.add_argument("--epochs", type=int, default=4000, help="Number of optimizer iterations")
    parser.add_argument("--num-collocation", type=int, default=1024, help="Number of log-sampled collocation points")
    parser.add_argument("--num-eval", type=int, default=600, help="Number of log-sampled evaluation points")
    parser.add_argument("--t-min", type=float, default=1.0e-5, help="Lower bound for log sampling (must be > 0)")
    parser.add_argument("--t-max", type=float, default=1.0e5, help="Final time")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.t_min <= 0.0:
        raise ValueError("--t-min must be strictly positive for logarithmic sampling")
    if args.t_min >= args.t_max:
        raise ValueError("--t-min must be smaller than --t-max")
    run(
        epochs=args.epochs,
        num_collocation=args.num_collocation,
        num_eval=args.num_eval,
        t_min=args.t_min,
        t_max=args.t_max,
    )



