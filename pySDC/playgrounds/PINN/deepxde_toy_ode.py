"""Minimal DeepXDE toy problem for upcoming SDC + PINN experiments.

This script solves y'(t) - y(t) = 0 on t in [0, 1], y(0) = 1.
The exact solution is y(t) = exp(t).
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Pick a backend explicitly to make the example reproducible.
os.environ.setdefault("DDE_BACKEND", "pytorch")

import deepxde as dde


def main(epochs: int = 2000, n_plot: int = 200) -> None:
    """Train a small PINN and compare it to the exact solution."""

    geom = dde.geometry.TimeDomain(0.0, 1.0)

    def ode(t, y):
        dy_dt = dde.grad.jacobian(y, t)
        return dy_dt - y

    ic = dde.icbc.IC(geom, lambda t: np.ones((len(t), 1)), lambda _, on_initial: on_initial)

    data = dde.data.PDE(geom, ode, [ic], num_domain=64, num_boundary=2, num_test=200)
    net = dde.nn.FNN([1] + [32] * 2 + [1], "tanh", "Glorot uniform")
    model = dde.Model(data, net)

    model.compile("adam", lr=1e-3)
    loss_history, _ = model.train(iterations=epochs, display_every=max(epochs // 10, 1))

    t = np.linspace(0.0, 1.0, n_plot)[:, None]
    y_pred = model.predict(t)
    y_true = np.exp(t)
    l2_rel = np.linalg.norm(y_pred - y_true) / np.linalg.norm(y_true)

    print(f"DeepXDE backend: {dde.backend.backend_name}")
    print(f"Relative L2 error: {l2_rel:.3e}")
    print(f"Final training loss: {loss_history.loss_train[-1]}")

    out_dir = Path(__file__).with_name("data")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "deepxde_toy_ode_solution.png"

    plt.figure(figsize=(6, 4))
    plt.plot(t, y_true, "k-", label="exact exp(t)")
    plt.plot(t, y_pred, "r--", label="PINN prediction")
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.title("DeepXDE toy ODE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()

