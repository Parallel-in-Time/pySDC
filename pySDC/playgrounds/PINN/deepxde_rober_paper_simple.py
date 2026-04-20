"""Simple DeepXDE regular PINN for ROBER using paper-style settings.

This script intentionally keeps only the core ingredients from the paper setup:
- full ROBER system (no QSSA),
- kinetics constants k1=0.04, k2=3e7, k3=1e4,
- log-time domain t in [1e-5, 1e5],
- 2500 residual points sampled uniformly in log scale,
- hard IC ansatz y(t) = y0 + (t / t_scale) * S * NN(log(t / t_scale)),
- 3 hidden layers x 128 neurons, GELU activation,
- Adam lr=1e-3, minibatch size 128.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

os.environ.setdefault("DDE_BACKEND", "pytorch")

import deepxde as dde


def rober_rhs(_t: float, y: np.ndarray) -> np.ndarray:
    y1, y2, y3 = y
    return np.array(
        [
            -0.04 * y1 + 1.0e4 * y2 * y3,
            0.04 * y1 - 1.0e4 * y2 * y3 - 3.0e7 * y2 * y2,
            3.0e7 * y2 * y2,
        ],
        dtype=float,
    )


def make_residual():
    def residual(t, y):
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

    return residual


def sample_uniform_log_points(t_min: float, t_max: float, n: int, seed: int) -> np.ndarray:
    """Sample points uniformly in log scale, with enrichment in early time region [1e-5, 1e-2]
    where y2 is most active and stiff."""
    rng = np.random.default_rng(seed)
    split = 1.0e-2

    if t_max <= split or t_min >= split:
        points = np.power(10.0, rng.uniform(np.log10(t_min), np.log10(t_max), size=n))
        points.sort()
        return points.reshape(-1, 1)

    # Interval straddles split: bias toward early-time window where y2 is most active.
    n_early = max(1, int(0.7 * n))
    n_late = max(1, n - n_early)
    if n_early + n_late > n:
        n_early = n - n_late

    early_points = np.power(10.0, rng.uniform(np.log10(t_min), np.log10(split), size=n_early))
    late_points = np.power(10.0, rng.uniform(np.log10(split), np.log10(t_max), size=n_late))

    points = np.concatenate([early_points, late_points])
    points.sort()
    return points.reshape(-1, 1)


def set_backend_defaults(force_cpu: bool) -> None:
    default_float = "float64"
    if dde.backend.backend_name == "pytorch":
        import torch

        if force_cpu:
            torch.set_default_device("cpu")
        elif torch.backends.mps.is_available():
            default_float = "float32"
    dde.config.set_default_float(default_float)


def get_reference_solution(t_eval: np.ndarray, t_max: float) -> np.ndarray:
    y_ref = solve_ivp(
        rober_rhs,
        (0.0, t_max),
        [1.0, 0.0, 0.0],
        method="BDF",
        t_eval=t_eval,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    if not y_ref.success:
        raise RuntimeError(f"Reference solve failed: {y_ref.message}")
    return y_ref.y.T


def make_transforms(
    t_start: float,
    t_end: float,
    y_start: np.ndarray,
    species_scale: np.ndarray,
):
    t_start = float(t_start)
    t_end = float(t_end)
    t_span = max(t_end - t_start, 1.0e-14)
    log_denom = max(np.log(t_end / t_start), 1.0e-14)

    def feature_transform(t):
        if dde.backend.backend_name == "pytorch":
            import torch

            return torch.log(t / t_start) / log_denom
        return np.log(t / t_start) / log_denom

    def output_transform(t, y_raw):
        if dde.backend.backend_name == "pytorch":
            import torch

            y0 = torch.tensor(y_start, dtype=y_raw.dtype, device=y_raw.device).reshape(1, 3)
            s = torch.tensor(species_scale, dtype=y_raw.dtype, device=y_raw.device).reshape(1, 3)
            return y0 + ((t - t_start) / t_span) * s * y_raw

        y0 = dde.backend.as_tensor(y_start.reshape(1, 3))
        s = dde.backend.as_tensor(species_scale.reshape(1, 3))
        return y0 + ((t - t_start) / t_span) * s * y_raw

    return feature_transform, output_transform


def train_global_pinn(
    t_eval: np.ndarray,
    y_ref: np.ndarray,
    iterations: int,
    num_points: int,
    batch_size: int,
    lr: float,
    t_min: float,
    t_max: float,
    seed: int,
    adaptive_weights: bool = True,
):
    geom = dde.geometry.TimeDomain(t_min, t_max)
    anchors = sample_uniform_log_points(t_min=t_min, t_max=t_max, n=num_points, seed=seed)

    data = dde.data.PDE(
        geom,
        make_residual(),
        bcs=[],
        num_domain=0,
        num_boundary=0,
        anchors=anchors,
        num_test=min(2000, t_eval.size),
    )

    net = dde.nn.FNN([1, 128, 128, 128, 3], "gelu", "Glorot uniform")
    species_scale = np.maximum(np.max(np.abs(y_ref), axis=0), 1.0e-12)
    feature_transform, output_transform = make_transforms(t_min, t_max, np.array([1.0, 0.0, 0.0]), species_scale)
    net.apply_feature_transform(feature_transform)
    net.apply_output_transform(output_transform)

    model = dde.Model(data, net)
    # Adaptive weighting for ODE residuals based on magnitude
    # y2 is tiny (~1e-5) so needs much higher weight to balance loss
    if adaptive_weights:
        y_max = np.maximum(np.max(np.abs(y_ref), axis=0), 1.0e-12)
        loss_weights = [1.0, 10.0 / (y_max[1] + 1.0e-12), 1.0]
    else:
        loss_weights = [1.0, 1.0, 1.0]
    model.compile("adam", lr=lr, loss_weights=loss_weights)
    loss_history, _ = model.train(iterations=iterations, batch_size=batch_size, display_every=max(iterations // 10, 1))
    y_pred = model.predict(t_eval.reshape(-1, 1))
    total_loss = np.sum(np.array(loss_history.loss_train), axis=1)
    return y_pred, np.array(loss_history.steps), total_loss, []


def build_slab_edges(t_min: float, t_max: float, num_slabs: int) -> np.ndarray:
    if num_slabs < 1:
        raise ValueError("num_slabs must be >= 1")
    return np.geomspace(t_min, t_max, num_slabs + 1)


def get_irk_tableau(order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if order == 2:
        a = np.array([[0.5]], dtype=float)
        b = np.array([1.0], dtype=float)
        c = np.array([0.5], dtype=float)
        return a, b, c
    if order == 4:
        s3 = np.sqrt(3.0)
        a = np.array(
            [
                [0.25, 0.25 - s3 / 6.0],
                [0.25 + s3 / 6.0, 0.25],
            ],
            dtype=float,
        )
        b = np.array([0.5, 0.5], dtype=float)
        c = np.array([0.5 - s3 / 6.0, 0.5 + s3 / 6.0], dtype=float)
        return a, b, c
    raise ValueError("Only IRK orders 2 and 4 are supported")


def irk_step_raw(t_n: float, y_n: np.ndarray, h: float, order: int) -> np.ndarray:
    if h <= 0.0:
        raise ValueError("IRK step size must be positive")

    a, b, c = get_irk_tableau(order)
    stages = b.size
    dim = y_n.size
    k0 = np.zeros(stages * dim, dtype=float)
    for i in range(stages):
        k0[i * dim : (i + 1) * dim] = rober_rhs(t_n + c[i] * h, y_n)

    def stage_residual(k_flat: np.ndarray) -> np.ndarray:
        k = k_flat.reshape(stages, dim)
        res = np.zeros_like(k)
        for i in range(stages):
            y_stage = y_n + h * np.sum(a[i, :, None] * k, axis=0)
            t_stage = t_n + c[i] * h
            res[i] = k[i] - rober_rhs(t_stage, y_stage)
        return res.ravel()

    sol = root(stage_residual, k0, method="hybr", tol=1.0e-12)
    if not sol.success:
        sol = root(stage_residual, k0, method="lm", tol=1.0e-12)
    if not sol.success:
        raise RuntimeError(f"IRK stage solve failed at t={t_n:.8e}: {sol.message}")
    k = sol.x.reshape(stages, dim)
    y_np1 = y_n + h * np.sum(b[:, None] * k, axis=0)
    return y_np1


def irk_step(t_n: float, y_n: np.ndarray, h: float, order: int, max_splits: int = 8) -> np.ndarray:
    # Substep automatically when stiff nonlinear stage solves fail on large intervals.
    for split_level in range(max_splits + 1):
        n_substeps = 2**split_level
        h_sub = h / n_substeps
        y_cur = y_n.copy()
        t_cur = t_n
        try:
            for _ in range(n_substeps):
                y_cur = irk_step_raw(t_cur, y_cur, h_sub, order)
                t_cur += h_sub
            return y_cur
        except RuntimeError:
            continue

    # Last-resort robust fallback for rare pathological stage solves on very stiff steps.
    fallback = solve_ivp(
        rober_rhs,
        (t_n, t_n + h),
        y_n,
        method="Radau",
        t_eval=np.array([t_n + h], dtype=float),
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    if fallback.success:
        return fallback.y[:, -1]
    raise RuntimeError(f"IRK failed after substepping retries at t={t_n:.8e}, h={h:.8e}")


def integrate_irk_nodes(t_nodes: np.ndarray, y_start: np.ndarray, irk_order: int) -> np.ndarray:
    y_vals = np.zeros((t_nodes.size, y_start.size), dtype=float)
    y_vals[0] = y_start
    for i in range(1, t_nodes.size):
        h = float(t_nodes[i] - t_nodes[i - 1])
        y_vals[i] = irk_step(float(t_nodes[i - 1]), y_vals[i - 1], h, irk_order)
    return y_vals


def train_local_slab(
    t_start: float,
    t_end: float,
    y_start: np.ndarray,
    t_eval_local: np.ndarray,
    iterations: int,
    num_points: int,
    batch_size: int,
    lr: float,
    seed: int,
    irk_order: int,
    steps_per_slab: int,
    irk_weight: float,
    interface_weight: float,
    lbfgs_iters: int,
):
    geom = dde.geometry.TimeDomain(t_start, t_end)
    anchors = sample_uniform_log_points(t_min=t_start, t_max=t_end, n=num_points, seed=seed)

    # Use an implicit RK one-step method to generate slab-local guide states.
    irk_nodes = max(3 + irk_order, steps_per_slab + 1)
    t_irk = np.geomspace(t_start, t_end, irk_nodes)
    y_irk = integrate_irk_nodes(t_nodes=t_irk, y_start=y_start, irk_order=irk_order)

    data_x = t_irk.reshape(-1, 1)
    guide_bcs = [dde.icbc.PointSetBC(data_x, y_irk[:, i : i + 1], component=i) for i in range(3)]
    endpoint_bcs = []
    if interface_weight > 0.0:
        t_end_x = np.array([[t_end]], dtype=float)
        endpoint_bcs = [dde.icbc.PointSetBC(t_end_x, y_irk[-1:, i : i + 1], component=i) for i in range(3)]
    bcs = guide_bcs + endpoint_bcs

    data = dde.data.PDE(
        geom,
        make_residual(),
        bcs=bcs,
        num_domain=0,
        num_boundary=0,
        anchors=anchors,
        num_test=min(2000, t_eval_local.size),
    )

    net = dde.nn.FNN([1, 128, 128, 128, 3], "gelu", "Glorot uniform")
    species_scale = np.maximum(np.max(np.abs(y_irk), axis=0), 1.0e-12)
    feature_transform, output_transform = make_transforms(t_start, t_end, y_start, species_scale)
    net.apply_feature_transform(feature_transform)
    net.apply_output_transform(output_transform)

    model = dde.Model(data, net)
    ode_terms = 3
    # Adaptive weighting for ODE residuals based on magnitude
    # y2 is tiny (~1e-5) so needs much higher weight to balance loss
    y_max = np.maximum(np.max(np.abs(y_irk), axis=0), 1.0e-12)
    ode_weights = np.array([1.0, 10.0 / (y_max[1] + 1.0e-12), 1.0])
    loss_weights = ode_weights.tolist()
    loss_weights += [float(irk_weight)] * len(guide_bcs)
    loss_weights += [float(interface_weight)] * len(endpoint_bcs)
    model.compile("adam", lr=lr, loss_weights=loss_weights)
    loss_history_adam, _ = model.train(iterations=iterations, batch_size=batch_size, display_every=max(iterations // 10, 1))

    steps = np.array(loss_history_adam.steps)
    total_loss = np.sum(np.array(loss_history_adam.loss_train), axis=1)

    if lbfgs_iters > 0:
        dde.optimizers.set_LBFGS_options(maxiter=lbfgs_iters)
        model.compile("L-BFGS", loss_weights=loss_weights)
        loss_history_lbfgs, _ = model.train(display_every=max(lbfgs_iters // 10, 1))
        lbfgs_steps = np.array(loss_history_lbfgs.steps)
        lbfgs_total = np.sum(np.array(loss_history_lbfgs.loss_train), axis=1)
        if lbfgs_steps.size > 0:
            shift = steps[-1] if steps.size > 0 else 0
            steps = np.concatenate([steps, lbfgs_steps + shift])
            total_loss = np.concatenate([total_loss, lbfgs_total])

    y_eval = model.predict(t_eval_local.reshape(-1, 1))
    y_end = model.predict(np.array([[t_end]], dtype=float))[0]
    return y_eval, y_end, steps, total_loss, y_irk[-1], t_irk, y_irk


def train_slab_irk_pinn(
    t_eval: np.ndarray,
    y_ref: np.ndarray,
    iterations: int,
    num_points: int,
    batch_size: int,
    lr: float,
    t_min: float,
    t_max: float,
    seed: int,
    num_slabs: int,
    irk_order: int,
    steps_per_slab: int,
    irk_weight: float,
    interface_weight: float,
    lbfgs_iters: int,
    use_pinn_ic: bool = False,
):
    if irk_order not in (2, 4):
        raise ValueError("Only IRK order 2 or 4 are supported")

    edges = build_slab_edges(t_min=t_min, t_max=t_max, num_slabs=num_slabs)
    y_pred = np.zeros_like(y_ref)
    loss_steps_all = []
    total_loss_all = []
    slab_notes = []
    step_offset = 0

    y_start = np.array([1.0, 0.0, 0.0], dtype=float)
    for slab_idx in range(num_slabs):
        t_start = float(edges[slab_idx])
        t_end = float(edges[slab_idx + 1])

        if slab_idx == num_slabs - 1:
            eval_mask = (t_eval >= t_start) & (t_eval <= t_end)
        else:
            eval_mask = (t_eval >= t_start) & (t_eval < t_end)

        t_eval_local = t_eval[eval_mask]
        if t_eval_local.size == 0:
            continue

        (
            y_eval,
            y_end_pred,
            steps,
            total_loss,
            y_end_irk,
            _t_irk,
            _y_irk,
        ) = train_local_slab(
            t_start=t_start,
            t_end=t_end,
            y_start=y_start,
            t_eval_local=t_eval_local,
            iterations=iterations,
            num_points=num_points,
            batch_size=batch_size,
            lr=lr,
            seed=seed + slab_idx,
            irk_order=irk_order,
            steps_per_slab=steps_per_slab,
            irk_weight=irk_weight,
            interface_weight=interface_weight,
            lbfgs_iters=lbfgs_iters,
        )

        y_pred[eval_mask] = y_eval
        loss_steps_all.append(steps + step_offset)
        total_loss_all.append(total_loss)
        step_offset += int(steps[-1]) if steps.size else 0

        slab_rmse = np.sqrt(np.mean((y_eval - y_ref[eval_mask]) ** 2, axis=0))
        slab_notes.append(
            {
                "slab": slab_idx,
                "t_start": t_start,
                "t_end": t_end,
                "rmse": slab_rmse,
                "end_gap_to_irk": np.abs(y_end_pred - y_end_irk),
            }
        )

        # Propagate IC to next slab: use IRK endpoint (accurate) or PINN endpoint (test mode).
        y_start = y_end_pred if use_pinn_ic else y_end_irk

    if loss_steps_all:
        loss_steps = np.concatenate(loss_steps_all)
        total_loss = np.concatenate(total_loss_all)
    else:
        loss_steps = np.array([], dtype=float)
        total_loss = np.array([], dtype=float)

    return y_pred, loss_steps, total_loss, slab_notes, edges


def run(
    iterations: int = 10000,
    num_points: int = 2500,
    batch_size: int = 128,
    lr: float = 1.0e-3,
    t_min: float = 1.0e-5,
    t_max: float = 1.0e5,
    num_eval: int = 600,
    seed: int = 42,
    run_tag: str = "",
    force_cpu: bool = True,
    approach: str = "global",
    num_slabs: int = 8,
    irk_order: int = 2,
    steps_per_slab: int = 40,
    irk_weight: float = 1.0,
    interface_weight: float = 0.0,
    lbfgs_iters: int = 0,
    use_pinn_ic: bool = False,
    adaptive_weights: bool = True,
) -> None:
    set_backend_defaults(force_cpu=force_cpu)
    dde.config.set_random_seed(seed)

    t_eval = np.geomspace(t_min, t_max, num_eval)
    y_ref = get_reference_solution(t_eval=t_eval, t_max=t_max)

    slab_notes: Sequence[dict] = []
    slab_edges = np.array([], dtype=float)
    if approach == "global":
        y_pred, steps, total_loss, slab_notes = train_global_pinn(
            t_eval=t_eval,
            y_ref=y_ref,
            iterations=iterations,
            num_points=num_points,
            batch_size=batch_size,
            lr=lr,
            t_min=t_min,
            t_max=t_max,
            seed=seed,
            adaptive_weights=adaptive_weights,
        )
    elif approach in ("slab_irk", "slab_bdf"):
        y_pred, steps, total_loss, slab_notes, slab_edges = train_slab_irk_pinn(
            t_eval=t_eval,
            y_ref=y_ref,
            iterations=iterations,
            num_points=num_points,
            batch_size=batch_size,
            lr=lr,
            t_min=t_min,
            t_max=t_max,
            seed=seed,
            num_slabs=num_slabs,
            irk_order=irk_order,
            steps_per_slab=steps_per_slab,
            irk_weight=irk_weight,
            interface_weight=interface_weight,
            lbfgs_iters=lbfgs_iters,
            use_pinn_ic=use_pinn_ic,
        )
    else:
        raise ValueError(f"Unknown approach: {approach}")

    rmse = np.sqrt(np.mean((y_pred - y_ref) ** 2, axis=0))

    out_dir = Path(__file__).with_name("data")
    out_dir.mkdir(exist_ok=True)
    suffix = f"_{run_tag}" if run_tag else ""

    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    labels = ["y1", "y2", "y3"]
    y_limits = [(0.0, 1.0), (0.0, 5.0e-5), (0.0, 1.0)]
    for i, ax in enumerate(axes):
        ax.semilogx(t_eval, y_ref[:, i], "k-", label=f"BDF {labels[i]}")
        ax.semilogx(t_eval, y_pred[:, i], "r--", label=f"PINN {labels[i]}")
        ax.set_ylabel(labels[i])
        ax.set_ylim(*y_limits[i])
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best")
    axes[-1].set_xlabel("t [s]")
    title = "ROBER regular PINN (DeepXDE, paper-style)"
    if approach in ("slab_irk", "slab_bdf"):
        title = f"ROBER slab PINN + IRK{irk_order} guides (slabs={num_slabs})"
    fig.suptitle(title)
    fig.tight_layout()
    fig_path = out_dir / f"deepxde_rober_paper_simple{suffix}_solution.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    species_plot_paths = []
    for i, label in enumerate(labels):
        fig_s, ax_s = plt.subplots(figsize=(7, 4))
        ax_s.semilogx(t_eval, y_ref[:, i], "k-", label=f"BDF {label}")
        ax_s.semilogx(t_eval, y_pred[:, i], "r--", label=f"PINN {label}")
        ax_s.set_xlabel("t [s]")
        ax_s.set_ylabel(label)
        ax_s.set_ylim(*y_limits[i])
        ax_s.grid(True, which="both", alpha=0.3)
        ax_s.legend(loc="best")
        fig_s.tight_layout()
        species_path = out_dir / f"deepxde_rober_paper_simple{suffix}_{label}.png"
        fig_s.savefig(species_path, dpi=200)
        plt.close(fig_s)
        species_plot_paths.append(species_path)

    loss_path = out_dir / f"deepxde_rober_paper_simple{suffix}_loss.png"
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    if steps.size:
        ax2.semilogy(steps, total_loss, "b-")
    ax2.set_xlabel("Parameter updates")
    ax2.set_ylabel("Total train loss")
    ax2.grid(True, which="both", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(loss_path, dpi=200)
    plt.close(fig2)

    metrics_path = out_dir / f"deepxde_rober_paper_simple{suffix}_metrics.txt"
    with metrics_path.open("w", encoding="ascii") as f:
        f.write(f"backend={dde.backend.backend_name}\n")
        f.write(f"seed={seed}\n")
        f.write(f"iterations={iterations}\n")
        f.write(f"num_points={num_points}\n")
        f.write(f"batch_size={batch_size}\n")
        f.write(f"lr={lr}\n")
        f.write(f"approach={approach}\n")
        f.write(f"t_min={t_min}\n")
        f.write(f"t_max={t_max}\n")
        if approach in ("slab_irk", "slab_bdf"):
            f.write(f"num_slabs={num_slabs}\n")
            f.write(f"irk_order={irk_order}\n")
            f.write(f"steps_per_slab={steps_per_slab}\n")
            f.write(f"irk_weight={irk_weight}\n")
            f.write(f"interface_weight={interface_weight}\n")
            f.write(f"lbfgs_iters={lbfgs_iters}\n")
            if slab_edges.size:
                edge_line = ",".join(f"{x:.8e}" for x in slab_edges)
                f.write(f"slab_edges={edge_line}\n")
        f.write(f"rmse_y1={rmse[0]:.16e}\n")
        f.write(f"rmse_y2={rmse[1]:.16e}\n")
        f.write(f"rmse_y3={rmse[2]:.16e}\n")
        for note in slab_notes:
            f.write(
                "slab_{slab}_t=[{t0:.8e},{t1:.8e}] rmse=[{r0:.8e},{r1:.8e},{r2:.8e}] "
                "end_gap_to_irk=[{g0:.8e},{g1:.8e},{g2:.8e}]\n".format(
                    slab=note["slab"],
                    t0=note["t_start"],
                    t1=note["t_end"],
                    r0=note["rmse"][0],
                    r1=note["rmse"][1],
                    r2=note["rmse"][2],
                    g0=note["end_gap_to_irk"][0],
                    g1=note["end_gap_to_irk"][1],
                    g2=note["end_gap_to_irk"][2],
                )
            )

    print(f"Saved: {fig_path}")
    for path in species_plot_paths:
        print(f"Saved: {path}")
    print(f"Saved: {loss_path}")
    print(f"Saved: {metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple DeepXDE ROBER regular PINN (paper-style)")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--num-points", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--t-min", type=float, default=1.0e-5)
    parser.add_argument("--t-max", type=float, default=1.0e5)
    parser.add_argument("--num-eval", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--no-force-cpu", action="store_true")
    parser.add_argument("--approach", choices=["global", "slab_irk", "slab_bdf"], default="global")
    parser.add_argument("--num-slabs", type=int, default=8)
    parser.add_argument("--irk-order", type=int, choices=[2, 4], default=2)
    parser.add_argument("--steps-per-slab", type=int, default=40)
    parser.add_argument("--irk-weight", type=float, default=1.0)
    parser.add_argument("--interface-weight", type=float, default=0.0)
    parser.add_argument("--lbfgs-iters", type=int, default=0)
    parser.add_argument("--use-pinn-ic", action="store_true",
                        help="Propagate slab IC from PINN endpoint instead of IRK endpoint")
    parser.add_argument("--no-adaptive-weights", action="store_true",
                        help="Disable adaptive y2 loss weighting (use uniform weights [1,1,1])")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        iterations=args.iterations,
        num_points=args.num_points,
        batch_size=args.batch_size,
        lr=args.lr,
        t_min=args.t_min,
        t_max=args.t_max,
        num_eval=args.num_eval,
        seed=args.seed,
        run_tag=args.run_tag,
        force_cpu=not args.no_force_cpu,
        approach=args.approach,
        num_slabs=args.num_slabs,
        irk_order=args.irk_order,
        steps_per_slab=args.steps_per_slab,
        irk_weight=args.irk_weight,
        interface_weight=args.interface_weight,
        lbfgs_iters=args.lbfgs_iters,
        use_pinn_ic=args.use_pinn_ic,
        adaptive_weights=not args.no_adaptive_weights,
    )

