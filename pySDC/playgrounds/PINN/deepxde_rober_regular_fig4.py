"""Regular PINN setup for ROBER, targeting Figure-4-style reproduction.

This script follows the regular PINN ingredients described in the paper:
- full ROBER system (no QSSA in the PINN equations),
- logarithmic time domain t in [1e-5, 1e5],
- hard-coded initial condition architecture y(t) = y0 + (t / t_scale) * S * NN(log(t / t_scale)),
- 3 hidden layers with 128 neurons and GELU activation,
- Adam optimizer with learning rate 1e-3 and minibatch size 128,
- residual points sampled uniformly in logarithmic scale,
- comparison against a BDF reference solution.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# Keep backend explicit but overridable from the shell.
os.environ.setdefault("DDE_BACKEND", "pytorch")

import deepxde as dde


class StopIfHighLoss(dde.callbacks.Callback):
    """Stop training when the total training loss stays above a threshold."""

    def __init__(
        self,
        threshold: float,
        start_from_iteration: int = 0,
        patience: int = 1,
        check_every: int = 1,
    ):
        super().__init__()
        self.threshold = float(threshold)
        self.start_from_iteration = int(start_from_iteration)
        self.patience = int(max(1, patience))
        self.check_every = int(max(1, check_every))
        self.triggered = False
        self.stop_iteration = -1
        self.stop_loss = float("nan")
        self.stop_reason = ""
        self._breaches = 0

    def on_train_begin(self):
        self.triggered = False
        self.stop_iteration = -1
        self.stop_loss = float("nan")
        self.stop_reason = ""
        self._breaches = 0

    def on_epoch_end(self):
        iteration = self.model.train_state.iteration
        if iteration < self.start_from_iteration:
            return
        # DeepXDE refreshes train/test loss at display/test cadence.
        # Checking every step can repeatedly reuse stale values and trigger false stops.
        if iteration % self.check_every != 0:
            return

        losses = self.model.train_state.loss_train
        if losses is None or len(losses) == 0:
            return

        current = float(np.sum(losses))
        if not np.isfinite(current):
            current = float("inf")

        if current > self.threshold:
            self._breaches += 1
        else:
            self._breaches = 0

        if self._breaches >= self.patience:
            self.triggered = True
            self.stop_iteration = iteration
            self.stop_loss = current
            self.stop_reason = "high_loss"
            self.model.stop_training = True


class StopIfLossExplodes(dde.callbacks.Callback):
    """Stop training when loss becomes non-finite or clearly explodes."""

    def __init__(self, max_abs_loss: float, start_from_iteration: int = 0, check_every: int = 1):
        super().__init__()
        self.max_abs_loss = float(max_abs_loss)
        self.start_from_iteration = int(start_from_iteration)
        self.check_every = int(max(1, check_every))
        self.triggered = False
        self.stop_iteration = -1
        self.stop_loss = float("nan")
        self.stop_reason = ""

    def on_train_begin(self):
        self.triggered = False
        self.stop_iteration = -1
        self.stop_loss = float("nan")
        self.stop_reason = ""

    def on_epoch_end(self):
        iteration = self.model.train_state.iteration
        if iteration < self.start_from_iteration:
            return
        if iteration % self.check_every != 0:
            return

        losses = self.model.train_state.loss_train
        if losses is None or len(losses) == 0:
            return

        current = float(np.sum(losses))
        if not np.isfinite(current):
            self.triggered = True
            self.stop_iteration = iteration
            self.stop_loss = float("inf")
            self.stop_reason = "non_finite_loss"
            self.model.stop_training = True
            return

        if current > self.max_abs_loss:
            self.triggered = True
            self.stop_iteration = iteration
            self.stop_loss = current
            self.stop_reason = "loss_explosion"
            self.model.stop_training = True


class TrainingDynamicsSnapshot(dde.callbacks.Callback):
    """Print compact trajectory diagnostics at selected iterations."""

    def __init__(self, t_eval: np.ndarray, y_ref: np.ndarray, checkpoints: set[int]):
        super().__init__()
        self.t_eval = t_eval.reshape(-1, 1)
        self.y_ref = y_ref
        self.checkpoints = set(int(k) for k in checkpoints if int(k) > 0)
        self.snapshots: dict[int, dict[str, np.ndarray]] = {}

    def _snapshot(self, iteration: int) -> None:
        y_pred = self.model.predict(self.t_eval)
        pred_range = np.ptp(y_pred, axis=0)
        ref_range = np.maximum(np.ptp(self.y_ref, axis=0), 1.0e-16)
        dynamic_ratio = pred_range / ref_range
        species_mean = np.mean(y_pred, axis=0)
        self.snapshots[iteration] = {
            "dynamic_ratio": dynamic_ratio,
            "species_mean": species_mean,
        }
        print(
            "[diag] iter="
            f"{iteration}: dyn_ratio(y1,y2,y3)=({dynamic_ratio[0]:.3e}, {dynamic_ratio[1]:.3e}, {dynamic_ratio[2]:.3e}), "
            f"mean(y1,y2,y3)=({species_mean[0]:.3e}, {species_mean[1]:.3e}, {species_mean[2]:.3e})"
        )

    def on_train_begin(self):
        self._snapshot(0)

    def on_epoch_end(self):
        iteration = self.model.train_state.iteration
        if iteration in self.checkpoints:
            self._snapshot(iteration)


def rober_rhs(_t: float, y: np.ndarray) -> np.ndarray:
    """RHS of the ROBER problem."""

    y1, y2, y3 = y
    return np.array(
        [
            -0.04 * y1 + 1.0e4 * y2 * y3,
            0.04 * y1 - 1.0e4 * y2 * y3 - 3.0e7 * y2 * y2,
            3.0e7 * y2 * y2,
        ]
    )


def make_rober_residual(residual_scales: np.ndarray | None = None):
    """Build residuals in physical time t, optionally rescaled per species."""

    if residual_scales is None:
        residual_scales = np.ones(3, dtype=float)
    residual_scales = np.asarray(residual_scales, dtype=float).reshape(3)

    def residual(t, y):
        y1 = y[:, 0:1]
        y2 = y[:, 1:2]
        y3 = y[:, 2:3]

        dy1_dt = dde.grad.jacobian(y, t, i=0, j=0)
        dy2_dt = dde.grad.jacobian(y, t, i=1, j=0)
        dy3_dt = dde.grad.jacobian(y, t, i=2, j=0)

        eq1 = (dy1_dt + 0.04 * y1 - 1.0e4 * y2 * y3) * residual_scales[0]
        eq2 = (dy2_dt - 0.04 * y1 + 1.0e4 * y2 * y3 + 3.0e7 * y2 * y2) * residual_scales[1]
        eq3 = (dy3_dt - 3.0e7 * y2 * y2) * residual_scales[2]

        return [eq1, eq2, eq3]

    return residual


def sample_time_points(t_min: float, t_max: float, num_points: int, mode: str, seed: int) -> np.ndarray:
    """Sample residual points in physical time."""

    if mode == "geomspace":
        return np.geomspace(t_min, t_max, num_points).reshape(-1, 1)

    if mode != "random_log":
        raise ValueError(f"Unknown sampling mode: {mode}")

    rng = np.random.default_rng(seed)
    anchors = np.power(10.0, rng.uniform(np.log10(t_min), np.log10(t_max), size=num_points))
    anchors.sort()
    return anchors.reshape(-1, 1)


def compute_reference_solution(t_eval: np.ndarray, t_max: float) -> np.ndarray:
    """Reference BDF solution used for evaluation and scale estimation."""

    y0 = np.array([1.0, 0.0, 0.0])
    ref = solve_ivp(
        rober_rhs,
        (0.0, t_max),
        y0,
        method="BDF",
        t_eval=t_eval,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    if not ref.success:
        raise RuntimeError(f"Reference solve failed: {ref.message}")
    return ref.y.T


def set_default_float_for_backend(force_cpu: bool) -> None:
    """Choose a backend-compatible float precision."""

    default_float = "float64"
    if dde.backend.backend_name == "pytorch":
        import torch

        if force_cpu:
            # Matching CPU execution avoids MPS float64 limitations on macOS.
            torch.set_default_device("cpu")
        # Apple MPS currently does not support float64 tensors.
        elif torch.backends.mps.is_available():
            default_float = "float32"
    dde.config.set_default_float(default_float)


def run(
    iterations: int = 20000,
    num_points: int = 2500,
    batch_size: int | None = 128,
    lr: float = 1.0e-3,
    t_min: float = 1.0e-5,
    t_max: float = 1.0e5,
    num_eval: int = 600,
    seed: int = 42,
    log_base: float = 10.0,
    force_cpu: bool = True,
    display_every: int = 100,
    max_loss_stop: float | None = None,
    max_divergence_loss: float | None = None,
    divergence_warmup_iterations: int = 50,
    loss_warmup_iterations: int = 10000,
    loss_breach_patience: int = 2,
    min_dynamic_ratio: float = 1.0e-2,
    zero_last_layer_init: bool = False,
    hard_ic: bool = True,
    use_log_input: bool = True,
    use_reference_scaling: bool = True,
    use_residual_scaling: bool = True,
    sampling_mode: str = "random_log",
    time_scale: float | None = None,
    run_tag: str = "",
) -> None:
    """Train a regular PINN for ROBER and compare with BDF reference."""

    set_default_float_for_backend(force_cpu=force_cpu)
    dde.config.set_random_seed(seed)

    t_eval = np.geomspace(t_min, t_max, num_eval)
    y_ref = compute_reference_solution(t_eval=t_eval, t_max=t_max)
    y0 = np.array([1.0, 0.0, 0.0])
    time_scale_value = float(t_max if time_scale is None else time_scale)
    if time_scale_value <= 0.0:
        raise ValueError("time_scale must be strictly positive")
    species_scale = np.ones(3, dtype=float)
    if use_reference_scaling:
        species_scale = np.maximum(np.max(np.abs(y_ref), axis=0), 1.0e-12)
    residual_scales = np.ones(3, dtype=float)
    if use_residual_scaling:
        residual_scales = 1.0 / species_scale

    geom_t_min = t_min if hard_ic else 0.0
    geom = dde.geometry.TimeDomain(geom_t_min, t_max)

    # The paper samples residual points uniformly in logarithmic scale.
    anchors = sample_time_points(t_min=t_min, t_max=t_max, num_points=num_points, mode=sampling_mode, seed=seed)

    bcs = []
    if not hard_ic:
        initial_t = np.array([[0.0]])
        bcs = [
            dde.icbc.PointSetBC(initial_t, np.array([[1.0]]), component=0),
            dde.icbc.PointSetBC(initial_t, np.array([[0.0]]), component=1),
            dde.icbc.PointSetBC(initial_t, np.array([[0.0]]), component=2),
        ]

    data = dde.data.PDE(
        geom,
        make_rober_residual(residual_scales=residual_scales),
        bcs=bcs,
        num_domain=0,
        num_boundary=0,
        anchors=anchors,
        num_test=min(2000, num_eval),
    )

    net = dde.nn.FNN([1, 128, 128, 128, 3], "gelu", "Glorot uniform")
    if zero_last_layer_init and dde.backend.backend_name == "pytorch" and hasattr(net, "linears"):
        import torch

        # Start close to y(t)=y0 in the hard-IC ansatz y=y0+t*NN(log t).
        with torch.no_grad():
            torch.nn.init.zeros_(net.linears[-1].weight)
            torch.nn.init.zeros_(net.linears[-1].bias)

    def feature_transform(t):
        shifted_t = t if hard_ic else (t + t_min)
        if dde.backend.backend_name == "pytorch":
            import torch

            scaled_t = shifted_t / time_scale_value
            return torch.log(scaled_t) / np.log(log_base) if use_log_input else scaled_t
        scaled_t = shifted_t / time_scale_value
        return np.log(scaled_t) / np.log(log_base) if use_log_input else scaled_t

    if use_log_input:
        net.apply_feature_transform(feature_transform)

    def output_transform(t, y_raw):
        if dde.backend.backend_name == "pytorch":
            import torch

            scale_t = torch.tensor(species_scale, dtype=y_raw.dtype, device=y_raw.device).reshape(1, 3)
            time_t = t / time_scale_value
            y0_t = torch.tensor([1.0, 0.0, 0.0], dtype=y_raw.dtype, device=y_raw.device)
            if hard_ic:
                return y0_t.reshape(1, 3) + time_t * scale_t * y_raw
            return scale_t * y_raw
        scale_b = dde.backend.as_tensor(species_scale.reshape(1, 3))
        if hard_ic:
            y0_b = dde.backend.as_tensor(np.array([[1.0, 0.0, 0.0]]))
            return y0_b + (t / time_scale_value) * scale_b * y_raw
        return scale_b * y_raw

    if hard_ic or use_reference_scaling:
        net.apply_output_transform(output_transform)

    model = dde.Model(data, net)
    model.compile("adam", lr=lr)

    callbacks = []
    loss_guard = None
    divergence_guard = None
    checkpoints = {max(1, iterations // 2), iterations}
    dynamics_snapshot = TrainingDynamicsSnapshot(t_eval=t_eval, y_ref=y_ref, checkpoints=checkpoints)
    callbacks.append(dynamics_snapshot)
    if max_divergence_loss is not None:
        divergence_guard = StopIfLossExplodes(
            max_abs_loss=max_divergence_loss,
            start_from_iteration=divergence_warmup_iterations,
            check_every=max(display_every, 1),
        )
        callbacks.append(divergence_guard)

    if max_loss_stop is not None:
        loss_guard = StopIfHighLoss(
            threshold=max_loss_stop,
            start_from_iteration=loss_warmup_iterations,
            patience=loss_breach_patience,
            check_every=max(display_every, 1),
        )
        callbacks.append(loss_guard)

    minibatch_mode = bool(batch_size is not None and batch_size > 0)
    if minibatch_mode:
        try:
            loss_history, _ = model.train(
                iterations=iterations,
                batch_size=batch_size,
                display_every=max(display_every, 1),
                callbacks=callbacks,
            )
        except TypeError as exc:
            if "batch_size" not in str(exc):
                raise
            # Some DeepXDE/PDE combinations ignore or do not support PDE mini-batches.
            loss_history, _ = model.train(
                iterations=iterations,
                display_every=max(display_every, 1),
                callbacks=callbacks,
            )
            minibatch_mode = False
    else:
        loss_history, _ = model.train(
            iterations=iterations,
            display_every=max(display_every, 1),
            callbacks=callbacks,
        )

    y_pred = model.predict(t_eval.reshape(-1, 1))

    rel_l2 = []
    for i in range(3):
        denom = max(np.linalg.norm(y_ref[:, i]), 1.0e-16)
        rel_l2.append(np.linalg.norm(y_pred[:, i] - y_ref[:, i]) / denom)

    mass_drift = np.max(np.abs(np.sum(y_pred, axis=1) - 1.0))
    final_total_loss = float(np.sum(loss_history.loss_train[-1])) if len(loss_history.loss_train) > 0 else float("nan")
    early_stopped = loss_guard.triggered if loss_guard is not None else False
    diverged_stopped = divergence_guard.triggered if divergence_guard is not None else False
    pred_range = np.ptp(y_pred, axis=0)
    ref_range = np.maximum(np.ptp(y_ref, axis=0), 1.0e-16)
    dynamic_ratio = pred_range / ref_range
    trivial_solution = bool(
        dynamic_ratio[0] < min_dynamic_ratio
        and dynamic_ratio[2] < min_dynamic_ratio
        and dynamic_ratio[1] < min_dynamic_ratio
    )

    print(f"DeepXDE backend: {dde.backend.backend_name}")
    print(f"Force CPU mode: {force_cpu}")
    print(f"Minibatch mode active: {minibatch_mode}")
    print(f"Hard-coded ICs active: {hard_ic}")
    print(f"Log-time NN input active: {use_log_input}")
    print(f"Sampling mode: {sampling_mode}")
    print(f"Time scale in NN/log transform: {time_scale_value:.3e}")
    print(f"Species scale: {species_scale[0]:.3e}, {species_scale[1]:.3e}, {species_scale[2]:.3e}")
    print(f"Residual scale: {residual_scales[0]:.3e}, {residual_scales[1]:.3e}, {residual_scales[2]:.3e}")
    print(f"Relative L2 error y1: {rel_l2[0]:.3e}")
    print(f"Relative L2 error y2: {rel_l2[1]:.3e}")
    print(f"Relative L2 error y3: {rel_l2[2]:.3e}")
    print(f"Dynamic range ratio y1/y2/y3: {dynamic_ratio[0]:.3e}, {dynamic_ratio[1]:.3e}, {dynamic_ratio[2]:.3e}")
    print(f"Trivial-solution flag: {trivial_solution}")
    print(f"Max mass-conservation drift |y1+y2+y3-1|: {mass_drift:.3e}")
    print(f"Final total training loss: {final_total_loss:.3e}")
    print(f"Best-train-loss step (DeepXDE): {model.train_state.best_step}")
    print(f"Best total train loss (DeepXDE): {model.train_state.best_loss_train:.3e}")
    if diverged_stopped:
        print(
            "Early stopped due to divergence guard at iteration "
            f"{divergence_guard.stop_iteration}: {divergence_guard.stop_loss:.3e} "
            f"({divergence_guard.stop_reason})"
        )
    if early_stopped:
        print(
            f"Early stopped due to high loss at iteration {loss_guard.stop_iteration}: "
            f"{loss_guard.stop_loss:.3e} ({loss_guard.stop_reason})"
        )

    if 0 in dynamics_snapshot.snapshots:
        init_diag = dynamics_snapshot.snapshots[0]
        init_ratio = init_diag["dynamic_ratio"]
        init_mean = init_diag["species_mean"]
        print(
            "Initial model diagnostics: "
            f"dyn_ratio(y1,y2,y3)=({init_ratio[0]:.3e}, {init_ratio[1]:.3e}, {init_ratio[2]:.3e}), "
            f"mean(y1,y2,y3)=({init_mean[0]:.3e}, {init_mean[1]:.3e}, {init_mean[2]:.3e})"
        )

    out_dir = Path(__file__).with_name("data")
    out_dir.mkdir(exist_ok=True)
    tag_suffix = f"_{run_tag}" if run_tag else ""

    # Figure-4-style comparison plot.
    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    labels = ["y1", "y2", "y3"]
    y_limits = [(0.0, 1.0), (0.0, 5.0e-5), (0.0, 1.0)]
    for i, ax in enumerate(axes):
        ax.semilogx(t_eval, y_ref[:, i], "k-", label=f"BDF {labels[i]}")
        ax.semilogx(t_eval, y_pred[:, i], "r--", label=f"regular PINN {labels[i]}")
        ax.set_ylabel(labels[i])
        ax.set_ylim(*y_limits[i])
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best")
    axes[-1].set_xlabel("t [s]")
    fig.suptitle("ROBER: BDF vs regular PINN (Figure-4-style)")
    fig.tight_layout()
    fig4_path = out_dir / f"deepxde_rober_regular_fig4{tag_suffix}_solution.png"
    fig.savefig(fig4_path, dpi=200)
    plt.close(fig)

    # Figure-5-style loss history (log scale).
    steps = np.array(loss_history.steps)
    train_loss = np.array(loss_history.loss_train)
    total_loss = np.sum(train_loss, axis=1)

    fig_loss, ax_loss = plt.subplots(figsize=(7, 4))
    ax_loss.semilogy(steps, total_loss, "b-", label="total training loss")
    ax_loss.set_xlabel("Parameter updates")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Regular PINN loss history (Figure-5-style)")
    ax_loss.grid(True, which="both", alpha=0.3)
    ax_loss.legend(loc="best")
    fig_loss.tight_layout()
    fig5_path = out_dir / f"deepxde_rober_regular_fig4{tag_suffix}_loss.png"
    fig_loss.savefig(fig5_path, dpi=200)
    plt.close(fig_loss)

    metrics_path = out_dir / f"deepxde_rober_regular_fig4{tag_suffix}_metrics.txt"
    with metrics_path.open("w", encoding="ascii") as f:
        f.write(f"backend={dde.backend.backend_name}\n")
        f.write(f"seed={seed}\n")
        f.write(f"force_cpu={force_cpu}\n")
        f.write(f"iterations={iterations}\n")
        f.write(f"num_points={num_points}\n")
        f.write(f"batch_size={batch_size}\n")
        f.write(f"lr={lr}\n")
        f.write(f"display_every={display_every}\n")
        f.write(f"minibatch_mode={minibatch_mode}\n")
        f.write(f"hard_ic={hard_ic}\n")
        f.write(f"use_log_input={use_log_input}\n")
        f.write(f"use_reference_scaling={use_reference_scaling}\n")
        f.write(f"use_residual_scaling={use_residual_scaling}\n")
        f.write(f"sampling_mode={sampling_mode}\n")
        f.write(f"time_scale={time_scale_value}\n")
        f.write(f"max_loss_stop={max_loss_stop}\n")
        f.write(f"max_divergence_loss={max_divergence_loss}\n")
        f.write(f"divergence_warmup_iterations={divergence_warmup_iterations}\n")
        f.write(f"loss_warmup_iterations={loss_warmup_iterations}\n")
        f.write(f"loss_breach_patience={loss_breach_patience}\n")
        f.write(f"min_dynamic_ratio={min_dynamic_ratio}\n")
        f.write(f"zero_last_layer_init={zero_last_layer_init}\n")
        f.write(f"species_scale_y1={species_scale[0]:.16e}\n")
        f.write(f"species_scale_y2={species_scale[1]:.16e}\n")
        f.write(f"species_scale_y3={species_scale[2]:.16e}\n")
        f.write(f"residual_scale_y1={residual_scales[0]:.16e}\n")
        f.write(f"residual_scale_y2={residual_scales[1]:.16e}\n")
        f.write(f"residual_scale_y3={residual_scales[2]:.16e}\n")
        f.write(f"early_stopped={early_stopped}\n")
        f.write(f"diverged_stopped={diverged_stopped}\n")
        if loss_guard is not None:
            f.write(f"early_stop_iteration={loss_guard.stop_iteration}\n")
            f.write(f"early_stop_loss={loss_guard.stop_loss:.16e}\n")
            f.write(f"early_stop_reason={loss_guard.stop_reason}\n")
        if divergence_guard is not None:
            f.write(f"divergence_stop_iteration={divergence_guard.stop_iteration}\n")
            f.write(f"divergence_stop_loss={divergence_guard.stop_loss:.16e}\n")
            f.write(f"divergence_stop_reason={divergence_guard.stop_reason}\n")
        f.write(f"t_min={t_min}\n")
        f.write(f"t_max={t_max}\n")
        f.write(f"final_total_loss={final_total_loss:.16e}\n")
        f.write(f"best_step={model.train_state.best_step}\n")
        f.write(f"best_total_train_loss={model.train_state.best_loss_train:.16e}\n")
        f.write(f"rel_l2_y1={rel_l2[0]:.16e}\n")
        f.write(f"rel_l2_y2={rel_l2[1]:.16e}\n")
        f.write(f"rel_l2_y3={rel_l2[2]:.16e}\n")
        f.write(f"dynamic_ratio_y1={dynamic_ratio[0]:.16e}\n")
        f.write(f"dynamic_ratio_y2={dynamic_ratio[1]:.16e}\n")
        f.write(f"dynamic_ratio_y3={dynamic_ratio[2]:.16e}\n")
        f.write(f"trivial_solution={trivial_solution}\n")
        f.write(f"mass_drift={mass_drift:.16e}\n")
        for iteration in sorted(dynamics_snapshot.snapshots.keys()):
            diag = dynamics_snapshot.snapshots[iteration]
            dyn = diag["dynamic_ratio"]
            mean = diag["species_mean"]
            f.write(
                f"diag_iter_{iteration}_dynamic_ratio={dyn[0]:.16e},{dyn[1]:.16e},{dyn[2]:.16e}\n"
            )
            f.write(f"diag_iter_{iteration}_species_mean={mean[0]:.16e},{mean[1]:.16e},{mean[2]:.16e}\n")

    print(f"Saved Figure-4-style plot to: {fig4_path}")
    print(f"Saved Figure-5-style loss plot to: {fig5_path}")
    print(f"Saved metrics to: {metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regular PINN ROBER run for Figure-4-style reproduction")
    parser.add_argument("--iterations", type=int, default=20000, help="Number of Adam parameter updates")
    parser.add_argument("--num-points", type=int, default=2500, help="Number of log-time collocation points")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Mini-batch size (>0 enables mini-batching). Paper baseline uses 128.",
    )
    parser.add_argument("--lr", type=float, default=1.0e-3, help="Adam learning rate")
    parser.add_argument("--t-min", type=float, default=1.0e-5, help="Lower time bound (>0)")
    parser.add_argument("--t-max", type=float, default=1.0e5, help="Upper time bound")
    parser.add_argument("--num-eval", type=int, default=600, help="Number of evaluation points")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-base", type=float, default=10.0, help="Log base for transformed time coordinate")
    parser.add_argument("--display-every", type=int, default=100, help="Print/test cadence during training")
    parser.add_argument(
        "--max-loss-stop",
        type=float,
        default=-1.0,
        help="Early-stop threshold for total training loss. Negative value disables (paper baseline).",
    )
    parser.add_argument(
        "--max-divergence-loss",
        type=float,
        default=-1.0,
        help="Stop on non-finite/exploded loss. Negative value disables (paper baseline).",
    )
    parser.add_argument(
        "--divergence-warmup-iterations",
        type=int,
        default=50,
        help="Iterations to skip before activating divergence guard.",
    )
    parser.add_argument(
        "--loss-warmup-iterations",
        type=int,
        default=10000,
        help="Iterations to skip before activating high-loss early stopping (Fig.5-style long warmup).",
    )
    parser.add_argument(
        "--loss-breach-patience",
        type=int,
        default=2,
        help="Number of consecutive high-loss checks before stopping.",
    )
    parser.add_argument(
        "--min-dynamic-ratio",
        type=float,
        default=1.0e-2,
        help="Below this ratio (predicted-range/reference-range) for all species, mark solution as trivial.",
    )
    parser.add_argument(
        "--zero-last-layer-init",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use zero initialization of the final linear layer. Default is off (paper-style Xavier init).",
    )
    parser.add_argument(
        "--hard-ic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hard-code initial conditions via y=y0+t*S*NN(...). Paper baseline uses this.",
    )
    parser.add_argument(
        "--use-log-input",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use log(t) as the NN input feature. Equation (3.1) in the paper uses this.",
    )
    parser.add_argument(
        "--use-reference-scaling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scale NN outputs by reference species magnitudes, matching the authors' released code more closely.",
    )
    parser.add_argument(
        "--use-residual-scaling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scale residuals inversely to species magnitudes to reduce loss imbalance.",
    )
    parser.add_argument(
        "--sampling-mode",
        choices=["random_log", "geomspace"],
        default="random_log",
        help="How to place residual points in physical time. The paper describes uniform sampling in log scale.",
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=-1.0,
        help="Scale used inside log(t/time_scale) and y=y0+(t/time_scale)SNN. Negative value uses t_max.",
    )
    parser.add_argument("--run-tag", type=str, default="", help="Optional suffix for output filenames")
    parser.add_argument(
        "--no-force-cpu",
        action="store_true",
        help="Allow backend/device auto-selection instead of forcing PyTorch CPU.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.t_min <= 0.0:
        raise ValueError("--t-min must be strictly positive")
    if args.t_min >= args.t_max:
        raise ValueError("--t-min must be smaller than --t-max")
    if args.log_base <= 1.0:
        raise ValueError("--log-base must be larger than 1")
    max_loss_stop = None if args.max_loss_stop < 0 else args.max_loss_stop
    max_divergence_loss = None if args.max_divergence_loss < 0 else args.max_divergence_loss
    batch_size = None if args.batch_size <= 0 else args.batch_size
    time_scale = None if args.time_scale < 0 else args.time_scale
    run(
        iterations=args.iterations,
        num_points=args.num_points,
        batch_size=batch_size,
        lr=args.lr,
        t_min=args.t_min,
        t_max=args.t_max,
        num_eval=args.num_eval,
        seed=args.seed,
        log_base=args.log_base,
        force_cpu=not args.no_force_cpu,
        display_every=args.display_every,
        max_loss_stop=max_loss_stop,
        max_divergence_loss=max_divergence_loss,
        divergence_warmup_iterations=args.divergence_warmup_iterations,
        loss_warmup_iterations=args.loss_warmup_iterations,
        loss_breach_patience=args.loss_breach_patience,
        min_dynamic_ratio=args.min_dynamic_ratio,
        zero_last_layer_init=args.zero_last_layer_init,
        hard_ic=args.hard_ic,
        use_log_input=args.use_log_input,
        use_reference_scaling=args.use_reference_scaling,
        use_residual_scaling=args.use_residual_scaling,
        sampling_mode=args.sampling_mode,
        time_scale=time_scale,
        run_tag=args.run_tag,
    )


