"""Portable runner for the upstream Stiff-PINN Robertson QSSA example.

This is a small macOS-friendly port of the Robertson QSSA example from
https://github.com/DENG-MIT/Stiff-PINN. The original upstream code depends on
``assimulo`` and uses ad-hoc relative paths/checkpointing; this version keeps
the core PyTorch/QSSA PINN formulation while using SciPy for the reference
solution and writing artifacts into the playground ``data/`` directory.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import solve_ivp
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


K0 = 0.04
K1 = 3.0e7
K2 = 1.0e4


def robertson_rhs(_t: float, y: np.ndarray) -> np.ndarray:
    y1, y2, y3 = y
    return np.array(
        [
            -K0 * y1 + K2 * y2 * y3,
            K0 * y1 - K2 * y2 * y3 - K1 * y2 * y2,
            K1 * y2 * y2,
        ],
        dtype=float,
    )


def reference_solution(t_eval: np.ndarray, t_max: float) -> np.ndarray:
    ref = solve_ivp(
        robertson_rhs,
        (0.0, t_max),
        np.array([1.0, 0.0, 0.0], dtype=float),
        method="BDF",
        t_eval=t_eval,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    if not ref.success:
        raise RuntimeError(f"Reference solve failed: {ref.message}")
    return ref.y.T


def qssa_y2_from_y1_y3(y1: torch.Tensor, y3: torch.Tensor) -> torch.Tensor:
    delta = torch.clamp((K2 * y3) ** 2 + 4.0 * K0 * y1 * K1, min=0.0)
    return (-K2 * y3 + torch.sqrt(delta)) / (2.0 * K1)


def qssa_y2_from_numpy(y1: np.ndarray, y3: np.ndarray) -> np.ndarray:
    delta = np.clip((K2 * y3) ** 2 + 4.0 * K0 * y1 * K1, a_min=0.0, a_max=None)
    return (-K2 * y3 + np.sqrt(delta)) / (2.0 * K1)


class PINNModel(nn.Module):
    def __init__(self, y0: torch.Tensor, w_scale: torch.Tensor, x_scale: float, width: int, depth: int):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(1, width), nn.GELU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.GELU()])
        layers.append(nn.Linear(width, y0.shape[1]))
        self.seq = nn.Sequential(*layers)
        self.y0 = y0
        self.w_scale = w_scale
        self.x_scale = float(x_scale)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.seq:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        scaled_t = t / self.x_scale
        return self.y0 + self.w_scale * scaled_t * self.seq(torch.log(scaled_t))


def grad_norm(model: nn.Module) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        total += float(parameter.grad.detach().norm(2).item() ** 2)
    return total**0.5


def run(
    epochs: int = 2000,
    n_grid_train: int = 2500,
    n_grid_test: int = 100,
    batch_size: int = 512,
    width: int = 120,
    depth: int = 3,
    learning_rate: float = 1.0e-3,
    t_min: float = 1.0e-2,
    t_max: float = 1.0e5,
    num_eval: int = 400,
    print_every: int = 100,
    loss_scale: float = 1.0e5,
    seed: int = 0,
    run_tag: str = "",
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_default_dtype(torch.float64)
    device = torch.device("cpu")

    t_eval = np.logspace(np.log10(t_min), np.log10(t_max), num=num_eval)
    y_ref = reference_solution(t_eval=t_eval, t_max=t_max)
    y_ref_qssa = y_ref[:, [0, 2]]

    y0 = torch.tensor([[1.0, 0.0]], dtype=torch.float64, device=device)
    w_scale = torch.tensor(np.maximum(y_ref_qssa.max(axis=0), 1.0e-12), dtype=torch.float64, device=device)
    model = PINNModel(y0=y0, w_scale=w_scale.view(1, -1), x_scale=t_max, width=width, depth=depth).to(device)

    t_train_all = torch.logspace(
        start=np.log10(t_min),
        end=np.log10(t_max),
        steps=n_grid_train,
        dtype=torch.float64,
        device=device,
    ).unsqueeze(1)
    t_test = torch.logspace(
        start=np.log10(t_min),
        end=np.log10(t_max),
        steps=n_grid_test,
        dtype=torch.float64,
        device=device,
    ).unsqueeze(1)
    train_loader = DataLoader(TensorDataset(t_train_all), batch_size=batch_size, shuffle=True, drop_last=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=4.0e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        steps_per_epoch=max(len(train_loader), 1),
        epochs=epochs,
    )
    criterion = nn.MSELoss()

    history: dict[str, list[float]] = {"loss_train": [], "loss_test": [], "loss_y1": [], "loss_y3": [], "grad_norm": []}

    for epoch in range(1, epochs + 1):
        epoch_losses: list[float] = []
        epoch_loss_y1: list[float] = []
        epoch_loss_y3: list[float] = []
        epoch_grad_norms: list[float] = []

        for (t_batch,) in train_loader:
            t_batch = t_batch.detach().clone().requires_grad_(True)
            y_batch = model(t_batch).abs()
            y1 = y_batch[:, 0]
            y3 = y_batch[:, 1]
            y2 = qssa_y2_from_y1_y3(y1, y3)

            dy1_dt = torch.autograd.grad(y1.sum(), t_batch, retain_graph=True, create_graph=True)[0].view(-1)
            dy3_dt = torch.autograd.grad(y3.sum(), t_batch, retain_graph=True, create_graph=True)[0].view(-1)

            rhs1 = -K0 * y1 + K2 * y2 * y3
            rhs3 = K1 * y2 * y2

            loss_y1 = criterion(dy1_dt, rhs1)
            loss_y3 = criterion(dy3_dt, rhs3)
            loss_train = (loss_y1 + loss_y3) * loss_scale

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            scheduler.step()

            epoch_losses.append(float(loss_train.item()))
            epoch_loss_y1.append(float(loss_y1.item()))
            epoch_loss_y3.append(float(loss_y3.item()))
            epoch_grad_norms.append(grad_norm(model))

        t_test_req = t_test.detach().clone().requires_grad_(True)
        y_test = model(t_test_req).abs()
        y1_test = y_test[:, 0]
        y3_test = y_test[:, 1]
        dy1_test = torch.autograd.grad(y1_test.sum(), t_test_req, retain_graph=True, create_graph=False)[0].view(-1)
        dy3_test = torch.autograd.grad(y3_test.sum(), t_test_req, retain_graph=False, create_graph=False)[0].view(-1)
        y2_test = qssa_y2_from_y1_y3(y1_test, y3_test)
        rhs1_test = -K0 * y1_test + K2 * y2_test * y3_test
        rhs3_test = K1 * y2_test * y2_test
        loss_test = float(((criterion(dy1_test, rhs1_test) + criterion(dy3_test, rhs3_test)) * loss_scale).item())

        history["loss_train"].append(float(np.mean(epoch_losses)))
        history["loss_test"].append(loss_test)
        history["loss_y1"].append(float(np.mean(epoch_loss_y1)))
        history["loss_y3"].append(float(np.mean(epoch_loss_y3)))
        history["grad_norm"].append(float(np.mean(epoch_grad_norms)))

        if epoch == 1 or epoch % max(print_every, 1) == 0 or epoch == epochs:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"epoch={epoch:5d} train={history['loss_train'][-1]:.3e} test={history['loss_test'][-1]:.3e} "
                f"loss_y1={history['loss_y1'][-1]:.3e} loss_y3={history['loss_y3'][-1]:.3e} "
                f"grad_norm={history['grad_norm'][-1]:.3e} lr={current_lr:.3e}"
            )

    with torch.no_grad():
        t_eval_tensor = torch.tensor(t_eval, dtype=torch.float64, device=device).unsqueeze(1)
        y_eval_qssa = model(t_eval_tensor).abs().cpu().numpy()

    y1_pred = y_eval_qssa[:, 0]
    y3_pred = y_eval_qssa[:, 1]
    y2_pred = qssa_y2_from_numpy(y1_pred, y3_pred)
    y_pred = np.column_stack((y1_pred, y2_pred, y3_pred))

    rel_l2 = []
    for i in range(3):
        denom = max(np.linalg.norm(y_ref[:, i]), 1.0e-16)
        rel_l2.append(float(np.linalg.norm(y_pred[:, i] - y_ref[:, i]) / denom))
    mass_drift = float(np.max(np.abs(np.sum(y_pred, axis=1) - 1.0)))

    out_dir = Path(__file__).with_name("data")
    out_dir.mkdir(exist_ok=True)
    tag = f"_{run_tag}" if run_tag else ""

    fig, axes = plt.subplots(4, 1, figsize=(7.5, 10.0), sharex=False)
    labels = ["y1", "y2", "y3"]
    for i in range(3):
        axes[i].semilogx(t_eval, y_ref[:, i], "k-", label=f"BDF {labels[i]}")
        axes[i].semilogx(t_eval, y_pred[:, i], "r--", label=f"Stiff-PINN QSSA {labels[i]}")
        axes[i].set_ylabel(labels[i])
        axes[i].grid(True, which="both", alpha=0.3)
        axes[i].legend(loc="best")
    axes[3].semilogy(np.arange(1, epochs + 1), history["loss_train"], label="train")
    axes[3].semilogy(np.arange(1, epochs + 1), history["loss_test"], label="test")
    axes[3].set_xlabel("epoch")
    axes[3].set_ylabel("loss")
    axes[3].grid(True, which="both", alpha=0.3)
    axes[3].legend(loc="best")
    fig.suptitle("Upstream-style Stiff-PINN Robertson QSSA")
    fig.tight_layout()
    plot_path = out_dir / f"stiff_pinn_robertson_qssa{tag}_solution.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    model_path = out_dir / f"stiff_pinn_robertson_qssa{tag}_model.pt"
    torch.save(model.state_dict(), model_path)

    metrics_path = out_dir / f"stiff_pinn_robertson_qssa{tag}_metrics.txt"
    with metrics_path.open("w", encoding="ascii") as f:
        f.write(f"epochs={epochs}\n")
        f.write(f"n_grid_train={n_grid_train}\n")
        f.write(f"n_grid_test={n_grid_test}\n")
        f.write(f"batch_size={batch_size}\n")
        f.write(f"width={width}\n")
        f.write(f"depth={depth}\n")
        f.write(f"learning_rate={learning_rate}\n")
        f.write(f"t_min={t_min}\n")
        f.write(f"t_max={t_max}\n")
        f.write(f"num_eval={num_eval}\n")
        f.write(f"seed={seed}\n")
        f.write(f"loss_scale={loss_scale}\n")
        f.write(f"final_train_loss={history['loss_train'][-1]:.16e}\n")
        f.write(f"final_test_loss={history['loss_test'][-1]:.16e}\n")
        f.write(f"rel_l2_y1={rel_l2[0]:.16e}\n")
        f.write(f"rel_l2_y2={rel_l2[1]:.16e}\n")
        f.write(f"rel_l2_y3={rel_l2[2]:.16e}\n")
        f.write(f"mass_drift={mass_drift:.16e}\n")
        f.write(f"plot_path={plot_path}\n")
        f.write(f"model_path={model_path}\n")

    print(f"Relative L2 error y1: {rel_l2[0]:.3e}")
    print(f"Relative L2 error y2: {rel_l2[1]:.3e}")
    print(f"Relative L2 error y3: {rel_l2[2]:.3e}")
    print(f"Max mass-conservation drift |y1+y2+y3-1|: {mass_drift:.3e}")
    print(f"Saved plot to: {plot_path}")
    print(f"Saved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the upstream-style Stiff-PINN Robertson QSSA example")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--n-grid-train", type=int, default=2500)
    parser.add_argument("--n-grid-test", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--width", type=int, default=120)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--t-min", type=float, default=1.0e-2)
    parser.add_argument("--t-max", type=float, default=1.0e5)
    parser.add_argument("--num-eval", type=int, default=400)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--loss-scale", type=float, default=1.0e5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-tag", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.t_min <= 0.0:
        raise ValueError("--t-min must be strictly positive")
    if args.t_min >= args.t_max:
        raise ValueError("--t-min must be smaller than --t-max")
    if args.depth < 1:
        raise ValueError("--depth must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    run(
        epochs=args.epochs,
        n_grid_train=args.n_grid_train,
        n_grid_test=args.n_grid_test,
        batch_size=args.batch_size,
        width=args.width,
        depth=args.depth,
        learning_rate=args.learning_rate,
        t_min=args.t_min,
        t_max=args.t_max,
        num_eval=args.num_eval,
        print_every=args.print_every,
        loss_scale=args.loss_scale,
        seed=args.seed,
        run_tag=args.run_tag,
    )
