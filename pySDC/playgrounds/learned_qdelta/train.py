"""Train a baseline PyTorch model for one-sweep SDC correction proposals."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pySDC.playgrounds.learned_qdelta.dataset import (
    dahlquist_lambda_dt,
    dahlquist_regime_labels,
    load_npz_samples,
    make_train_val_datasets,
)
from pySDC.playgrounds.learned_qdelta.models import build_model


def residual_penalty(batch, pred_dU, device):
    """Residual norm after predicted update for scalar linear test equation samples."""
    if batch['u0'].shape[-1] != 1:
        # Currently only implemented for scalar ODE states.
        return torch.zeros((), device=device)

    u0 = batch['u0'].to(device).squeeze(-1)
    Uk = batch['U_k'].to(device).squeeze(-1)
    dt = batch['dt'].to(device).squeeze(-1)
    qmat = batch['qmat'].to(device)
    lam = batch['problem_params'].to(device)[:, 0]

    Utrial = Uk + pred_dU
    Ftrial = lam[:, None] * Utrial
    qf = torch.bmm(qmat, Ftrial.unsqueeze(-1)).squeeze(-1)
    res = u0[:, None] - Utrial + dt[:, None] * qf
    return (res**2).mean()


def evaluate(model, loader, y_std, y_mean, device, residual_weight):
    model.eval()
    mse_total = 0.0
    loss_total = 0.0
    n = 0

    with torch.no_grad():
        for batch in loader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            pred = model(x)

            mse = ((pred - y) ** 2).mean()
            loss = mse

            if residual_weight > 0:
                pred_phys = pred * y_std + y_mean
                loss = loss + residual_weight * residual_penalty(batch, pred_phys, device)

            bs = x.shape[0]
            mse_total += mse.item() * bs
            loss_total += loss.item() * bs
            n += bs

    return loss_total / n, mse_total / n


def parse_float_list(raw: str | None) -> tuple[float, ...]:
    if raw is None or raw.strip() == '':
        return tuple()
    return tuple(float(part.strip()) for part in raw.split(',') if part.strip())


def summarize_dahlquist_regimes(samples, regime_edges):
    lamdt = dahlquist_lambda_dt(samples)
    labels = dahlquist_regime_labels(lamdt, regime_edges=regime_edges)
    out = {}
    for label in np.unique(labels):
        mask = labels == label
        out[int(label)] = {
            'count': int(mask.sum()),
            'lamdt_min': float(lamdt[mask].min()),
            'lamdt_max': float(lamdt[mask].max()),
        }
    return out


def get_git_commit() -> str:
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        return commit
    except Exception:
        return 'unknown'


def make_curriculum_loaders(train_ds, batch_size, thresholds):
    if not thresholds:
        return [(None, DataLoader(train_ds, batch_size=batch_size, shuffle=True))]

    lamdt = np.abs(train_ds.samples['problem_params'][:, 0] * train_ds.samples['dt'][:, 0])
    loaders = []
    for threshold in thresholds:
        idx = np.where(lamdt <= threshold)[0]
        if idx.size == 0:
            continue
        subset = torch.utils.data.Subset(train_ds, idx.tolist())
        loaders.append((float(threshold), DataLoader(subset, batch_size=batch_size, shuffle=True)))

    if not loaders:
        loaders = [(None, DataLoader(train_ds, batch_size=batch_size, shuffle=True))]
    return loaders


def train(args):
    samples = load_npz_samples(args.data)

    # Auto-detect z-parameterisation from dataset metadata
    use_z_param = bool(args.use_z_param)
    if 'use_z_param' in samples and not use_z_param:
        use_z_param = bool(int(samples['use_z_param'].reshape(-1)[0]))

    regime_edges = parse_float_list(args.regime_edges) or (1.0, 5.0, 15.0)
    holdout_z_interval = None
    if args.holdout_z_interval:
        parts = parse_float_list(args.holdout_z_interval)
        if len(parts) == 2:
            holdout_z_interval = tuple(parts)

    train_ds, val_ds, x_norm, y_norm = make_train_val_datasets(
        samples,
        val_fraction=args.val_fraction,
        seed=args.seed,
        split_strategy=args.split_strategy,
        regime_edges=tuple(regime_edges),
        holdout_regime=args.holdout_regime,
        holdout_z_interval=holdout_z_interval,
        use_z_param=use_z_param,
        dimension_agnostic=args.dimension_agnostic,
    )

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    input_dim = train_ds[0]['x'].numel()
    output_dim = train_ds[0]['y'].numel()

    model_cfg = {
        'name': args.model,
        'hidden_dim': args.hidden_dim,
        'depth': args.depth,
    }

    device = torch.device(args.device)
    model = build_model(model_cfg, input_dim=input_dim, output_dim=output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    y_mean = torch.from_numpy(y_norm.mean).float().to(device)
    y_std = torch.from_numpy(y_norm.std).float().to(device)

    best_val = float('inf')
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    curriculum_thresholds = parse_float_list(args.curriculum_thresholds)
    train_loaders = make_curriculum_loaders(train_ds, args.batch_size, curriculum_thresholds)

    stage_len = max(1, args.epochs // len(train_loaders))

    history = []  # list of per-epoch dicts

    manifest = {
        'data_path': args.data,
        'split_strategy': args.split_strategy,
        'holdout_regime': args.holdout_regime,
        'regime_edges': list(regime_edges),
        'train_size': int(len(train_ds)),
        'val_size': int(len(val_ds)),
        'seed': int(args.seed),
        'git_commit': get_git_commit(),
        'model_config': model_cfg,
        'use_z_param': use_z_param,
        'dimension_agnostic': args.dimension_agnostic,
        'dahlquist_regimes': summarize_dahlquist_regimes(samples, tuple(regime_edges)) if samples['u0'].shape[1] == 1 else {},
        'curriculum_thresholds': list(curriculum_thresholds),
    }
    with open(out_dir / 'train_manifest.json', 'w') as fobj:
        json.dump(manifest, fobj, indent=2)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_mse = 0.0
        count = 0

        stage_idx = min((epoch - 1) // stage_len, len(train_loaders) - 1)
        stage_thr, train_loader = train_loaders[stage_idx]

        for batch in train_loader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)

            pred = model(x)
            mse = ((pred - y) ** 2).mean()
            loss = mse

            if args.residual_weight > 0:
                pred_phys = pred * y_std + y_mean
                loss = loss + args.residual_weight * residual_penalty(batch, pred_phys, device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = x.shape[0]
            train_loss += loss.item() * bs
            train_mse += mse.item() * bs
            count += bs

        scheduler.step()

        train_loss /= count
        train_mse /= count

        val_loss, val_mse = evaluate(model, val_loader, y_std, y_mean, device, args.residual_weight)

        print(
            f'epoch={epoch:4d}  train_loss={train_loss:.3e}  train_mse={train_mse:.3e}  '
            f'val_loss={val_loss:.3e}  val_mse={val_mse:.3e}  '
            f'curriculum_thr={stage_thr if stage_thr is not None else "all"}'
        )

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_mse': train_mse,
            'val_loss': val_loss,
            'val_mse': val_mse,
            'curriculum_thr': stage_thr,
        })

        checkpoint = {
            'model_state': model.state_dict(),
            'model_config': model_cfg,
            'input_dim': input_dim,
            'output_dim': output_dim,
            'x_normalizer': x_norm.state_dict(),
            'y_normalizer': y_norm.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss,
            'manifest': manifest,
            'use_z_param': use_z_param,
            'dimension_agnostic': args.dimension_agnostic,
            'history': history,
        }
        torch.save(checkpoint, out_dir / 'last.pt')

        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, out_dir / 'best.pt')

    print(f'Best validation loss: {best_val:.3e}')


def main():
    parser = argparse.ArgumentParser(description='Train learned Q_delta prototype model.')
    parser.add_argument('--data', type=str, default='pySDC/playgrounds/learned_qdelta/data/sdc_sweeps.npz')
    parser.add_argument('--output-dir', type=str, default='pySDC/playgrounds/learned_qdelta/checkpoints')
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--val-fraction', type=float, default=0.2)
    parser.add_argument('--split-strategy', type=str, choices=['random', 'dahlquist_regime'], default='random')
    parser.add_argument('--regime-edges', type=str, default='1.0,5.0,15.0')
    parser.add_argument('--holdout-regime', type=str, default=None)
    parser.add_argument('--holdout-z-interval', type=str, default=None,
                        help='Hold out a z interval, e.g. "-6,-4" to exclude z in [-6, -4]')
    parser.add_argument('--curriculum-thresholds', type=str, default='')
    parser.add_argument('--residual-weight', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--use-z-param', action='store_true', default=False,
                        help='Use z=lambda*dt as sole problem parameter; drop dt from feature vector')
    parser.add_argument('--dimension-agnostic', action='store_true', default=False,
                        help='Use dimension-agnostic features (scalar summaries). Generalizes across grid sizes.')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    train(args)


if __name__ == '__main__':
    main()


