"""Train an FNO-based sweep-correction model.

Compared to the MLP baseline, the FNO operates on spatial fields directly:
  input:  (B, 1 + 2M, N)  -> u0 + M Uk channels + M Rk channels
  output: (B, M,      N)  -> M correction fields dU

This trainer adds solver-aligned objectives beyond plain regression:
  - projection loss along residual directions (proxy for residual improvement)
  - cosine-margin penalty (proxy for gate acceptance robustness)
  - optional diffusion-CFL balanced sampling to avoid easy-regime dominance
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from pySDC.playgrounds.learned_qdelta.dataset import (
    ChannelNormalizer,
    load_npz_samples,
    make_fno_datasets,
)
from pySDC.playgrounds.learned_qdelta.models import FNO1d


def get_git_commit() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        return 'unknown'


def parse_float_csv(raw: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in raw.split(',') if part.strip())


def build_cfl_balanced_sampler(train_ds, edges: tuple[float, ...], seed: int) -> WeightedRandomSampler | None:
    """Build weighted sampler that balances diffusion-CFL regimes."""
    if len(train_ds) < 2:
        return None

    dt = np.asarray(train_ds.samples['dt'][:, 0], dtype=np.float64)
    nvars = np.asarray([s.shape[0] for s in train_ds.samples['u0']], dtype=np.float64)
    nu = np.asarray(train_ds.samples['problem_params'][:, 0], dtype=np.float64)
    dx = 1.0 / np.maximum(nvars + 1.0, 1.0)
    cfl = nu * dt / np.maximum(dx * dx, 1e-16)

    labels = np.digitize(cfl, np.asarray(edges, dtype=np.float64), right=False)
    unique, counts = np.unique(labels, return_counts=True)
    count_map = {int(k): int(v) for k, v in zip(unique, counts)}
    weights = np.asarray([1.0 / max(count_map[int(lbl)], 1) for lbl in labels], dtype=np.float64)
    weights = weights / max(weights.mean(), 1e-16)

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def evaluate_fno(
    model: FNO1d,
    loader: DataLoader,
    y_norm: ChannelNormalizer,
    device: torch.device,
    cosine_gate_margin: float,
) -> dict[str, float]:
    """Return validation metrics in physical (un-normalised) space."""
    model.eval()
    total_mse, total_proj_mse, total_norm_mse, total_gate, total_cos, count = 0.0, 0.0, 0.0, 0.0, 0.0, 0
    y_std_t = torch.from_numpy(y_norm.std).to(device)
    y_mean_t = torch.from_numpy(y_norm.mean).to(device)

    with torch.no_grad():
        for batch in loader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            r_k = batch['R_k'].to(device)

            pred_norm = model(x)

            pred_phys = pred_norm * y_std_t + y_mean_t
            y_phys = y * y_std_t + y_mean_t
            mse = ((pred_phys - y_phys) ** 2).mean().item()

            proj_pred = torch.sum(pred_phys * r_k, dim=-1)
            proj_true = torch.sum(y_phys * r_k, dim=-1)
            proj_denom = torch.mean(proj_true**2).item() + 1e-12
            proj_mse = torch.mean((proj_pred - proj_true) ** 2).item() / proj_denom

            pred_norm_mag = torch.linalg.vector_norm(pred_phys.reshape(pred_phys.shape[0], pred_phys.shape[1], -1), dim=-1)
            true_norm_mag = torch.linalg.vector_norm(y_phys.reshape(y_phys.shape[0], y_phys.shape[1], -1), dim=-1)
            norm_denom = torch.mean(true_norm_mag**2).item() + 1e-12
            norm_mse = torch.mean((pred_norm_mag - true_norm_mag) ** 2).item() / norm_denom

            cos = F.cosine_similarity(
                pred_phys.reshape(pred_phys.shape[0], pred_phys.shape[1], -1),
                y_phys.reshape(y_phys.shape[0], y_phys.shape[1], -1),
                dim=-1,
            )
            margin = torch.tensor(cosine_gate_margin, dtype=cos.dtype, device=cos.device)
            gate = torch.relu(margin - cos).mean().item()

            total_mse += mse * x.shape[0]
            total_proj_mse += proj_mse * x.shape[0]
            total_norm_mse += norm_mse * x.shape[0]
            total_gate += gate * x.shape[0]
            total_cos += cos.mean().item() * x.shape[0]
            count += x.shape[0]

    if count == 0:
        return {
            'val_mse_phys': float('inf'),
            'val_proj_mse': float('inf'),
            'val_norm_mse': float('inf'),
            'val_gate_loss': float('inf'),
            'val_cosine': -1.0,
            'val_composite': float('inf'),
        }

    val_mse = total_mse / count
    val_proj = total_proj_mse / count
    val_norm = total_norm_mse / count
    val_gate = total_gate / count
    val_cos = total_cos / count
    return {
        'val_mse_phys': val_mse,
        'val_proj_mse': val_proj,
        'val_norm_mse': val_norm,
        'val_gate_loss': val_gate,
        'val_cosine': val_cos,
        'val_composite': val_mse + val_proj + 0.2 * val_gate + 0.1 * val_norm,
    }


def train(args):
    samples = load_npz_samples(args.data)

    train_ds, val_ds, x_norm, y_norm = make_fno_datasets(
        samples,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    sampler = None
    if args.balance_cfl:
        sampler = build_cfl_balanced_sampler(train_ds, parse_float_csv(args.cfl_edges), seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    # Derive channel counts from data
    sample0 = train_ds[0]
    in_channels  = sample0['x'].shape[0]   # 1 + 2M
    out_channels = sample0['y'].shape[0]   # M

    resume_ckpt = None
    start_epoch = 0
    history = []
    best_val = float('inf')

    model_cfg = {
        'name': 'fno',
        'in_channels': in_channels,
        'out_channels': out_channels,
        'width': args.width,
        'modes': args.modes,
        'depth': args.depth,
    }

    if args.resume_from:
        try:
            resume_ckpt = torch.load(args.resume_from, map_location='cpu', weights_only=False)
        except TypeError:
            resume_ckpt = torch.load(args.resume_from, map_location='cpu')
        if resume_ckpt.get('model_type', 'fno') != 'fno':
            raise ValueError(f'Resume checkpoint {args.resume_from} is not an FNO checkpoint')
        model_cfg = dict(resume_ckpt['model_config'])

    device = torch.device(args.device)
    model = FNO1d(
        in_channels=in_channels,
        out_channels=out_channels,
        width=int(model_cfg['width']),
        modes=int(model_cfg['modes']),
        depth=int(model_cfg['depth']),
    ).to(device)

    if resume_ckpt is not None:
        if int(model_cfg['in_channels']) != in_channels or int(model_cfg['out_channels']) != out_channels:
            raise ValueError(
                'Resume checkpoint channel configuration does not match current dataset '
                f'({model_cfg["in_channels"]}/{model_cfg["out_channels"]} vs {in_channels}/{out_channels})'
            )
        model.load_state_dict(resume_ckpt['model_state'])
        start_epoch = int(resume_ckpt.get('epoch', 0))
        history = list(resume_ckpt.get('history', [])) if not args.reset_history else []
        if history:
            best_val = min(float(item['selection_metric']) for item in history)
        elif 'val_metrics' in resume_ckpt:
            if args.checkpoint_metric == 'val_mse':
                best_val = float(resume_ckpt['val_metrics']['val_mse_phys'])
            else:
                best_val = float(resume_ckpt['val_metrics'].get('val_composite', float('inf')))
        print(f'Resuming from {args.resume_from} at epoch {start_epoch}')

    if start_epoch >= args.epochs:
        raise ValueError(f'--epochs ({args.epochs}) must be larger than resume epoch ({start_epoch})')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs - start_epoch, 1))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        'data_path':   args.data,
        'model_cfg':   model_cfg,
        'seed':        args.seed,
        'epochs':      args.epochs,
        'lr':          args.lr,
        'batch_size':  args.batch_size,
        'git_commit':  get_git_commit(),
        'train_size':  len(train_ds),
        'val_size':    len(val_ds),
        'balance_cfl': bool(args.balance_cfl),
        'cfl_edges': list(parse_float_csv(args.cfl_edges)),
        'residual_proj_weight': float(args.residual_proj_weight),
        'correction_norm_weight': float(args.correction_norm_weight),
        'cosine_gate_weight': float(args.cosine_gate_weight),
        'cosine_gate_margin': float(args.cosine_gate_margin),
        'checkpoint_metric': str(args.checkpoint_metric),
        'resume_from': str(args.resume_from) if args.resume_from else None,
        'start_epoch': int(start_epoch),
    }
    with open(out_dir / 'train_manifest.json', 'w') as fh:
        json.dump(manifest, fh, indent=2)

    y_std_t  = torch.from_numpy(y_norm.std).to(device)
    y_mean_t = torch.from_numpy(y_norm.mean).to(device)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        train_loss, count = 0.0, 0
        train_mse, train_proj, train_norm, train_gate, train_cos = 0.0, 0.0, 0.0, 0.0, 0.0

        for batch in train_loader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            r_k = batch['R_k'].to(device)

            pred_norm = model(x)
            mse_loss = ((pred_norm - y) ** 2).mean()

            pred_phys = pred_norm * y_std_t + y_mean_t
            y_phys = y * y_std_t + y_mean_t

            proj_pred = torch.sum(pred_phys * r_k, dim=-1)
            proj_true = torch.sum(y_phys * r_k, dim=-1)
            proj_denom = torch.mean(proj_true**2).detach() + 1e-12
            proj_loss = torch.mean((proj_pred - proj_true) ** 2) / proj_denom

            pred_norm_mag = torch.linalg.vector_norm(pred_phys.reshape(pred_phys.shape[0], pred_phys.shape[1], -1), dim=-1)
            true_norm_mag = torch.linalg.vector_norm(y_phys.reshape(y_phys.shape[0], y_phys.shape[1], -1), dim=-1)
            norm_denom = torch.mean(true_norm_mag**2).detach() + 1e-12
            norm_loss = torch.mean((pred_norm_mag - true_norm_mag) ** 2) / norm_denom

            cos = F.cosine_similarity(
                pred_phys.reshape(pred_phys.shape[0], pred_phys.shape[1], -1),
                y_phys.reshape(y_phys.shape[0], y_phys.shape[1], -1),
                dim=-1,
            )
            gate_loss = torch.relu(args.cosine_gate_margin - cos).mean()

            loss = (
                mse_loss
                + args.residual_proj_weight * proj_loss
                + args.correction_norm_weight * norm_loss
                + args.cosine_gate_weight * gate_loss
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * x.shape[0]
            train_mse += mse_loss.item() * x.shape[0]
            train_proj += proj_loss.item() * x.shape[0]
            train_norm += norm_loss.item() * x.shape[0]
            train_gate += gate_loss.item() * x.shape[0]
            train_cos += cos.mean().item() * x.shape[0]
            count += x.shape[0]

        scheduler.step()
        train_loss /= count
        train_mse /= count
        train_proj /= count
        train_norm /= count
        train_gate /= count
        train_cos /= count

        val_metrics = evaluate_fno(model, val_loader, y_norm, device, args.cosine_gate_margin)
        if args.checkpoint_metric == 'val_mse':
            selection_metric = val_metrics['val_mse_phys']
        else:
            selection_metric = val_metrics['val_composite']

        print(
            f'epoch={epoch:4d}  train_loss={train_loss:.3e}  train_mse={train_mse:.3e}'
            f'  train_proj={train_proj:.3e}  train_norm={train_norm:.3e}'
            f'  train_gate={train_gate:.3e}  train_cos={train_cos:.3f}'
            f'  val_mse_phys={val_metrics["val_mse_phys"]:.3e}  val_proj={val_metrics["val_proj_mse"]:.3e}'
            f'  val_norm={val_metrics["val_norm_mse"]:.3e}'
            f'  val_gate={val_metrics["val_gate_loss"]:.3e}  val_cos={val_metrics["val_cosine"]:.3f}'
        )

        history.append(
            {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_mse': train_mse,
                'train_proj_loss': train_proj,
                'train_norm_loss': train_norm,
                'train_gate_loss': train_gate,
                'train_cosine': train_cos,
                **val_metrics,
                'selection_metric': float(selection_metric),
            }
        )

        checkpoint = {
            'model_state':   model.state_dict(),
            'model_config':  model_cfg,
            'x_normalizer':  x_norm.state_dict(),
            'y_normalizer':  y_norm.state_dict(),
            'epoch':         epoch,
            'val_loss':      val_metrics['val_mse_phys'],
            'val_metrics':   val_metrics,
            'manifest':      manifest,
            'model_type':    'fno',
            'in_channels':   in_channels,
            'out_channels':  out_channels,
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'history':       history,
        }
        torch.save(checkpoint, out_dir / 'last.pt')
        if selection_metric < best_val:
            best_val = selection_metric
            torch.save(checkpoint, out_dir / 'best.pt')

    print(f'Best checkpoint metric ({args.checkpoint_metric}): {best_val:.3e}')


def main():
    parser = argparse.ArgumentParser(description='Train FNO sweep-correction model.')
    parser.add_argument('--data', type=str,
                        default='pySDC/playgrounds/learned_qdelta/data/heat1d_agnostic_1k.npz')
    parser.add_argument('--output-dir', type=str,
                        default='pySDC/playgrounds/learned_qdelta/checkpoints/heat1d_fno_v1')
    parser.add_argument('--width',  type=int, default=64)
    parser.add_argument('--modes',  type=int, default=16)
    parser.add_argument('--depth',  type=int, default=4)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr',     type=float, default=1e-3)
    parser.add_argument('--val-fraction', type=float, default=0.2)
    parser.add_argument('--seed',   type=int, default=7)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--balance-cfl', action='store_true',
                        help='Use weighted sampling to balance diffusion-CFL regimes')
    parser.add_argument('--cfl-edges', type=str, default='0.2,1.0,4.0',
                        help='Comma-separated diffusion-CFL regime edges')
    parser.add_argument('--residual-proj-weight', type=float, default=0.2,
                        help='Weight for projection loss along residual directions')
    parser.add_argument('--correction-norm-weight', type=float, default=0.1,
                        help='Weight for correction-norm loss to reduce oversized/undersized proposals')
    parser.add_argument('--cosine-gate-weight', type=float, default=0.05,
                        help='Weight for cosine-margin gate surrogate penalty')
    parser.add_argument('--cosine-gate-margin', type=float, default=0.2,
                        help='Cosine similarity margin used in gate surrogate loss')
    parser.add_argument('--checkpoint-metric', type=str, choices=['val_mse', 'composite'], default='composite',
                        help='Metric used to select best.pt')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Resume model weights/history from an existing FNO checkpoint and continue to a larger epoch count')
    parser.add_argument('--reset-history', action='store_true',
                        help='Ignore stored history when resuming and track only the new fine-tuning segment')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    train(args)


if __name__ == '__main__':
    main()


