"""Run a Dahlquist matrix benchmark for classical vs learned sweeper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.playgrounds.learned_qdelta.evaluate import run_setup
from pySDC.playgrounds.learned_qdelta.learned_sweeper import LearnedQDeltaSweeper


def parse_float_list(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(',') if part.strip()]


def regime_label(lamdt: float, edges: list[float]) -> int:
    return int(np.digitize([lamdt], np.asarray(edges, dtype=np.float64), right=False)[0])


def aggregate_by_regime(rows: list[dict], edges: list[float]) -> dict:
    out = {}
    for row in rows:
        label = regime_label(abs(row['lam'] * row['dt']), edges)
        out.setdefault(label, []).append(row)

    summary = {}
    for label, items in out.items():
        summary[str(label)] = {
            'count': len(items),
            'baseline_avg_niter': float(np.mean([me['baseline_avg_niter'] for me in items])),
            'learned_avg_niter': float(np.mean([me['learned_avg_niter'] for me in items])),
            'baseline_avg_runtime': float(np.mean([me['baseline_runtime'] for me in items])),
            'learned_avg_runtime': float(np.mean([me['learned_runtime'] for me in items])),
            'learned_acceptance_rate': float(np.nanmean([me['learned_acceptance_rate'] for me in items])),
        }
    return summary


def run_matrix(args):
    # Support --zvals as an alias: fixes dt=1, uses z as lambda
    if args.zvals:
        lambdas = parse_float_list(args.zvals)
        dts = [1.0]
        use_z_sweep = True
    else:
        lambdas = parse_float_list(args.lambdas)
        dts = parse_float_list(args.dts)
        use_z_sweep = False
    edges = parse_float_list(args.regime_edges)

    classical_params = {'quad_type': 'RADAU-RIGHT', 'num_nodes': args.num_nodes, 'QI': 'IE'}
    learned_params = {
        'quad_type': 'RADAU-RIGHT',
        'num_nodes': args.num_nodes,
        'QI': 'IE',
        'model_checkpoint': args.checkpoint,
        'accept_factor': args.accept_factor,
        'model_device': args.device,
        'fallback_sweeper_class': generic_implicit,
    }

    rows = []
    for lam in lambdas:
        for dt in dts:
            baseline = run_setup(
                generic_implicit,
                classical_params,
                dt,
                args.Tend,
                args.maxiter,
                problem='dahlquist',
                lam=lam,
                nu=0.0,
            )
            learned = run_setup(
                LearnedQDeltaSweeper,
                learned_params,
                dt,
                args.Tend,
                args.maxiter,
                problem='dahlquist',
                lam=lam,
                nu=0.0,
            )

            rows.append(
                {
                    'lam': lam,
                    'dt': dt,
                    'lamdt_abs': abs(lam * dt),
                    'baseline_avg_niter': baseline['avg_niter'],
                    'learned_avg_niter': learned['avg_niter'],
                    'baseline_runtime': baseline['runtime'],
                    'learned_runtime': learned['runtime'],
                    'baseline_avg_residual_ratio': baseline['avg_residual_ratio'],
                    'learned_avg_residual_ratio': learned['avg_residual_ratio'],
                    'learned_acceptance_rate': learned['acceptance_rate'],
                }
            )

    out = {
        'meta': {
            'lambdas': lambdas,
            'dts': dts,
            'regime_edges': edges,
            'Tend': args.Tend,
            'maxiter': args.maxiter,
            'num_nodes': args.num_nodes,
            'accept_factor': args.accept_factor,
            'checkpoint': args.checkpoint,
            'use_z_sweep': use_z_sweep,
        },
        'rows': rows,
        'regime_summary': aggregate_by_regime(rows, edges),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as fobj:
        json.dump(out, fobj, indent=2)

    print(f'Wrote benchmark report to {out_path}')
    for label, info in out['regime_summary'].items():
        count = int(info.get('count', 0))
        acc = float(info.get('learned_acceptance_rate', np.nan))
        print(f'regime={label}: n={count} acc={acc:.3f}')


def main():
    parser = argparse.ArgumentParser(description='Dahlquist benchmark matrix for learned Q_delta prototype.')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='pySDC/playgrounds/learned_qdelta/results/dahlquist_matrix.json')
    parser.add_argument('--lambdas', type=str, default='-2,-5,-10,-20')
    parser.add_argument('--dts', type=str, default='0.02,0.05,0.1,0.2')
    parser.add_argument('--zvals', type=str, default='',
                        help='Comma-separated z=lambda*dt values to sweep (fixes dt=1). '
                             'Overrides --lambdas/--dts. Use =... syntax for negatives.')
    parser.add_argument('--regime-edges', type=str, default='1.0,5.0,15.0')
    parser.add_argument('--Tend', type=float, default=1.0)
    parser.add_argument('--maxiter', type=int, default=6)
    parser.add_argument('--num-nodes', type=int, default=3)
    parser.add_argument('--accept-factor', type=float, default=0.95)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    run_matrix(args)


if __name__ == '__main__':
    main()

