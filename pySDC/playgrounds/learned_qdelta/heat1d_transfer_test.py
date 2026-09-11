"""Apply trained Dahlquist model to 1D heat equation (transfer test).

This script takes a model trained on scalar Dahlquist (y' = λy) and applies it
to a completely different problem: 1D heat equation with spatial coupling.

Expected behavior:
  - Acceptance rate drops (OOD problem)
  - Some proposals might be bad (learned structure doesn't match)
  - Fallback gate should reject worst proposals
  - Result: neutral or slight degradation, but no catastrophic failure

This validates the safety-critical design.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.playgrounds.learned_qdelta.hooks import LearnedQDeltaHook
from pySDC.playgrounds.learned_qdelta.learned_sweeper import LearnedQDeltaSweeper


def run_heat1d(sweeper_class, sweeper_params, problem_name: str, description: str):
    """Run 1D heat equation with given sweeper."""
    level_params = {'dt': 0.01, 'restol': 1e-10}
    step_params = {'maxiter': 4}
    problem_params = {
        'nvars': 127,
        'nu': 0.1,
        'freq': 2,
        'bc': 'dirichlet-zero',
    }

    description_dict = {
        'problem_class': heatNd_unforced,
        'problem_params': problem_params,
        'sweeper_class': sweeper_class,
        'sweeper_params': sweeper_params,
        'level_params': level_params,
        'step_params': step_params,
    }

    controller_params = {'logger_level': 30, 'hook_class': [LearnedQDeltaHook]}
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description_dict)

    P = controller.MS[0].levels[0].prob
    u0 = getattr(P, 'u_exact')(0.0)

    t0 = time.perf_counter()
    _, stats = controller.run(u0=u0, t0=0.0, Tend=0.05)
    runtime = time.perf_counter() - t0

    # Extract metrics
    from pySDC.helpers.stats_helper import get_sorted
    residuals = [v for _, v in get_sorted(stats, type='residual_post_sweep', sortby='time')]
    niters = [v for _, v in get_sorted(stats, type='niter', sortby='time')]
    accepts = [v for _, v in get_sorted(stats, type='learned_accept', sortby='time')]

    result = {
        'sweeper': problem_name,
        'description': description,
        'runtime': runtime,
        'avg_residual_ratio': float(np.mean(np.array(residuals[1:]) / np.array(residuals[:-1]))) if len(residuals) > 1 else np.nan,
        'avg_niter': float(np.mean(niters)) if niters else np.nan,
        'acceptance_rate': float(np.mean(accepts)) if accepts else np.nan,
        'num_steps': len(niters),
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Transfer test: apply Dahlquist-trained model to 1D heat equation.'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained Dahlquist model (best.pt)')
    parser.add_argument('--output', type=str, default='heat1d_transfer_test.json',
                        help='Output JSON with results')
    parser.add_argument('--accept-factor', type=float, default=1.0)
    args = parser.parse_args()

    print("=" * 80)
    print("TRANSFER TEST: Dahlquist Model → 1D Heat Equation")
    print("=" * 80)
    print()
    print(f"Model checkpoint: {args.checkpoint}")
    print(f"Target problem:  1D heat equation (nvars=127, nu=0.1)")
    print(f"Accept factor:   {args.accept_factor}")
    print()

    # Classical baseline
    print("Running CLASSICAL sweeper on heat1d...")
    classical_params = {'quad_type': 'RADAU-RIGHT', 'num_nodes': 3, 'QI': 'IE'}
    baseline = run_heat1d(generic_implicit, classical_params, 'classical', 'Baseline (no model)')
    print(f"  Runtime: {baseline['runtime']:.2f}s")
    print(f"  Avg niter: {baseline['avg_niter']:.2f}")
    print(f"  Avg residual ratio: {baseline['avg_residual_ratio']:.4f}")
    print()

    # Learned with transfer
    print("Running LEARNED sweeper (Dahlquist model, OOD transfer) on heat1d...")
    learned_params = {
        'quad_type': 'RADAU-RIGHT',
        'num_nodes': 3,
        'QI': 'IE',
        'model_checkpoint': args.checkpoint,
        'accept_factor': args.accept_factor,
        'model_device': 'cpu',
        'fallback_sweeper_class': generic_implicit,
    }
    learned = run_heat1d(LearnedQDeltaSweeper, learned_params, 'learned', 'Trained Dahlquist model (OOD transfer)')
    print(f"  Runtime: {learned['runtime']:.2f}s")
    print(f"  Avg niter: {learned['avg_niter']:.2f}")
    print(f"  Avg residual ratio: {learned['avg_residual_ratio']:.4f}")
    print(f"  Acceptance rate: {learned['acceptance_rate']:.3f}")
    print()

    # Comparison
    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print()
    print(f"{'Metric':<25} {'Classical':<15} {'Learned (OOD)':<15} {'Delta':<15}")
    print("-" * 70)

    niter_delta = learned['avg_niter'] - baseline['avg_niter']
    res_ratio_delta = learned['avg_residual_ratio'] - baseline['avg_residual_ratio']
    runtime_delta = learned['runtime'] - baseline['runtime']

    print(f"{'Avg iterations':<25} {baseline['avg_niter']:<15.2f} {learned['avg_niter']:<15.2f} {niter_delta:+.2f}")
    print(f"{'Avg residual ratio':<25} {baseline['avg_residual_ratio']:<15.4f} {learned['avg_residual_ratio']:<15.4f} {res_ratio_delta:+.4f}")
    print(f"{'Runtime (s)':<25} {baseline['runtime']:<15.2f} {learned['runtime']:<15.2f} {runtime_delta:+.2f}")
    print(f"{'Acceptance rate':<25} {'N/A':<15} {learned['acceptance_rate']:<15.3f}")
    print()

    # Interpretation
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print()

    if learned['acceptance_rate'] < 0.3:
        print(f"⚠️  Low acceptance ({learned['acceptance_rate']:.1%})")
        print("   → Model proposals mostly rejected (OOD problem structure)")
        print("   → This is EXPECTED for transfer to coupled system")
        print("   → Fallback gate is working (protecting against bad proposals)")
    elif learned['acceptance_rate'] > 0.5:
        print(f"✓ Moderate acceptance ({learned['acceptance_rate']:.1%})")
        print("   → Model generalizes somewhat to coupled problem")
        print("   → Learned structure has some universal appeal")

    if abs(niter_delta) < 0.1:
        print()
        print(f"✓ No iteration degradation (Δ={niter_delta:+.2f})")
        print("   → Fallback gate is effective")
        print("   → Even OOD proposals don't hurt classical performance")
    elif niter_delta < 0:
        print()
        print(f"✓ Learned improves iterations (Δ={niter_delta:+.2f})")
        print("   → Model found universal patterns across problem classes!")
    else:
        print()
        print(f"✗ Learned slightly worse (Δ={niter_delta:+.2f})")
        print("   → Some proposals degrade convergence")
        print("   → Accept threshold may be too permissive")

    if abs(res_ratio_delta) > 0.1:
        print()
        print(f"⚠️  Residual ratio difference: {res_ratio_delta:+.4f}")
        if res_ratio_delta > 0:
            print("   → Learned does NOT reduce residuals as well as classical")
        else:
            print("   → Learned reduces residuals better than classical (rare!)")

    print()

    # Save results
    results = {
        'test_name': 'Dahlquist model transfer to 1D heat',
        'model_checkpoint': args.checkpoint,
        'problem': '1D heat equation (nvars=127, nu=0.1, dt=0.01)',
        'baseline': baseline,
        'learned': learned,
        'comparison': {
            'niter_delta': niter_delta,
            'residual_ratio_delta': res_ratio_delta,
            'runtime_delta': runtime_delta,
            'acceptance_rate': learned['acceptance_rate'],
        },
        'notes': [
            'Model trained on scalar Dahlquist (y\' = λy)',
            'Applied to 1D heat PDE (127 coupled spatial modes)',
            'No retraining - purely out-of-distribution transfer test',
            f'Accept factor: {args.accept_factor}',
        ],
    }

    import pathlib
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {out_path}")


if __name__ == '__main__':
    main()

