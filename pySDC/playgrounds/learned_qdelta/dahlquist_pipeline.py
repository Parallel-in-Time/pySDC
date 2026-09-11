"""End-to-end Dahlquist pipeline: generate data, train model, run matrix benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from pySDC.playgrounds.learned_qdelta.dahlquist_benchmark import run_matrix
from pySDC.playgrounds.learned_qdelta.data_generation import generate_dataset
from pySDC.playgrounds.learned_qdelta.train import train


def run_pipeline(config_path: str, output_root: str):
    with open(config_path, 'r') as fobj:
        cfg = json.load(fobj)

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    data_path = root / 'dahlquist_sweeps.npz'
    ckpt_dir = root / 'checkpoints'
    benchmark_path = root / 'dahlquist_matrix.json'

    data_cfg = cfg['data']
    generate_dataset(output_path=str(data_path), **data_cfg)

    train_cfg = cfg['train'].copy()
    train_cfg['data'] = str(data_path)
    train_cfg['output_dir'] = str(ckpt_dir)
    train(SimpleNamespace(**train_cfg))

    bm_cfg = cfg['benchmark'].copy()
    bm_cfg['checkpoint'] = str(ckpt_dir / 'best.pt')
    bm_cfg['output'] = str(benchmark_path)
    run_matrix(SimpleNamespace(**bm_cfg))

    print(f'Pipeline done. Results in {root}')


def main():
    parser = argparse.ArgumentParser(description='Run the Dahlquist learned-Qdelta pipeline from a JSON config.')
    parser.add_argument(
        '--config',
        type=str,
        default='pySDC/playgrounds/learned_qdelta/config_dahlquist_baseline.json',
    )
    parser.add_argument('--output-root', type=str, default='pySDC/playgrounds/learned_qdelta/results/pipeline_run')
    args = parser.parse_args()

    run_pipeline(config_path=args.config, output_root=args.output_root)


if __name__ == '__main__':
    main()

