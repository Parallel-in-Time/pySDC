from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


# Tensor layout convention used in this prototype:
# - u0:         (N, state_dim)
# - U_k:        (N, M, state_dim)
# - R_k:        (N, M, state_dim)
# - target_dU:  (N, M, state_dim)
# - dt:         (N, 1)
# - nodes:      (N, M)
# - qmat:       (N, M, M)
# - problem_params: (N, p)

DATA_CONTRACT_VERSION = 1
REQUIRED_SAMPLE_KEYS = ('u0', 'U_k', 'R_k', 'dt', 'nodes', 'qmat', 'problem_params', 'target_dU')


def load_npz_samples(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        samples = {k: data[k] for k in data.files}

    missing = [k for k in REQUIRED_SAMPLE_KEYS if k not in samples]
    if missing:
        raise KeyError(f'Dataset is missing required keys: {missing}')

    if 'contract_version' in samples:
        version = int(samples['contract_version'].reshape(-1)[0])
        if version != DATA_CONTRACT_VERSION:
            raise ValueError(
                f'Unsupported contract version {version}, expected {DATA_CONTRACT_VERSION}. '
                'Regenerate data or update the loader.'
            )

    return samples


def dahlquist_lambda_dt(samples: dict[str, np.ndarray]) -> np.ndarray:
    """Compute |lambda * dt| for scalar Dahlquist samples."""
    lam = np.asarray(samples['problem_params'][:, 0]).reshape(-1)
    dt = np.asarray(samples['dt'][:, 0]).reshape(-1)
    return np.abs(lam * dt)


def dahlquist_regime_labels(lambda_dt: np.ndarray, regime_edges: tuple[float, ...]) -> np.ndarray:
    """Map |lambda*dt| to discrete regime labels using edge thresholds."""
    return np.digitize(lambda_dt, np.asarray(regime_edges, dtype=np.float64), right=False)


def build_feature_vector_dimension_agnostic_legacy(sample: dict[str, np.ndarray]) -> np.ndarray:
    """Legacy 12D dimension-agnostic feature vector (kept for compatibility)."""
    u0 = sample['u0'].reshape(-1)
    U_k = sample['U_k'].reshape(-1)
    R_k = sample['R_k'].reshape(-1)
    qmat = sample['qmat'].reshape(-1)
    dt = float(sample['dt'].reshape(-1)[0])
    problem_params = sample['problem_params'].reshape(-1)

    features = [
        np.mean(u0),
        np.std(u0),
        np.linalg.norm(u0),
        np.mean(U_k),
        np.std(U_k),
        np.linalg.norm(U_k),
        np.mean(R_k),
        np.linalg.norm(R_k),
        np.max(np.abs(R_k)),
        np.mean(np.abs(qmat)),
        dt,
        np.mean(problem_params),
    ]

    return np.array(features, dtype=np.float32)


def build_feature_vector_dimension_agnostic_per_node(sample: dict[str, np.ndarray]) -> np.ndarray:
    """Build dimension-agnostic feature vector (invariant to nvars).

    Instead of flattening the full state (which scales with problem dimension),
    use scalar summaries of state, collocation, and residuals.

    Features per-node (for M nodes, 5D each):
      - U_k[m]:  mean, std, norm, min, max
      - R_k[m]:  mean, norm, max_abs, sign_ratio, std
    Global features (7D):
      - u0: mean, std, norm
      - qmat: mean abs, max abs
      - dt, avg problem_params

    Total: M*10 + 7 features (for M=3: 37D)
    Works for ANY spatial dimension (127D, 254D, 2D, 3D, etc.)
    """
    u0 = sample['u0'].reshape(-1)
    U_k = np.asarray(sample['U_k'])   # (M, state_dim)
    R_k = np.asarray(sample['R_k'])   # (M, state_dim)
    qmat = sample['qmat'].reshape(-1)
    dt = float(sample['dt'].reshape(-1)[0])
    problem_params = sample['problem_params'].reshape(-1)

    M = U_k.shape[0]
    per_node = []
    for m in range(M):
        u = U_k[m].reshape(-1)
        r = R_k[m].reshape(-1)
        per_node.extend([
            # Per-node state features
            float(np.mean(u)),
            float(np.std(u)),
            float(np.linalg.norm(u)),
            float(np.min(u)),
            float(np.max(u)),
            # Per-node residual features
            float(np.mean(r)),
            float(np.linalg.norm(r)),
            float(np.max(np.abs(r))),
            float(np.std(r)),
            float(np.sum(r > 0) / (len(r) + 1e-30) - 0.5),  # sign ratio centered
        ])

    global_features = [
        # Initial state
        float(np.mean(u0)),
        float(np.std(u0)),
        float(np.linalg.norm(u0)),
        # Collocation matrix
        float(np.mean(np.abs(qmat))),
        float(np.max(np.abs(qmat))),
        # Problem
        dt,
        float(np.mean(problem_params)),
    ]

    return np.array(per_node + global_features, dtype=np.float32)


def build_feature_vector(sample: dict[str, np.ndarray], use_z_param: bool = False,
                         dimension_agnostic: bool = False,
                         agnostic_feature_variant: str = 'legacy12') -> np.ndarray:
    """Build a flat feature vector from one sample dict.

    Args:
        sample: Dictionary with required sample keys
        use_z_param: If True, omit dt (absorbed into z) and append log|z|
        dimension_agnostic: If True, use scalar summaries (works for any nvars)
                           If False, use full flattened state (current behavior)

    When *use_z_param* is True the step size ``dt`` is omitted from the
    feature vector because it is already absorbed into the ``z = lambda*dt``
    entry that is stored in ``problem_params``. Additionally, log|z| is
    appended as an extra feature to help the MLP distinguish stiffness regimes.

    When *dimension_agnostic* is True, use scalar summaries that are invariant
    to problem dimension (spatial grid size). Allows training once, using for
    any grid refinement. Performance: ~70-90% of full-state model.
    """
    if dimension_agnostic:
        if agnostic_feature_variant in ('legacy12', 'v1'):
            return build_feature_vector_dimension_agnostic_legacy(sample)
        if agnostic_feature_variant in ('per_node', 'per_node37', 'v2'):
            return build_feature_vector_dimension_agnostic_per_node(sample)
        raise ValueError(f'Unknown agnostic_feature_variant={agnostic_feature_variant}')

    # Original full-state behavior (dimension-dependent)
    parts = [
        sample['u0'].reshape(-1),
        sample['U_k'].reshape(-1),
        sample['R_k'].reshape(-1),
    ]
    if not use_z_param:
        parts.append(sample['dt'].reshape(-1))
    parts.extend([
        sample['nodes'].reshape(-1),
        sample['problem_params'].reshape(-1),
    ])

    # For z-param mode, append log|z| as a separate feature
    if use_z_param:
        pp = sample['problem_params'].reshape(-1)
        z = float(pp[0])
        log_abs_z = np.log(np.abs(z) + 1e-30)
        parts.append(np.array([log_abs_z], dtype=np.float64))

    return np.concatenate(parts, axis=0).astype(np.float32)


def build_target_vector(sample: dict[str, np.ndarray],
                        dimension_agnostic: bool = False,
                        agnostic_target_variant: str = 'norm_ratio') -> np.ndarray:
    """Build target vector from sample.

    In dimension-agnostic mode, returns one scalar per collocation node.

    Variants:
      - ``norm_ratio`` (default, legacy):
          ||target_dU[m]|| / (||R_k[m]|| + eps)
      - ``signed_projection``:
          <target_dU[m], R_k[m]> / (||R_k[m]||^2 + eps)

    At inference, the full correction is reconstructed as
    ΔU[m] = pred[m] * R_k[m].
    """
    if dimension_agnostic:
        dU = sample['target_dU']   # (M, state_dim)
        R_k = sample['R_k']        # (M, state_dim)
        eps = 1e-14
        dU_flat = dU.reshape(dU.shape[0], -1)
        R_flat = R_k.reshape(R_k.shape[0], -1)

        if agnostic_target_variant in ('norm_ratio', 'v1'):
            dU_norms = np.linalg.norm(dU_flat, axis=1)
            R_norms = np.linalg.norm(R_flat, axis=1)
            scales = dU_norms / (R_norms + eps)
        elif agnostic_target_variant in ('signed_projection', 'v2'):
            numer = np.sum(dU_flat * R_flat, axis=1)
            denom = np.sum(R_flat * R_flat, axis=1)
            scales = numer / (denom + eps)
        else:
            raise ValueError(f'Unknown agnostic_target_variant={agnostic_target_variant}')

        return scales.astype(np.float32)
    return sample['target_dU'].reshape(-1).astype(np.float32)


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, data: np.ndarray, eps: float = 1e-8) -> 'Standardizer':
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        std[std < eps] = 1.0
        return cls(mean=mean, std=std)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverse(self, data: np.ndarray) -> np.ndarray:
        return data * self.std + self.mean

    def state_dict(self) -> dict[str, np.ndarray]:
        return {'mean': self.mean, 'std': self.std}

    @classmethod
    def from_state_dict(cls, state: dict[str, np.ndarray]) -> 'Standardizer':
        return cls(mean=state['mean'], std=state['std'])


class SweepCorrectionDataset(Dataset):
    """PyTorch dataset for one-sweep correction learning."""

    def __init__(
        self,
        samples: dict[str, np.ndarray],
        indices: np.ndarray,
        x_normalizer: Standardizer,
        y_normalizer: Standardizer,
        use_z_param: bool = False,
        dimension_agnostic: bool = False,
        agnostic_target_variant: str = 'norm_ratio',
    ):
        self.samples = {k: samples[k][indices] for k in REQUIRED_SAMPLE_KEYS}
        self.use_z_param = use_z_param
        self.dimension_agnostic = dimension_agnostic
        self.agnostic_target_variant = agnostic_target_variant

        x_raw = np.stack(
            [
                build_feature_vector({k: self.samples[k][i] for k in self.samples.keys()}, use_z_param=use_z_param, dimension_agnostic=dimension_agnostic)
                for i in range(len(indices))
            ],
            axis=0,
        )
        y_raw = np.stack(
            [
                build_target_vector(
                    {k: self.samples[k][i] for k in self.samples.keys()},
                    dimension_agnostic=dimension_agnostic,
                    agnostic_target_variant=agnostic_target_variant,
                )
                for i in range(len(indices))
            ],
            axis=0,
        )

        self.x = x_normalizer.transform(x_raw).astype(np.float32)
        self.y = y_normalizer.transform(y_raw).astype(np.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {
            'x': torch.from_numpy(self.x[idx]),
            'y': torch.from_numpy(self.y[idx]),
            'u0': torch.from_numpy(self.samples['u0'][idx]).float(),
            'U_k': torch.from_numpy(self.samples['U_k'][idx]).float(),
            'dt': torch.from_numpy(self.samples['dt'][idx]).float(),
            'qmat': torch.from_numpy(self.samples['qmat'][idx]).float(),
            'problem_params': torch.from_numpy(self.samples['problem_params'][idx]).float(),
        }
        return item


def make_train_val_datasets(
    samples: dict[str, np.ndarray],
    val_fraction: float = 0.2,
    seed: int = 7,
    split_strategy: str = 'random',
    regime_edges: tuple[float, ...] = (1.0, 5.0, 15.0),
    holdout_regime: int | str | None = None,
    holdout_z_interval: tuple[float, float] | None = None,
    use_z_param: bool = False,
    dimension_agnostic: bool = False,
    agnostic_target_variant: str = 'norm_ratio',
) -> tuple[SweepCorrectionDataset, SweepCorrectionDataset, Standardizer, Standardizer]:
    n = samples['u0'].shape[0]
    rng = np.random.default_rng(seed)
    indices = np.arange(n)

    if split_strategy == 'random':
        rng.shuffle(indices)
        n_val = max(1, int(val_fraction * n))
        val_idx = np.sort(indices[:n_val])
        train_idx = np.sort(indices[n_val:])

    elif split_strategy == 'dahlquist_regime':
        lamdt = dahlquist_lambda_dt(samples)
        labels = dahlquist_regime_labels(lamdt, regime_edges=regime_edges)

        if holdout_z_interval is not None:
            # Hold out a specific z interval (e.g., [-6, -4])
            z_vals = lamdt
            z_min, z_max = holdout_z_interval
            val_mask = (z_vals >= z_min) & (z_vals <= z_max)
            val_idx = np.sort(indices[val_mask])
            train_idx = np.sort(indices[~val_mask])
        elif holdout_regime is not None:
            if isinstance(holdout_regime, str):
                if holdout_regime.lower() == 'stiff':
                    holdout = int(labels.max())
                else:
                    holdout = int(holdout_regime)
            else:
                holdout = int(holdout_regime)

            val_mask = labels == holdout
            val_idx = np.sort(indices[val_mask])
            train_idx = np.sort(indices[~val_mask])
        else:
            val_parts = []
            train_parts = []
            for label in np.unique(labels):
                idx = indices[labels == label]
                rng.shuffle(idx)
                n_val = max(1, int(val_fraction * idx.size)) if idx.size > 1 else idx.size
                val_parts.append(idx[:n_val])
                train_parts.append(idx[n_val:])
            val_idx = np.sort(np.concatenate(val_parts)) if val_parts else np.array([], dtype=int)
            train_idx = np.sort(np.concatenate(train_parts)) if train_parts else np.array([], dtype=int)
    else:
        raise ValueError(f'Unknown split_strategy={split_strategy}')

    if train_idx.size == 0:
        raise ValueError('Need at least one training sample after split.')
    if val_idx.size == 0:
        raise ValueError('Need at least one validation sample after split.')

    train_x_raw = np.stack(
        [build_feature_vector({k: samples[k][i] for k in REQUIRED_SAMPLE_KEYS}, use_z_param=use_z_param, dimension_agnostic=dimension_agnostic) for i in train_idx],
        axis=0,
    )
    train_y_raw = np.stack(
        [
            build_target_vector(
                {k: samples[k][i] for k in REQUIRED_SAMPLE_KEYS},
                dimension_agnostic=dimension_agnostic,
                agnostic_target_variant=agnostic_target_variant,
            )
            for i in train_idx
        ],
        axis=0,
    )

    x_norm = Standardizer.fit(train_x_raw)
    y_norm = Standardizer.fit(train_y_raw)

    train_ds = SweepCorrectionDataset(
        samples,
        train_idx,
        x_norm,
        y_norm,
        use_z_param=use_z_param,
        dimension_agnostic=dimension_agnostic,
        agnostic_target_variant=agnostic_target_variant,
    )
    val_ds = SweepCorrectionDataset(
        samples,
        val_idx,
        x_norm,
        y_norm,
        use_z_param=use_z_param,
        dimension_agnostic=dimension_agnostic,
        agnostic_target_variant=agnostic_target_variant,
    )
    return train_ds, val_ds, x_norm, y_norm


# ===========================================================================
# FNO dataset — spatial-field tensors instead of flat vectors
# ===========================================================================

class FNOSweepDataset(Dataset):
    """Dataset that returns spatial-field tensors suitable for FNO training.

    Each sample produces:
      x: (in_channels, N)  — stacked input fields
         channel layout: [u0, U_k[0..M-1], R_k[0..M-1]]   → 1+M+M channels
      y: (M, N)            — correction fields ΔU[0..M-1]

    Optionally a 1-D normalisation (per-field mean/std computed over the
    training split) is applied along the spatial axis via ``x_field_norm``
    and ``y_field_norm`` (both are ``Standardizer`` objects operating on
    the channel-flattened representation, or ``None`` to skip).

    This dataset is resolution-invariant: the FNO accepts any spatial size
    N because all spatial operations are global FFT + pointwise conv.
    """

    def __init__(
        self,
        samples: dict[str, np.ndarray],
        indices: np.ndarray,
        x_channel_norm: 'ChannelNormalizer | None' = None,
        y_channel_norm: 'ChannelNormalizer | None' = None,
    ):
        self.samples = {k: samples[k][indices] for k in REQUIRED_SAMPLE_KEYS}
        self.x_channel_norm = x_channel_norm
        self.y_channel_norm = y_channel_norm
        self._indices = indices

    def __len__(self):
        return self._indices.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        u0 = self.samples['u0'][idx]       # (state_dim,)
        U_k = self.samples['U_k'][idx]     # (M, state_dim)
        R_k = self.samples['R_k'][idx]     # (M, state_dim)
        dU  = self.samples['target_dU'][idx]  # (M, state_dim)
        dt  = float(self.samples['dt'][idx, 0])
        pp  = self.samples['problem_params'][idx]  # (p,)

        # Build (in_channels, N) input tensor
        # channels: u0, U_k[0], …, U_k[M-1], R_k[0], …, R_k[M-1]
        x_fields = np.concatenate(
            [u0[None, :], U_k, R_k], axis=0
        ).astype(np.float32)   # (1+2M, N)

        y_fields = dU.astype(np.float32)   # (M, N)

        if self.x_channel_norm is not None:
            x_fields = self.x_channel_norm.transform(x_fields)
        if self.y_channel_norm is not None:
            y_fields = self.y_channel_norm.transform(y_fields)

        return {
            'x': torch.from_numpy(x_fields),
            'y': torch.from_numpy(y_fields),
            'R_k': torch.from_numpy(R_k.astype(np.float32)),
            'target_dU': torch.from_numpy(dU.astype(np.float32)),
            'dt': torch.tensor([dt], dtype=torch.float32),
            'nvars': torch.tensor([u0.shape[0]], dtype=torch.float32),
            'problem_params': torch.from_numpy(pp.astype(np.float32)),
        }


@dataclass
class ChannelNormalizer:
    """Per-channel mean/std normalisation for (C, N) spatial-field tensors.

    ``mean`` and ``std`` have shape (C, 1) so broadcasting works over N.
    """
    mean: np.ndarray   # (C, 1)
    std:  np.ndarray   # (C, 1)

    @classmethod
    def fit(cls, fields: np.ndarray, eps: float = 1e-8) -> 'ChannelNormalizer':
        """Fit on a batch of fields with shape (num_samples, C, N)."""
        flat = fields.reshape(fields.shape[0], fields.shape[1], -1)  # (S, C, N)
        mean = flat.mean(axis=(0, 2), keepdims=False)[:, None]        # (C, 1)
        std  = flat.std(axis=(0, 2), keepdims=False)[:, None]         # (C, 1)
        std[std < eps] = 1.0
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, fields: np.ndarray) -> np.ndarray:
        """fields: (C, N)"""
        return (fields - self.mean) / self.std

    def inverse(self, fields: np.ndarray) -> np.ndarray:
        """fields: (C, N)"""
        return fields * self.std + self.mean

    def state_dict(self) -> dict[str, np.ndarray]:
        return {'mean': self.mean, 'std': self.std}

    @classmethod
    def from_state_dict(cls, state: dict) -> 'ChannelNormalizer':
        return cls(mean=np.asarray(state['mean']), std=np.asarray(state['std']))


def make_fno_datasets(
    samples: dict[str, np.ndarray],
    val_fraction: float = 0.2,
    seed: int = 7,
) -> tuple['FNOSweepDataset', 'FNOSweepDataset', ChannelNormalizer, ChannelNormalizer]:
    """Build train/val FNOSweepDatasets with per-channel spatial normalisation.

    Returns (train_ds, val_ds, x_norm, y_norm).
    The normalizers are grid-size agnostic (mean/std over channel × all spatial
    points in the training set) so the same normalizer can be applied to any
    spatial resolution at inference time.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f'val_fraction must lie in (0, 1), got {val_fraction}')

    n = samples['u0'].shape[0]
    if n < 2:
        raise ValueError('Need at least two samples to build train/val FNO datasets.')

    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    n_val = max(1, int(val_fraction * n))
    val_idx = np.sort(indices[:n_val])
    train_idx = np.sort(indices[n_val:])

    if train_idx.size == 0:
        raise ValueError('Need at least one training sample after FNO split.')
    if val_idx.size == 0:
        raise ValueError('Need at least one validation sample after FNO split.')

    # Collect training fields for fitting normalizers
    u0_tr = samples['u0'][train_idx]        # (S, N)
    U_k_tr = samples['U_k'][train_idx]      # (S, M, N)
    R_k_tr = samples['R_k'][train_idx]      # (S, M, N)
    dU_tr  = samples['target_dU'][train_idx] # (S, M, N)

    x_stack = np.concatenate(
        [u0_tr[:, None, :], U_k_tr, R_k_tr], axis=1
    ).astype(np.float32)   # (S, 1+2M, N)

    x_norm = ChannelNormalizer.fit(x_stack)
    y_norm = ChannelNormalizer.fit(dU_tr.astype(np.float32))

    train_ds = FNOSweepDataset(samples, train_idx, x_norm, y_norm)
    val_ds   = FNOSweepDataset(samples, val_idx,   x_norm, y_norm)
    return train_ds, val_ds, x_norm, y_norm

