from __future__ import annotations

import copy
import logging
import time

import numpy as np

from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.playgrounds.learned_qdelta.dataset import Standardizer, build_feature_vector, ChannelNormalizer
from pySDC.playgrounds.learned_qdelta.models import build_model, FNO1d
from pySDC.playgrounds.learned_qdelta.sweeper_utils import (
    compute_residual_vectors,
    extract_problem_params,
    stack_nodes,
    state_to_numpy,
)


class DataCollectingImplicitSweeper(generic_implicit):
    """Generic implicit sweeper that records one-sweep correction samples."""

    def __init__(self, params, level):
        super().__init__(params, level)
        self.samples = []
        self.use_z_param = bool(params.get('use_z_param', False))

    def update_nodes(self):
        L = self.level
        P = L.prob
        M = self.coll.num_nodes

        u0_vec = state_to_numpy(L.u[0])
        old_u = [P.dtype_u(L.u[m + 1]) for m in range(M)]
        old_f = [P.dtype_f(L.f[m + 1]) for m in range(M)]

        U_k = stack_nodes(old_u)
        R_k, _ = compute_residual_vectors(L, old_u, old_f)

        super().update_nodes()

        new_u = stack_nodes([L.u[m + 1] for m in range(M)])
        target_dU = new_u - U_k

        _dt_for_param = L.dt if self.use_z_param else None
        sample = {
            'u0': u0_vec,
            'U_k': U_k,
            'R_k': R_k,
            'dt': np.array([L.dt], dtype=np.float64),
            'nodes': np.asarray(self.coll.nodes, dtype=np.float64),
            'qmat': np.asarray(self.coll.Qmat[1:, 1:], dtype=np.float64),
            'problem_params': extract_problem_params(L.prob, dt=_dt_for_param),
            'target_dU': target_dU,
        }
        self.samples.append(sample)

    def drain_samples(self):
        out = self.samples
        self.samples = []
        return out


class LearnedQDeltaSweeper(generic_implicit):
    """Sweeper with learned one-sweep proposal and robust classical fallback."""

    def __init__(self, params, level):
        super().__init__(params, level)

        self.logger = logging.getLogger('learned_qdelta_sweeper')
        self.accept_factor = float(params.get('accept_factor', 0.95))
        self.model_device = params.get('model_device', 'cpu')
        self.model_checkpoint = params.get('model_checkpoint', None)
        self.model_enabled = False
        self.use_z_param = False  # overridden by checkpoint

        self.last_old_residual = None
        self.last_trial_residual = None
        self.last_accepted = False
        self.last_used_model = False

        self._x_normalizer = None
        self._y_normalizer = None
        self._model = None
        self._output_dim = None
        self._agnostic_feature_variant = 'legacy12'

        self._fallback_class = params.get('fallback_sweeper_class', generic_implicit)
        self._fallback_sweeper = None

        fallback_params = copy.deepcopy(params)
        for key in [
            'accept_factor',
            'accept_factor_min',
            'accept_factor_max',
            'accept_factor_slope',
            'accept_factor_center',
            'confidence_ratio_max',
            'model_device',
            'model_checkpoint',
            'fallback_sweeper_class',
        ]:
            fallback_params.pop(key, None)

        if self._fallback_class is not type(self):
            self._fallback_sweeper = self._fallback_class(fallback_params, level)

        self._load_model()

    def _load_model(self):
        if self.model_checkpoint is None:
            self.logger.warning('No model checkpoint provided. Learned proposal disabled.')
            return
        ckpt_path = self.model_checkpoint

        try:
            import torch

            try:
                checkpoint = torch.load(ckpt_path, map_location=self.model_device, weights_only=False)
            except TypeError:
                checkpoint = torch.load(ckpt_path, map_location=self.model_device)
            model_cfg = checkpoint['model_config']
            input_dim = int(checkpoint['input_dim'])
            output_dim = int(checkpoint['output_dim'])

            self._model = build_model(model_cfg, input_dim=input_dim, output_dim=output_dim)
            self._model.load_state_dict(checkpoint['model_state'])
            self._model.to(self.model_device)
            self._model.eval()

            self._x_normalizer = Standardizer.from_state_dict(checkpoint['x_normalizer'])
            self._y_normalizer = Standardizer.from_state_dict(checkpoint['y_normalizer'])
            self.use_z_param = bool(checkpoint.get('use_z_param', False))
            # dimension_agnostic: use checkpoint flag; auto-detect from input_dim for old checkpoints
            if 'dimension_agnostic' in checkpoint:
                self.dimension_agnostic = bool(checkpoint['dimension_agnostic'])
            else:
                self.dimension_agnostic = (input_dim <= 20)

            if self.dimension_agnostic:
                if input_dim == 12:
                    self._agnostic_feature_variant = 'legacy12'
                elif input_dim == self.coll.num_nodes * 10 + 7:
                    self._agnostic_feature_variant = 'per_node37'
                else:
                    self.logger.warning(
                        'Unknown agnostic input_dim=%s; defaulting to legacy12 feature builder.',
                        input_dim,
                    )
                    self._agnostic_feature_variant = 'legacy12'

            self._output_dim = output_dim
            self.model_enabled = True
        except Exception as exc:  # pragma: no cover - prototype safety net
            self.logger.warning('Failed to load learned model (%s). Falling back to classical sweeps.', exc)
            self.model_enabled = False

    def _run_fallback(self, attempted=False):
        self.last_used_model = bool(attempted)
        self.last_accepted = False
        if self._fallback_sweeper is None:
            super().update_nodes()
        else:
            self._fallback_sweeper.update_nodes()

    def _apply_trial_nodes(self, trial_u):
        L = self.level
        for m, candidate in enumerate(trial_u, start=1):
            L.u[m] = candidate
            t = L.time + L.dt * self.coll.nodes[m - 1]
            L.f[m] = L.prob.eval_f(candidate, t)
        L.status.updated = True

    def update_nodes(self):
        L = self.level
        P = L.prob
        M = self.coll.num_nodes

        if not self.model_enabled:
            self.last_gate_reason_code = self.GATE_REASON_MODEL_DISABLED
            self._run_fallback(attempted=False)
            return

        old_u = [P.dtype_u(L.u[m + 1]) for m in range(M)]
        old_f = [P.dtype_f(L.f[m + 1]) for m in range(M)]
        old_residual_vec, old_residual = compute_residual_vectors(L, old_u, old_f)

        sample = {
            'u0': state_to_numpy(L.u[0]),
            'U_k': stack_nodes(old_u),
            'R_k': old_residual_vec,
            'dt': np.array([L.dt], dtype=np.float64),
            'nodes': np.asarray(self.coll.nodes, dtype=np.float64),
            'qmat': np.asarray(self.coll.Qmat[1:, 1:], dtype=np.float64),
            'problem_params': extract_problem_params(L.prob, dt=L.dt if self.use_z_param else None),
        }

        x = build_feature_vector(
            sample,
            use_z_param=self.use_z_param,
            dimension_agnostic=self.dimension_agnostic,
            agnostic_feature_variant=self._agnostic_feature_variant,
        )[None, :]
        x = self._x_normalizer.transform(x).astype(np.float32)

        try:
            import torch

            with torch.no_grad():
                x_t = torch.from_numpy(x).to(self.model_device)
                pred_norm = self._model(x_t).cpu().numpy()
        except Exception as exc:  # pragma: no cover - runtime safety
            self.logger.warning('Learned proposal failed at runtime (%s). Using fallback.', exc)
            self.last_old_residual = old_residual
            self.last_trial_residual = float('nan')
            self._run_fallback(attempted=True)
            return

        pred = self._y_normalizer.inverse(pred_norm)[0]

        trial_u = []
        if self.dimension_agnostic and self._output_dim == M:
            # pred is M per-node scale factors: ΔU[m] = pred[m] * R_k[m]
            # This reconstructs a correction proportional to the residual,
            # scaled by a learned magnitude — fully independent of spatial dim.
            scales = pred.reshape(M)
            R_k = old_residual_vec  # (M, state_dim)
            for m in range(M):
                trial = P.dtype_u(L.u[m + 1])
                correction = scales[m] * R_k[m].reshape(np.asarray(old_u[m]).shape)
                trial[:] = np.asarray(old_u[m]) + correction
                trial_u.append(trial)
        else:
            # Full-state mode: pred is flattened ΔU vector (M * state_dim)
            pred = pred.reshape(M, -1)
            for m in range(M):
                trial = P.dtype_u(L.u[m + 1])
                trial[:] = np.asarray(old_u[m]) + pred[m].reshape(np.asarray(old_u[m]).shape)
                trial_u.append(trial)

        trial_f = [P.eval_f(trial_u[m], L.time + L.dt * self.coll.nodes[m]) for m in range(M)]
        _, trial_residual = compute_residual_vectors(L, trial_u, trial_f)

        self.last_old_residual = old_residual
        self.last_trial_residual = trial_residual
        self.last_used_model = True

        if trial_residual <= self.accept_factor * max(old_residual, 1e-16):
            self._apply_trial_nodes(trial_u)
            self.last_accepted = True
        else:
            self.last_accepted = False
            self._run_fallback(attempted=True)


class FNOLearnedQDeltaSweeper(generic_implicit):
    """Sweeper backed by a Fourier Neural Operator (FNO).

    The FNO operates on the *spatial fields* u0, U_k, R_k directly and
    outputs correction fields ΔU — one spatial field per collocation node.
    Because the model uses global FFT + pointwise convolutions it works for
    **any grid size N** without retraining.

    Checkpoint format
    -----------------
    Created by ``train_fno.py``.  Expected keys:
      - ``model_state``   – FNO state dict
      - ``model_config``  – dict with ``name='fno'``, ``in_channels``, etc.
      - ``x_normalizer``  – ChannelNormalizer state dict  (C_in, 1)
      - ``y_normalizer``  – ChannelNormalizer state dict  (M, 1)
      - ``model_type``    – 'fno'
      - ``in_channels``   – int
      - ``out_channels``  – int (= M)

    Parameters (sweeper_params)
    ---------------------------
    model_checkpoint : str   path to .pt file
    accept_factor    : float residual-gate threshold (default 0.95)
    model_device     : str   'cpu' or 'cuda'  (default 'cpu')
    fallback_sweeper_class : type  (default generic_implicit)

    Gating details
    --------------
    The residual gate can adapt with residual scale:
      eff_accept = clip(base + slope * (log10(old_residual) - center), min, max)
    and a confidence pre-gate rejects corrections with very large norm ratio
    compared to the current residual before trial evaluation.
    """

    GATE_REASON_ACCEPTED = 0
    GATE_REASON_RESIDUAL_FAIL = 1
    GATE_REASON_CONFIDENCE_PRE_REJECT = 2
    GATE_REASON_INFERENCE_ERROR = 3
    GATE_REASON_MODEL_DISABLED = 4
    GATE_REASON_SKIPPED_BY_SWEEP_LIMIT = 5

    def __init__(self, params, level):
        super().__init__(params, level)

        self.logger = logging.getLogger('fno_learned_qdelta_sweeper')
        self.accept_factor   = float(params.get('accept_factor', 0.95))
        self.accept_factor_min = float(params.get('accept_factor_min', self.accept_factor))
        self.accept_factor_max = float(params.get('accept_factor_max', self.accept_factor))
        self.accept_factor_slope = float(params.get('accept_factor_slope', 0.0))
        self.accept_factor_center = float(params.get('accept_factor_center', -6.0))
        self.confidence_ratio_max = float(params.get('confidence_ratio_max', 4.0))
        self.learned_max_sweeps_per_step = int(params.get('learned_max_sweeps_per_step', 0))
        self.model_device    = params.get('model_device', 'cpu')
        self.model_checkpoint = params.get('model_checkpoint', None)
        self.model_enabled   = False

        self.last_old_residual   = None
        self.last_trial_residual = None
        self.last_accepted       = False
        self.last_used_model     = False
        self.last_effective_accept_factor = self.accept_factor
        self.last_confidence_ratio = float('nan')
        self.last_gate_reason_code = self.GATE_REASON_MODEL_DISABLED
        self.last_inference_time = 0.0
        self.last_trial_eval_time = 0.0

        self._model      = None
        self._x_ch_norm  = None   # ChannelNormalizer for input fields
        self._y_ch_norm  = None   # ChannelNormalizer for target fields
        self._out_channels = None

        self._fallback_class = params.get('fallback_sweeper_class', generic_implicit)
        self._fallback_sweeper = None

        fallback_params = copy.deepcopy(params)
        for key in [
            'accept_factor',
            'accept_factor_min',
            'accept_factor_max',
            'accept_factor_slope',
            'accept_factor_center',
            'confidence_ratio_max',
            'learned_max_sweeps_per_step',
            'model_device',
            'model_checkpoint',
            'fallback_sweeper_class',
        ]:
            fallback_params.pop(key, None)

        if self._fallback_class is not type(self):
            self._fallback_sweeper = self._fallback_class(fallback_params, level)

        self._load_model()

    # ------------------------------------------------------------------
    def _load_model(self):
        if self.model_checkpoint is None:
            self.logger.warning('No FNO checkpoint provided. Falling back to classical.')
            return
        ckpt_path = self.model_checkpoint
        try:
            import torch
            try:
                ckpt = torch.load(ckpt_path, map_location=self.model_device, weights_only=False)
            except TypeError:
                ckpt = torch.load(ckpt_path, map_location=self.model_device)

            if ckpt.get('model_type', 'mlp') != 'fno':
                raise ValueError(
                    f'Checkpoint model_type={ckpt.get("model_type")} is not "fno". '
                    'Use LearnedQDeltaSweeper for MLP checkpoints.'
                )

            cfg = ckpt['model_config']
            self._model = FNO1d(
                in_channels  = int(cfg['in_channels']),
                out_channels = int(cfg['out_channels']),
                width  = int(cfg.get('width',  64)),
                modes  = int(cfg.get('modes',  16)),
                depth  = int(cfg.get('depth',   4)),
            )
            self._model.load_state_dict(ckpt['model_state'])
            self._model.to(self.model_device)
            self._model.eval()

            self._x_ch_norm   = ChannelNormalizer.from_state_dict(ckpt['x_normalizer'])
            self._y_ch_norm   = ChannelNormalizer.from_state_dict(ckpt['y_normalizer'])
            self._out_channels = int(ckpt['out_channels'])
            self.model_enabled = True
            self.logger.info(
                'FNO checkpoint loaded: in_ch=%s out_ch=%s width=%s modes=%s depth=%s',
                cfg['in_channels'], cfg['out_channels'],
                cfg.get('width'), cfg.get('modes'), cfg.get('depth'),
            )
        except Exception as exc:
            self.logger.warning('Failed to load FNO model (%s). Falling back to classical.', exc)
            self.model_enabled = False

    # ------------------------------------------------------------------
    def _run_fallback(self, attempted: bool = False):
        self.last_used_model = bool(attempted)
        self.last_accepted   = False
        if self._fallback_sweeper is None:
            super().update_nodes()
        else:
            self._fallback_sweeper.update_nodes()

    def _apply_trial_nodes(self, trial_u):
        L = self.level
        for m, candidate in enumerate(trial_u, start=1):
            L.u[m] = candidate
            t = L.time + L.dt * self.coll.nodes[m - 1]
            L.f[m] = L.prob.eval_f(candidate, t)
        L.status.updated = True

    # ------------------------------------------------------------------
    def update_nodes(self):
        L = self.level
        P = L.prob
        M = self.coll.num_nodes

        self.last_inference_time = 0.0
        self.last_trial_eval_time = 0.0
        self.last_confidence_ratio = float('nan')
        self.last_effective_accept_factor = self.accept_factor

        if not self.model_enabled:
            self.last_gate_reason_code = self.GATE_REASON_MODEL_DISABLED
            self._run_fallback(attempted=False)
            return

        # Optionally restrict learned proposals to early sweeps within each step.
        # A value of 0 means "no limit".
        current_sweep = int(getattr(L.status, 'sweep', 0))
        if self.learned_max_sweeps_per_step > 0 and current_sweep > self.learned_max_sweeps_per_step:
            self.last_gate_reason_code = self.GATE_REASON_SKIPPED_BY_SWEEP_LIMIT
            self._run_fallback(attempted=False)
            return

        old_u = [P.dtype_u(L.u[m + 1]) for m in range(M)]
        old_f = [P.dtype_f(L.f[m + 1]) for m in range(M)]
        old_residual_vec, old_residual = compute_residual_vectors(L, old_u, old_f)

        # Build (1, C_in, N) spatial-field input tensor
        u0_np = state_to_numpy(L.u[0])           # (N,)
        U_k_np = stack_nodes(old_u)              # (M, N)
        R_k_np = old_residual_vec                # (M, N)

        x_fields = np.concatenate(
            [u0_np[None, :], U_k_np, R_k_np], axis=0
        ).astype(np.float32)                     # (1+2M, N)
        x_fields = self._x_ch_norm.transform(x_fields)   # normalise

        try:
            import torch
            infer_t0 = time.perf_counter()
            with torch.inference_mode():
                x_t = torch.from_numpy(x_fields[None]).to(self.model_device)  # (1, C_in, N)
                pred_norm = self._model(x_t).cpu().numpy()[0]                  # (M, N)
            self.last_inference_time = time.perf_counter() - infer_t0
        except Exception as exc:
            self.logger.warning('FNO proposal failed (%s). Using fallback.', exc)
            self.last_old_residual   = old_residual
            self.last_trial_residual = float('nan')
            self.last_gate_reason_code = self.GATE_REASON_INFERENCE_ERROR
            self._run_fallback(attempted=True)
            return

        # Inverse-normalise back to physical space
        pred_dU = self._y_ch_norm.inverse(pred_norm)   # (M, N)

        # Apply correction
        trial_u = []
        for m in range(M):
            trial = P.dtype_u(L.u[m + 1])
            trial[:] = np.asarray(old_u[m]) + pred_dU[m].reshape(np.asarray(old_u[m]).shape)
            trial_u.append(trial)

        pred_norm_val = float(np.linalg.norm(pred_dU.reshape(-1)))
        residual_norm_val = float(np.linalg.norm(old_residual_vec.reshape(-1)))
        self.last_confidence_ratio = pred_norm_val / max(residual_norm_val, 1e-16)
        if self.last_confidence_ratio > self.confidence_ratio_max:
            self.last_old_residual = old_residual
            self.last_trial_residual = float('nan')
            self.last_used_model = True
            self.last_accepted = False
            self.last_gate_reason_code = self.GATE_REASON_CONFIDENCE_PRE_REJECT
            self._run_fallback(attempted=True)
            return

        trial_t0 = time.perf_counter()
        trial_f = [P.eval_f(trial_u[m], L.time + L.dt * self.coll.nodes[m]) for m in range(M)]
        _, trial_residual = compute_residual_vectors(L, trial_u, trial_f)
        self.last_trial_eval_time = time.perf_counter() - trial_t0

        log_res = np.log10(max(float(old_residual), 1e-30))
        eff_accept = self.accept_factor + self.accept_factor_slope * (log_res - self.accept_factor_center)
        eff_accept = float(np.clip(eff_accept, self.accept_factor_min, self.accept_factor_max))
        self.last_effective_accept_factor = eff_accept

        self.last_old_residual   = old_residual
        self.last_trial_residual = trial_residual
        self.last_used_model     = True

        if trial_residual <= eff_accept * max(old_residual, 1e-16):
            self._apply_trial_nodes(trial_u)
            self.last_accepted = True
            self.last_gate_reason_code = self.GATE_REASON_ACCEPTED
        else:
            self.last_accepted = False
            self.last_gate_reason_code = self.GATE_REASON_RESIDUAL_FAIL
            self._run_fallback(attempted=True)
