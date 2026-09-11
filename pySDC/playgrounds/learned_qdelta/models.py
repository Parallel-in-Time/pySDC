from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPPreconditioner(nn.Module):
    """Small baseline MLP used for one-sweep correction prediction."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128, depth: int = 2, use_batchnorm: bool = True):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Fourier Neural Operator (1D)
# ---------------------------------------------------------------------------

class SpectralConv1d(nn.Module):
    """Complex-weight spectral mixing over the first `modes` Fourier modes."""

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        scale = 1.0 / (in_channels * out_channels) ** 0.5
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, 2)
        )

    def _complex_matmul(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, modes)  complex
        # w: (C_in, C_out, modes) complex
        # returns (B, C_out, modes) complex
        wr, wi = w[..., 0], w[..., 1]
        xr, xi = x.real, x.imag
        return torch.complex(
            torch.einsum('bim,iom->bom', xr, wr) - torch.einsum('bim,iom->bom', xi, wi),
            torch.einsum('bim,iom->bom', xr, wi) + torch.einsum('bim,iom->bom', xi, wr),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, N)
        N = x.shape[-1]
        x_ft = torch.fft.rfft(x, dim=-1)          # (B, C_in, N//2+1)
        modes = min(self.modes, x_ft.shape[-1])

        out_ft = torch.zeros(
            x.shape[0], self.out_channels, x_ft.shape[-1],
            dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :modes] = self._complex_matmul(x_ft[:, :, :modes], self.weights[:, :, :modes, :])
        return torch.fft.irfft(out_ft, n=N, dim=-1)   # (B, C_out, N)


class FNOBlock1d(nn.Module):
    """One FNO layer: spectral branch + bypass pointwise conv + activation."""

    def __init__(self, channels: int, modes: int):
        super().__init__()
        self.spectral = SpectralConv1d(channels, channels, modes)
        self.bypass = nn.Conv1d(channels, channels, kernel_size=1)
        # GroupNorm(1, C) is equivalent to LayerNorm over channels and works
        # for any spatial size N≥1 (InstanceNorm1d requires N>1 during training).
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.spectral(x) + self.bypass(x)))


class FNO1d(nn.Module):
    """1-D Fourier Neural Operator for SDC sweep correction.

    Input  shape: (B, in_channels,  N)  — in_channels spatial fields (u0, Uk, Rk, …)
    Output shape: (B, out_channels, N)  — correction fields ΔU per collocation node

    Works for **any** spatial grid size N because all operations are either
    pointwise or via global FFT.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int = 64,
        modes: int = 16,
        depth: int = 4,
    ):
        super().__init__()
        self.lift = nn.Conv1d(in_channels, width, kernel_size=1)
        self.blocks = nn.ModuleList([FNOBlock1d(width, modes) for _ in range(depth)])
        self.proj1 = nn.Conv1d(width, width * 2, kernel_size=1)
        self.proj2 = nn.Conv1d(width * 2, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, N)
        x = self.lift(x)
        for block in self.blocks:
            x = block(x)
        x = F.gelu(self.proj1(x))
        return self.proj2(x)   # (B, C_out, N)


def build_model(config: dict, input_dim: int = 0, output_dim: int = 0):
    """Factory for correction models.

    MLP: input_dim / output_dim are flat vector sizes.
    FNO: uses config keys in_channels, out_channels, width, modes, depth.
         input_dim / output_dim are ignored (grid-size agnostic).
    """
    name = config.get('name', 'mlp').lower()

    if name == 'mlp':
        return MLPPreconditioner(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=int(config.get('hidden_dim', 128)),
            depth=int(config.get('depth', 2)),
            use_batchnorm=bool(config.get('use_batchnorm', True)),
        )

    if name == 'fno':
        num_nodes = int(config.get('num_nodes', 3))
        return FNO1d(
            in_channels=int(config.get('in_channels', 1 + 2 * num_nodes)),
            out_channels=int(config.get('out_channels', num_nodes)),
            width=int(config.get('width', 64)),
            modes=int(config.get('modes', 16)),
            depth=int(config.get('depth', 4)),
        )

    raise ValueError(f'Unknown model name: {name}')

