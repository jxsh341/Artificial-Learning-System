"""State-space temporal model (spec §10).

    s_t = f(A s_{t-1} + B z_t)
    y_t = C s_t + D z_t

Exposes the required interface:

    state = ssm.initial_state(batch_size)
    output, state = ssm.step(input_t, state)
    outputs, final_state = ssm(sequence)

Also contains stateless / transformer temporal baselines used by the
ablation matrix (spec §20).
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

_ACTIVATIONS = {
    "none": nn.Identity,
    "identity": nn.Identity,
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
}


class StateSpaceModel(nn.Module):
    """Discrete-time linear state-space model with optional state nonlinearity.

    The state transition ``A`` is constrained to spectral radius
    ``<= max_radius``: directly for the diagonal parameterisation, and via
    power-iteration spectral-norm rescaling for the dense parameterisation.
    """

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        output_dim: int | None = None,
        diagonal: bool = True,
        max_radius: float = 0.99,
        init_decay: float = 0.9,
        nonlinearity: str = "tanh",
        power_iters: int = 5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.output_dim = output_dim or state_dim
        self.diagonal = diagonal
        self.max_radius = float(max_radius)
        self.power_iters = power_iters

        if diagonal:
            # a = max_radius * tanh(raw / 2); invert for init_decay.
            init = min(max(init_decay / self.max_radius, 1e-4), 0.9999)
            self.A_raw = nn.Parameter(torch.full((state_dim,), 2.0 * math.atanh(init)))
        else:
            self.A_matrix = nn.Parameter(
                torch.randn(state_dim, state_dim) / math.sqrt(state_dim)
            )
        self.B = nn.Linear(input_dim, state_dim, bias=False)
        self.C = nn.Linear(state_dim, self.output_dim)
        self.D = nn.Linear(input_dim, self.output_dim, bias=False)

        if nonlinearity not in _ACTIVATIONS:
            raise ValueError(f"unknown nonlinearity {nonlinearity!r}")
        self.act = _ACTIVATIONS[nonlinearity]()

    def effective_decay(self) -> Tensor:
        """Per-dimension decay rates (diagonal parameterisation only)."""
        if not self.diagonal:
            raise RuntimeError("effective_decay only defined for diagonal SSM")
        return self.max_radius * torch.tanh(self.A_raw / 2.0)

    def _A_effective(self) -> Tensor:
        if self.diagonal:
            return torch.diag(self.effective_decay())
        A = self.A_matrix
        v = torch.ones(A.shape[1], device=A.device, dtype=A.dtype) / math.sqrt(A.shape[1])
        for _ in range(self.power_iters):
            v = A.t() @ (A @ v)
            v = v / v.norm().clamp(min=1e-8)
        sigma = (A @ v).norm()
        scale = (self.max_radius / sigma.clamp(min=1e-8)).clamp(max=1.0)
        return A * scale

    def initial_state(
        self,
        batch_size: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        dev = device if device is not None else self.B.weight.device
        dt = dtype if dtype is not None else self.B.weight.dtype
        return torch.zeros(batch_size, self.state_dim, device=dev, dtype=dt)

    def reset_state(self, state: Tensor, mask: Tensor | None = None) -> Tensor:
        """Full or partial state reset (spec §19 temporal perturbations)."""
        fresh = torch.zeros_like(state)
        if mask is None:
            return fresh
        return torch.where(mask.unsqueeze(-1), fresh, state) if mask.ndim == 1 else torch.where(
            mask, fresh, state
        )

    def step(self, z: Tensor, state: Tensor | None) -> tuple[Tensor, Tensor]:
        """One temporal update. ``z``: ``[B, input_dim]``."""
        if state is None:
            state = self.initial_state(z.shape[0], z.device, z.dtype)
        if self.diagonal:
            s = state * self.effective_decay() + self.B(z)
        else:
            s = state @ self._A_effective().t() + self.B(z)
        s = self.act(s)
        y = self.C(s) + self.D(z)
        return y, s

    def forward(
        self,
        sequence: Tensor,
        initial_state: Tensor | None = None,
        return_states: bool = False,
    ):
        """Process a sequence ``[T, B, input_dim]`` step by step."""
        if sequence.ndim != 3:
            raise ValueError(f"sequence must be [T, B, input_dim], got {tuple(sequence.shape)}")
        state = initial_state
        outputs, states = [], []
        for t in range(sequence.shape[0]):
            y, state = self.step(sequence[t], state)
            outputs.append(y)
            states.append(state)
        out = torch.stack(outputs, dim=0)
        if return_states:
            return out, state, torch.stack(states, dim=0)
        return out, state


class NullTemporal(nn.Module):
    """Stateless temporal placeholder used by no-SSM ablations."""

    def initial_state(self, batch_size: int, device=None, dtype=None):
        return None

    def step(self, z: Tensor, state):
        return z, state

    def reset_state(self, state, mask=None):
        return None

    def forward(self, sequence: Tensor, initial_state=None, return_states: bool = False):
        if return_states:
            return sequence, None, None
        return sequence, None


class TransformerTemporal(nn.Module):
    """Causal transformer over the embedding sequence (ablation M5).

    ``step`` recomputes attention over the cached prefix — simple and correct
    for the short sequences used here, not meant for production scale.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        max_len: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.output_dim = d_model
        self.state_dim = d_model
        self.max_len = max_len
        self.proj = nn.Linear(input_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self._cache: list[Tensor] | None = None

    def initial_state(self, batch_size: int, device=None, dtype=None):
        self._cache = []
        return None

    def step(self, z: Tensor, state):
        if self._cache is None:
            self.initial_state(z.shape[0])
        self._cache.append(z)
        if len(self._cache) > self.max_len:
            self._cache = self._cache[-self.max_len :]
        x = torch.stack(self._cache, dim=1)  # [B, T, H]
        t = x.shape[1]
        mask = torch.triu(
            torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1
        )
        h = self.encoder(self.proj(x) + self.pos[:, :t], mask=mask)
        return h[:, -1], None

    def forward(self, sequence: Tensor, initial_state=None, return_states: bool = False):
        self.initial_state(sequence.shape[1], sequence.device)
        x = sequence.transpose(0, 1)  # [B, T, H]
        t = x.shape[1]
        mask = torch.triu(
            torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1
        )
        h = self.encoder(self.proj(x) + self.pos[:, :t], mask=mask)
        return h.transpose(0, 1), None
