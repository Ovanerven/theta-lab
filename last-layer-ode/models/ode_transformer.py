from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.attention import sdpa_kernel, SDPBackend

from scaffolds import MechanisticScaffold


def gamma(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return lo + (hi - lo) * torch.sigmoid(x)


class OdeTransformer(nn.Module):
    """
    Causal Transformer encoder replacing the GRU history encoder.

    At each step k, attends over the last `context_len` lifted feature vectors
    using a causal mask and decodes theta_k from the output at the current
    position. Bolus application and RK4 ODE integration are identical to OdeRNN.

    Architecture:
      lift : (U+P) -> lift_dim (SiLU) -> hidden
      pos  : Embedding(context_len, hidden)  — window-relative positions
      enc  : TransformerEncoder, d_model=hidden, nhead=hidden//32, num_layers
      head : hidden -> theta_dim

    Note: jit_scripting must be false (nn.TransformerEncoder is not scriptable).
    """

    def __init__(
        self,
        *,
        U: int,
        rhs: MechanisticScaffold,
        u_to_y_jump: torch.Tensor,
        hidden: int = 128,
        lift_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.0,
        ff_mult: int = 2,
        theta_lo: float = 1e-3,
        theta_hi: float = 2.0,
        n_substeps: int = 1,
        use_basal: bool = False,
        context_len: int = 64,
        **kwargs,
    ):
        super().__init__()
        self.U         = int(U)
        self.P         = int(rhs.P)
        self.theta_dim = int(rhs.theta_dim)
        self.rhs       = rhs
        self.n_substeps = int(n_substeps)
        self.use_basal  = bool(use_basal)
        self.theta_lo   = float(theta_lo)
        self.theta_hi   = float(theta_hi)
        self.hidden     = int(hidden)
        self.context_len = int(context_len)

        self.lift = nn.Sequential(
            nn.Linear(self.U + self.P, lift_dim),
            nn.SiLU(),
            nn.Linear(lift_dim, hidden),
        )

        self.pos_embed = nn.Embedding(self.context_len, hidden)

        nhead = max(1, hidden // 32)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=nhead,
            dim_feedforward=hidden * ff_mult,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN: more stable at small batch sizes
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=max(1, num_layers),
            enable_nested_tensor=False,
        )

        head_out = self.theta_dim + self.P if use_basal else self.theta_dim
        self.head = nn.Linear(hidden, head_out)
        # Small-weight init: keeps raw≈0 at start so gamma maps to the midpoint
        # of [theta_lo, theta_hi] regardless of transformer output magnitude.
        # Default Kaiming init on a large-residual-stack output saturates theta
        # toward theta_hi, destabilising the ODE on the first forward pass.
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

        # Pos-embed at N(0,0.02) instead of the default N(0,1) — standard
        # transformer practice (GPT-2 / BERT style) to reduce initial variance.
        nn.init.normal_(self.pos_embed.weight, std=0.02)

        if u_to_y_jump.shape != (self.U, self.P):
            raise ValueError(
                f"u_to_y_jump must be (U,P)=({self.U},{self.P}), "
                f"got {tuple(u_to_y_jump.shape)}"
            )
        self.register_buffer("u_to_y_jump", u_to_y_jump.float(), persistent=True)

    def forward(
        self,
        y0: torch.Tensor,                      # (B, P)
        u_seq: torch.Tensor,                   # (B, K, U)
        dt_seq: torch.Tensor,                  # (B, K)
        obs_idx: torch.Tensor,
        y_seq: Optional[torch.Tensor] = None,  # (B, K, P) for teacher forcing
        teacher_forcing: bool = True,
        tf_every: int = 50,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, K, _ = u_seq.shape
        device, dtype = y0.device, y0.dtype

        y_out    = torch.empty(B, K, self.P,         device=device, dtype=dtype)
        th_out   = torch.empty(B, K, self.theta_dim, device=device, dtype=dtype)
        beta_out = torch.zeros(B, K, self.P,         device=device, dtype=dtype)

        use_partial = obs_idx.numel() > 0
        y_prev = y0

        feat_history: List[torch.Tensor] = []  # (B, hidden) tensors, grows to context_len

        for k in range(K):
            u_k  = u_seq[:, k, :]
            dt_k = dt_seq[:, k]

            y_in = y_prev.detach()
            if teacher_forcing and k > 0 and (k % tf_every == 0) and y_seq is not None:
                if use_partial:
                    y_in = y_prev.clone()
                    idx  = obs_idx.to(device=y_in.device, dtype=torch.long)
                    y_in[:, idx] = y_seq[:, k - 1, idx].to(dtype=y_in.dtype).detach()
                else:
                    y_in = y_seq[:, k - 1, :].to(dtype=y_prev.dtype).detach()

            feat = self.lift(torch.cat([u_k, y_in], dim=-1))  # (B, hidden)
            feat_history.append(feat)

            # Detach features that have scrolled out of the context window so
            # the backward graph stays bounded at O(K * context_len) tensors.
            if len(feat_history) > self.context_len:
                feat_history[-self.context_len - 1] = (
                    feat_history[-self.context_len - 1].detach()
                )

            start  = max(0, len(feat_history) - self.context_len)
            window: List[torch.Tensor] = feat_history[start:]
            W = len(window)

            seq = torch.stack(window, dim=1)  # (B, W, hidden)

            # Window-relative positional embedding: 0 = oldest, W-1 = newest
            pos_ids = torch.arange(W, device=device, dtype=torch.long)
            seq = seq + self.pos_embed(pos_ids).unsqueeze(0)

            # Don't need the causal mask
            # causal_mask = torch.triu(
            #     torch.full((W, W), float("-inf"), device=device, dtype=dtype),
            #     diagonal=1,
            # )

            with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):  # raises if flash can't run
                out = self.transformer(seq, is_causal=True)

            # out = self.transformer(seq, is_causal=True)  # (B, W, hidden)
            z   = out[:, -1, :]                            # (B, hidden)
            raw = self.head(z)

            if self.use_basal:
                theta_k = gamma(raw[:, :self.theta_dim], self.theta_lo, self.theta_hi)
                beta_k  = raw[:, self.theta_dim:] * (y_prev / (y_prev + 1.0))
                beta_out[:, k, :] = beta_k
                y = y_prev + (u_k @ self.u_to_y_jump)
                y = self._rk4_substeps_basal(y, dt_k, theta_k, beta_k)
            else:
                theta_k = gamma(raw, self.theta_lo, self.theta_hi)
                y = y_prev + (u_k @ self.u_to_y_jump)
                y = self._rk4_substeps(y, dt_k, theta_k)

            y_out[:, k, :]  = y
            th_out[:, k, :] = theta_k
            y_prev = y

        return y_out, th_out, beta_out

    def _rk4_substeps(
        self, y: torch.Tensor, dt: torch.Tensor, theta: torch.Tensor
    ) -> torch.Tensor:
        rhs   = self.rhs
        n_sub = self.n_substeps
        dt    = dt.unsqueeze(1)
        hdt   = dt / float(n_sub)
        for _ in range(n_sub):
            k1 = rhs(y,                   theta)
            k2 = rhs(y + 0.5 * hdt * k1, theta)
            k3 = rhs(y + 0.5 * hdt * k2, theta)
            k4 = rhs(y +       hdt * k3,  theta)
            y  = y + (hdt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return torch.clamp_min(y, 0.0)

    def _rk4_substeps_basal(
        self,
        y: torch.Tensor,
        dt: torch.Tensor,
        theta: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        rhs   = self.rhs
        n_sub = self.n_substeps
        dt    = dt.unsqueeze(1)
        hdt   = dt / float(n_sub)
        for _ in range(n_sub):
            k1 = rhs(y,                   theta) + beta
            k2 = rhs(y + 0.5 * hdt * k1, theta) + beta
            k3 = rhs(y + 0.5 * hdt * k2, theta) + beta
            k4 = rhs(y +       hdt * k3,  theta) + beta
            y  = y + (hdt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return torch.clamp_min(y, 0.0)