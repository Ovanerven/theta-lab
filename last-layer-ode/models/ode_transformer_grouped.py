from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from scaffolds import MechanisticScaffold


def gamma(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return lo + (hi - lo) * torch.sigmoid(x)


def log_gamma(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    return lo * torch.exp(torch.log(hi / lo) * torch.sigmoid(x))


class OdeTransformerGrouped(nn.Module):
    """
    Causal Transformer encoder with GROUPED teacher forcing for training
    throughput.

    Architecture is identical to OdeTransformer (lift -> pos+causal encoder
    -> head -> gamma -> RK4). The only change is the training-time forward
    schedule: trajectories are processed in alternating phases

        [TF group (G steps, parallel)] -> [AR gap (A steps, sequential)] -> ...

    During a TF group, all G inputs are lifted in a single batched call
    using ground-truth (y_seq[k-1]) as the previous-state input. The
    Transformer then runs ONCE over the current attention window with a
    causal mask, and the last G output positions yield G theta vectors.
    Each of the G RK4 integrations is independent — each starts from its
    own ground-truth y_seq[k-1] — so they are batched as (B*G, P) in a
    single RK4 call.

    This replaces G sequential encoder forward passes (+ G small RK4 calls)
    with ONE encoder forward over a (B, W, hidden) tensor and ONE batched
    RK4 over (B*G, P). On the Transformer this is close to a G× reduction
    in wall-clock time during teacher-forced phases. Per-step supervision
    is unchanged — theta_k is still computed independently for each step,
    and loss is still per-step MSE against y_seq.

    AR gaps between groups ensure the model sees its own predictions as
    input, so it learns to self-correct at inference time.

    At teacher_forcing=False (validation / test / inference), the forward
    falls back to the fully sequential autoregressive path — behaviour is
    identical to OdeTransformer.

    Init-only kwargs
    ----------------
    tf_group_size : int
        Number of steps to process in parallel per TF phase. Must be
        >= 1 and <= context_len. Default 32.
    ar_gap : int
        Number of sequential AR steps between TF groups. Set to 0 to
        train with pure parallel TF throughout (fast but no self-correction
        signal). Default 4.

    Notes
    -----
    - tf_every from the forward() call is IGNORED in grouped mode. The
      schedule is fully determined by (tf_group_size, ar_gap).
    - TF within a group is FULL teacher forcing: both the lift input AND
      the RK4 starting state come from y_seq. This breaks BPTT through
      integration at group boundaries but is what enables the parallelism.
      BPTT is intact WITHIN AR segments.
    - jit_scripting must be false (nn.TransformerEncoder is not scriptable).
    - Compatible with torch.autocast(bfloat16) + flash attention via the
      is_causal=True path used in encode.
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
        tf_group_size: int = 32,
        ar_gap: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.U             = int(U)
        self.P             = int(rhs.P)
        self.theta_dim     = int(rhs.theta_dim)
        self.rhs           = rhs
        self.n_substeps    = int(n_substeps)
        self.use_basal     = bool(use_basal)
        self.theta_lo      = float(theta_lo)
        self.theta_hi      = float(theta_hi)
        self.hidden        = int(hidden)
        self.context_len   = int(context_len)
        self.tf_group_size = int(tf_group_size)
        self.ar_gap        = int(ar_gap)

        if rhs.theta_lo_vec is not None and rhs.theta_hi_vec is not None:
            lo = torch.tensor(rhs.theta_lo_vec, dtype=torch.float32)
            hi = torch.tensor(rhs.theta_hi_vec, dtype=torch.float32)
        else:
            lo = torch.full((self.theta_dim,), theta_lo)
            hi = torch.full((self.theta_dim,), theta_hi)
        self.register_buffer("theta_lo_vec", lo)
        self.register_buffer("theta_hi_vec", hi)

        if self.tf_group_size < 1:
            raise ValueError(f"tf_group_size must be >= 1, got {tf_group_size}")
        if self.tf_group_size > self.context_len:
            raise ValueError(
                f"tf_group_size={tf_group_size} > context_len={context_len}; "
                "group cannot exceed the attention window."
            )
        if self.ar_gap < 0:
            raise ValueError(f"ar_gap must be >= 0, got {ar_gap}")

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
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=max(1, num_layers),
            enable_nested_tensor=False,
        )

        head_out = self.theta_dim + self.P if use_basal else self.theta_dim
        self.head = nn.Linear(hidden, head_out)

        self.register_buffer(
            "_causal_mask",
            nn.Transformer.generate_square_subsequent_mask(self.context_len),
            persistent=False,
        )

        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.pos_embed.weight, std=0.02)

        if u_to_y_jump.shape != (self.U, self.P):
            raise ValueError(
                f"u_to_y_jump must be (U,P)=({self.U},{self.P}), "
                f"got {tuple(u_to_y_jump.shape)}"
            )
        self.register_buffer("u_to_y_jump", u_to_y_jump.float(), persistent=True)

    # --------------------------------------------------------------- utils

    def _prune_history(self, feat_history: List[torch.Tensor]) -> None:
        """Detach the feature one slot behind the window; matches OdeTransformer."""
        cutoff = len(feat_history) - self.context_len - 1
        if 0 <= cutoff < len(feat_history):
            t = feat_history[cutoff]
            if t.requires_grad:
                feat_history[cutoff] = t.detach()

    def _encode_last_n(
        self,
        feat_history: List[torch.Tensor],
        n_out: int,
    ) -> torch.Tensor:
        """
        Run the Transformer over the last context_len features and return
        raw head outputs for the last `n_out` positions. Shape: (B, n_out, head_out).
        """
        start = max(0, len(feat_history) - self.context_len)
        window = feat_history[start:]
        W = len(window)

        seq = torch.stack(window, dim=1)  # (B, W, hidden)
        pos_ids = torch.arange(W, device=seq.device, dtype=torch.long)
        seq = seq + self.pos_embed(pos_ids).unsqueeze(0)

        mask = self._causal_mask[:W, :W].to(dtype=seq.dtype)
        out = self.transformer(seq, mask=mask, is_causal=True)  # (B, W, hidden)

        return self.head(out[:, -n_out:, :])  # (B, n_out, head_out)

    # --------------------------------------------------------------- forward

    def forward(
        self,
        y0: torch.Tensor,                      # (B, P)
        u_seq: torch.Tensor,                   # (B, K, U)
        dt_seq: torch.Tensor,                  # (B, K)
        obs_idx: torch.Tensor,                 # unused here; kept for API parity
        y_seq: Optional[torch.Tensor] = None,  # (B, K, P) required for TF
        teacher_forcing: bool = True,
        tf_every: int = 50,                    # ignored in grouped mode
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, K, _ = u_seq.shape
        device, dtype = y0.device, y0.dtype

        if lengths is not None:
            lengths = lengths.to(device=device, dtype=torch.long).clamp(min=0, max=K)

        y_out    = torch.empty(B, K, self.P,         device=device, dtype=dtype)
        th_out   = torch.empty(B, K, self.theta_dim, device=device, dtype=dtype)
        beta_out = torch.zeros(B, K, self.P,         device=device, dtype=dtype)

        do_grouped = bool(teacher_forcing) and (y_seq is not None)

        feat_history: List[torch.Tensor] = []
        y_prev = y0

        if not do_grouped:
            # Pure sequential AR — same behaviour as OdeTransformer @ TF=False.
            y_prev = self._ar_segment(
                k_start=0, n_steps=K,
                y_prev=y_prev, u_seq=u_seq, dt_seq=dt_seq,
                y_out=y_out, th_out=th_out, beta_out=beta_out,
                feat_history=feat_history, lengths=lengths,
            )
            return y_out, th_out, beta_out

        k = 0
        while k < K:
            # ---- TF group
            g = min(self.tf_group_size, K - k)
            y_prev = self._tf_group(
                k_start=k, g=g,
                y0=y0, y_prev_in=y_prev,
                u_seq=u_seq, dt_seq=dt_seq, y_seq=y_seq, lengths=lengths,  # type: ignore[arg-type]
                y_out=y_out, th_out=th_out, beta_out=beta_out,
                feat_history=feat_history,
            )
            k += g
            if k >= K:
                break

            # ---- AR gap
            a = min(self.ar_gap, K - k)
            if a > 0:
                y_prev = self._ar_segment(
                    k_start=k, n_steps=a,
                    y_prev=y_prev, u_seq=u_seq, dt_seq=dt_seq,
                    y_out=y_out, th_out=th_out, beta_out=beta_out,
                    feat_history=feat_history, lengths=lengths,
                )
                k += a

        return y_out, th_out, beta_out

    # --------------------------------------------------------------- phases

    def _tf_group(
        self,
        *,
        k_start: int,
        g: int,
        y0: torch.Tensor,
        y_prev_in: torch.Tensor,
        u_seq: torch.Tensor,
        dt_seq: torch.Tensor,
        y_seq: torch.Tensor,
        lengths: Optional[torch.Tensor],
        y_out: torch.Tensor,
        th_out: torch.Tensor,
        beta_out: torch.Tensor,
        feat_history: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Process g consecutive steps from k_start..k_start+g-1 in parallel.
        Uses y_seq[k-1] (or y0 at k=0) as the previous-state for every step.

        Returns the model's predicted y at step k_start+g-1 (to seed the
        next AR segment).
        """
        B = u_seq.shape[0]
        P, U = self.P, self.U
        dtype = u_seq.dtype

        if lengths is None:
            active_mask = torch.ones(B, g, device=u_seq.device, dtype=torch.bool)
        else:
            k_pos = k_start + torch.arange(g, device=u_seq.device, dtype=torch.long)
            active_mask = lengths.unsqueeze(1) > k_pos.unsqueeze(0)

        # Ground-truth "previous state" per step in the group.  (B, g, P)
        if k_start == 0:
            y_in_group = torch.cat(
                [y0.unsqueeze(1), y_seq[:, : g - 1, :]], dim=1
            ) if g > 1 else y0.unsqueeze(1)
        else:
            y_in_group = y_seq[:, k_start - 1 : k_start + g - 1, :]
        y_in_group = y_in_group.detach().to(dtype=dtype)

        u_group  = u_seq[:,  k_start : k_start + g, :]      # (B, g, U)
        dt_group = dt_seq[:, k_start : k_start + g]         # (B, g)

        # Single batched lift: (B, g, U+P) -> (B, g, hidden)
        feat_group = self.lift(torch.cat([u_group, y_in_group], dim=-1))

        # Extend history with the g new features (in temporal order).
        # Pruning is safe: cutoff is always strictly behind k_start since
        # g <= context_len (enforced in __init__).
        for i in range(g):
            feat_history.append(feat_group[:, i, :])
            self._prune_history(feat_history)

        # One Transformer pass over the current window; take last g outputs.
        raw = self._encode_last_n(feat_history, n_out=g)  # (B, g, head_out)

        # Batched RK4: fold the g dim into batch so each step integrates
        # independently from its own detached ground-truth starting state.
        # For variable-length batches we run RK4 only on active flattened rows.
        Bg = B * g
        active_flat = active_mask.reshape(Bg)
        y_in_flat = y_in_group.reshape(Bg, P)
        u_flat    = u_group.reshape(Bg, U)
        dt_flat   = dt_group.reshape(Bg)
        y_flat_out = y_in_flat.clone()
        theta_flat_out = torch.zeros(Bg, self.theta_dim, device=y_in_flat.device, dtype=y_in_flat.dtype)

        if self.use_basal:
            raw_theta_flat = raw[:, :, : self.theta_dim].reshape(Bg, self.theta_dim)
            raw_beta_flat = raw[:, :, self.theta_dim :].reshape(Bg, P)
            beta_flat_out = torch.zeros(Bg, P, device=y_in_flat.device, dtype=y_in_flat.dtype)

            if active_flat.any():
                theta_active = log_gamma(raw_theta_flat[active_flat], self.theta_lo_vec, self.theta_hi_vec)
                beta_active = raw_beta_flat[active_flat] * (
                    y_in_flat[active_flat] / (y_in_flat[active_flat] + 1.0)
                )
                y_active = y_in_flat[active_flat] + (u_flat[active_flat] @ self.u_to_y_jump)
                y_active = self._rk4_substeps_basal(y_active, dt_flat[active_flat], theta_active, beta_active)
                y_flat_out[active_flat] = y_active.to(dtype=y_flat_out.dtype)
                theta_flat_out[active_flat] = theta_active.to(dtype=theta_flat_out.dtype)
                beta_flat_out[active_flat] = beta_active.to(dtype=beta_flat_out.dtype)

            beta_out[:, k_start : k_start + g, :] = beta_flat_out.reshape(B, g, P)
        else:
            raw_theta_flat = raw.reshape(Bg, self.theta_dim)
            if active_flat.any():
                theta_active = log_gamma(raw_theta_flat[active_flat], self.theta_lo_vec, self.theta_hi_vec)
                y_active = y_in_flat[active_flat] + (u_flat[active_flat] @ self.u_to_y_jump)
                y_active = self._rk4_substeps(y_active, dt_flat[active_flat], theta_active)
                y_flat_out[active_flat] = y_active.to(dtype=y_flat_out.dtype)
                theta_flat_out[active_flat] = theta_active.to(dtype=theta_flat_out.dtype)

        y_group     = y_flat_out.reshape(B, g, P)
        theta_group = theta_flat_out.reshape(B, g, self.theta_dim)

        y_out[:,  k_start : k_start + g, :] = y_group
        th_out[:, k_start : k_start + g, :] = theta_group

        # Seed the next AR segment with the model's last prediction.
        last_active = active_mask[:, -1]
        if last_active.any():
            y_next = y_prev_in.clone()
            y_next[last_active] = y_group[last_active, -1, :]
            return y_next
        return y_prev_in

    def _ar_segment(
        self,
        *,
        k_start: int,
        n_steps: int,
        y_prev: torch.Tensor,
        u_seq: torch.Tensor,
        dt_seq: torch.Tensor,
        y_out: torch.Tensor,
        th_out: torch.Tensor,
        beta_out: torch.Tensor,
        feat_history: List[torch.Tensor],
        lengths: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Sequential autoregressive segment of n_steps starting at k_start.
        Mirrors the per-step loop of OdeTransformer: lift uses detached
        y_prev; RK4 uses un-detached y_prev so BPTT flows through the
        integration chain.
        """
        for offset in range(n_steps):
            k = k_start + offset
            u_k  = u_seq[:, k, :]
            dt_k = dt_seq[:, k]

            active = None if lengths is None else (lengths > k)
            if active is not None and not active.any():
                y_out[:, k, :] = y_prev
                th_out[:, k, :] = 0.0
                continue

            y_in = y_prev.detach()

            feat = self.lift(torch.cat([u_k, y_in], dim=-1))  # (B, hidden)
            feat_history.append(feat)
            self._prune_history(feat_history)

            raw = self._encode_last_n(feat_history, n_out=1).squeeze(1)  # (B, head_out)

            y_next = y_prev.clone()
            theta_store = torch.zeros(
                y_prev.shape[0], self.theta_dim, device=y_prev.device, dtype=y_prev.dtype
            )

            if self.use_basal:
                theta_raw = log_gamma(raw[:, : self.theta_dim], self.theta_lo_vec, self.theta_hi_vec)
                if active is None:
                    beta_k = raw[:, self.theta_dim :] * (y_prev / (y_prev + 1.0))
                    beta_out[:, k, :] = beta_k
                    y = y_prev + (u_k @ self.u_to_y_jump)
                    y = self._rk4_substeps_basal(y, dt_k, theta_raw, beta_k)
                    y_next = y
                    theta_store = theta_raw
                else:
                    beta_k = torch.zeros_like(y_prev)
                    beta_k[active] = raw[active, self.theta_dim :] * (
                        y_prev[active] / (y_prev[active] + 1.0)
                    )
                    beta_out[:, k, :] = beta_k
                    y_active = y_prev[active] + (u_k[active] @ self.u_to_y_jump)
                    y_active = self._rk4_substeps_basal(
                        y_active, dt_k[active], theta_raw[active], beta_k[active]
                    )
                    y_next[active] = y_active.to(dtype=y_next.dtype)
                    theta_store[active] = theta_raw[active].to(dtype=theta_store.dtype)
            else:
                theta_raw = log_gamma(raw, self.theta_lo_vec, self.theta_hi_vec)
                if active is None:
                    y = y_prev + (u_k @ self.u_to_y_jump)
                    y = self._rk4_substeps(y, dt_k, theta_raw)
                    y_next = y
                    theta_store = theta_raw
                else:
                    y_active = y_prev[active] + (u_k[active] @ self.u_to_y_jump)
                    y_active = self._rk4_substeps(y_active, dt_k[active], theta_raw[active])
                    y_next[active] = y_active.to(dtype=y_next.dtype)
                    theta_store[active] = theta_raw[active].to(dtype=theta_store.dtype)

            y_out[:, k, :]  = y_next
            th_out[:, k, :] = theta_store
            y_prev = y_next

        return y_prev

    # --------------------------------------------------------------- RK4

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