# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 00:19:55 2026

@author: bobva
"""

def Training_loop(self, *, epochs=200, lr=1e-3,
                     hidden=128, num_layers=1,
                     model_cls=None, batch=300, normalize=False,decay = 0.0085,
                     save_path: str | Path = None, teacher_forcing = True,
                     val_frac: float = 0.15):   # ← NEW: hold-out from training as validation
        assert hasattr(self, "train_idx"), "Call make_train_test_split() first."
    
        # --- split current train_idx into train/val (shuffle already done earlier) ---
        if val_frac > 0.0 and len(self.train_idx) > 1:
            n_val = max(1, int(len(self.train_idx) * val_frac))
            val_idx   = self.train_idx[:n_val]
            train_idx = self.train_idx[n_val:]
        else:
            val_idx, train_idx = [], list(self.train_idx)
    
        train_ds = self._make_subset_ds(train_idx)
        train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                                  num_workers=0, collate_fn=collate,
                                  pin_memory=True)
    
        val_loader = None
        if len(val_idx) > 0:
            val_ds = self._make_subset_ds(val_idx)
            val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False,
                                    num_workers=0, collate_fn=collate,
                                    pin_memory=True)
    
        model = model_cls(len(self.input_cols) - 1,
                          hidden = hidden, num_layers = num_layers).to(self.device)
        
        try:
            model = torch.jit.script(model)
            print('The model compiled successfully')
        except AttributeError:
            print('The model did not compile please check')
    
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)
        best_val = float('inf')
        self.best_model = copy.deepcopy(model)
        for ep in range(1, epochs + 1):
            if ep == 250:
                teacher_forcing = False
            # ------------------------- TRAIN -------------------------
            model.train()
            train_total = 0.0
            for x0, u_seq, dna_seq, dt_seq, y_seq, lengths in train_loader:
                x0, u_seq, dna_seq, dt_seq, y_seq, lengths = (
                    x0.to(self.device), u_seq.to(self.device),
                    dna_seq.to(self.device), dt_seq.to(self.device),
                    y_seq.to(self.device), lengths.to(self.device)
                )
    
                opt.zero_grad()
                pred, params = model(x0, u_seq, dna_seq, dt_seq, y_seq, teacher_forcing = teacher_forcing)
    

                y_clamped    = y_seq.clamp_min(1.0)
                pred_clamped = pred.clamp_min(1.0)
                log_y, log_pred = torch.log1p(y_clamped), torch.log1p(pred_clamped)
                extra_mse_chan    = ((pred_last - y_last) ** 2).mean(0, keepdim=True)
                err2     = (log_pred_all - log_y_all).pow(2) * w
                mse_chan = err2.sum((0, 1)) / w.sum((0, 1)).clamp_min(1)    
                mse_chan = torch.cat([mse_chan,extra_mse_chan])                        
                
                weighted = self.loss_weight * mse_chan
                mask = self.loss_weight.ne(0)# boolean mask of active terms
                den  = mask.sum().clamp_min(1).to(weighted.dtype)  # avoid /0
                loss = weighted[mask].sum() / den
                
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                opt.step()
                train_total += float(loss.item())
    
        # repeat for validation.....

                     
@torch.jit.script
def ODE_MODEL_ANALYTICAL(   
    m_prev:  torch.Tensor,   # (B,1) immature m
    mm_prev: torch.Tensor,   # (B,1) mature mRNA (bright)
    p_prev:  torch.Tensor,   # (B,1) immature protein
    pm_prev: torch.Tensor,   # (B,1) mature protein
    R_prev:  torch.Tensor,   # (B,1)
    dt_k:    torch.Tensor,   # (B,1)
    dna_cum_total: torch.Tensor,  # (B,1) constant within step
    lam_k:   torch.Tensor,   # (B,1)
    VTXmax:  torch.Tensor,   # (B,1)
    kdm:     torch.Tensor,   # (B,1)
    VTLmax:  torch.Tensor,   # (B,1)
    kmt:     torch.Tensor,   # (B,1) protein maturation rate
    kmatm:   torch.Tensor,   # (B,1) mRNA maturation rate (m -> mm)
    eps: float = 1e-9
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    One analytic IVTT step with mRNA maturation.
    Returns: (m_curr, mm_curr, p_curr, pm_curr, R_curr), each (B,1).
    """
    # --- resource decay (use end-of-step as effective within step) ---
    rho    = torch.exp(-lam_k * dt_k)     # (B,1)
    R_curr = R_prev * rho

    # --- effective catalytic rates and source S ----------------------
    VTX_eff = R_curr * VTXmax
    VTL_eff = R_curr * VTLmax
    S       = VTX_eff * dna_cum_total     # (B,1), constant within step

    # --- m (immature) ------------------------------------------------
    alpha  = (kdm + kmatm).clamp_min(eps) # (B,1)
    m_inf  = S / alpha
    exp_a  = torch.exp(-alpha * dt_k)
    m_curr = torch.clamp_min(m_inf + (m_prev - m_inf) * exp_a, 0.0)

    # --- mm (matured mRNA) ------------------------------------------
    exp_d  = torch.exp(-kdm   * dt_k)     # e^{-kdm Δt}
    exp_mr = torch.exp(-kmatm * dt_k)     # e^{-kmatm Δt}
    term1  = mm_prev * exp_d
    term2  = m_inf * (kmatm / (kdm + eps)) * (1.0 - exp_d)
    term3  = (m_prev - m_inf) * exp_d * (1.0 - exp_mr)
    mm_curr = torch.clamp_min(term1 + term2 + term3, 0.0)

    # --- total mRNA M for protein formulas --------------------------
    M_prev = m_prev + mm_prev
    M_inf  = S / (kdm + eps)
    exp_M  = exp_d

    # --- protein p ---------------------------------------------------
    eta    = torch.exp(-kmt * dt_k)       # e^{-kmt Δt}
    delta  = (kdm - kmt)
    same   = torch.abs(delta) < 1e-6

    int_M_eq  = (M_inf * (1.0 - eta) / (kmt + eps)
                 + (M_prev - M_inf) * dt_k * eta)
    int_M_gen = (M_inf * (1.0 - eta) / (kmt + eps)
                 + (M_prev - M_inf) * (eta - exp_M) / (delta + eps))
    int_M_conv = torch.where(same, int_M_eq, int_M_gen)

    p_curr  = torch.clamp_min(p_prev * eta + VTL_eff * int_M_conv, 0.0)

    # --- mature protein p* (conservation) ---------------------------
    int_M_total = (M_inf * dt_k
                  + (M_prev - M_inf) / (kdm + eps) * (1.0 - exp_M))
    pm_curr = torch.clamp_min(pm_prev + (p_prev - p_curr) + VTL_eff * int_M_total, 0.0)

    return m_curr, mm_curr, p_curr, pm_curr, R_curr



class RNN_model(nn.Module):
    """
    Closed-loop model with Δt-aware GRU latent.
    Input at step k: [u_k, sqrt(mm_{k-1}), sqrt(p*_{k-1})] -> lift(feat_k)
    -> (decay hidden by exp(-λ Δt)) -> GRUCell -> z_k
    -> θ_k = [lam, VTX_max, kdm, VTL_max, kmt, kmatm] (bounded via gamma)
    -> ivtt_step_mRNA_maturation() -> next states

    Returns:
        out    : (B,K,3)  channels [mm, p, p*]
        params : (B,K,7)  [VTX_max, kdm, VTL_max, kmt, kmatm, R, lam]
    """
    def __init__(
        self,
        in_u: int,
        hidden: int = 128,  # GRU hidden size
        d_h: int = 32,      # lift dimension for h_k
        dropout: float = 0.01
    ):
        super().__init__()
        self.hidden = hidden
        self.d_h = d_h
        self.in_dim = in_u + 2  # [u_k, sqrt(mm_prev), sqrt(pm_prev)]

        # Small feature lift
        self.lift = nn.Sequential(
            nn.Linear(self.in_dim, d_h),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        # Δt-aware decay on hidden is explicit (per-neuron λ ≥ 0 via softplus)
        # Init near small positive so exp(-λ Δt) ≈ 1 early on
        self.lambda_raw = nn.Parameter(torch.full((hidden,), -2.0))

        # GRU over lifted features; hidden is the latent
        self.gru = nn.GRUCell(input_size=d_h, hidden_size=hidden)

        # Head to raw params
        self.head = nn.Linear(hidden, 6)  # [lam, VTX_mag, kdm, VTL_mag, kmt, kmatm]

        # Inits
        for m in self.lift:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        x0: torch.Tensor,          # (B,2): [m_obs0, p*0]
        u_seq: torch.Tensor,       # (B,K,U)
        dna_raw: torch.Tensor,     # (B,K,1)
        dt_seq: torch.Tensor,      # (B,K)
        y_seq: torch.Tensor,       # (B,K,C) with mm at [:, :, 0], p* at [:, :, 2] or [:, :, 1]
        teacher_forcing: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, K, U = u_seq.shape
        dev = u_seq.device
        dtype = u_seq.dtype

        # outputs
        mm_out  = torch.empty(B, K, 1, device=dev, dtype=dtype)
        p_out   = torch.empty_like(mm_out)
        pm_out  = torch.empty_like(mm_out)

        # diagnostics
        VTX_seq   = torch.empty_like(mm_out)
        kdm_seq   = torch.empty_like(mm_out)
        VTL_seq   = torch.empty_like(mm_out)
        kmt_seq   = torch.empty_like(mm_out)
        kmatm_seq = torch.empty_like(mm_out)
        R_seq     = torch.empty_like(mm_out)
        lam_seq   = torch.empty_like(mm_out)

        # initial biological states
        mm_prev = x0[:, 0:1] + 0.01
        m_prev  = torch.zeros_like(mm_prev) + 0.01
        pm_prev = x0[:, 1:2] + 0.01
        p_prev  = torch.zeros_like(mm_prev) + 0.01
        R_prev  = torch.ones_like(mm_prev)

        # cumulative DNA
        dna_cum_total = torch.cumsum(dna_raw, dim=1)[:, -1, :]  # (B,1)

        # latent state (GRU hidden)
        z = torch.zeros(B, self.hidden, device=dev, dtype=dtype)

        # positive per-neuron decay rates λ_i
        lambda_pos = F.softplus(self.lambda_raw)  # (hidden,)

        # ground-truth slices for occasional teacher forcing
        y_mm = y_seq[:, :, 0:1]  # (B,K,1)
        y_pm = y_seq[:, :, 2:3] if y_seq.shape[-1] >= 3 else y_seq[:, :, 1:2]

        """you can see I solve the ODE in the forward loop!"""
        for k in range(K):
            dt_k = dt_seq[:, k:k+1]        # (B,1)
            u_k  = u_seq[:, k]             # (B,U)

            # teacher forcing cadence (safe at k=0)
            if teacher_forcing and k > 0 and (k % 50 == 0):
                det_mm = y_mm[:, k-1, :].detach()
                det_pm = y_pm[:, k-1, :].detach()
            else:
                det_mm = mm_prev.detach()
                det_pm = pm_prev.detach()

            # features
            feat_k = torch.cat([
                torch.sqrt(u_k),                                     # (B,U)
                torch.sqrt(det_mm).clamp_min(0),         # (B,1)
                torch.sqrt(det_pm).clamp_min(0),         # (B,1)
            ], dim=-1)                                   # (B, U+2)

            h_k = self.lift(feat_k)                      # (B, d_h)

            # ---- Δt-aware GRU update ----
            # decay hidden first, then apply GRUCell
            decay = torch.exp(-lambda_pos.unsqueeze(0) * dt_k)  # (B, hidden)
            z = self.gru(h_k, z)                        # (B, hidden)

            # ---- head -> raw params ----
            raw = self.head(z)
            lam_raw, VTX_mag, kdm_raw, VTL_mag, kmt_raw, kmatm_raw = raw.split(1, dim=-1)

            # ---- bounded θ via gamma ----
            lam_k  = gamma(lam_raw,   1e-6, 5e-4)
            VTXmax = gamma(VTX_mag,   3e-5, 0.12)
            VTLmax = gamma(VTL_mag,   3e-5, 0.08)
            kdm_k  = gamma(kdm_raw,   1e-5, 1e-2)
            kmt_k  = gamma(kmt_raw,   1e-5, 3.5e-4)
            kmatm_k= gamma(kmatm_raw, 5e-5, 3.5e-3)

            # ---- one scripted ODE step ---- this is the ODE solving step!
            m_curr, mm_curr, p_curr, pm_curr, R_curr = ivtt_step_mRNA_maturation(
                m_prev, mm_prev, p_prev, pm_prev, R_prev,
                dt_k, dna_cum_total,
                lam_k, VTXmax, kdm_k, VTLmax, kmt_k, kmatm_k
            )

            # store
            mm_out[:, k:k+1, :]   = mm_curr.unsqueeze(1)
            p_out[:,  k:k+1, :]   = p_curr.unsqueeze(1)
            pm_out[:, k:k+1, :]   = pm_curr.unsqueeze(1)
            VTX_seq[:, k:k+1, :]  = VTXmax.unsqueeze(1)
            kdm_seq[:, k:k+1, :]  = kdm_k.unsqueeze(1)
            VTL_seq[:, k:k+1, :]  = VTLmax.unsqueeze(1)
            kmt_seq[:, k:k+1, :]  = kmt_k.unsqueeze(1)
            kmatm_seq[:,k:k+1, :] = kmatm_k.unsqueeze(1)
            R_seq[:,   k:k+1, :]  = R_curr.unsqueeze(1)
            lam_seq[:, k:k+1, :]  = lam_k.unsqueeze(1)

            # roll biological states
            m_prev, mm_prev, p_prev, pm_prev, R_prev = m_curr, mm_curr, p_curr, pm_curr, R_curr

        out    = torch.cat([mm_out, p_out, pm_out], dim=-1)  # (B,K,3)
        params = torch.cat([VTX_seq, kdm_seq, VTL_seq, kmt_seq, kmatm_seq, R_seq, lam_seq], dim=-1)  # (B,K,7)
        return (out, params) if self.training else (out, params.detach())

