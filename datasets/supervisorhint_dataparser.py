

from __future__ import annotations
from utils import *
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import os
import torch
import copy
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from typing import Mapping, Hashable, Sequence, Dict, List, Tuple
from sklearn.preprocessing import StandardScaler,RobustScaler,MinMaxScaler
from sklearn.metrics import r2_score

# -----------------------------------------------------------------------------
# 1) Vectorized Dataset + Caching
# -----------------------------------------------------------------------------
class Datastruct(Dataset):
    def __init__(
        self,
        x0_list : List[torch.Tensor],
        u_list  : List[torch.Tensor],
        dna_raw : List[torch.Tensor],   #                       
        dt_list : List[torch.Tensor],
        y_list  : List[torch.Tensor],
    ):
        self.x0_list  = x0_list
        self.u_list   = u_list
        self.dna_raw  = dna_raw
        self.dt_list  = dt_list
        self.y_list   = y_list

    def __len__(self): 
         return len(self.x0_list)

    def __getitem__(self, idx):
        return (
            self.x0_list[idx],
            self.u_list[idx],
            self.dna_raw[idx],
            self.dt_list[idx],
            self.y_list[idx],
        )


# ---------- compute μ/σ for observed channels on TRAIN only ----------
def _fit_obs_mu_std(y_list, transform="", eps=1e-8, device="cpu"):
    """
    y_list: list of (K,3) tensors with channels [mRNA(mm), p(unused), protein(pm)]
    train_idx: iterable of indices to include (training split)
    Returns four Python floats: mu_mm, std_mm, mu_pm, std_pm (on transformed data)
    """
    mm_vals, pm_vals = [], []
    for i in range(len(y_list)):
        y = y_list[i].to(device)      # (K,3)
        mm = y[:, 0].clamp_min(0.0)   # mRNA (observed)
        pm = y[:, 2].clamp_min(0.0)   # mature protein (observed)
        mm_vals.append(mm)
        pm_vals.append(pm)

    mm_all = torch.cat(mm_vals, dim=0)  # (N,)
    pm_all = torch.cat(pm_vals, dim=0)  # (N,)

    # variance-stabilizing transform
    if transform == "sqrt":
        mm_t = torch.sqrt(mm_all)
        pm_t = torch.sqrt(pm_all)
    elif transform == "log1p":
        mm_t = torch.log1p(mm_all)
        pm_t = torch.log1p(pm_all)
    elif transform == "anscombe":
        mm_t = 2.0 * torch.sqrt(mm_all + 0.375)
        pm_t = 2.0 * torch.sqrt(pm_all + 0.375)
    else:
        mm_t, pm_t = mm_all, pm_all

    # drop non-finite just in case
    mm_t = mm_t[torch.isfinite(mm_t)]
    pm_t = pm_t[torch.isfinite(pm_t)]

    mu_mm  = float(mm_t.mean())
    std_mm = float(mm_t.std(unbiased=False).clamp_min(eps))
    mu_pm  = float(pm_t.mean())
    std_pm = float(pm_t.std(unbiased=False).clamp_min(eps))
    return mu_mm, std_mm, mu_pm, std_pm

def smooth_gaussian_preserve_total(
    u_seq: torch.Tensor,                  # (B, K, U)
    sigma: float = 3.0,                   # std in steps
    radius: int | None = None,            # kernel radius; default ~ 3σ
    lengths: torch.Tensor | None = None,  # (B,) valid lengths; optional
) -> torch.Tensor:
    """
    Symmetric Gaussian smoothing in time (left+right neighbors), per-channel,
    preserving *per-(batch,channel)* total volume over valid steps.

    Returns: smoothed tensor of shape (B, K, U).
    """
    B, K, U = u_seq.shape
    device, dtype = u_seq.device, u_seq.dtype
    if radius is None:
        radius = max(1, int(round(3.0 * sigma)))  # ~99.7% mass

    # 1) Build normalized symmetric Gaussian kernel
    t = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / (k.sum() + 1e-12)                      # sum = 1
    weight = k.view(1, 1, -1).repeat(U, 1, 1)      # (U,1,L) for grouped conv

    # 2) Prepare data as (B, U, K); optionally extend tail to last valid frame
    x = u_seq.transpose(1, 2).contiguous()         # (B, U, K)

    if lengths is not None:
        # make a mask over time for valid steps
        tgrid = torch.arange(K, device=device).unsqueeze(0)              # (1,K)
        valid_mask = (tgrid < lengths.view(-1, 1)).unsqueeze(1).to(dtype)  # (B,1,K)

        # repeat last valid frame across the padded tail so blur doesn't see zeros
        for b in range(B):
            L = int(lengths[b].item())
            if L < 1: continue
            if L < K:
                last = x[b, :, L-1:L]                                    # (U,1)
                x[b, :, L:] = last                                       # extend
    else:
        valid_mask = torch.ones(B, 1, K, device=device, dtype=dtype)

    # 3) Symmetric blur with reflect padding (no future leakage beyond K)
    xpad = F.pad(x, (radius, radius), mode='reflect')   # (B,U,K+2r)
    y = F.conv1d(xpad, weight, padding=0, groups=U)     # (B,U,K), per-channel

    # 4) EXACT total preservation per (B,U) over valid steps
    s_orig   = (x * valid_mask).sum(dim=2)              # (B,U)
    s_smooth = (y * valid_mask).sum(dim=2) + 1e-12      # (B,U)
    scale = (s_orig / s_smooth).unsqueeze(2)            # (B,U,1)
    y = y * scale

    # 5) Zero-out padded region (if any) to keep shapes consistent
    y = y * valid_mask + 0.0 * (1.0 - valid_mask)

    return y.transpose(1, 2).contiguous()               # (B,K,U)

# -----------------------------------------------------------------------------
# 1) Model/ML infrastructure
# -----------------------------------------------------------------------------
from pathlib import Path
import random, numpy as np, torch
from torch.utils.data import DataLoader
from torch import nn
from sklearn.preprocessing import StandardScaler

class NeuralSpline:
    def __init__(
        self,
        path: str | Path = None,
        inputs=None,
        outputs=None,
        time_label: str = "Time_seconds",
        varlist=('DNA c',),
        rescale_inputs: bool = True, w = 1,
    ):
        
        # ---------- load  ------------------------------------------------
        if inputs is None or outputs is None:
            from parse_IVTT_data import load_parsed_io
            self.inputs_df, self.outputs_df, self.metadata_dict = \
                load_parsed_io(f"/home/bob-van-sluijs/Desktop/{path}/")
        else:
            self.inputs_df, self.outputs_df = inputs, output
        self.w = w

        # ---------- diffs & intervals per run ---------------------------
        input_list, interval_list, dna_raw_list = [], [], []
        for tag, df in self.inputs_df.items():
            cols = sorted([c for c in df.columns if c != time_label])
            diffs_raw = df[cols].to_numpy()[0:-1]          # (K,U)
            interval_list.append(list(zip(df[time_label][:-1],
                                          df[time_label][0:])))
            dna_cum = np.cumsum(diffs_raw[:, [cols.index('DNA c')]])

            dna_raw_list.append(diffs_raw[:, [cols.index('DNA c')]])

            diffs_raw = np.delete(diffs_raw, cols.index('DNA c'), 1)    # this is the toal concentration of DNA we need it for the mechanistic model but NOT for the AI as an input, we can leave it out of the model
            input_list.append(diffs_raw)

        
        # ---------- optional log‑standard‑scale (excluding DNA) ----------        
        rescale_inputs = rescale_inputs
        if rescale_inputs:
            all_arr      = np.concatenate(input_list, 0)
            self.scaler       = MinMaxScaler().fit(all_arr)
            for i, arr in enumerate(input_list): #this is where you can scale the data to fit a format e.g. min max, log etc.
                input_list[i] = self.scaler.transform(arr)
                print('rescale',i, input_list[i] , arr)
 
        # ---------- build tensor lists (3‑state) -------------------------
        self.x0_list, self.u_list, self.dna_list, self.dt_list, self.y_list = \
            [], [], [], [], []

        self.input_cols = sorted({c for d in self.inputs_df.values()
                                  for c in d.columns if c != time_label})
        self.outputs_list = [
            self.outputs_df[tag].sort_values(time_label).reset_index(drop=True)
            for tag in self.inputs_df.keys()
            ]
        
        for (intervals, diffs_scaled), dna_raw, df_out in zip(
                zip(interval_list, input_list), dna_raw_list, self.outputs_list):

            times = df_out[time_label].to_numpy()
            bro   = df_out["Broccoli [RFU]"].to_numpy() # mRNA, here the light intensity matches concentration exactly
            mch   = df_out["mCherry [RFU]"].to_numpy()/2    # mature protein this is divided by to because that is how the observed light is converted to concentration

            dt_seq  = torch.tensor(times[1:] - times[:-1], dtype=torch.float32)
            u_seq   = torch.tensor(diffs_scaled,          dtype=torch.float32)
            dna_seq = torch.tensor(dna_raw,               dtype=torch.float32)
            self.dna_fixed  = dna_raw
            y_arr = np.stack([bro[:-1],
                              np.zeros_like(bro[:-1]),      # p unmeasured
                              mch[:-1]], axis=1)
            y_seq = torch.tensor(y_arr, dtype=torch.float32)
            
            # if mch[-1] < 7000: #Here you could filter out datapoints if need be, i.e. if the final yield is over xxx then...
            self.x0_list.append(y_seq[0])   # (3,)
            self.u_list.append(u_seq)       # (K,U‑DNA)
            self.dna_list.append(dna_seq)   # (K,1)
            self.dt_list.append(dt_seq)     # (K,)
            self.y_list.append(y_seq)       # (K,3)
            
        self.ctrl_idx = [self.input_cols.index(v) for v in varlist]
        self.device   = torch.device("cuda")
        self.make_train_test_split()

        
    def make_train_test_split(self, *, test_frac=0.125, val_frac=0.125):
        N = len(self.x0_list)
        self.choice = random.choice(range(100))
        rng = random.Random(self.choice)
        idx = list(range(N))
        rng.shuffle(idx)
    
        n_test = int(N * test_frac)
        self.test_idx = idx[:n_test]
        rest = idx[n_test:]
    
        n_val = int(len(rest) * val_frac)
        self.val_idx = rest[:n_val] if n_val > 0 else []
        self.train_idx = rest[n_val:]
    
        msg = f"Split → {len(self.train_idx)} train"
        if self.val_idx: msg += f" | {len(self.val_idx)} val"
        msg += f" | {len(self.test_idx)} test"
        print(self.test_idx, self.val_idx, self.train_idx)
        
        
        
        
path  = 'Data parsed pruned'       
listset = [
    # (2000,0.008,150,300,4,True,True),
    (2000,0.0018,200,200,2,True,True)]
   
for i in range(0,1,1):
     epoch, decay, hidden, batch, num_layers, rescale, normalize = listset[i]
     Data = NeuralSpline(path = path,rescale_inputs = rescale)
     
     
     
#below you will find the neural net
# # --------------- Closed-loop stacked-GRU with scripted step ------------------
# class GRU_model_latent_decay_simple_fb_mat(nn.Module):
#     """
#     Closed-loop, stacked GRUCells with state feedback:
#       input at step k: [u_k, dna_k, dt_k, log1p(mm_{k-1}), log1p(p*_{k-1})]
#       -> stacked GRUCells
#       -> θ_k = [λ, VTX_max, kdm, VTL_max, kmt, kmatm]  (bounded)
#       -> ivtt_step_mRNA_maturation() -> states for next step

#     Returns:
#       out    : (B,K,3) with channels [mm, p, p*]
#       params : (B,K,7) with [VTX_max, kdm, VTL_max, kmt, kmatm, R, lam]
#     """
#     def __init__(self,
#                  in_u: int,
#                  hidden: int = 128,
#                  num_layers: int = 2,
#                  dropout: float = 0.2 ):
#         super().__init__()
#         assert num_layers >= 1
#         self.hidden = hidden
#         self.num_layers = num_layers

#         in_dim = in_u+2
#         # stacked GRUCells
#         cells = [nn.GRUCell(in_dim, hidden)]
#         cells += [nn.GRUCell(hidden, hidden) for _ in range(num_layers - 1)]
#         self.cells = nn.ModuleList(cells)
#         self.dropout = nn.Dropout(dropout)
        
#         self.bottle = nn.Sequential(nn.Linear(hidden,  120), nn.SiLU(),
#                                     nn.Linear(120,  40), nn.SiLU(),   
#                                     )
#         self.head = nn.Linear(40, 6)  # [lam, VTX_mag, kdm, VTL_mag, kmt, kmatm]
                

#         for c in self.cells:
#             nn.init.orthogonal_(c.weight_hh)
#             nn.init.xavier_uniform_(c.weight_ih)
#             nn.init.zeros_(c.bias_hh)
#             nn.init.zeros_(c.bias_ih)
#         nn.init.xavier_uniform_(self.head.weight)
#         nn.init.zeros_(self.head.bias)

#     def forward(self,
#                 x0:      torch.Tensor,   # (B,2): [m_obs0, p*0]  (obs mRNA is matured)
#                 u_seq:   torch.Tensor,   # (B,K,U)
#                 dna_raw: torch.Tensor,   # (B,K,1)
#                 dt_seq:  torch.Tensor,   # (B,K)
#                 y_seq:   torch.Tensor,
#                 teacher_forcing: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:

#         B, K, U = u_seq.shape
#         dev = u_seq.device
#         dtype = u_seq.dtype

#         # outputs (we return mm, p, pm)
#         mm_out = torch.empty(B, K, 1, device=dev, dtype=dtype)
#         p_out  = torch.empty_like(mm_out)
#         pm_out = torch.empty_like(mm_out)

#         # diagnostics
#         VTX_seq = torch.empty_like(mm_out)
#         kdm_seq = torch.empty_like(mm_out)
#         VTL_seq = torch.empty_like(mm_out)
#         kmt_seq = torch.empty_like(mm_out)
#         kmatm_seq = torch.empty_like(mm_out)
#         R_seq   = torch.empty_like(mm_out)
#         lam_seq = torch.empty_like(mm_out)


#         mm_prev = x0[:, 0:1]  + 0.01                # measured mRNA is mature
#         m_prev  = torch.zeros_like(mm_prev) + 0.01 # start immature m at 0
#         pm_prev = x0[:, 2:3]  + 0.01 
#         p_prev  = torch.zeros_like(mm_prev)  + 0.01 
#         R_prev  = torch.ones_like(mm_prev)

#         # constant DNA per sequence (matches integrator)
#         dna_cum_total = torch.cumsum(dna_raw, dim=1)[:, -1, :]  # (B,1)

#         # hidden states
#         hs = [torch.zeros(B, self.hidden, device=dev, dtype=dtype) for _ in range(self.num_layers)]
#         y_mm = y_seq[:, :, 0:1]  # (B,K,1)
#         y_pm = y_seq[:, :, 2:3]  # (B,K,1)
#         # This loop solver the whole thing 
#         for k in range(K):
#             dt_k  = dt_seq[:, k:k+1]      # (B,1)
#             u_k   = u_seq[:, k]           # (B,U)
#             mm_gt_prev = mm_prev
#             pm_gt_prev = pm_prev
            
#             if k == 0:
#                 # for step 0, use the provided initial obs x0
#                 mm_gt_prev = mm_prev   # mature mRNA
#                 pm_gt_prev = pm_prev  # mature protein
  

#             if k%100 == 0 and teacher_forcing == True:
#                 # use ground-truth from previous step if available
#                 mm_gt_prev = y_mm[:, k-1, :] 
#                 pm_gt_prev = y_pm[:, k-1, :] 
                
#             det_mm = mm_gt_prev.detach()
#             det_pm = pm_gt_prev.detach() #these are the features that get plugged in. you can keep them raw or sqrt them etc.
#             feat_k = torch.cat([torch.sqrt(u_k).clamp_min(0),
#                                 torch.sqrt(det_mm).clamp_min(0),
#                                 torch.sqrt(det_pm).clamp_min(0),
#                                 # torch.sqrt(m_prev),
#                                 # torch.sqrt(p_prev),
#                                 ]
#                                , dim=-1)  # (B, in_dim) 
            
#             x = feat_k
#             for li, cell in enumerate(self.cells):
#                 hs[li] = cell(x, hs[li])
#                 x = self.dropout(hs[li]) if self.training else hs[li]
#             h = x  # (B, hidden)

#             # head → raw params
#             z = self.bottle(h)
#             raw = self.head(z)
#             lam_raw, VTX_mag, kdm_raw, VTL_mag, kmt_raw, kmatm_raw = raw.split(1, dim=-1)

                 
#             lam_k = gamma(lam_raw, 1e-6, 0.0005)
#             VTXmax = gamma(VTX_mag, 3e-5, 0.12)
#             VTLmax = gamma(VTL_mag, 3e-5, 0.08)
#             kdm_k = gamma(kdm_raw, 1e-5, 1e-2)
#             kmt_k = gamma(kmt_raw, 1e-5, 0.00035)
#             kmatm_k = gamma(kmatm_raw, 5e-5, 0.0035)
       
            
#             # ---- one scripted step ----

#             m_curr, mm_curr, p_curr, pm_curr, R_curr = ivtt_step_mRNA_maturation(
#                 m_prev, mm_prev, p_prev, pm_prev, R_prev,
#                 dt_k, dna_cum_total,
#                 lam_k, VTXmax, kdm_k, VTLmax, kmt_k, kmatm_k
#             )
            
#             # store
#             mm_out[:, k:k+1, :]   = mm_curr.unsqueeze(1)
#             p_out[:,  k:k+1,  :]  = p_curr.unsqueeze(1)
#             pm_out[:, k:k+1,  :]  = pm_curr.unsqueeze(1)

#             VTX_seq[:,  k:k+1, :] = VTXmax.unsqueeze(1)
#             kdm_seq[:,  k:k+1, :] = kdm_k.unsqueeze(1)
#             VTL_seq[:,  k:k+1, :] = VTLmax.unsqueeze(1)
#             kmt_seq[:,  k:k+1, :] = kmt_k.unsqueeze(1)
#             kmatm_seq[:,k:k+1, :] = kmatm_k.unsqueeze(1)
#             R_seq[:,    k:k+1, :] = R_curr.unsqueeze(1)
#             lam_seq[:,  k:k+1, :] = lam_k.unsqueeze(1)

#             # roll
#             m_prev, mm_prev, p_prev, pm_prev, R_prev = m_curr, mm_curr, p_curr, pm_curr, R_curr

#         out    = torch.cat([mm_out, p_out, pm_out], dim=-1)  # (B,K,3): [mm, p, p*]
#         params = torch.cat([VTX_seq, kdm_seq, VTL_seq, kmt_seq, kmatm_seq, R_seq, lam_seq], dim=-1)  # (B,K,7)

#         return (out, params) if self.training else (out, params.detach())




# then you will find the actual ODE model but analytically solved!!! so this is like the ODE function you plug into RK4
# %%

# @torch.jit.script
# def ivtt_step_mRNA_maturation(
#     m_prev:  torch.Tensor,   # (B,1) immature m
#     mm_prev: torch.Tensor,   # (B,1) mature mRNA (bright)
#     p_prev:  torch.Tensor,   # (B,1) immature protein
#     pm_prev: torch.Tensor,   # (B,1) mature protein
#     R_prev:  torch.Tensor,   # (B,1)
#     dt_k:    torch.Tensor,   # (B,1)
#     dna_cum_total: torch.Tensor,  # (B,1) constant within step
#     lam_k:   torch.Tensor,   # (B,1)
#     VTXmax:  torch.Tensor,   # (B,1)
#     kdm:     torch.Tensor,   # (B,1)
#     VTLmax:  torch.Tensor,   # (B,1)
#     kmt:     torch.Tensor,   # (B,1) protein maturation rate
#     kmatm:   torch.Tensor,   # (B,1) mRNA maturation rate (m -> mm)
#     eps: float = 1e-9
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#     """
#     One analytic IVTT step with mRNA maturation.
#     Returns: (m_curr, mm_curr, p_curr, pm_curr, R_curr), each (B,1).
#     """
#     # --- resource decay (use end-of-step as effective within step) ---
#     rho    = torch.exp(-lam_k * dt_k)     # (B,1)
#     R_curr = R_prev * rho

#     # --- effective catalytic rates and source S ----------------------
#     VTX_eff = R_curr * VTXmax
#     VTL_eff = R_curr * VTLmax
#     S       = VTX_eff * dna_cum_total     # (B,1), constant within step

#     # --- m (immature) ------------------------------------------------
#     alpha  = (kdm + kmatm).clamp_min(eps) # (B,1)
#     m_inf  = S / alpha
#     exp_a  = torch.exp(-alpha * dt_k)
#     m_curr = torch.clamp_min(m_inf + (m_prev - m_inf) * exp_a, 0.0)

#     # --- mm (matured mRNA) ------------------------------------------
#     exp_d  = torch.exp(-kdm   * dt_k)     # e^{-kdm Δt}
#     exp_mr = torch.exp(-kmatm * dt_k)     # e^{-kmatm Δt}
#     term1  = mm_prev * exp_d
#     term2  = m_inf * (kmatm / (kdm + eps)) * (1.0 - exp_d)
#     term3  = (m_prev - m_inf) * exp_d * (1.0 - exp_mr)
#     mm_curr = torch.clamp_min(term1 + term2 + term3, 0.0)

#     # --- total mRNA M for protein formulas --------------------------
#     M_prev = m_prev + mm_prev
#     M_inf  = S / (kdm + eps)
#     exp_M  = exp_d

#     # --- protein p ---------------------------------------------------
#     eta    = torch.exp(-kmt * dt_k)       # e^{-kmt Δt}
#     delta  = (kdm - kmt)
#     same   = torch.abs(delta) < 1e-6

#     int_M_eq  = (M_inf * (1.0 - eta) / (kmt + eps)
#                  + (M_prev - M_inf) * dt_k * eta)
#     int_M_gen = (M_inf * (1.0 - eta) / (kmt + eps)
#                  + (M_prev - M_inf) * (eta - exp_M) / (delta + eps))
#     int_M_conv = torch.where(same, int_M_eq, int_M_gen)

#     p_curr  = torch.clamp_min(p_prev * eta + VTL_eff * int_M_conv, 0.0)

#     # --- mature protein p* (conservation) ---------------------------
#     int_M_total = (M_inf * dt_k
#                   + (M_prev - M_inf) / (kdm + eps) * (1.0 - exp_M))
#     pm_curr = torch.clamp_min(pm_prev + (p_prev - p_curr) + VTL_eff * int_M_total, 0.0)

#     return m_curr, mm_curr, p_curr, pm_curr, R_curr
# # %%


#Helper functions
# def gamma(raw: torch.Tensor, lo: float, hi: float, tau: float = 1, cap: int = 50) -> torch.Tensor:
#     # map R → (0,1) with gentler slope
#     s = torch.sigmoid(raw / tau)
#     log_lo = math.log(lo); log_hi = math.log(hi)
#     return torch.exp(log_lo + (log_hi - log_lo) * s)

# def build_x0_ext_on(x0: torch.Tensor) -> torch.Tensor:
#     """
#     Append two gate states initialised to 1 (ON).

#     Returns
#     -------
#     x0_ext : (B, 5)  [m0, p0, p*0, hTX0, hTL0]
#     """
#     ones = torch.ones_like(x0[:, :1])        # (B,1)  filled with 1
#     x0_ext = torch.cat((x0, ones, ones), dim=1)
#     return x0_ext