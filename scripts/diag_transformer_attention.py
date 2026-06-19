"""Transformer attention diagnostic: does attention attend to boluses, and why doesn't it help?

Captures the self-attention weights (averaged over heads, then layers) from the FINAL per-step
forward (window = full trajectory), so attn[q,k] = how much the theta-decode at step q attends to
step k. Marks the bolus timesteps (raw reagent deltas) and the u_open event. Outputs:
  - bolus-attention enrichment  = mean(attn on bolus keys) / mean(attn on all keys)  (1.0 = no preference)
  - recency mass               = attn within the last few steps of each query (locality)
  - a 3-panel figure: attention map | final-query attention vs boluses | per-query bolus-attn

Usage: python scripts/diag_transformer_attention.py [run_dir] [sample_idx]
Output: figures/diag/trans_attn_<run>.{png,pdf} + printed stats
"""
import sys, glob, types
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path("/Users/olivervanerven/Documents/Thesis/theta-lab"); sys.path.insert(0, str(ROOT/"last-layer-ode"))
from plot_diagnostics import rebuild_model_from_experiment, _maybe_lift, _filter_model_kwargs, load_yaml
from train import collate, collate_varlen
from metrics.endpoint_r2 import _apply_channel_min_gate as _gate

dev = torch.device("cpu")
EXP = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(glob.glob(str(
      ROOT/"experiments_final/FINAL_coarse_transformer_decay_replicate/*TR_pulse_s1*")))[0]
EXP = Path(EXP)
model, ds, *r = rebuild_model_from_experiment(EXP, dev); model.eval()
lift = r[2] or {}; cfg = load_yaml(EXP/"config.yaml")
raw = ds.dataset if isinstance(ds, torch.utils.data.Subset) else ds
print("model:", EXP.name, "| u_transform:", cfg.get("u_transform"), "| nlayers:", len(model.transformer.layers))

# ---- capture attention: replace each layer's forward with a manual Pre-LN block that calls
#      self_attn with need_weights=True (the fused/fast path otherwise skips self_attn entirely) ----
model.transformer.enable_nested_tensor = False
ATT = []
def make_fwd(layer):
    def fwd(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        x = src; n = self.norm1(x)
        a, w = self.self_attn(n, n, n, attn_mask=src_mask, need_weights=True,
                              average_attn_weights=True, is_causal=False)
        ATT.append(w.detach())                       # (B, W, W) averaged over heads
        x = x + self.dropout1(a)
        x = x + self._ff_block(self.norm2(x))
        return x
    return fwd
nlayer = len(model.transformer.layers)
for L in model.transformer.layers:
    L.forward = types.MethodType(make_fwd(L), L)

# ---- one sample ----
SAMPLE = int(sys.argv[2]) if len(sys.argv) > 2 else int(np.load(EXP/"split.npz")["test_idx"][0])
sub = torch.utils.data.Subset(raw, [SAMPLE]); cf = collate_varlen if getattr(raw,"variable_length",False) else collate
b = next(iter(torch.utils.data.DataLoader(sub, batch_size=1, collate_fn=cf)))
y0, u, y, Ls = b[0], b[1], b[2], b[3]
dt = b[5] if len(b) >= 6 else torch.from_numpy(raw.dt[:u.shape[1]])[None].expand(y0.shape[0], -1)
u_raw = u[0].clone()                                  # (K,U) raw deltas (pre-gate, pre-lift) for bolus detection
if bool(cfg.get("subtract_channel_min", False)):
    c = cfg.get("subtract_channel_min_cols"); c = [int(x) for x in c] if c else None; y0, y = _gate(y0, y, c, Ls)
y0, y = _maybe_lift(y0, y, lift); oi = torch.tensor(lift["scaffold_obs_idx"]) if lift else torch.arange(y0.shape[-1])
mk = {"y_seq": None, "teacher_forcing": False, "u_transform": str(cfg.get("u_transform","none")), "y_transform": str(cfg.get("y_transform","none"))}
ATT.clear()
with torch.no_grad():
    pred, theta, _ = model(y0, u, dt, oi, **_filter_model_kwargs(model, mk))
K = int(Ls[0]); t = np.cumsum(dt[0,:K].numpy())/60.0

# final-step attention map (last nlayer captures = window of full trajectory), avg over layers
final = torch.stack(ATT[-nlayer:]).mean(0)[0].numpy()   # (W,W); W==K at the final step
A = final[:K, :K]                                       # (K,K) causal: A[q,k]=attn of query q to key k

# ---- bolus timesteps: any encoder reagent column has a nonzero raw delta ----
ucols = model.gru_u_cols if model.gru_u_cols is not None else list(range(u_raw.shape[1]))
ureag = u_raw[:K][:, ucols].abs().sum(1).numpy()        # total reagent delta per step
bolus = np.where(ureag > 1e-6)[0]
OI = [str(c) for c in np.load(ROOT/"datasets/cell-free/txtl_native_real_only_coarsenold.npz", allow_pickle=True)["control_names"]].index("u_open")
uopen_steps = np.where(u_raw[:K, OI].numpy() > 0)[0]
print(f"K={K} steps | {len(bolus)} bolus steps at {bolus[:12].tolist()}{'...' if len(bolus)>12 else ''} | u_open step {uopen_steps.tolist()}")

# ---- metrics ----
# per-query attention mass on bolus keys vs uniform expectation (#bolus/#valid)
enrich = []
for q in range(1, K):
    row = A[q, :q+1]; row = row/ (row.sum()+1e-12)
    bk = [k for k in bolus if k <= q]
    if not bk: continue
    got = row[bk].sum(); exp = len(bk)/(q+1)
    enrich.append(got/ (exp+1e-12))
enrich = np.array(enrich)
# recency: fraction of each query's attention within its last 10 steps
rec = np.array([A[q, max(0,q-9):q+1].sum()/(A[q,:q+1].sum()+1e-12) for q in range(1,K)])
print(f"bolus-attention enrichment (1.0=no preference): mean={enrich.mean():.2f}  median={np.median(enrich):.2f}")
print(f"recency mass (attn in last 10 steps): mean={rec.mean():.2f}  (1.0 = fully local/recent)")
print(f"final-query attn: top-5 attended steps = {np.argsort(A[K-1,:K])[-5:][::-1].tolist()}")

# ---- figure ----
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
im = ax[0].imshow(np.log10(A+1e-4), aspect="auto", origin="lower", cmap="magma")
for bs in bolus: ax[0].axhline(bs, color="cyan", lw=0.2, alpha=0.3); ax[0].axvline(bs, color="cyan", lw=0.2, alpha=0.3)
for uo in uopen_steps: ax[0].axvline(uo, color="lime", lw=0.8)
ax[0].set_title("attention map  log$_{10}$ A[query,key]\n(cyan=bolus, green=u_open)", fontsize=8)
ax[0].set_xlabel("key step"); ax[0].set_ylabel("query step"); fig.colorbar(im, ax=ax[0], fraction=0.046)

fq = A[K-1, :K]
ax[1].plot(np.arange(K), fq, color="#4E79A7", lw=0.8)
for bs in bolus: ax[1].axvline(bs, color="#E15759", lw=0.5, alpha=0.5)
for uo in uopen_steps: ax[1].axvline(uo, color="green", lw=1.2, ls="--")
ax[1].set_title(f"final-query attention vs boluses\n(enrich={enrich.mean():.2f})", fontsize=8)
ax[1].set_xlabel("key step"); ax[1].set_ylabel("attention")

ax[2].plot(np.arange(1,K), rec, color="#59A14F", lw=0.8, label=f"recency (last10)={rec.mean():.2f}")
ax[2].axhline(enrich.mean(), color="#E15759", lw=1.0, ls="--", label=f"bolus enrich={enrich.mean():.2f}")
ax[2].axhline(1.0, color="0.6", lw=0.6)
ax[2].set_title("locality vs bolus-targeting per query", fontsize=8)
ax[2].set_xlabel("query step"); ax[2].set_ylim(0, None); ax[2].legend(fontsize=6, loc="upper right")

out = ROOT/"figures"/"diag"; out.mkdir(parents=True, exist_ok=True)
tag = EXP.name.split("_",2)[-1] if "_" in EXP.name else EXP.name
fig.suptitle(f"{EXP.name}  ({cfg.get('u_transform')})", fontsize=9)
fig.tight_layout(rect=[0,0,1,0.94])
fig.savefig(out/f"trans_attn_{tag}.png", dpi=140, bbox_inches="tight")
fig.savefig(out/f"trans_attn_{tag}.pdf", bbox_inches="tight")
print("wrote", out/f"trans_attn_{tag}.png")
