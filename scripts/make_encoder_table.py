"""Encoder-comparison table (R^2) -> LaTeX, in the SAME "typical and best" format as the scaffold
ladder / baselines tables: each cell = seeded mean$_{\\pm sd}$ ~ (best single run), bold = best mean
per column, underline = best single run per column. 6 columns: protein{all,deoxy,oxy}, mRNA{...}.

All encoders share the matched recipe (M4 / h400 / pulse_cumsum_sqrt / 3 seeds):
  GRU/LSTM/sLSTM  -> FINAL_coarse_encoder_zoo_light  (ode_rnn / lstm_rnn / ode_slstm)
  Mamba           -> FINAL_coarse_encoder_zoo_heavy   (ode_mamba, custom SSM; the library wrappers collapse)
  Transformer     -> FINAL_coarse_transformer_decay_replicate (TR_pulse, transformer_final_norm)

Output: results/coarse_ladder/encoder_table_combined.tex (+ .md preview)
"""
import glob, csv
from pathlib import Path
import numpy as np, yaml

ROOT = Path(__file__).resolve().parent.parent
ZL = ROOT / "experiments_final/FINAL/FINAL_coarse_encoder_zoo_light"
ZH = ROOT / "experiments_final/FINAL/FINAL_coarse_encoder_zoo_heavy"
TR = ROOT / "experiments_final/FINAL_coarse_transformer_decay_replicate"
METR = ("pm", "p_sealed", "p_opened", "mm", "m_sealed", "m_opened")

# (display, dir, model_class, exp_name substring filter)  — order = RNN -> SSM -> attention
ENCODERS = [
    ("GRU",         ZL, "ode_rnn",         None),
    ("LSTM",        ZL, "lstm_rnn",        None),
    ("sLSTM",       ZL, "ode_slstm",       None),
    ("Mamba",       ZH, "ode_mamba",       "EZ_mamba_s"),     # custom SSM (lib wrappers excluded)
    ("Transformer", TR, "ode_transformer", "TR_pulse_s"),     # pulse_cumsum_sqrt + final-norm fix
]

def read(d):
    r = list(csv.DictReader(open(Path(d) / "r2_cache.csv")))[-1]
    return dict(pm=float(r["r2_protein_final"]), p_sealed=float(r["r2_protein_old"]),
                p_opened=float(r["r2_protein_new"]), mm=float(r["r2_mrna_max"]),
                m_sealed=float(r["r2_mrna_old"]), m_opened=float(r["r2_mrna_new"]))

rows = []   # (display, [seed dicts])
for disp, D, mc, filt in ENCODERS:
    runs = []
    for cfgp in glob.glob(str(D / "*" / "config.yaml")):
        d = Path(cfgp).parent; c = yaml.safe_load(open(cfgp))
        if str(c.get("model_class")) != mc: continue
        if filt and filt not in str(c.get("exp_name", "")): continue
        if (d / "r2_cache.csv").exists(): runs.append(read(d))
    rows.append((disp, runs))
    print(f"{disp:12s} n={len(runs)}  protein mean={np.mean([r['pm'] for r in runs]):.3f}  max={max(r['pm'] for r in runs):.3f}"
          if runs else f"{disp:12s}  NONE")

def raw_vals(rs, mode):
    if mode == "mean":
        return {m: (float(np.mean([r[m] for r in rs])), float(np.std([r[m] for r in rs]))) for m in METR}
    br = max(rs, key=lambda r: r["pm"])
    return {m: (float(br[m]), None) for m in METR}

def fmt_mean(point, sd, bold):
    s = f"{point:.2f}$_{{\\pm{sd:.2f}}}$"
    return f"\\textbf{{{s}}}" if bold else s
def fmt_max(point, best):
    s = f"{point:.2f}"
    return f"\\underline{{{s}}}" if best else s

mean_rv = {d: raw_vals(rs, "mean") for d, rs in rows if rs}
max_rv  = {d: raw_vals(rs, "max")  for d, rs in rows if rs}
best_mean = {m: max(v[m][0] for v in mean_rv.values()) for m in METR}
best_max  = {m: max(v[m][0] for v in max_rv.values())  for m in METR}

cap = (r"\textbf{History encoder comparison (typical and best).} Held-out $R^2$ for the history "
       r"encoder $F_\phi$ on the M4 mechanistic vector field, at matched capacity ($h{=}400$) and with "
       r"an identical recipe (pulse-cumsum input features). Reported over all test trajectories "
       r"(\emph{all}) and by protocol---\emph{deoxygenated} (tube closed) and \emph{oxygenated} (tube "
       r"reopened at steady state, admitting O$_2$). Each cell gives the seeded mean$\,\pm\,$s.d.\ over "
       r"3 seeds and, in parentheses, the single best run (the seed with the highest overall protein "
       r"$R^2$, with that run's score in this column). \textbf{Bold} = best mean per column; "
       r"\underline{underline} = best single run per column.")
L = [r"\begin{table}[t]", r"\caption{" + cap + r"}", r"\label{tab:encoder}",
     r"\begin{tabular}{l ccc ccc}", r"\toprule",
     r" & \multicolumn{3}{c}{Protein $R^2_{\mathrm{endpoint}}$} & \multicolumn{3}{c}{mRNA $R^2_{\mathrm{peak}}$} \\",
     r"Encoder & all & deoxygenated & oxygenated & all & deoxygenated & oxygenated \\", r"\midrule"]
md = ["| Encoder | P:all | P:deoxy | P:oxy | mRNA:all | mRNA:deoxy | mRNA:oxy |", "|"+"---|"*7]
for disp, rs in rows:
    if not rs:
        L.append(f"{disp} & " + " & ".join(["--"]*6) + r" \\"); continue
    mrv, xrv = mean_rv[disp], max_rv[disp]
    cells = [fmt_mean(*mrv[m], mrv[m][0] == best_mean[m]) + r"~{\footnotesize (" +
             fmt_max(xrv[m][0], xrv[m][0] == best_max[m]) + r")}" for m in METR]
    L.append(f"{disp} & " + " & ".join(cells) + r" \\")
    md.append(f"| {disp} | " + " | ".join(f"{mrv[m][0]:.2f}±{mrv[m][1]:.2f} ({xrv[m][0]:.2f})" for m in METR) + " |")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
out = ROOT / "results" / "coarse_ladder"; out.mkdir(parents=True, exist_ok=True)
(out / "encoder_table_combined.tex").write_text("\n".join(L) + "\n")
(out / "encoder_table_combined.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
print(f"\nwrote {out}/encoder_table_combined.tex")
