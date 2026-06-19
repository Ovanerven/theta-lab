"""Baselines table: static (RF on recipe) -> black-box (neural-ODE) -> mechanistic (CMVF M4).
All scored on the same observables (mRNA, protein). Emits BOTH mean(+/-sd) and max(best-seed) tables.

  results/coarse_ladder/baselines_table_mean.{tex,md}   (mean +/- s.d. over seeds)
  results/coarse_ladder/baselines_table_max.{tex,md}    (best-protein seed per model)
"""
import glob, csv, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from collections import defaultdict
import numpy as np, yaml
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parent.parent
FINAL = ROOT / "experiments_final" / "FINAL"
DS = ROOT / "datasets/cell-free/txtl_native_real_only_coarsenold.npz"
METR = ("pm", "p_sealed", "p_opened", "mm", "m_sealed", "m_opened")   # protein{all,deox,ox}, mRNA{all,deox,ox}

def read(d):
    r = list(csv.DictReader(open(d / "r2_cache.csv")))[-1]
    return dict(pm=float(r["r2_protein_final"]), p_sealed=float(r["r2_protein_old"]),
                p_opened=float(r["r2_protein_new"]), mm=float(r["r2_mrna_max"]),
                m_sealed=float(r["r2_mrna_old"]), m_opened=float(r["r2_mrna_new"]))

# ---------- RF static floor: recipe totals -> gated endpoint, 3 random seeds ----------
def rf_rows():
    z = np.load(DS, allow_pickle=True)
    u, y, L = z["u_seq"], z["y_seq"], z["lengths"]
    src = np.array([1 if str(s) == "new" else 0 for s in z["source_label"]])  # 0 sealed,1 opened
    B = u.shape[0]
    X = np.array([u[i, :int(L[i])].sum(0) for i in range(B)])                  # reagent totals
    Pf = np.zeros(B); Mmx = np.zeros(B)
    for i in range(B):
        Li = int(L[i]); seg = y[i, :Li, :].copy()
        for ch in (3, 5): seg[:, ch] -= seg[:, ch].min()                       # channel-min gate
        Pf[i] = seg[Li-1, 5]; Mmx[i] = seg[:, 3].max()
    sp = np.load(next(FINAL.glob("scaffold_ladder/*NG_plain_s0/split.npz")))
    tr, te = sp["train_idx"], sp["test_idx"]; keep = X[tr].std(0) > 1e-12
    so, sn = src[te] == 0, src[te] == 1
    def r2(t, p, m): t, p = t[m], p[m]; ss = np.sum((t-t.mean())**2); return 1-np.sum((t-p)**2)/ss if ss>1e-9 else np.nan
    out = []
    for seed in (0, 1, 2):
        def fit(Y):
            m = RandomForestRegressor(n_estimators=400, random_state=seed, n_jobs=-1).fit(X[tr][:, keep], Y[tr])
            return m.predict(X[te][:, keep])
        pP, pM = fit(Pf), fit(Mmx)
        out.append(dict(pm=r2(Pf[te], pP, np.ones(len(te), bool)),
                        p_sealed=r2(Pf[te], pP, so), p_opened=r2(Pf[te], pP, sn),
                        mm=r2(Mmx[te], pM, np.ones(len(te), bool)),
                        m_sealed=r2(Mmx[te], pM, so), m_opened=r2(Mmx[te], pM, sn)))
    return out

# ---------- CMVF rows (M4 + M5), each at its best hidden ----------
CMVF = {"txtl_model4_three_state": ("CMVF (M4)", "NG_plain"),     # plain arm only
        "txtl_resource_and_maturation_dna": ("CMVF (M5)", None)}
cmvf = {sk: defaultdict(list) for sk in CMVF}
for cfgp in glob.glob(str(FINAL / "scaffold_ladder/*/config.yaml")):
    d = Path(cfgp).parent; c = yaml.safe_load(open(cfgp)); sk = str(c.get("scaffold"))
    if sk not in CMVF: continue
    nf = CMVF[sk][1]
    if nf and nf not in str(c.get("exp_name", "")): continue
    if (d / "r2_cache.csv").exists(): cmvf[sk][c.get("hidden")].append(read(d))
cmvf_best = {sk: max(cmvf[sk], key=lambda h: np.mean([x["pm"] for x in cmvf[sk][h]])) for sk in CMVF}

# ---------- nODE baselines ----------
NODE = {"neural_ode_mlp": "NODE-MLP", "neural_ode_gru": "NODE-GRU", "neural_ode_correction": "NODE-corr"}
nd = defaultdict(list)
for cfgp in glob.glob(str(FINAL / "node_baselines/*/config.yaml")):
    d = Path(cfgp).parent; c = yaml.safe_load(open(cfgp)); mc = str(c.get("model_class"))
    if mc in NODE and (d / "r2_cache.csv").exists(): nd[mc].append(read(d))

# rows: static -> black-box -> mechanistic
rows = [("RF (static)", rf_rows())]
for mc in ["neural_ode_mlp", "neural_ode_gru", "neural_ode_correction"]:
    if nd[mc]: rows.append((NODE[mc], nd[mc]))
for sk in ["txtl_model4_three_state", "txtl_resource_and_maturation_dna"]:
    rows.append((CMVF[sk][0], cmvf[sk][cmvf_best[sk]]))

def emit(mode):
    def val(vals, met):
        if mode == "mean": return float(np.mean([v[met] for v in vals]))
        return max(vals, key=lambda v: v["pm"])[met]                          # best-protein seed's value
    best = {m: max(val(v, m) for _, v in rows) for m in METR}                  # per-column best (bold)
    def fmt(vals, met):
        if mode == "mean":
            a = np.array([v[met] for v in vals]); s = f"{a.mean():.2f}$_{{\\pm{f'{a.std():.2f}'[1:]}}}$"
            sm = f"{a.mean():.2f} ± {a.std():.2f}"
        else:
            s = sm = f"{val(vals, met):.2f}"
        bold = abs(val(vals, met) - best[met]) < 1e-9
        return (f"\\textbf{{{s}}}" if bold else s), (f"**{sm}**" if bold else sm)
    agg = "mean ± s.d." if mode == "mean" else "best-protein seed"
    md = [f"# Baselines (static -> black-box -> mechanistic) — {agg}", "",
          "| model | protein: all | sealed | opened | mRNA | n |", "|" + "---|"*6]
    L = [r"\begin{table}[t]",
         r"\caption{\textbf{Baselines.} Endpoint $R^2$: static recipe regression (random forest) "
         r"$\rightarrow$ black-box neural ODEs $\rightarrow$ the mechanistic CMVF (M4). All score the "
         r"same observables (mRNA, protein) on the same test split. RF predicts the gated endpoint from "
         r"recipe totals only (no dynamics); NODE-MLP/GRU learn a black-box vector field; "
         r"NODE-corr is global-$\theta$ mechanism + a state-only neural residual (no history "
         r"encoder). " + (r"Mean$\,\pm\,$s.d.\ over seeds" if mode == "mean"
                          else r"Best-protein seed per model") + r"; \textbf{bold} = best per column.}",
         r"\label{tab:baselines" + ("" if mode == "mean" else "_max") + r"}",
         r"\begin{tabular}{l ccc c}", r"\toprule",
         r" & \multicolumn{3}{c}{Protein $R^2_{\mathrm{endpoint}}$} & mRNA $R^2_{\mathrm{peak}}$ \\",
         r"Model & all & sealed & opened & all \\", r"\midrule"]
    met4 = METR[:4]                                       # legacy 4-col layout: protein{all,deox,ox} + mRNA(all)
    for name, vals in rows:
        tex = [fmt(vals, m)[0] for m in met4]; mdc = [fmt(vals, m)[1] for m in met4]
        md.append(f"| {name} | " + " | ".join(mdc) + f" | {len(vals)} |")
        L.append(f"{name} & " + " & ".join(tex) + r" \\")
        if name in ("RF (static)", "NODE-corr"): L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out = ROOT / "results" / "coarse_ladder"; out.mkdir(parents=True, exist_ok=True)
    (out / f"baselines_table_{mode}.tex").write_text("\n".join(L) + "\n")
    (out / f"baselines_table_{mode}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md) + "\n")

def emit_combined():
    """Single table in the scaffold-ladder 'paren' style: one row per model. Each cell gives the seeded
    mean +/- s.d. with the single best-protein seed in parentheses. Bold = best mean per column;
    underline = best single run per column. Protein and mRNA are each split all/deoxy/oxy."""
    def mean_of(vals, m):
        a = np.array([v[m] for v in vals]); return float(a.mean()), float(a.std())
    def best_of(vals, m):
        return max(vals, key=lambda v: v["pm"])[m]        # value of the best-protein seed in column m
    best_mean = {m: max(mean_of(v, m)[0] for _, v in rows) for m in METR}
    best_max  = {m: max(best_of(v, m)    for _, v in rows) for m in METR}
    def fmt_mean(point, sd, bold):
        s = f"{point:.2f}$_{{\\pm{sd:.2f}}}$"
        return f"\\textbf{{{s}}}" if bold else s
    def fmt_max(point, bold):
        s = f"{point:.2f}"
        return f"\\underline{{{s}}}" if bold else s
    def cell(vals, m):
        mu, sd = mean_of(vals, m); bv = best_of(vals, m)
        mean_s = fmt_mean(mu, sd, abs(mu - best_mean[m]) < 1e-9)
        max_s  = fmt_max(bv, abs(bv - best_max[m]) < 1e-9)
        return mean_s + r"~{\footnotesize (" + max_s + r")}"
    cap = (r"\textbf{Baselines.} Endpoint $R^2$: static recipe regression (random forest) $\rightarrow$ "
           r"black-box neural ODEs $\rightarrow$ the mechanistic CMVF (M4--M5). All score the same "
           r"observables (mRNA, protein) on the same test split. RF predicts the gated endpoint from "
           r"recipe totals only (no dynamics); NODE-MLP/GRU learn a black-box vector field; "
           r"NODE-corr is global-$\theta$ mechanism + a state-only neural residual (no history "
           r"encoder). Reported over all test trajectories (\emph{all}) and by protocol---"
           r"\emph{deoxygenated} (tube closed) and \emph{oxygenated} (tube reopened). Each cell gives the "
           r"seeded mean$\,\pm\,$s.d.\ over 3 seeds and, in parentheses, the single best run (the seed "
           r"with the highest overall protein $R^2$, with that run's score in this column). "
           r"\textbf{Bold} = best mean per column; \underline{underline} = best single run per column.")
    L = [r"\begin{table}[t]", r"\caption{" + cap + r"}", r"\label{tab:baselines}",
         r"\begin{tabular}{l ccc ccc}", r"\toprule",
         r" & \multicolumn{3}{c}{Protein $R^2_{\mathrm{endpoint}}$} & \multicolumn{3}{c}{mRNA $R^2_{\mathrm{peak}}$} \\",
         r"Model & all & deoxygenated & oxygenated & all & deoxygenated & oxygenated \\", r"\midrule"]
    SECTION = {"RF (static)", "NODE-corr"}          # class boundaries (static | black-box | mechanistic)
    for name, vals in rows:
        L.append(f"{name} & " + " & ".join(cell(vals, m) for m in METR) + r" \\")
        if name in SECTION: L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out = ROOT / "results" / "coarse_ladder"; out.mkdir(parents=True, exist_ok=True)
    (out / "baselines_table_combined.tex").write_text("\n".join(L) + "\n")
    print("\n".join(L) + "\n")

for mode in ("mean", "max"):
    emit(mode)
emit_combined()
for sk in CMVF:
    print(f"{CMVF[sk][0]} h={cmvf_best[sk]} (n={len(cmvf[sk][cmvf_best[sk]])})")
print("RF=3 random seeds; nODE=3 seeds each")
