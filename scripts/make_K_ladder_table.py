"""Sparse-θ K-anchor ladder table -> LaTeX + markdown, aggregated over seeds (mean±sd AND max).

Rows = scaffolds at their dense ladder-best hidden. Columns = θ-anchor count:
  dense (full per-timestep θ) | K1 | K2 | K3 | K6
Cell = endpoint protein R², aggregated over the 3 seeds:
  mean table -> mean$_{\\pm sd}$ ;  max table -> best-protein seed.
A sealed/opened companion (markdown) is written for inspection.

M9 is OMITTED: its dense row is now event_dark (lr0.002) while its K-runs are still the old
dark_stable — mixing them would be misleading. Add M9 back once FINAL_coarse_K_M9ed lands.

Reads experiments_final/FINAL (dense scaffold_ladder dirs + FINAL_coarse_K_sweep + K6 sweep).
Usage:  python scripts/make_K_ladder_table.py
Output: results/coarse_ladder/K_ladder_{mean,max}.{tex,md}
"""
import glob, os, csv
from pathlib import Path
from collections import defaultdict
import numpy as np, yaml

ROOT = Path(__file__).resolve().parent.parent
FINAL = ROOT / "experiments_final" / "FINAL"
SCAF = {"txtl_model3_two_state": "M3", "txtl_model4_three_state": "M4",
        "txtl_resource_and_maturation_dna": "M5", "txtl_model7_bg_fixed": "M7",
        "txtl_model8_bg_fixed": "M8", "txtl_model9_event_dark": "M9"}
DISP = {"M3": "M3", "M4": "M4", "M5": "M5", "M7": "M7", "M8": "M8", "M9": "M9"}
BEST = {"M3": 400, "M4": 300, "M5": 400, "M7": 600, "M8": 600, "M9": 400}
ORDER = ["M3", "M4", "M5", "M7", "M8", "M9"]
COLS = ["dense", "K1", "K2", "K3", "K6"]

# (scaf, col) -> list of {pm,old,new} across seeds
data = defaultdict(list)
for cfgp in glob.glob(str(FINAL / "**" / "config.yaml"), recursive=True):
    d = Path(cfgp).parent; cfg = yaml.safe_load(open(cfgp))
    sc = SCAF.get(str(cfg.get("scaffold", "")))
    if not sc: continue
    rc = d / "r2_cache.csv"
    if not (rc.exists() and os.path.getsize(rc) > 50): continue
    r = list(csv.DictReader(open(rc)))[-1]
    vals = dict(pm=float(r["r2_protein_final"]), old=float(r["r2_protein_old"]), new=float(r["r2_protein_new"]))
    K = cfg.get("n_theta_anchors"); en = str(cfg.get("exp_name", ""))
    if K is None:
        if "scaffold_ladder" not in cfgp: continue          # dense ref ONLY from the ladder
        if sc == "M4" and "lateP" in en: continue           #   (excludes theta_freeze/node_baselines)
        if sc == "M9" and "oxy01" in en: continue           #   M9 dense = event_dark oxy00 only
        if cfg.get("hidden") == BEST[sc]: data[(sc, "dense")].append(vals)
    elif int(K) in (1, 2, 3, 6):
        data[(sc, f"K{int(K)}")].append(vals)

def pick(vals_list, mode, key):
    """mean -> (mean, sd); max -> (best-protein seed's value, None). None if no data."""
    if not vals_list: return None
    if mode == "mean":
        a = np.array([v[key] for v in vals_list]); return (float(a.mean()), float(a.std()))
    return (float(max(vals_list, key=lambda v: v["pm"])[key]), None)

def fmt(c, tex):
    if c is None: return "--"
    pt, sd = c
    if sd is None: return f"{pt:.2f}"
    return f"{pt:.2f}$_{{\\pm{sd:.2f}}}$" if tex else f"{pt:.2f} ± {sd:.2f}"

def emit(mode):
    agg = "mean$\\,\\pm\\,$s.d." if mode == "mean" else "best-protein seed"
    # protein-R² main table
    md = [f"# K-anchor ladder — endpoint protein R² ({mode}); dense = full per-timestep θ", "",
          "| Scaffold | " + " | ".join(COLS) + " |", "|" + "---|"*(len(COLS)+1)]
    L = [r"\begin{table}[t]",
         r"\caption{\textbf{Sparse-$\theta$ anchor ladder.} Endpoint protein $R^2$ as the number of "
         r"$\theta(t)$ time-anchors $K$ is reduced from dense (full per-timestep) down to $K{=}1$ "
         r"(one constant $\theta$). Each scaffold at its ladder-best hidden; " + agg + r" over 3 seeds.}",
         r"\label{tab:k_ladder" + ("" if mode == "mean" else "_max") + r"}",
         r"{\small", r"\begin{tabular}{l ccccc}", r"\toprule",
         r"Scaffold & dense & $K{=}1$ & $K{=}2$ & $K{=}3$ & $K{=}6$ \\", r"\midrule"]
    for sc in ORDER:
        cells = [pick(data.get((sc, c), []), mode, "pm") for c in COLS]
        md.append(f"| {DISP[sc]} | " + " | ".join(fmt(c, False) for c in cells) + " |")
        L.append(f"{DISP[sc]} & " + " & ".join(fmt(c, True) for c in cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    # sealed/opened companion (markdown only)
    md2 = [f"\n## sealed / opened protein R² ({mode})", "",
           "| Scaffold | " + " | ".join(f"{c} (sealed/opened)" for c in COLS) + " |", "|" + "---|"*(len(COLS)+1)]
    for sc in ORDER:
        row = [f"{fmt(pick(data.get((sc,c),[]),mode,'old'),False)} / {fmt(pick(data.get((sc,c),[]),mode,'new'),False)}" for c in COLS]
        md2.append(f"| {DISP[sc]} | " + " | ".join(row) + " |")
    out = ROOT / "results" / "coarse_ladder"; out.mkdir(parents=True, exist_ok=True)
    (out / f"K_ladder_{mode}.tex").write_text("\n".join(L) + "\n")
    (out / f"K_ladder_{mode}.md").write_text("\n".join(md) + "\n" + "\n".join(md2) + "\n")
    print("\n".join(md) + "\n")

for mode in ("mean", "max"):
    emit(mode)
# seed counts per cell (sanity) + missing
print("seed counts per (scaffold, K):")
for sc in ORDER:
    print(f"  {sc}: " + "  ".join(f"{c}={len(data.get((sc,c),[]))}" for c in COLS))
print("M9 = event_dark (oxy00 dense + K_M9ed K-runs).")
