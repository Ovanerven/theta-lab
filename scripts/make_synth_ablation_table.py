"""Synthetic-data ablation table: real-only vs synth (no L_zero) vs synth (+L_zero), per scaffold.
Same "typical and best" format as the ladder/baseline tables (mean$_{pm sd}$ (best), bold=best mean).
Hidden is FIXED per scaffold (M4 h300, M5 h400, M7 h600, M8 h600) so the three conditions are
compared at equal capacity. Re-run after the overnight FINAL_coarse_synth_ablation_fill lands.

Output: results/coarse_ladder/synth_ablation_table.tex (+ .md)
"""
import glob, csv, os
from pathlib import Path
import numpy as np, yaml
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
DIRS = [
    ROOT / "experiments_final/FINAL/scaffold_ladder",            # real-only (M4/M5/M7real/M8real)
    ROOT / "experiments_final/FINAL/synth_ablation",             # synth +Lz (moved combined M7/M8)
    ROOT / "experiments_final/FINAL_coarse_m4m5_synth",          # synth +Lz (M4/M5)
    ROOT / "experiments_final/FINAL_coarse_synth_ablation_fill", # synth -Lz (+ M5 +Lz)  [overnight]
]
SCAF = {  # display -> (scaffold string, fixed hidden)
    "M4": ("txtl_model4_three_state", 300),
    "M5": ("txtl_resource_and_maturation_dna", 400),
    "M7": ("txtl_model7_bg_fixed", 600),
    "M8": ("txtl_model8_bg_fixed", 600),
}
COND = ["real", "synth_nolz", "synth_lz"]   # column order
def classify(cfg):
    if "real_only" in str(cfg.get("dataset_path", "")): return "real"
    return "synth_lz" if float(cfg.get("lambda_zero_traj", 0)) > 0 else "synth_nolz"

cells = defaultdict(lambda: defaultdict(list))   # cells[scaf][cond] = [protein_final,...]
for D in DIRS:
    for cfgp in glob.glob(str(D / "*" / "config.yaml")):
        c = yaml.safe_load(open(cfgp)); d = Path(cfgp).parent
        sk = str(c.get("scaffold")); h = c.get("hidden")
        disp = next((k for k, (s, hh) in SCAF.items() if s == sk and hh == h), None)
        if disp is None: continue
        rc = d / "r2_cache.csv"
        if not rc.exists(): continue
        try: r2 = float(list(csv.DictReader(open(rc)))[-1]["r2_protein_final"])
        except Exception: continue
        cells[disp][classify(c)].append(r2)

# print status + build table
print(f"{'scaf':4s} " + "  ".join(f"{c:>18s}" for c in COND))
for disp in ["M4", "M5", "M7", "M8"]:
    line = f"{disp:4s} "
    for cond in COND:
        v = cells[disp][cond]
        line += f"  {('%.2f±%.2f(n=%d)'%(np.mean(v),np.std(v),len(v))) if v else '-- (n=0)':>18s}"
    print(line)

HDR = {"real": "real-only", "synth_nolz": r"synth, no $\mathcal{L}_0$", "synth_lz": r"synth $+\mathcal{L}_0$"}
best = {cond: max((np.mean(cells[s][cond]) for s in SCAF if cells[s][cond]), default=None) for cond in COND}
def cell(v, cond):
    if not v: return "--"
    m, sd, mx = np.mean(v), np.std(v), max(v)
    s = f"{m:.2f}$_{{\\pm{sd:.2f}}}$"
    if best[cond] is not None and abs(m - best[cond]) < 1e-9: s = f"\\textbf{{{s}}}"
    return s + r"~{\footnotesize (" + f"{mx:.2f}" + r")}"
L = [r"\begin{table}[h]",
     r"\caption{\textbf{Adding synthetic non-expressing data (M4--M8).} Endpoint protein $R^2$ (all "
     r"test trajectories) for each mechanistic vector field trained on real data only, on real$+$synthetic "
     r"data without the no-go loss ($\mathcal{L}_0$), and on real$+$synthetic data with it. Hidden fixed "
     r"per scaffold (equal capacity); seeded mean$\,\pm\,$s.d., best single run in parentheses. "
     r"\textbf{Bold} = best mean per column.}",
     r"\label{tab:synth_ablation}", r"\centering", r"\begin{tabular}{l ccc}", r"\toprule",
     r"ODE & " + " & ".join(HDR[c] for c in COND) + r" \\", r"\midrule"]
md = ["| ODE | " + " | ".join(HDR[c] for c in COND) + " |", "|"+"---|"*4]
for disp in ["M4", "M5", "M7", "M8"]:
    L.append(f"{disp} & " + " & ".join(cell(cells[disp][c], c) for c in COND) + r" \\")
    md.append(f"| {disp} | " + " | ".join((f"{np.mean(cells[disp][c]):.2f}" if cells[disp][c] else "--") for c in COND) + " |")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
out = ROOT / "results" / "coarse_ladder"; out.mkdir(parents=True, exist_ok=True)
(out / "synth_ablation_table.tex").write_text("\n".join(L) + "\n")
(out / "synth_ablation_table.md").write_text("\n".join(md) + "\n")
print("\nwrote", out / "synth_ablation_table.tex")
