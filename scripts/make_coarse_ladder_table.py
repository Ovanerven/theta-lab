"""Seeded coarse dense-θ scaffold-ladder table (R²) -> LaTeX + markdown preview.

Scans the dense-θ ladder runs (model_class=ode_rnn, coarse dataset), and for EACH scaffold
AUTO-SELECTS the hidden dimension with the best seeded-mean protein R² (argmax over the swept
hiddens present), then aggregates that hidden's seeds (mean±sd). Re-run as seeds/hiddens land.

Table columns: Scaffold | protein | sealed | opened | mRNA  (the selected hidden + seed count are
printed to console for the methods text, not shown in the table). sealed = tube kept closed;
opened = tube reopened at steady state (O2 influx -> protein bump).

Metrics (from r2_cache.csv): r2_protein_final (all / old=sealed / new=opened) + r2_mrna_max.
Usage:  python scripts/make_coarse_ladder_table.py [extra_study_dir ...]
Output: results/coarse_ladder/coarse_ladder_table_{mean,max}.{tex,md}
"""
import sys, glob, csv
from pathlib import Path
from collections import defaultdict
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
DIRS = [ROOT / "experiments_final" / "FINAL" / "scaffold_ladder"] + [Path(a) for a in sys.argv[1:]]

# scaffold key -> (display name, optional exp_name substring filter)
SCAF = {  # bare IDs for now (single-column fit); rename later
    "txtl_model3_two_state":            ("M3", None),
    "txtl_model4_three_state":          ("M4", "NG_plain"),  # plain arm only
    "txtl_resource_and_maturation_dna": ("M5", None),
    "txtl_model7_bg_fixed":             ("M7", None),
    "txtl_model8_bg_fixed":             ("M8", None),
    "txtl_model9_event_dark":           ("M9", "oxy00"),   # canonical M9 = event_dark lr0.002, no O2 penalty
}
ORDER = ["txtl_model3_two_state", "txtl_model4_three_state", "txtl_resource_and_maturation_dna",
         "txtl_model7_bg_fixed", "txtl_model8_bg_fixed", "txtl_model9_event_dark"]

# collect runs grouped by (scaffold, hidden)
by_hid = {k: defaultdict(list) for k in SCAF}
for D in DIRS:
    for cfgp in glob.glob(str(Path(D) / "*" / "config.yaml")):
        d = Path(cfgp).parent; cfg = yaml.safe_load(open(cfgp))
        sk = str(cfg.get("scaffold", ""))
        if sk not in SCAF or str(cfg.get("model_class", "")) != "ode_rnn": continue
        if "coarsenold" not in str(cfg.get("dataset_path", "")): continue
        _, namefilt = SCAF[sk]
        if namefilt and namefilt not in str(cfg.get("exp_name", "")): continue
        rc = d / "r2_cache.csv"
        if not rc.exists(): continue
        r = list(csv.DictReader(open(rc)))[-1]
        by_hid[sk][cfg.get("hidden")].append(dict(
            pm=float(r["r2_protein_final"]), p_sealed=float(r["r2_protein_old"]),
            p_opened=float(r["r2_protein_new"]), mm=float(r["r2_mrna_max"]),
            m_sealed=float(r["r2_mrna_old"]), m_opened=float(r["r2_mrna_new"])))

# AUTO-SELECT best hidden per scaffold = argmax seeded-mean protein R²
tbl = []   # (display, selected_hidden, rows)
for sk in ORDER:
    hids = by_hid[sk]
    if not hids: tbl.append((SCAF[sk][0], None, None)); continue
    best_h = max(hids, key=lambda h: np.mean([r["pm"] for r in hids[h]]))
    tbl.append((SCAF[sk][0], best_h, hids[best_h]))

out = ROOT / "results" / "coarse_ladder"; out.mkdir(parents=True, exist_ok=True)
METR = ("pm", "p_sealed", "p_opened", "mm", "m_sealed", "m_opened")   # protein{all,sealed,opened}, mRNA{...}

def raw_vals(rs, mode):
    """Return {metric: (point, sd_or_None)} for the row."""
    if mode == "mean":
        return {met: (float(np.mean([r[met] for r in rs])), float(np.std([r[met] for r in rs]))) for met in METR}
    br = max(rs, key=lambda r: r["pm"])                  # best-protein seed (one coherent model)
    return {met: (float(br[met]), None) for met in METR}

def fmt_cell(point, sd, tex, bold):
    if sd is None:
        s = f"{point:.2f}"
    else:
        s = f"{point:.2f}$_{{\\pm{sd:.2f}}}$" if tex else f"{point:.2f} ± {sd:.2f}"
    if not bold: return s
    return f"\\textbf{{{s}}}" if tex else f"**{s}**"

def emit(mode):
    title = ("seeded mean ± sd" if mode == "mean"
             else "best-protein seed (one model: protein-R² argmax, with its sealed/opened/mRNA)")
    raw = {disp: raw_vals(rs, mode) for disp, _, rs in tbl if rs}
    best = {met: max(v[met][0] for v in raw.values()) for met in METR}   # best per column (bold)
    md = [f"# Coarse dense-θ scaffold ladder ({title}); hidden auto-selected per scaffold", "",
          "| ODE | P:all | P:deoxygenated | P:oxygenated | mRNA:all | mRNA:deoxygenated | mRNA:oxygenated |",
          "|" + "---|"*7]
    # grouped header (no \cmidrule per request); consistent $R^2_{type}$ labels; mRNA overall col = 'all'.
    # {\small + tight tabcolsep, scoped} so the table fits one column of a 2-col layout.
    agg = (r"Mean$\,\pm\,$s.d.\ over seeds" if mode == "mean" else r"Best-protein seed per scaffold")
    cap = (r"\textbf{Cell-free mechanistic vector field ladder.} Held-out $R^2$ for mechanistic vector "
           r"fields M3--M9, each at its selected encoder hidden dimension, comparing CMVF predictions to "
           r"the two measurable channels: endpoint protein (the target product) and peak mRNA (which "
           r"drives translation). Each is reported over all test trajectories (\emph{all}) and by "
           r"protocol---\emph{deoxygenated} (tube closed) and \emph{oxygenated} (tube reopened at steady "
           r"state, admitting O$_2$). " + agg + r"; \textbf{bold} = best per column.")
    lab = "tab:scaffold_ladder" + ("" if mode == "mean" else "_max")
    # plain natural-width tabular (no tabular*). Add \centering or wrap in your full-width breakout
    # (\beginwide ... \endwide) for placement; with 6 columns it won't fit one multicol column.
    L = [r"\begin{table}[t]",
         r"\caption{" + cap + r"}", r"\label{" + lab + r"}",   # caption ON TOP
         r"\begin{tabular}{l ccc ccc}", r"\toprule",
         r" & \multicolumn{3}{c}{Protein $R^2_{\mathrm{endpoint}}$} & \multicolumn{3}{c}{mRNA $R^2_{\mathrm{peak}}$} \\",
         r"ODE & all & deoxygenated & oxygenated & all & deoxygenated & oxygenated \\", r"\midrule"]
    for disp, hid, rs in tbl:
        if not rs:
            md.append(f"| {disp} |" + " — |"*6)
            L.append(f"{disp} & " + " & ".join(["--"]*6) + r" \\"); continue
        rv = raw_vals(rs, mode)
        bmd = [fmt_cell(*rv[m], False, rv[m][0] == best[m]) for m in METR]
        btex = [fmt_cell(*rv[m], True, rv[m][0] == best[m]) for m in METR]
        md.append(f"| {disp} | " + " | ".join(bmd) + " |")
        L.append(f"{disp} & " + " & ".join(btex) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out / f"coarse_ladder_table_{mode}.tex").write_text("\n".join(L) + "\n")
    (out / f"coarse_ladder_table_{mode}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md) + "\n")

for mode in ("mean", "max"):
    emit(mode)

# ---------------------------------------------------------------------------
# Combined mean+max variants (one table showing both): "stacked" and "paren".
# Each cell: seeded mean ± s.d. as the headline number; the best single model
# (protein-R² argmax seed, its value in this column) as a secondary gray number.
# bold = best per column, computed separately for the mean and for the best-model row.
# NOTE: the combined .tex needs \usepackage{makecell} and \usepackage{xcolor}.
# ---------------------------------------------------------------------------
def fmt_mean_tex(point, sd, bold):
    s = f"{point:.2f}$_{{\\pm{sd:.2f}}}$"
    return f"\\textbf{{{s}}}" if bold else s

def fmt_max_tex(point, best, underline=False):
    s = f"{point:.2f}"
    if not best: return s
    return f"\\underline{{{s}}}" if underline else f"\\textbf{{{s}}}"

def emit_combined(layout):
    mean_rv = {disp: raw_vals(rs, "mean") for disp, _, rs in tbl if rs}
    max_rv  = {disp: raw_vals(rs, "max")  for disp, _, rs in tbl if rs}
    best_mean = {met: max(v[met][0] for v in mean_rv.values()) for met in METR}
    best_max  = {met: max(v[met][0] for v in max_rv.values())  for met in METR}
    is_paren = not layout.startswith("stacked")
    where = (r"in parentheses" if is_paren else r"below it")
    mark  = (r"\textbf{Bold} = best mean per column; \underline{underline} = best single run per column."
             if is_paren else
             r"\textbf{Bold} = best per column (mean and best run ranked separately).")
    cap = (r"\textbf{Cell-free mechanistic vector field ladder (typical and best).} Held-out $R^2$ for "
           r"mechanistic vector fields M3--M9, each at its selected encoder hidden dimension, on the two "
           r"measurable channels: endpoint protein (the target product) and peak mRNA (which drives "
           r"translation). Reported over all test trajectories (\emph{all}) and by protocol---"
           r"\emph{deoxygenated} (tube closed) and \emph{oxygenated} (tube reopened at steady state, "
           r"admitting O$_2$). Each cell gives the seeded mean$\,\pm\,$s.d.\ and, " + where + r", the "
           r"single best run: the seed with the highest overall protein $R^2$, with that run's score in "
           r"this column. " + mark)
    lab = "tab:scaffold_ladder_combined_" + layout
    L = [r"\begin{table}[t]", r"\caption{" + cap + r"}", r"\label{" + lab + r"}",
         r"\begin{tabular}{l ccc ccc}", r"\toprule",
         r" & \multicolumn{3}{c}{Protein $R^2_{\mathrm{endpoint}}$} & \multicolumn{3}{c}{mRNA $R^2_{\mathrm{peak}}$} \\",
         r"ODE & all & deoxygenated & oxygenated & all & deoxygenated & oxygenated \\",
         r"\midrule"]
    groups = [(disp, rs) for disp, _, rs in tbl if rs]
    for gi, (disp, rs) in enumerate(groups):
        mrv, xrv = mean_rv[disp], max_rv[disp]
        cells = []
        for m in METR:
            mean_s = fmt_mean_tex(*mrv[m], mrv[m][0] == best_mean[m])
            max_s  = fmt_max_tex(xrv[m][0], xrv[m][0] == best_max[m], underline=is_paren)
            if not is_paren:
                cells.append(r"\makecell[l]{" + mean_s + r"\\[-1.5pt]" + r"{\footnotesize " + max_s + r"}}")
            else:  # paren: underline marks the best max per column (bold reserved for the mean)
                cells.append(mean_s + r"~" + r"{\footnotesize (" + max_s + r")}")
        L.append(f"{disp} & " + " & ".join(cells) + r" \\")
        if layout == "stacked_div" and gi != len(groups) - 1:
            L.append(r"\cmidrule(lr){1-7}")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out / f"coarse_ladder_table_combined_{layout}.tex").write_text("\n".join(L) + "\n")
    return "\n".join(L)


def emit_combined_rows():
    """Two rows per ODE: a 'mean' row (mean ± s.d., the headline) and a 'best' row (the single
    best-protein seed). Numeric columns are right-aligned and every non-negative number
    reserves a leading sign slot (\\phantom{-}), so decimal points align across positive and
    negative rows; the best row pads the missing s.d. subscript so it aligns under the mean.
    Bold marks the best value per column among the mean rows only; groups are set off by a thin
    space, not a full-width rule (less busy). Needs \\usepackage{multirow}."""
    mean_rv = {disp: raw_vals(rs, "mean") for disp, _, rs in tbl if rs}
    max_rv  = {disp: raw_vals(rs, "max")  for disp, _, rs in tbl if rs}
    best_mean = {met: max(v[met][0] for v in mean_rv.values()) for met in METR}
    cap = (r"\textbf{Cell-free mechanistic vector field ladder (typical and best).} Held-out $R^2$ for "
           r"mechanistic vector fields M3--M9, each at its selected encoder hidden dimension, on the two "
           r"measurable channels: endpoint protein (the target product) and peak mRNA (which drives "
           r"translation). Reported over all test trajectories (\emph{all}) and by protocol---"
           r"\emph{deoxygenated} (tube closed) and \emph{oxygenated} (tube reopened at steady state, "
           r"admitting O$_2$). Each ODE has two rows: \emph{mean} is the average over seeds, $\pm$ s.d.; "
           r"\emph{best} is the single run with the highest overall protein $R^2$, with that same "
           r"run's score shown in every column. Because \emph{best} is one fixed run, its entry can sit "
           r"below the \emph{mean}. \textbf{Bold} marks the best value per column across the \emph{mean} rows.")
    # right-aligned numerics; reserve a '-' slot on non-negatives so decimals line up across signs.
    PAD  = r"\phantom{$_{\pm0.00}$}"                       # width of the mean's s.d. subscript
    sgn  = lambda x: "" if x < 0 else r"\phantom{-}"       # reserve the minus column on positives
    L = [r"\begin{table}[t]", r"\caption{" + cap + r"}", r"\label{tab:scaffold_ladder_combined_rows}",
         r"\begin{tabular}{l l rrr rrr}", r"\toprule",
         r" & & \multicolumn{3}{c}{Protein $R^2_{\mathrm{endpoint}}$} & \multicolumn{3}{c}{mRNA $R^2_{\mathrm{peak}}$} \\",
         r"ODE & & all & deoxygenated & oxygenated & all & deoxygenated & oxygenated \\",
         r"\midrule"]
    groups = [(disp, rs) for disp, _, rs in tbl if rs]
    for gi, (disp, rs) in enumerate(groups):
        mrv, xrv = mean_rv[disp], max_rv[disp]
        mean_cells = [sgn(mrv[m][0]) + fmt_mean_tex(*mrv[m], mrv[m][0] == best_mean[m]) for m in METR]
        best_cells = [sgn(xrv[m][0]) + f"{xrv[m][0]:.2f}" + PAD for m in METR]
        L.append(r"\multirow{2}{*}{" + disp + r"} & mean & " + " & ".join(mean_cells) + r" \\")
        L.append(r" & best & " + " & ".join(best_cells) + r" \\")
        if gi != len(groups) - 1:
            L.append(r"\addlinespace[3pt]")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out / "coarse_ladder_table_combined_rows.tex").write_text("\n".join(L) + "\n")
    return "\n".join(L)

combined = {lay: emit_combined(lay) for lay in ("stacked", "paren", "stacked_div")}
combined["rows"] = emit_combined_rows()

# standalone preview document with all variants, so they can be compared visually
preview = [
    r"\documentclass[11pt]{article}",
    r"\usepackage[margin=1in,landscape]{geometry}",
    r"\usepackage{booktabs,makecell,amsmath,multirow}",
    r"\usepackage[table]{xcolor}",
    r"\begin{document}",
    r"\section*{Variant C --- two rows per ODE (mean / best), stat column}",
    combined["rows"],
    r"\vspace{2em}",
    r"\section*{Variant A$'$ --- stacked cell with a divider between ODEs}",
    combined["stacked_div"],
    r"\vfill",
    r"\section*{Variant A --- stacked cell (no divider)}",
    combined["stacked"],
    r"\vspace{2em}",
    r"\section*{Variant B --- inline parenthetical}",
    combined["paren"],
    r"\end{document}",
]
(out / "combined_preview.tex").write_text("\n".join(preview) + "\n")

print("auto-selected hidden per scaffold (for the methods text):")
for disp, hid, rs in tbl:
    print(f"  {disp:26s} h={hid}  (n_seeds={len(rs) if rs else 0})")
print(f"wrote: {out}/coarse_ladder_table_{{mean,max,combined_stacked,combined_paren}}.tex")
print(f"       {out}/combined_preview.tex  (standalone, both variants)")
