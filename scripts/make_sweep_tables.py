#!/usr/bin/env python3
"""Build A4-friendly LaTeX appendix tables from the consolidated long table
(results/sweep_consolidated/all_results_long.csv, produced by
aggregate_from_cache.py).

Emits several layouts into separate subfolders so they can be compared:

  tables/per_scaffold/   one compact table per scaffold (rows=model,
                         cols=n; full-obs block then first-last block)
  tables/per_system/     one wide table per system (rows=model,
                         cols = scaffold x n) — landscape for wide ones

Each layout is produced for BOTH metrics (median, mean). Every folder also
gets a `_preview.md` (fixed-width, no LaTeX needed) and the LaTeX folders get
an `appendix_<layout>_<metric>.tex` wrapper that \\input's the tabulars with
captions, plus a README.md noting which tables need landscape.

Cell = overall NRMSE (mean over species of the per-species median|mean across
trajectories). Best (min) value per n-column-group is bolded. Missing cells
show as "--".
"""
from __future__ import annotations
import csv, os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LONG = os.path.join(ROOT, "results", "sweep_consolidated", "all_results_long.csv")
TBL  = os.path.join(ROOT, "results", "sweep_consolidated", "tables")
N_ORDER = [3, 10, 100, 1000]

DATA_MODEL_ORDER = ["baseline","l1reg","unbounded","node_gru","node_mlp","node_corr","fixed_theta","sample_theta","b1_static"]
ARCH_MODEL_ORDER = ["gru","lstm","slstm","transformer","mamba"]

# system grouping for the per-system layout: (group_title, [scaffold_label in order])
DATA_SYSTEMS = [
    ("Single enzyme", ["Enzyme/2","Enzyme/4","Enzyme/6"]),
    ("MOF synthesis", ["MOF/4","MOF/6","MOF/8","MOF/12"]),
    ("Glycolysis",    ["Glyc/4","Glyc/8","Glyc/12","Glyc/22"]),
]
ARCH_SYSTEMS = [
    ("Architecture sweep — single enzyme", ["Enzyme/2"]),
    ("Architecture sweep — MOF synthesis", ["MOF/4","MOF/6"]),
    ("Architecture sweep — glycolysis",    ["Glyc/8","Glyc/12"]),
]

SKIP_MODEL_KEYS = {"l2reg"}  # not run across all scaffolds — drop from tables

def _is_redundant(r):
    """Enzyme/2 (single_enzyme_lumped) is 2-state, so 'first_last' observation
    is identical to 'full' (the two observed species ARE first+last). Drop the
    first_last block for that scaffold in BOTH families (data ablation and the
    one stray arch-sweep GRU/FL run copied over from A6_baseline)."""
    return (r["scaffold_label"] == "Enzyme/2"
            and r["supervision"] == "first_last")

def load():
    rows = list(csv.DictReader(open(LONG)))
    rows = [r for r in rows
            if r["model_key"] not in SKIP_MODEL_KEYS and not _is_redundant(r)]
    for r in rows:
        r["n"] = int(r["n"]);
        for k in ("nrmse_median","nrmse_mean"):
            r[k] = float(r[k])
    return rows

def dedup(rows):
    """One value per (study, model_key, supervision, n). Prefer seed 42, else first."""
    g = defaultdict(list)
    for r in rows:
        g[(r["study"], r["model_key"], r["supervision"], r["n"])].append(r)
    out = {}
    for k, lst in g.items():
        lst.sort(key=lambda r: (r["seed"] != "42" and r["seed"] != 42, r["run"]))
        out[k] = lst[0]
    return out

def esc(label):
    """Escape underscores for LaTeX, but leave math (\\theta etc.) alone."""
    return label if "$" in label else label.replace("_", r"\_")

def fmt(x, tex=False):
    """Compact NRMSE cell (3 decimal places). Large (diverged) values -> scientific
    so the table stays legible and the blow-up is obvious."""
    if x != x: return "--"
    if x >= 1000:
        from math import floor, log10
        e = int(floor(log10(x))); m = x / 10**e
        return f"${m:.1f}\\!\\times\\!10^{{{e}}}$" if tex else f"{m:.1f}e{e}"
    return f"{x:.3f}"

# ── per-scaffold compact tables ──────────────────────────────────────────────
def label_for(rows, model_key):
    for r in rows:
        if r["model_key"] == model_key: return r["model_label"]
    return model_key

def build_per_scaffold(rows, ded, metric):
    """Return {scaffold_label: latex_tabular_str} and markdown preview blocks."""
    fld = f"nrmse_{metric}"
    by_scaf = defaultdict(list)
    for r in rows: by_scaf[(r["family"], r["scaffold_label"])].append(r)
    tex, md = {}, {}
    for (family, scaf), rs in sorted(by_scaf.items()):
        order = DATA_MODEL_ORDER if family=="data_ablation" else ARCH_MODEL_ORDER
        study = rs[0]["study"]
        sups = [s for s in ("full","first_last") if any(r["supervision"]==s for r in rs)]
        # build cells: model_key -> sup -> n -> val
        cell = defaultdict(lambda: defaultdict(dict))
        labels = {}
        for r in rs:
            cell[r["model_key"]][r["supervision"]][r["n"]] = r[fld]
            labels[r["model_key"]] = r["model_label"]
        present = [m for m in order if m in cell] + [m for m in cell if m not in order]
        # LaTeX
        L = ["\\begin{tabular}{l" + "r"*len(N_ORDER) + "}", "\\toprule",
             "Model & " + " & ".join(f"$n{{=}}{n}$" for n in N_ORDER) + r" \\"]
        M = [f"### {family} — {scaf}  (NRMSE {metric})", "",
             "| Model | " + " | ".join(f"n={n}" for n in N_ORDER) + " |",
             "|" + "---|"*(len(N_ORDER)+1)]
        for si, sup in enumerate(sups):
            suplabel = "Full observation" if sup=="full" else "First--last observation"
            L.append("\\midrule")
            L.append(f"\\multicolumn{{{1+len(N_ORDER)}}}{{l}}{{\\emph{{{suplabel}}}}}" + r" \\")
            M.append(f"| *{suplabel}* | | | | |")
            # best per column
            best = {}
            for n in N_ORDER:
                vals = [cell[m][sup].get(n) for m in present if cell[m][sup].get(n) is not None]
                vals = [v for v in vals if v==v]
                best[n] = min(vals) if vals else None
            for m in present:
                if sup not in cell[m]: continue
                row_tex, row_md = [esc(labels[m])], [labels[m]]
                for n in N_ORDER:
                    v = cell[m][sup].get(n)
                    s = fmt(v, tex=True) if v is not None else "--"
                    smd = fmt(v, tex=False) if v is not None else "--"
                    if v is not None and best[n] is not None and abs(v-best[n])<1e-12:
                        s = f"\\textbf{{{s}}}"; smd = f"**{smd}**"
                    row_tex.append(s); row_md.append(smd)
                L.append(" & ".join(row_tex) + r" \\")
                M.append("| " + " | ".join(row_md) + " |")
        L += ["\\bottomrule", "\\end{tabular}"]
        tex[(family,scaf)] = "\n".join(L)
        md[(family,scaf)]  = "\n".join(M)
    return tex, md

# ── per-system stacked tables (A4 portrait, longtable) ───────────────────────
# One narrow table per group: columns = the four n values; scaffolds (and, for
# the arch sweep, supervision settings) become labelled row-blocks separated by
# \midrule. Uses longtable so a long group flows across pages with a repeating
# header — the standard academic way to combine many sub-conditions in one table.
def build_per_system(rows, ded, metric, systems, family):
    fld = f"nrmse_{metric}"
    cell = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    labels = {}
    for r in rows:
        if r["family"]!=family: continue
        cell[r["scaffold_label"]][r["model_key"]][r["supervision"]][r["n"]] = r[fld]
        labels[r["model_key"]] = r["model_label"]
    order = DATA_MODEL_ORDER if family=="data_ablation" else ARCH_MODEL_ORDER
    present_all = [m for m in order if m in labels]
    # data ablation: one table per (system, supervision) — keeps each ~one page.
    # arch sweep: one table per scenario, supervision shown as inner row-blocks.
    split_by_sup = (family == "data_ablation")
    # tabular* with @{\extracolsep{\fill}} stretches to fill its minipage width
    # exactly => every sibling table is the SAME width regardless of content.
    colspec = "@{\\extracolsep{\\fill}}l" + "r"*len(N_ORDER)
    ncol = 1 + len(N_ORDER)
    header = "Model & " + " & ".join(f"$n={n}$" for n in N_ORDER) + r" \\"

    def block(s, sup, show_sup):
        """Return (tex_lines, md_lines) for one scaffold[/sup] block, or ([],[])."""
        if not any(cell[s][m].get(sup) for m in present_all): return [], []
        lab = s if not show_sup else f"{s} — {'full obs' if sup=='full' else 'first--last'}"
        best = {}
        for n in N_ORDER:
            vs = [cell[s][m][sup].get(n) for m in present_all if cell[s][m][sup].get(n) is not None]
            vs = [v for v in vs if v == v]; best[n] = min(vs) if vs else None
        T = [f"\\multicolumn{{{ncol}}}{{l}}{{\\textbf{{{lab}}}}} \\\\"]
        M = [f"| **{lab}** |" + " |"*len(N_ORDER)]
        for m in present_all:
            if not cell[s][m].get(sup): continue
            rt, rm = [esc(labels[m])], [labels[m]]
            for n in N_ORDER:
                v = cell[s][m][sup].get(n)
                t  = fmt(v, tex=True)  if v is not None else "--"
                md = fmt(v, tex=False) if v is not None else "--"
                if v is not None and best[n] is not None and abs(v-best[n]) < 1e-12:
                    t = f"\\textbf{{{t}}}"; md = f"**{md}**"
                rt.append(t); rm.append(md)
            T.append(" & ".join(rt) + r" \\")
            M.append("| " + " | ".join(rm) + " |")
        return T, M

    tex, md = {}, {}
    for title, scafs in systems:
        scafs = [s for s in scafs if s in cell]
        if not scafs: continue
        sups_present = [s for s in ("full","first_last") if any(cell[sc][m][s] for sc in scafs for m in cell[sc])]
        tables = [(sup, [sup]) for sup in sups_present] if split_by_sup else [("all", sups_present)]
        for suptag, sup_list in tables:
            blocks_t, blocks_m = [], []
            show_sup = len(sup_list) > 1
            for s in scafs:
                for sup in sup_list:
                    bt, bm = block(s, sup, show_sup)
                    if bt: blocks_t.append(bt); blocks_m.append(bm)
            if not blocks_t: continue
            captsup = (", full obs" if suptag=="full" else ", first--last") if split_by_sup else ""
            caption = f"{title} — NRMSE ({metric}{captsup})."
            L = [f"% {caption}",
                 f"\\begin{{tabular*}}{{\\linewidth}}{{{colspec}}}",
                 "\\toprule", header, "\\midrule"]
            for bi, bt in enumerate(blocks_t):
                if bi: L.append("\\midrule")
                L += bt
            L += ["\\bottomrule", "\\end{tabular*}"]
            Mh = [f"### {caption}", "", "| Model | " + " | ".join(f"n={n}" for n in N_ORDER) + " |",
                  "|" + "---|"*(ncol)]
            for bi, bm in enumerate(blocks_m):
                Mh += bm
            tex[(title, suptag)] = "\n".join(L)
            md[(title, suptag)]  = "\n".join(Mh)
    return tex, md

def slug(s):
    import re
    return re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_").lower()

def main():
    import shutil
    shutil.rmtree(TBL, ignore_errors=True)  # clean stale tables before regenerating
    rows = load(); ded = dedup(rows)
    for metric in ("median","mean"):
        # per scaffold
        d = os.path.join(TBL,"per_scaffold",metric); os.makedirs(d,exist_ok=True)
        tex,md = build_per_scaffold(rows,ded,metric)
        wrap=["% auto-generated; \\usepackage{booktabs}","",]
        preview=[f"# Per-scaffold tables — NRMSE {metric}",""]
        for (family,scaf),t in sorted(tex.items()):
            fn=f"{family}_{slug(scaf)}_{metric}.tex"
            open(os.path.join(d,fn),"w").write(t+"\n")
            wrap.append("\\begin{table}[t]\\centering\\footnotesize")
            wrap.append(f"\\caption{{{family.replace('_',' ')}: {scaf} — NRMSE ({metric}).}}")
            wrap.append(f"\\input{{{fn}}}")
            wrap.append("\\end{table}")
            preview.append(md[(family,scaf)]); preview.append("")
        open(os.path.join(d,f"appendix_per_scaffold_{metric}.tex"),"w").write("\n".join(wrap)+"\n")
        open(os.path.join(d,"_preview.md"),"w").write("\n".join(preview)+"\n")

        # per system (stacked tabular, A4 portrait — paired two-up via minipages
        # so every page shows two side-by-side tables; scaffolds are row-blocks).
        d2 = os.path.join(TBL,"per_system",metric); os.makedirs(d2,exist_ok=True)
        wrap2=["% auto-generated; \\usepackage{booktabs,caption,subcaption}",
               "\\captionsetup{font=footnotesize,labelfont=bf,skip=4pt}",
               "\\captionsetup[sub]{font=footnotesize,labelfont=bf,skip=2pt}",""]
        preview2=[f"# Per-system tables — NRMSE {metric}", "",
                  "_Abbreviations: CMVF-unb. = CMVF-unbounded; IC (θ) = initial-condition θ; NODE-corr. = NODE-correction._", ""]

        # Build per-family dict: {title: {sup: (fn, md_str)}}
        by_family = {}
        for family,systems in (("data_ablation",DATA_SYSTEMS),("arch_sweep",ARCH_SYSTEMS)):
            tex2,md2 = build_per_system(rows,ded,metric,systems,family)
            entries = {}
            for (title,sup),t in tex2.items():
                fn = f"{family}_{slug(title)}_{sup}_{metric}.tex"
                open(os.path.join(d2,fn),"w").write(t+"\n")
                entries.setdefault(title,{})[sup] = (fn, md2[(title,sup)])
            by_family[family] = entries

        # Data ablation: one float per SYSTEM with ONE main caption and (a)(b)
        # subcaptions for full / first-last side-by-side.
        for title, _ in DATA_SYSTEMS:
            ent = by_family.get("data_ablation",{}).get(title)
            if not ent: continue
            sups = [s for s in ("full","first_last") if s in ent]
            wrap2.append("\\begin{table}[t]\\centering\\footnotesize")
            wrap2.append("\\setlength{\\tabcolsep}{2pt}")
            wrap2.append(f"\\caption{{{title} --- NRMSE ({metric}).}}")
            for j, sup in enumerate(sups):
                fn, _ = ent[sup]
                lab = "Full observation" if sup=="full" else "First--last observation"
                wrap2.append("\\begin{minipage}[t]{0.49\\textwidth}\\centering")
                wrap2.append(f"\\subcaption{{{lab}}}")
                wrap2.append(f"\\input{{{fn}}}")
                wrap2.append("\\end{minipage}")
                if j == 0 and len(sups) == 2:
                    wrap2.append("\\hfill")
            wrap2.append("\\end{table}")
            for sup in sups:
                preview2.append(ent[sup][1]); preview2.append("")

        # Arch sweep: one float per SCENARIO (full + first-last are already inner
        # row-blocks). Pair scenarios 2-up with separate captions (unrelated).
        arch_items = []
        for title, _ in ARCH_SYSTEMS:
            ent = by_family.get("arch_sweep",{}).get(title)
            if not ent: continue
            sup_key = next(iter(ent.keys()))
            fn, md_str = ent[sup_key]
            arch_items.append((title, fn, md_str))
        for i in range(0, len(arch_items), 2):
            pair = arch_items[i:i+2]
            wrap2.append("\\begin{table}[t]\\centering\\footnotesize")
            wrap2.append("\\setlength{\\tabcolsep}{2pt}")
            for j, (title, fn, _) in enumerate(pair):
                wrap2.append("\\begin{minipage}[t]{0.49\\textwidth}\\centering")
                wrap2.append(f"\\captionof{{table}}{{{title} --- NRMSE ({metric}).}}")
                wrap2.append(f"\\input{{{fn}}}")
                wrap2.append("\\end{minipage}")
                if j == 0 and len(pair) == 2:
                    wrap2.append("\\hfill")
            wrap2.append("\\end{table}")
            for _, _, md_str in pair:
                preview2.append(md_str); preview2.append("")
        open(os.path.join(d2,f"appendix_per_system_{metric}.tex"),"w").write("\n".join(wrap2)+"\n")
        open(os.path.join(d2,"_preview.md"),"w").write("\n".join(preview2)+"\n")

    # top-level README
    readme = f"""# Sweep result tables

Auto-generated by `scripts/make_sweep_tables.py` from
`results/sweep_consolidated/all_results_long.csv`.

Cell = overall NRMSE = mean over species of the per-species (median|mean across
trajectories). Best (min) per n-column is **bold**. `--` = no completed run.
Diverged runs show as `m x 10^e`. Model labels: data-ablation uses
CMVF / CMVF-L1 / CMVF-L2 / CMVF-unbounded / NODE-GRU / NODE-MLP / NODE-correction
/ Global ($\\theta$) / Initial-condition ($\\theta$); the arch sweep uses
GRU / LSTM / sLSTM / Transformer / Mamba.

## Layouts (pick whichever reads best)
- `per_scaffold/{{median,mean}}/` — one compact table per scaffold (rows=model,
  cols=n; full then first-last block). 2-3 tile per portrait A4. Start here.
- `per_system/{{median,mean}}/` — **stacked `longtable` per group, A4 portrait**.
  Columns = the four $n$ values (just 4 numeric columns); scaffolds (and, for
  the arch sweep, observation settings) are labelled row-blocks separated by
  `\\midrule`. Data ablation = one table per (system, observation); architecture
  sweep = one table per scenario (single enzyme / MOF / glycolysis) with
  observation as inner row-blocks. Page-spanning is automatic via `longtable`
  with repeating headers. No landscape, no resize.

## Compiled PDFs
Run `bash scripts/build_table_pdfs.sh` to compile each layout/metric into
`preview_<layout>_<metric>.pdf` here. (Run it AFTER make_sweep_tables.py, which
wipes this folder.) All four currently compile with 0 overfull boxes on A4.

Each folder also has a `_preview.md` (rendered numbers, no LaTeX needed) and an
`appendix_*.tex` wrapper that \\input's the tabulars with captions. Preamble:
`\\usepackage{{booktabs}}` (+ `pdflscape,graphicx` for per_system).
"""
    open(os.path.join(TBL,"README.md"),"w").write(readme)
    print("Wrote tables under", os.path.relpath(TBL,ROOT))

if __name__ == "__main__":
    main()
