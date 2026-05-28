#!/usr/bin/env python3
"""Torch-free re-aggregation of sweep results straight from per-run
`nrmse_cache.csv` files (no model rebuild needed).

Reproduces the recipe in last-layer-ode/metrics/compare_runs.py:
  per-run primary NRMSE = mean over species of the per-species
  (median | mean across trajectories), which is already cached.

For each study it writes legacy-format `<study>_{median,mean}.csv`
(run, nrmse, <species...>) and contributes to one consolidated long-form
table. It also validates the run NAME's _n token against the dataset path
recorded in config.yaml and reports any mismatch or missing cache.

Usage:
    python scripts/aggregate_from_cache.py            # all studies
    python scripts/aggregate_from_cache.py --report   # validation report only
"""
from __future__ import annotations
import argparse, csv, os, re
import yaml
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP  = os.path.join(ROOT, "experiments")
OUT_CSV_DIR = os.path.join(ROOT, "results", "sweep_consolidated", "csv")
OUT_LONG    = os.path.join(ROOT, "results", "sweep_consolidated", "all_results_long.csv")

# ── study registry ───────────────────────────────────────────────────────────
# family, scaffold-display, list of (n, folder). n=100 lives in the un-suffixed base.
DATA_STUDIES = [  # (out_stem, family, scaffold_label, folder)
    ("single_enzyme_lumped_data_ablation", "data_ablation", "Enzyme/2", "single_enzyme_lumped_data_ablation"),
    ("single_enzyme_4_data_ablation",      "data_ablation", "Enzyme/4", "single_enzyme_4_data_ablation"),
    ("single_enzyme_6_data_ablation",      "data_ablation", "Enzyme/6", "single_enzyme_6_data_ablation"),
    ("mof_synthesis_4_data_ablation",      "data_ablation", "MOF/4",    "mof_synthesis_4_data_ablation"),
    ("mof_synthesis_6_data_ablation",      "data_ablation", "MOF/6",    "mof_synthesis_6_data_ablation"),
    ("mof_synthesis_8_data_ablation",      "data_ablation", "MOF/8",    "mof_synthesis_8_data_ablation"),
    ("mof_synthesis_12_data_ablation",     "data_ablation", "MOF/12",   "mof_synthesis_12_data_ablation"),
    ("glycolysis_reduced4_data_ablation",  "data_ablation", "Glyc/4",   "glycolysis_reduced4_data_ablation"),
    ("glycolysis_reduced8_data_ablation",  "data_ablation", "Glyc/8",   "glycolysis_reduced8_data_ablation"),
    ("glycolysis_reduced12_data_ablation", "data_ablation", "Glyc/12",  "glycolysis_reduced12_data_ablation"),
    ("glycolysis_full22_data_ablation",    "data_ablation", "Glyc/22",  "glycolysis_full22_data_ablation"),
]
# arch sweep: out_stem (per n) -> folder ; base folder = n100
ARCH_STUDIES = [  # (study_key, family, scaffold_label, {n: folder})
    ("single_enzyme_lumped_arch_sweep", "arch_sweep", "Enzyme/2", {
        100:"single_enzyme_lumped_arch_sweep",3:"single_enzyme_lumped_arch_sweep_n3",
        10:"single_enzyme_lumped_arch_sweep_n10",1000:"single_enzyme_lumped_arch_sweep_n1000"}),
    ("mof_synthesis_4_arch_sweep", "arch_sweep", "MOF/4", {
        100:"mof_synthesis_4_arch_sweep",3:"mof_synthesis_4_arch_sweep_n3",
        10:"mof_synthesis_4_arch_sweep_n10",1000:"mof_synthesis_4_arch_sweep_n1000"}),
    ("mof_synthesis_6_arch_sweep", "arch_sweep", "MOF/6", {
        100:"mof_synthesis_6_arch_sweep",3:"mof_synthesis_6_arch_sweep_n3",
        10:"mof_synthesis_6_arch_sweep_n10",1000:"mof_synthesis_6_arch_sweep_n1000"}),
    ("glycolysis_reduced8_arch_sweep", "arch_sweep", "Glyc/8", {
        100:"glycolysis_reduced8_arch_sweep",3:"glycolysis_reduced8_arch_sweep_n3",
        10:"glycolysis_reduced8_arch_sweep_n10",1000:"glycolysis_reduced8_arch_sweep_n1000"}),
    ("glycolysis_reduced12_arch_sweep", "arch_sweep", "Glyc/12", {
        100:"glycolysis_reduced12_arch_sweep",3:"glycolysis_reduced12_arch_sweep_n3",
        10:"glycolysis_reduced12_arch_sweep_n10",1000:"glycolysis_reduced12_arch_sweep_n1000"}),
]

# ── label canonicalisation ───────────────────────────────────────────────────
DATA_ABL_LABEL = {
    "A1_fixed_theta": ("fixed_theta", r"Global ($\theta$)"),
    "A2_sample_theta": ("sample_theta", r"Initial-condition ($\theta$)"),
    "A3_neural_ode_gru": ("node_gru", "NODE-GRU"),
    "A4_neural_ode_mlp": ("node_mlp", "NODE-MLP"),
    "A5_unbounded": ("unbounded", "CMVF-unbounded"),
    "A6_baseline": ("baseline", "CMVF"),
    "A7_l1reg": ("l1reg", "CMVF-L1"),
    "A8_l2reg": ("l2reg", "CMVF-L2"),
    "A9_neural_ode_correction": ("node_corr", "NODE-correction"),
}
# arch sweep tokens (order matters: check A6_baseline & slstm before lstm)
ARCH_TOKENS = [
    ("A6_baseline", "gru", "GRU"),
    ("slstm", "slstm", "sLSTM"),
    ("lstm", "lstm", "LSTM"),
    ("trans", "transformer", "Transformer"),
    ("mamba", "mamba", "Mamba"),
]

def strip_ts(name): return re.sub(r"^\d{8}_\d{6}_", "", name)
def n_from(s):
    m = re.search(r"_n(\d+)(?:_|\.|$)", s); return int(m.group(1)) if m else None

def read_cache(d):
    p = os.path.join(d, "nrmse_cache.csv")
    if not os.path.exists(p): return None
    rows = list(csv.DictReader(open(p)))
    if not rows: return None
    species = {}
    for r in rows:
        try:
            species[r["species"]] = (float(r["median"]), float(r["mean"]), int(r["n_samples"]))
        except (ValueError, KeyError):
            return None
    return species

def read_cfg(d):
    p = os.path.join(d, "config.yaml")
    if not os.path.exists(p): return {}
    try: return yaml.safe_load(open(p)) or {}
    except Exception: return {}

def classify(name, family, cfg):
    """Return (model_key, model_label, supervision)."""
    sup = "first_last" if ("first_last" in name or re.search(r"_fl(_|$)", name)
                           or (cfg.get("obs_idx") not in (None, "null"))) else "full"
    if family == "data_ablation":
        for tok,(k,lab) in DATA_ABL_LABEL.items():
            if name.startswith(tok): return k, lab, sup
        return "unknown", name, sup
    else:  # arch_sweep
        for tok,k,lab in ARCH_TOKENS:
            if name.startswith(tok) or re.search(rf"(^|_){tok}(_|$)", name):
                return k, lab, sup
        return "unknown", name, sup

def primary(species, stat_idx):
    vals = [v[stat_idx] for v in species.values()]
    vals = [x for x in vals if x == x]  # drop NaN
    return sum(vals)/len(vals) if vals else float("nan")

# ── main aggregation ─────────────────────────────────────────────────────────
def collect(folder, family):
    """Return list of run records for one folder."""
    recs = []
    full = os.path.join(EXP, folder)
    if not os.path.isdir(full): return recs, [f"FOLDER MISSING: {folder}"]
    notes = []
    for d in sorted(os.listdir(full)):
        rd = os.path.join(full, d)
        if not os.path.isdir(rd) or not os.path.exists(os.path.join(rd,"config.yaml")):
            continue
        name = strip_ts(d)
        cfg = read_cfg(rd)
        species = read_cache(rd)
        n_name = n_from(name)
        n_cfg  = n_from(str(cfg.get("dataset_path","")))
        if species is None:
            notes.append(f"NO/UNREADABLE CACHE: {folder}/{d}")
            continue
        if n_name is None:
            notes.append(f"NAME LACKS _n: {folder}/{d} (dataset n={n_cfg})")
        if n_name is not None and n_cfg is not None and n_name != n_cfg:
            notes.append(f"N MISMATCH name={n_name} cfg={n_cfg}: {folder}/{d}")
        k, lab, sup = classify(name, family, cfg)
        seed_m = re.search(r"_s(\d+)(?:_|$)", name)
        recs.append(dict(
            run=name, model_key=k, model_label=lab, supervision=sup,
            n=n_name if n_name is not None else n_cfg,
            seed=int(seed_m.group(1)) if seed_m else None,
            species=species,
            nrmse_median=primary(species,0), nrmse_mean=primary(species,1),
            scaffold=cfg.get("scaffold",""), dataset_path=cfg.get("dataset_path",""),
        ))
    return recs, notes

def write_legacy_csv(recs, stem, stat):  # stat: 'median'|'mean'
    idx = 0 if stat=="median" else 1
    # union of species preserving first-seen order
    species_order = []
    for r in recs:
        for s in r["species"]:
            if s not in species_order: species_order.append(s)
    recs_sorted = sorted(recs, key=lambda r: r[f"nrmse_{stat}"])
    os.makedirs(OUT_CSV_DIR, exist_ok=True)
    out = os.path.join(OUT_CSV_DIR, f"{stem}_{stat}.csv")
    with open(out,"w",newline="") as f:
        w = csv.writer(f); w.writerow(["run","nrmse"]+species_order)
        for r in recs_sorted:
            row=[r["run"], f"{r[f'nrmse_{stat}']:.6f}"]
            for s in species_order:
                v=r["species"].get(s); row.append(f"{v[idx]:.6f}" if v else "")
            w.writerow(row)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="only print validation report")
    args = ap.parse_args()

    long_rows = []
    all_notes = []
    written = []

    # data ablation: one folder per study
    for stem, family, label, folder in DATA_STUDIES:
        recs, notes = collect(folder, family)
        all_notes += notes
        if recs and not args.report:
            written.append(write_legacy_csv(recs, stem, "median"))
            written.append(write_legacy_csv(recs, stem, "mean"))
        for r in recs:
            long_rows.append((family, stem, label, r["scaffold"], r["model_key"], r["model_label"],
                              r["supervision"], r["n"], r["seed"], r["nrmse_median"], r["nrmse_mean"],
                              len(r["species"]), r["run"], r["dataset_path"]))

    # arch sweep: combine all n-folders into one study, but also write per-n legacy CSVs
    for study_key, family, label, nmap in ARCH_STUDIES:
        combined = []
        for n, folder in sorted(nmap.items()):
            recs, notes = collect(folder, family)
            all_notes += notes
            # override n by folder mapping when the run name lacks it
            for r in recs:
                if r["n"] is None: r["n"] = n
            combined += recs
            if recs and not args.report:
                suffix = "" if n==100 else f"_n{n}"
                written.append(write_legacy_csv(recs, f"{study_key}{suffix}", "median"))
                written.append(write_legacy_csv(recs, f"{study_key}{suffix}", "mean"))
        for r in combined:
            long_rows.append((family, study_key, label, r["scaffold"], r["model_key"], r["model_label"],
                              r["supervision"], r["n"], r["seed"], r["nrmse_median"], r["nrmse_mean"],
                              len(r["species"]), r["run"], r["dataset_path"]))

    if not args.report:
        os.makedirs(os.path.dirname(OUT_LONG), exist_ok=True)
        with open(OUT_LONG,"w",newline="") as f:
            w=csv.writer(f)
            w.writerow(["family","study","scaffold_label","scaffold","model_key","model_label",
                        "supervision","n","seed","nrmse_median","nrmse_mean","n_species","run","dataset_path"])
            for row in long_rows: w.writerow(row)
        print(f"Wrote {len(written)} per-study CSVs to {os.path.relpath(OUT_CSV_DIR,ROOT)}/")
        print(f"Wrote consolidated long table: {os.path.relpath(OUT_LONG,ROOT)}  ({len(long_rows)} rows)")

    print(f"\n=== VALIDATION ({len(all_notes)} note(s)) ===")
    for nse in all_notes: print("  "+nse)
    if not all_notes: print("  clean — every run cached, every name's _n matches its dataset.")

if __name__ == "__main__":
    main()
