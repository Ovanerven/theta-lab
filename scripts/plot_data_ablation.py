"""
Generate comprehensive data-ablation plots for dataset groups.

Output structures:
  1. Individual plots: figures/data_ablation/individual/<dataset>/...
  2. 1x3 Row plots:    figures/data_ablation/1x3_rows/<group_name>/...
  3. 3x3 Full Grid:    figures/data_ablation/3x3_grid/...
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import re
import os

# ── NeurIPS True-to-Print rcParams ──────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 7,             
    "axes.labelsize": 7,        
    "axes.titlesize": 7,
    "xtick.labelsize": 7,       
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,     
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.4,
    "lines.linewidth": 0.9,     # Reduced global line thickness
    "lines.markersize": 3.0,    # Slightly smaller markers to match thinner lines
})

# ── Architecture config ─────────────────────────────────────────────────────
# Linewidths explicitly reduced here for a cleaner, thinner look
ARCH_STYLE = {
    "A6_baseline":              ("CMVF (Ours)",          "#e41a1c", 5, 1.4),
    "A7_l1reg":                 ("CMVF-L1",              "#4dac26", 3, 0.9),
    "A5_unbounded":             ("CMVF-unbounded",       "#984ea3", 3, 0.9),
    "A3_neural_ode_gru":        ("NODE-GRU",             "#377eb8", 2, 0.9),
    "A4_neural_ode_mlp":        ("NODE-MLP",             "#ff7f00", 2, 0.9),
    "A9_neural_ode_correction": ("NODE-correction",      "#a65628", 2, 0.9),
    "A1_fixed_theta":           (r"Global ($\theta$)",   "#999999", 1, 0.7),
    "A2_sample_theta":          (r"IC ($\theta$)",       "#f781bf", 1, 0.7),
}

# ── Base Directories ─────────────────────────────────────────────────────────
BASE_DATA = os.path.join(os.path.dirname(__file__), "..", "results", "final_data_ablation")
BASE_FIGS = os.path.join(os.path.dirname(__file__), "..", "figures", "data_ablation")

COMBINED_FIGURES = {
    "mof_combined": [
        ("mof_synthesis_6_data_ablation",   "MOF/6-state"),
        ("mof_synthesis_8_data_ablation",   "MOF/8-state"),
        ("mof_synthesis_12_data_ablation",  "MOF/12-state"),
    ],
    "glycolysis_combined": [
        ("glycolysis_reduced4_data_ablation",   "Glycolysis/4-state"),
        ("glycolysis_reduced8_data_ablation",   "Glycolysis/8-state"),
        ("glycolysis_reduced12_data_ablation",  "Glycolysis/12-state"),
    ],
    "single_enzyme_combined": [
        ("single_enzyme_lumped_data_ablation",  "Enzyme/2-state"),
        ("single_enzyme_4_data_ablation",       "Enzyme/4-state"),
        ("single_enzyme_6_data_ablation",       "Enzyme/6-state"),
    ]
}

N_VALUES    = [3, 10, 100, 1000]
DISPLAY_CAP = 10000000000000000

_ARCH_PAT = "(" + "|".join(re.escape(k) for k in ARCH_STYLE) + ")"
RUN_PATTERN_STD  = re.compile(rf"^{_ARCH_PAT}_n(\d+)$")
RUN_PATTERN_FL   = re.compile(rf"^{_ARCH_PAT}_first_last_n(\d+)$")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def parse_runs(df: pd.DataFrame, first_last: bool = False) -> dict:
    pattern = RUN_PATTERN_FL if first_last else RUN_PATTERN_STD
    data = {k: {} for k in ARCH_STYLE}
    for _, row in df.iterrows():
        m = pattern.match(str(row["run"]))
        if m:
            arch, n, nrmse = m.group(1), int(m.group(2)), float(row["nrmse"])
            if n in N_VALUES and nrmse > 0:
                data[arch][n] = nrmse
    return data

def extract_universal_legend(axes_list):
    handles_dict = {}
    for ax in axes_list:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in handles_dict:
                handles_dict[label] = handle
                
    ordered_labels, ordered_handles = [], []
    for arch_key, (arch_label, _, _, _) in ARCH_STYLE.items():
        if arch_label in handles_dict:
            ordered_labels.append(arch_label)
            ordered_handles.append(handles_dict[arch_label])
    return ordered_handles, ordered_labels

# =============================================================================
# 1. INDIVIDUAL PLOT GENERATOR (1x1)
# =============================================================================
def make_single_plot(title: str, arch_data: dict, stat_label: str, log_y: bool, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    
    for arch, (label, color, zo, lw) in ARCH_STYLE.items():
        pts = arch_data[arch]
        if len(pts) < 2: continue
        xs = [x for x in sorted(pts) if pts[x] <= DISPLAY_CAP]
        ys = [pts[x] for x in xs]
        if not xs: continue
        
        ax.plot(xs, ys, color=color, linestyle="-", linewidth=lw*1.5,
                marker="o", markersize=4, zorder=zo, label=label)

    ax.set_xscale("log")
    ax.set_xticks(N_VALUES)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

    if log_y:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation(labelOnlyBase=False))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ylabel = f"NRMSE ({stat_label}, log)"
    else:
        ylabel = f"NRMSE ({stat_label})"

    ax.set_xlim(2, 1500)
    ax.set_xlabel("N", labelpad=4)
    ax.set_ylabel(ylabel, labelpad=4)
    
    ax.legend(loc="upper right", frameon=True, fontsize=7.5)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)

# =============================================================================
# 2. ROW COMBINED PLOT GENERATOR (1x3)
# =============================================================================
def make_combined_plot(datasets_data: list, stat_label: str, log_y: bool, out_path: str) -> None:
    n_cols = len(datasets_data)
    # sharey removed! They will scale independently to fit the data perfectly.
    fig, axes = plt.subplots(1, n_cols, figsize=(5.5, 2.1))
    if n_cols == 1: axes = [axes]

    for i, (title, arch_data) in enumerate(datasets_data):
        ax = axes[i]
        for arch, (label, color, zo, lw) in ARCH_STYLE.items():
            pts = arch_data[arch]
            if len(pts) < 2: continue
            xs = [x for x in sorted(pts) if pts[x] <= DISPLAY_CAP]
            ys = [pts[x] for x in xs]
            if not xs: continue
            ax.plot(xs, ys, color=color, linestyle="-", linewidth=lw,
                    marker="o", markersize=plt.rcParams['lines.markersize'], zorder=zo, label=label)

        ax.set_xscale("log")
        ax.set_xticks(N_VALUES)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

        if log_y:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation(labelOnlyBase=False))
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())
            ylabel = f"NRMSE ({stat_label}, log)"
        else:
            ylabel = f"NRMSE ({stat_label})"

        ax.set_xlim(2, 1500)
        ax.set_xlabel("N", labelpad=4)
        
        # Because scaling is independent, every plot gets a Y label again
        ax.set_ylabel(ylabel, labelpad=4)
        
        subcaption = f"({chr(97 + i)}) {title}"
        ax.text(0.5, -0.38, subcaption, transform=ax.transAxes, fontsize=9, ha="center", va="top")

    handles, labels = extract_universal_legend(axes)
    if handles:
        # bbox_to_anchor Y lowered to 1.03 for a super tight hug to the top of the plot
        axes[1].legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.03), 
                       ncol=4, frameon=True, columnspacing=1.0, handletextpad=0.3, borderaxespad=0.0)

    # Removed the rigid 'top' margin to let tight_layout work dynamically
    plt.subplots_adjust(wspace=0.35, bottom=0.25)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)

# =============================================================================
# 3. FULL GRID PLOT GENERATOR (3x3)
# =============================================================================
def make_grid_plot(grid_data: list, stat_label: str, log_y: bool, out_path: str) -> None:
    n_rows = len(grid_data)
    n_cols = len(grid_data[0]) if n_rows > 0 else 0
    
    # sharey removed! They will scale independently to fit the data perfectly.
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5, 6.0))

    for row_idx, row_data in enumerate(grid_data):
        for col_idx, (title, arch_data) in enumerate(row_data):
            ax = axes[row_idx, col_idx]

            for arch, (label, color, zo, lw) in ARCH_STYLE.items():
                pts = arch_data[arch]
                if len(pts) < 2: continue
                xs = [x for x in sorted(pts) if pts[x] <= DISPLAY_CAP]
                ys = [pts[x] for x in xs]
                if not xs: continue
                ax.plot(xs, ys, color=color, linestyle="-", linewidth=lw,
                        marker="o", markersize=plt.rcParams['lines.markersize'], zorder=zo, label=label)

            ax.set_xscale("log")
            ax.set_xticks(N_VALUES)
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

            if log_y:
                ax.set_yscale("log")
                ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation(labelOnlyBase=False))
                ax.yaxis.set_minor_formatter(mticker.NullFormatter())
                ylabel = f"NRMSE ({stat_label}, log)"
            else:
                ylabel = f"NRMSE ({stat_label})"

            ax.set_xlim(2, 1500)
            ax.set_xlabel("N", labelpad=4)
            
            # Every plot gets a Y label to be mathematically clear when sharing is off
            ax.set_ylabel(ylabel, labelpad=4)

            subcaption = f"({chr(97 + row_idx * n_cols + col_idx)}) {title}"
            ax.text(0.5, -0.38, subcaption, transform=ax.transAxes, fontsize=7, ha="center", va="top")

    handles, labels = extract_universal_legend(axes.flatten())
    if handles:
        # bbox_to_anchor Y lowered to 1.03 for a super tight hug
        axes[0, 1].legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.03), 
                          ncol=4, frameon=True, columnspacing=1.0, handletextpad=0.3, borderaxespad=0.0)

    # Increased wspace slightly so the restored Y-labels don't bump into each other
    plt.subplots_adjust(wspace=0.35, hspace=0.6, bottom=0.1)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)

# =============================================================================
# MAIN EXECUTION LOOP
# =============================================================================
MASKING_VARIANTS = [
    ("standard",   False),
    ("first_last", True),
]

for mask_tag, first_last in MASKING_VARIANTS:
    for stat_label in ["mean", "median"]:
        for log_y, scale_tag in [(False, "linear"), (True, "log")]:
            
            grid_data_all_rows = []
            skip_grid = False

            for group_name, dataset_group in COMBINED_FIGURES.items():
                group_parsed_data = []
                missing_files = False

                for stem, title in dataset_group:
                    file_path = os.path.join(BASE_DATA, f"{stem}_{stat_label}.csv")
                    
                    if not os.path.exists(file_path):
                        print(f"  SKIP {stem} (missing {stat_label} file)")
                        missing_files = True
                        break
                    
                    df = pd.read_csv(file_path)
                    arch_data = parse_runs(df, first_last=first_last)
                    
                    # --- 1. GENERATE INDIVIDUAL 1x1 PLOT ---
                    out_single = os.path.join(BASE_FIGS, "individual", group_name, mask_tag, f"{stem}_{stat_label}_{scale_tag}.pdf")
                    make_single_plot(title, arch_data, stat_label, log_y, out_single)
                    
                    group_parsed_data.append((title, arch_data))

                if missing_files:
                    skip_grid = True
                    continue 

                # --- 2. GENERATE ROW 1x3 PLOT ---
                out_1x3 = os.path.join(BASE_FIGS, "1x3_rows", group_name, mask_tag, f"{stat_label}_{scale_tag}.pdf")
                make_combined_plot(group_parsed_data, stat_label, log_y, out_1x3)
                print(f"  saved 1x3: {out_1x3}")
                
                grid_data_all_rows.append(group_parsed_data)

            # --- 3. GENERATE FULL 3x3 GRID PLOT ---
            if not skip_grid and len(grid_data_all_rows) == 3:
                out_3x3 = os.path.join(BASE_FIGS, "3x3_grid", mask_tag, f"{stat_label}_{scale_tag}.pdf")
                make_grid_plot(grid_data_all_rows, stat_label, log_y, out_3x3)
                print(f"  saved 3x3: {out_3x3}")

print("\nDone.")