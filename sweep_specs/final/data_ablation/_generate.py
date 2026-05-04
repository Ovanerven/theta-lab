#!/usr/bin/env python3
"""
Generate one data-ablation sweep yaml per scaffold.

Each sweep: A1..A9 model variants × N in {1000, 100, 10, 3}, with
val_n / test_n set to ~10% of train (integer rounded).

Run:
    python sweep_specs/final/data_ablation/_generate.py
"""
from pathlib import Path

OUT = Path(__file__).parent

# (N, val_n, test_n) — 80/10/10 split of total N (val=test=10% of N).
N_SPLITS = [
    (1000, 100, 100),  # train 800
    (100,  10,  10),   # train 80
    (10,   1,   1),    # train 8
    (3,    1,   1),    # train 1  (proportions break at this scale; need ≥1 in each split)
]

# Common model variants. (suffix, run-dict, mask_variant).
# `mask_variant=True` means we *also* generate a masked-loss copy of this run
# with obs_idx=[0, P-1] (only first + last species supervised). Used to compare
# how each model class handles partial observability under shrinking N.
# Skipped for A1/A2 (fixed/sample theta — masking is not the relevant question
# for those ablations). Skipped for A9 unless explicitly requested.
ABLATIONS = [
    ("A1_fixed_theta",            {"model_class": "ode_fixed_theta"},                                         False),
    ("A2_sample_theta",           {"model_class": "ode_sample_theta"},                                        False),
    ("A3_neural_ode_gru",         {"model_class": "neural_ode_gru"},                                          True),
    ("A4_neural_ode_mlp",         {"model_class": "neural_ode_mlp"},                                          True),
    ("A5_unbounded",              {"model_class": "ode_rnn", "theta_bounded": False},                         True),
    ("A6_baseline",               {"model_class": "ode_rnn"},                                                 True),
    ("A7_l1reg",                  {"model_class": "ode_rnn", "l1_regularization": True, "lambda_reg": 0.1},   True),
    ("A9_neural_ode_correction",  {"model_class": "neural_ode_correction"},                                   True),
]


# (scaffold_name, P)  — drives the A10 obs_idx=[0, P-1] mask.
SCAFFOLD_P = {
    "single_enzyme_4":      4,
    "single_enzyme_6":      6,
    "single_enzyme_lumped": 2,   # degenerate: [0,1] = full state, kept for symmetry
    "glycolysis_oracle22":  22,
    "glycolysis_reduced12": 12,
    "glycolysis_reduced8":  8,
    "glycolysis_reduced4":  4,
    "mof_synthesis_4":      4,
    "mof_synthesis_6":      6,
    "mof_synthesis_8":      8,
    "mof_synthesis_12":     12,
}


def dataset_path(family: str, scaffold: str, N: int) -> str:
    """Map (scaffold, N) → dataset .npz path matching the conventions in
    datasets/<family>/<family>_datasets.sh."""
    if family == "single_enzyme":
        # SE files: single_enzyme_4.npz (n=1000), single_enzyme_4_n{N}.npz else
        tag = scaffold.replace("single_enzyme_", "")
        if N == 1000:
            return f"datasets/single_enzyme/single_enzyme_{tag}.npz"
        return f"datasets/single_enzyme/single_enzyme_{tag}_n{N}.npz"
    if family == "glycolysis":
        return f"datasets/glycolysis/{scaffold}_n{N}.npz"
    if family == "mof":
        return f"datasets/mof/{scaffold}_n{N}.npz"
    raise ValueError(family)


def base_config(family: str) -> str:
    return {
        "single_enzyme": "configs/single_enzyme/single_enzyme.yaml",
        "glycolysis":    "configs/glycolisis/glycolisis.yaml",
        "mof":           "configs/archs/gru_optimal.yaml",
    }[family]


def ablations_for(scaffold: str):
    """Expand ABLATIONS into (tag, run-dict). Each ablation marked with
    mask_variant=True spawns an additional `_first_last` run with the loss
    restricted to {first, last} species via obs_idx=[0, P-1]."""
    P = SCAFFOLD_P[scaffold]
    out = []
    for tag, run, mask in ABLATIONS:
        out.append((tag, run))
        if mask:
            masked = dict(run)
            masked["obs_idx"] = [0, P - 1]
            out.append((f"{tag}_first_last", masked))
    return out


def build_yaml(family: str, scaffold: str) -> str:
    study = f"{scaffold}_data_ablation"
    lines = []
    lines.append(f"# Data ablation for {scaffold}.")
    lines.append("# Sweep all ablation variants (A1..A9) at N in {1000,100,10,3}.")
    lines.append("# Splits: train=80% / val=10% / test=10% of N (clamped to ≥1 each).")
    lines.append("")
    lines.append(f"study: {study}")
    lines.append("runs_per_job: 4")
    lines.append('time: "01:30:00"')
    lines.append("partition: gpu_h100")
    lines.append("no_plot: false")
    lines.append("")
    lines.append("fixed:")
    lines.append(f"  base_config: {base_config(family)}")
    lines.append(f"  scaffold: {scaffold}")
    lines.append("  epochs: 100")
    lines.append("")
    lines.append("runs:")

    # Group by N for readability
    for N, val_n, test_n in N_SPLITS:
        lines.append(f"  # ── N={N}  (val_n={val_n}, test_n={test_n}) ──")
        ds = dataset_path(family, scaffold, N)
        for tag, run in ablations_for(scaffold):
            exp_name = f"{tag}_n{N}"
            extras = "".join(
                f", {k}: {repr_yaml(v)}"
                for k, v in run.items() if k != "model_class"
            )
            lines.append(
                f"  - {{model_class: {run['model_class']}, "
                f"exp_name: {exp_name}, "
                f"dataset_path: {ds}, "
                f"val_n: {val_n}, test_n: {test_n}{extras}}}"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def repr_yaml(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


SCAFFOLDS = [
    ("single_enzyme", "single_enzyme_4"),
    ("single_enzyme", "single_enzyme_6"),
    ("single_enzyme", "single_enzyme_lumped"),
    ("glycolysis",    "glycolysis_oracle22"),
    ("glycolysis",    "glycolysis_reduced12"),
    ("glycolysis",    "glycolysis_reduced8"),
    ("glycolysis",    "glycolysis_reduced4"),
    ("mof",           "mof_synthesis_4"),
    ("mof",           "mof_synthesis_6"),
    ("mof",           "mof_synthesis_8"),
    ("mof",           "mof_synthesis_12"),
]


def main():
    for family, scaffold in SCAFFOLDS:
        path = OUT / f"{scaffold}.yaml"
        path.write_text(build_yaml(family, scaffold))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
