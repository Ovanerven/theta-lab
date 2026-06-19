# LaTeX table generation (ablations)

This repo stores aggregated ablation results as CSVs under `final_data_ablation/`.

The script `scripts/make_latex_ablation_tables.py` converts these CSVs to `booktabs` LaTeX tables grouped by number of datapoints ($n$) with `\midrule` separators.

## Input assumptions

- CSV has a `run` column.
- `run` ends with `_n{int}` (e.g. `A6_baseline_n1000`).
- Optional supervision mode uses substring `first_last` in the run name (e.g. `..._first_last_n1000`).
- All other columns are numeric metrics.

## Common usage

Generate tables for all ablation CSVs:

```zsh
/usr/local/bin/python3 scripts/make_latex_ablation_tables.py final_data_ablation
```

Generate tables for one CSV:

```zsh
/usr/local/bin/python3 scripts/make_latex_ablation_tables.py final_data_ablation/glycolysis_reduced4_data_ablation_mean.csv
```

Write outputs elsewhere:

```zsh
/usr/local/bin/python3 scripts/make_latex_ablation_tables.py final_data_ablation --out-dir paper/tables
```

## Outputs

- If both supervision modes are present in the CSV, the script writes:
  - `*_full.tex`
  - `*_first_last.tex`
- Otherwise it writes a single `*.tex` file.

By default the best (minimum) value in the `nrmse` column is bolded *within each $n$ group*.
Disable bolding via:

```zsh
/usr/local/bin/python3 scripts/make_latex_ablation_tables.py final_data_ablation --bold-metric ""
```
