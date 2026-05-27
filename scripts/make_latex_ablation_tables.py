#!/usr/bin/env python3
"""Generate booktabs LaTeX tables from *_data_ablation_*.csv files.

Assumptions (matches `final/glycolysis_reduced4_data_ablation_mean.csv`):
- First column is `run`, remaining columns are numeric metrics.
- `run` encodes the data size as `_n{int}`.
- Optional supervision setting is encoded via substring `first_last` in `run`.

Outputs:
- If both supervision modes exist, writes two .tex files: *_full.tex and *_first_last.tex
- Otherwise writes a single .tex file.

Example:
  python scripts/make_latex_ablation_tables.py final/glycolysis_reduced4_data_ablation_mean.csv --out-dir final
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import os
import re
from collections import defaultdict
from typing import Sequence


_RUN_RE = re.compile(r"^(?P<prefix>.*)_n(?P<n>\d+)$")


@dataclasses.dataclass(frozen=True)
class Row:
    run: str
    n: int
    supervision: str  # "full" | "first_last"
    model_key: str
    display_name: str
    metrics: dict[str, float]


def _infer_supervision(run: str) -> str:
    return "first_last" if "first_last" in run else "full"


def _strip_prefix(run_prefix: str) -> str:
    # Typical prefix is like "A6_baseline" or "A3_neural_ode_gru_first_last".
    return re.sub(r"^A\d+_", "", run_prefix)


def _infer_model_key(run_prefix: str) -> str:
    # Use a normalized key for ordering; keep stable across datasets.
    key = _strip_prefix(run_prefix)
    key = key.replace("_first_last", "")
    return key


_DISPLAY_NAME_MAP: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^baseline$"), "Baseline"),
    (re.compile(r"^l1reg$"), "L1 reg"),
    (re.compile(r"^unbounded$"), "Unbounded"),
    (re.compile(r"^fixed_theta$"), r"Fixed $\theta$"),
    (re.compile(r"^sample_theta$"), r"Sampled $\theta$"),
    (re.compile(r"^neural_ode_gru$"), r"Neural ODE (GRU)"),
    (re.compile(r"^neural_ode_mlp$"), r"Neural ODE (MLP)"),
    (re.compile(r"^neural_ode_correction$"), r"Neural ODE (correction)"),
]


def _infer_display_name(model_key: str) -> str:
    for pattern, name in _DISPLAY_NAME_MAP:
        if pattern.match(model_key):
            return name
    # fallback: keep underscores but make it readable
    return model_key.replace("_", " ")


_MODEL_ORDER = [
    "baseline",
    "l1reg",
    "unbounded",
    "neural_ode_gru",
    "neural_ode_mlp",
    "neural_ode_correction",
    "fixed_theta",
    "sample_theta",
]


def _model_sort_key(model_key: str) -> tuple[int, str]:
    try:
        return (_MODEL_ORDER.index(model_key), model_key)
    except ValueError:
        return (len(_MODEL_ORDER), model_key)


def _render_latex_table(metrics: list[str], groups: list[dict]) -> str:
    # booktabs table; columns are: model + each metric
    cols = "l" + ("r" * len(metrics))
    lines: list[str] = []
    lines.append(f"\\begin{{tabular}}{{{cols}}}")
    lines.append("\\toprule")
    header = "Model" + "".join([f" & {m}" for m in metrics]) + r" \\"
    lines.append(header)
    lines.append("\\midrule")

    for i, g in enumerate(groups):
        if i != 0:
            lines.append("\\midrule")
        lines.append(
            f"\\multicolumn{{{1 + len(metrics)}}}{{l}}{{\\textbf{{$n={g['n']}$}}}}" + r" \\"
        )
        for row in g["rows"]:
            row_line = row["display_name"] + "".join([f" & {row['fmt'][m]}" for m in metrics]) + r" \\"
            lines.append(row_line)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def _read_rows(csv_path: str) -> tuple[list[str], list[Row]]:
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "run" not in reader.fieldnames:
            raise ValueError(f"Expected a 'run' column in {csv_path}")
        metric_names = [c for c in reader.fieldnames if c != "run"]
        rows: list[Row] = []
        for raw in reader:
            run = (raw.get("run") or "").strip()
            m = _RUN_RE.match(run)
            if not m:
                raise ValueError(f"Could not parse n from run='{run}'")
            run_prefix = m.group("prefix")
            n = int(m.group("n"))
            supervision = _infer_supervision(run_prefix)
            model_key = _infer_model_key(run_prefix)
            display_name = _infer_display_name(model_key)

            metrics: dict[str, float] = {}
            for name in metric_names:
                value_raw = raw.get(name)
                if value_raw is None or value_raw == "":
                    metrics[name] = float("nan")
                else:
                    metrics[name] = float(value_raw)

            rows.append(
                Row(
                    run=run,
                    n=n,
                    supervision=supervision,
                    model_key=model_key,
                    display_name=display_name,
                    metrics=metrics,
                )
            )

    return metric_names, rows


def _format_number(x: float, digits: int) -> str:
    # Keep it simple: plain decimals. If you prefer siunitx, swap to \num{...}.
    if x != x:  # NaN
        return "--"
    return f"{x:.{digits}f}"


def _group_for_latex(
    rows: Sequence[Row],
    metric_names: Sequence[str],
    digits: int,
    bold_metric: str | None,
    descending_n: bool,
) -> dict:
    # Split by n, then order models within n.
    rows_by_n: dict[int, list[Row]] = defaultdict(list)
    for r in rows:
        rows_by_n[r.n].append(r)

    ns = sorted(rows_by_n.keys(), reverse=descending_n)

    groups = []
    for n in ns:
        group_rows = sorted(rows_by_n[n], key=lambda r: _model_sort_key(r.model_key))

        # Bold best (min) per group for one metric (default: nrmse)
        best_value = None
        if bold_metric and bold_metric in metric_names:
            vals = [r.metrics[bold_metric] for r in group_rows if r.metrics[bold_metric] == r.metrics[bold_metric]]
            best_value = min(vals) if vals else None

        rendered_rows = []
        for r in group_rows:
            fmt: dict[str, str] = {}
            for m in metric_names:
                s = _format_number(r.metrics[m], digits)
                if best_value is not None and m == bold_metric and r.metrics[m] == best_value:
                    s = f"\\textbf{{{s}}}"
                fmt[m] = s
            rendered_rows.append({"display_name": r.display_name, "fmt": fmt, "model_key": r.model_key})

        groups.append({"n": n, "rows": rendered_rows})

    return {"metrics": list(metric_names), "groups": groups}


def _write_table_tex(out_path: str, ctx: dict) -> None:
    tex = _render_latex_table(metrics=ctx["metrics"], groups=ctx["groups"])
    with open(out_path, "w", newline="") as f:
        f.write(tex)
        f.write("\n")


def _default_out_base(csv_path: str) -> str:
    base = os.path.basename(csv_path)
    if base.lower().endswith(".csv"):
        base = base[:-4]
    return base


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "path",
        help="Path to a single ablation .csv (first column 'run') OR a directory containing .csv files",
    )
    p.add_argument("--out-dir", default=None, help="Output directory (default: same as csv)")
    p.add_argument("--digits", type=int, default=3, help="Decimal digits")
    p.add_argument(
        "--bold-metric",
        default="nrmse",
        help="Metric column to bold (best/min) within each n-group; set to '' to disable",
    )
    p.add_argument(
        "--descending-n",
        action="store_true",
        default=True,
        help="Sort n groups descending (default: true)",
    )
    p.add_argument(
        "--ascending-n",
        action="store_true",
        help="Sort n groups ascending",
    )

    args = p.parse_args(argv)

    in_path = args.path
    digits = args.digits
    bold_metric = args.bold_metric.strip() or None
    descending_n = True
    if args.ascending_n:
        descending_n = False

    csv_paths: list[str]
    if os.path.isdir(in_path):
        csv_paths = sorted(
            [os.path.join(in_path, f) for f in os.listdir(in_path) if f.lower().endswith(".csv")]
        )
        if not csv_paths:
            raise SystemExit(f"No .csv files found under directory: {in_path}")
    else:
        csv_paths = [in_path]

    all_written: list[str] = []
    for csv_path in csv_paths:
        out_dir = args.out_dir or os.path.dirname(csv_path)
        metric_names, rows = _read_rows(csv_path)

        # Split tables by supervision mode if needed.
        by_mode: dict[str, list[Row]] = defaultdict(list)
        for r in rows:
            by_mode[r.supervision].append(r)

        out_base = _default_out_base(csv_path)
        written: list[str] = []
        if set(by_mode.keys()) == {"full", "first_last"}:
            for mode in ("full", "first_last"):
                ctx = _group_for_latex(by_mode[mode], metric_names, digits, bold_metric, descending_n)
                out_path = os.path.join(out_dir, f"{out_base}_{mode}.tex")
                _write_table_tex(out_path, ctx)
                written.append(out_path)
        else:
            # Single table
            only_mode = sorted(by_mode.keys())[0]
            ctx = _group_for_latex(by_mode[only_mode], metric_names, digits, bold_metric, descending_n)
            out_path = os.path.join(out_dir, f"{out_base}.tex")
            _write_table_tex(out_path, ctx)
            written.append(out_path)

        all_written.extend(written)

    print("Wrote:")
    for pth in all_written:
        print(f"- {pth}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
