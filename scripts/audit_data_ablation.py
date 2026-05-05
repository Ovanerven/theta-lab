#!/usr/bin/env python3
"""
Audit data-ablation sweeps: for each exp_name defined in the spec, find the
most recent run and report whether it completed (model.pt exists), is in
progress (config only), or NaN'd (model_last.pt but no model.pt).

Usage:
    python scripts/audit_data_ablation.py                    # all specs in final/data_ablation_ran/
    python scripts/audit_data_ablation.py --spec sweep_specs/final/data_ablation_ran/glycolysis_reduced8.yaml
    python scripts/audit_data_ablation.py --missing-only     # only print incomplete exp_names

Output:
    One line per (study, exp_name). Exits with code = number of not-yet-complete entries.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import yaml

SPEC_DIR = Path("sweep_specs/final/data_ablation_ran")
EXP_ROOT = Path("experiments")

STATUS_COMPLETE  = "COMPLETE"
STATUS_NAN       = "NAN/CRASH"      # model_last.pt exists, no model.pt
STATUS_RUNNING   = "RUNNING?"       # config.yaml exists, no checkpoint at all
STATUS_EMPTY     = "EMPTY"          # directory exists but nothing in it
STATUS_MISSING   = "NEVER_RAN"      # no directory with this exp_name suffix exists


def check_run_dir(run_dir: Path) -> str:
    if (run_dir / "model.pt").exists():
        return STATUS_COMPLETE
    if (run_dir / "model_last.pt").exists():
        return STATUS_NAN
    if (run_dir / "config.yaml").exists():
        return STATUS_RUNNING
    return STATUS_EMPTY


def find_runs_for_exp(study_dir: Path, exp_name: str) -> list[Path]:
    """Return all run dirs whose name ends with _{exp_name}, sorted newest first."""
    if not study_dir.exists():
        return []
    return sorted(
        [d for d in study_dir.iterdir() if d.is_dir() and d.name.endswith(f"_{exp_name}")],
        reverse=True,
    )


def audit_spec(spec_path: Path, missing_only: bool) -> int:
    with open(spec_path) as f:
        spec = yaml.safe_load(f)

    study = spec["study"]
    study_dir = EXP_ROOT / study
    runs = spec.get("runs", [])

    n_incomplete = 0
    rows = []

    for run in runs:
        exp_name = run["exp_name"]
        candidates = find_runs_for_exp(study_dir, exp_name)

        if not candidates:
            status = STATUS_MISSING
            run_id = "-"
        else:
            # Use the most recent restart
            latest = candidates[0]
            status = check_run_dir(latest)
            run_id = latest.name
            # If latest failed but an earlier restart succeeded, call it COMPLETE
            if status != STATUS_COMPLETE:
                for older in candidates[1:]:
                    if check_run_dir(older) == STATUS_COMPLETE:
                        status = STATUS_COMPLETE
                        run_id = older.name + "  (older restart)"
                        break

        if status != STATUS_COMPLETE:
            n_incomplete += 1

        rows.append((status, study, exp_name, run_id))

    n_complete = sum(1 for r in rows if r[0] == STATUS_COMPLETE)
    header = f"{study}  {n_complete}/{len(rows)} complete"

    if missing_only:
        incomplete = [(s, e, r) for s, _, e, r in rows if s != STATUS_COMPLETE]
        if incomplete:
            print(f"\n--- {header} ---")
            for status, exp_name, run_id in incomplete:
                print(f"  [{status}]  {exp_name}  ({run_id})")
    else:
        print(f"\n{'='*70}")
        print(f"  {header}  ({spec_path.name})")
        print(f"{'='*70}")
        for status, study, exp_name, run_id in rows:
            tag = f"[{status}]"
            print(f"  {tag:<14}  {exp_name:<55}  {run_id}")

    return n_incomplete


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", help="Path to a single spec YAML (default: all in final/data_ablation_ran/)")
    parser.add_argument("--missing-only", action="store_true", help="Only print incomplete entries")
    args = parser.parse_args()

    if args.spec:
        specs = [Path(args.spec)]
    else:
        specs = sorted(SPEC_DIR.glob("*.yaml"))

    total_incomplete = 0
    for spec_path in specs:
        total_incomplete += audit_spec(spec_path, args.missing_only)

    print(f"\nTotal incomplete: {total_incomplete}")
    sys.exit(total_incomplete)


if __name__ == "__main__":
    main()
