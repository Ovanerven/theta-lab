#!/usr/bin/env python3
"""
Move run directories that don't have model.pt out of experiments/<study>/
into experiments_dead/<study>/. Reversible: just `mv` the dir back.

A run dir is archived iff:
  - it lives directly under experiments/<study>/
  - it has no model.pt
  - AND there is *no other* run dir (same study) whose name ends with the same
    `_<exp_name>` suffix that DOES have a model.pt (so we don't archive an old
    failed restart of a run that later succeeded — actually that one we DO want
    to archive; see --keep-superseded to flip behavior).

Usage:
    # See what would move (default dry-run):
    python scripts/archive_dead_runs.py --study glycolysis_reduced8_data_ablation

    # All studies referenced by data_ablation_ran specs:
    python scripts/archive_dead_runs.py --from-spec-dir sweep_specs/final/data_ablation_ran

    # Actually move:
    python scripts/archive_dead_runs.py --study glycolysis_reduced8_data_ablation --apply
"""
from __future__ import annotations
import argparse
import shutil
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from audit_data_ablation import EXP_ROOT, STATUS_COMPLETE, check_run_dir

DEAD_ROOT = Path("experiments_dead")


def recent_mtime(d: Path) -> float:
    """Most recent mtime of any file under d (recursively). 0 if dir is empty."""
    latest = 0.0
    try:
        for p in d.rglob("*"):
            try:
                m = p.stat().st_mtime
                if m > latest:
                    latest = m
            except OSError:
                continue
        # also include the dir itself
        latest = max(latest, d.stat().st_mtime)
    except OSError:
        pass
    return latest


def archive_study(study: str, apply: bool, keep_superseded: bool, min_age_min: float) -> int:
    study_dir = EXP_ROOT / study
    if not study_dir.exists():
        print(f"  [skip] {study} (no such dir)")
        return 0

    dead_dir = DEAD_ROOT / study
    n_moved = 0

    run_dirs = sorted([d for d in study_dir.iterdir() if d.is_dir()])
    now = time.time()
    cutoff = now - min_age_min * 60
    for d in run_dirs:
        if check_run_dir(d) == STATUS_COMPLETE:
            continue

        # Safety: don't touch anything modified recently — could still be running.
        m = recent_mtime(d)
        if m > cutoff:
            age_min = (now - m) / 60
            print(f"  [skip - active]  {d.name}  (mtime {age_min:.1f} min ago)")
            continue

        # Optionally skip if a sibling with the same _<exp_name> suffix has model.pt.
        if keep_superseded:
            # exp_name suffix = everything after the first 16 chars of the timestamp + "_"
            # run dir name format: YYYYMMDD_HHMMSS_<exp_name>
            parts = d.name.split("_", 2)
            exp_name = parts[2] if len(parts) == 3 else d.name
            siblings = [
                s for s in run_dirs
                if s != d and s.name.endswith(f"_{exp_name}")
            ]
            if any(check_run_dir(s) == STATUS_COMPLETE for s in siblings):
                # Don't archive — user wants to keep failed siblings of successful runs
                continue

        dest = dead_dir / d.name
        if apply:
            dead_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(d), str(dest))
        print(f"  {'MOVED' if apply else 'WOULD MOVE'}  {d}  ->  {dest}")
        n_moved += 1

    return n_moved


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--study", action="append", default=[],
                   help="Study folder name under experiments/. Repeatable.")
    p.add_argument("--from-spec-dir",
                   help="Read spec YAMLs from this dir; archive each spec's `study:` folder.")
    p.add_argument("--apply", action="store_true",
                   help="Actually move. Without this flag, just prints what would happen.")
    p.add_argument("--keep-superseded", action="store_true",
                   help="Don't archive failed runs whose exp_name has another COMPLETE sibling.")
    p.add_argument("--min-age-min", type=float, default=60.0,
                   help="Skip runs whose dir was modified within this many minutes "
                        "(default: 60). Protects live jobs.")
    args = p.parse_args()

    studies = list(args.study)
    if args.from_spec_dir:
        for spec in sorted(Path(args.from_spec_dir).glob("*.yaml")):
            with open(spec) as f:
                studies.append(yaml.safe_load(f)["study"])

    if not studies:
        p.error("Pass --study and/or --from-spec-dir")

    total = 0
    for s in dict.fromkeys(studies):  # dedupe, preserve order
        print(f"\n=== {s} ===")
        total += archive_study(s, args.apply, args.keep_superseded)

    verb = "Moved" if args.apply else "Would move"
    print(f"\n{verb} {total} run dirs.")
    if not args.apply:
        print("Re-run with --apply to actually move.")


if __name__ == "__main__":
    main()
