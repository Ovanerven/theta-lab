#!/usr/bin/env python3
"""
Emit a filtered sweep spec containing only the runs whose latest attempt is
*not* COMPLETE (no model.pt). Reuses the exact status logic from
audit_data_ablation.py.

Usage:
    python scripts/make_retry_spec.py sweep_specs/final/data_ablation_ran/glycolysis_reduced8.yaml \
        -o sweep_specs/final/retry/glycolysis_reduced8_retry.yaml \
        --time 04:00:00

By default writes to sweep_specs/final/retry/<orig_name>_retry.yaml.
The output preserves study/fixed/runs_per_job/etc. but only keeps incomplete runs.
If --time is given, overrides the top-level `time:` field (useful since most
failures here are slurm time-limit kills, not NaNs).

Optional:
    --skip-exp <name> [<name> ...]
        Don't include these exp_names (e.g. ones whose jobs are still running).
    --skip-from-slurm <slurm_out_path> [...]
        Auto-detect currently-running exp_names by reading the "Experiment:"
        line from slurm .out files and skip them.
    --archive
        Before writing the retry spec, MOVE each incomplete run's experiment
        directories (not in --skip-exp) to experiments_archive/<study>/...
        so a fresh retry doesn't see stale state.
    --partition <name>
        Override partition (e.g. gpu_h100).
"""
from __future__ import annotations
import argparse
import re
import shutil
import sys
from pathlib import Path

import yaml

# Reuse the audit logic
sys.path.insert(0, str(Path(__file__).parent))
from audit_data_ablation import (
    EXP_ROOT,
    STATUS_COMPLETE,
    check_run_dir,
    find_runs_for_exp,
)

ARCHIVE_ROOT = Path("experiments_archive")
# Matches the "Experiment: experiments/<study>/<run_id>" line emitted by training.
SLURM_EXPLINE = re.compile(r"^Experiment:\s+experiments/([^/]+)/([^\s]+)\s*$")
# Run dirs are named "<YYYYMMDD>_<HHMMSS>_<exp_name>" — strip the 15-char timestamp prefix.
RUNID_PREFIX = re.compile(r"^\d{8}_\d{6}_")


def study_and_exp_from_slurm(slurm_out: Path) -> tuple[str, str] | None:
    """Return (study, exp_name) parsed from an 'Experiment:' line, or None."""
    try:
        with open(slurm_out) as f:
            for line in f:
                m = SLURM_EXPLINE.match(line)
                if m:
                    return m.group(1), RUNID_PREFIX.sub("", m.group(2))
    except OSError:
        return None
    return None


def latest_status(study_dir: Path, exp_name: str) -> str:
    candidates = find_runs_for_exp(study_dir, exp_name)
    if not candidates:
        return "NEVER_RAN"
    status = check_run_dir(candidates[0])
    if status != STATUS_COMPLETE:
        for older in candidates[1:]:
            if check_run_dir(older) == STATUS_COMPLETE:
                return STATUS_COMPLETE
    return status


def process_spec(
    spec_path: Path,
    skip_by_study: dict,
    extra_skip: set,
    args,
) -> None:
    with open(spec_path) as f:
        spec = yaml.safe_load(f)

    study = spec["study"]
    study_dir = EXP_ROOT / study
    runs = spec.get("runs", [])

    skip = set(extra_skip) | set(skip_by_study.get(study, set()))

    keep = []
    skipped_running = []
    archived = []
    for run in runs:
        exp_name = run["exp_name"]
        status = latest_status(study_dir, exp_name)
        if args.include_status is not None:
            is_target = status in args.include_status
        else:
            is_target = status != STATUS_COMPLETE
        if not is_target:
            continue
        if exp_name in skip:
            skipped_running.append((status, exp_name))
            continue
        keep.append((status, run))

        if args.archive:
            for run_dir in find_runs_for_exp(study_dir, exp_name):
                if check_run_dir(run_dir) == STATUS_COMPLETE:
                    continue  # leave any older COMPLETE attempts alone
                dst = ARCHIVE_ROOT / study / run_dir.name
                archived.append((run_dir, dst))
                if not args.dry_run:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(run_dir), str(dst))

    print(f"\n=== {study}  ({spec_path.name}) ===")
    if archived:
        verb = "Would move" if args.dry_run else "Moved"
        print(f"  {verb} {len(archived)} incomplete run dir(s) to {ARCHIVE_ROOT}/:")
        for src, dst in archived:
            print(f"    {src}  ->  {dst}")
    if skipped_running:
        print(f"  Skipped (currently running): {len(skipped_running)}")
        for status, exp_name in skipped_running:
            print(f"    [{status}]  {exp_name}")

    if not keep:
        print(f"  Nothing to retry — all runs COMPLETE (or skipped).")
        return

    spec["runs"] = [r for _, r in keep]
    if args.time:
        spec["time"] = args.time
    if args.partition:
        spec["partition"] = args.partition

    if args.output and len(args._specs) == 1:
        out_path = Path(args.output)
    else:
        out_path = Path("sweep_specs/final/retry") / f"{spec_path.stem}_retry.yaml"

    if args.dry_run:
        print(f"  Would write {out_path}  ({len(keep)} runs)")
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            yaml.safe_dump(spec, f, sort_keys=False, default_flow_style=False, width=10**9)
        print(f"  Wrote {out_path}  ({len(keep)} runs)")
    for status, run in keep:
        print(f"    [{status}]  {run['exp_name']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("spec", nargs="*",
                   help="Path(s) to spec YAML. If omitted, --all processes all "
                        "specs under sweep_specs/final/data_ablation_ran/.")
    p.add_argument("--all", action="store_true",
                   help="Process every YAML under sweep_specs/final/data_ablation_ran/.")
    p.add_argument("-o", "--output",
                   help="Output spec path (only valid when exactly 1 spec is given). "
                        "Default: sweep_specs/final/retry/<name>_retry.yaml per spec.")
    p.add_argument("--time", help="Override top-level time: field (e.g. 04:00:00)")
    p.add_argument("--partition", help="Override top-level partition: field (e.g. gpu_h100)")
    p.add_argument("--include-status", nargs="*", default=None,
                   help="Statuses to include (default: anything not COMPLETE). "
                        "Choices: NAN/CRASH RUNNING? EMPTY NEVER_RAN")
    p.add_argument("--skip-exp", nargs="*", default=[],
                   help="Skip these exp_names across all studies (e.g. globally "
                        "known running jobs).")
    p.add_argument("--skip-from-slurm", nargs="*", default=[],
                   help="Slurm .out files of currently-running jobs. The "
                        "(study, exp_name) pair is parsed from the 'Experiment:' "
                        "line and applied only to the matching study.")
    p.add_argument("--archive", action="store_true",
                   help="Move incomplete run dirs to experiments_archive/<study>/ "
                        "before writing retry spec (skipped exp_names are left alone).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be moved/written, don't touch the filesystem.")
    args = p.parse_args()

    if args.all:
        specs = sorted(Path("sweep_specs/final/data_ablation_ran").glob("*.yaml"))
    else:
        specs = [Path(s) for s in args.spec]

    if not specs:
        p.error("No specs to process. Pass spec paths or --all.")

    if args.output and len(specs) != 1:
        p.error("--output is only valid when exactly one spec is provided.")

    args._specs = specs  # used by process_spec to decide output path handling

    # Build per-study skip sets from slurm out files.
    skip_by_study: dict[str, set[str]] = {}
    for slurm_out in args.skip_from_slurm:
        result = study_and_exp_from_slurm(Path(slurm_out))
        if result:
            study, exp_name = result
            skip_by_study.setdefault(study, set()).add(exp_name)
            print(f"Skip-from-slurm  {Path(slurm_out).name}: study={study} exp={exp_name}")
        else:
            print(f"WARNING: could not extract (study, exp_name) from {slurm_out}",
                  file=sys.stderr)

    extra_skip = set(args.skip_exp)

    for spec_path in specs:
        process_spec(spec_path, skip_by_study, extra_skip, args)


if __name__ == "__main__":
    main()
