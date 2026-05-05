#!/usr/bin/env python3
"""
Find non-COMPLETE run directories in experiments/<study>/, classify why they
failed by scanning the matching slurm .out file, log the verdict to a CSV, and
move the folder into experiments_archive/<study>/<run_id>/.

A run dir is "non-COMPLETE" if it has no model.pt. The matching slurm .out is
the file in slurm_outputs/<study>/ whose "Experiment:" line points at this
run dir.

Usage:
    python scripts/archive_failed_runs.py                # all data_ablation studies
    python scripts/archive_failed_runs.py --study STUDY  # single study
    python scripts/archive_failed_runs.py --skip-runid R1 R2 ...  # leave these alone (e.g. live jobs)
    python scripts/archive_failed_runs.py --dry-run

The CSV log goes to experiments_archive/failed_runs_log.csv (appended).
"""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import re
import shutil
import sys
from pathlib import Path

EXP_ROOT = Path("experiments")
SLURM_ROOT = Path("slurm_outputs")
ARCHIVE_ROOT = Path("experiments_archive")
LOG_PATH = ARCHIVE_ROOT / "failed_runs_log.csv"

EXP_LINE = re.compile(r"^Experiment:\s+experiments/([^/]+)/([^\s]+)\s*$")
NAN_PAT = re.compile(r"\bnan\b", re.IGNORECASE)
TIMELIMIT_PAT = re.compile(r"DUE TO TIME LIMIT|CANCELLED.*TIME LIMIT", re.IGNORECASE)
OOM_PAT = re.compile(r"out of memory|CUDA out of memory|OOM", re.IGNORECASE)
CANCELLED_PAT = re.compile(r"\bCANCELLED\b", re.IGNORECASE)


def status(run_dir: Path) -> str:
    if (run_dir / "model.pt").exists():
        return "COMPLETE"
    if (run_dir / "model_last.pt").exists():
        return "NAN/CRASH"
    if (run_dir / "config.yaml").exists():
        return "RUNNING?"
    return "EMPTY"


def index_slurm_outputs(study: str) -> dict[str, Path]:
    """Map run_id -> slurm .out path by parsing the 'Experiment:' line."""
    mapping: dict[str, Path] = {}
    study_slurm_dir = SLURM_ROOT / study
    if not study_slurm_dir.exists():
        return mapping
    for out_file in study_slurm_dir.glob("*.out"):
        try:
            with open(out_file) as f:
                for i, line in enumerate(f):
                    if i > 100:
                        break
                    m = EXP_LINE.match(line)
                    if m and m.group(1) == study:
                        mapping[m.group(2)] = out_file
                        break
        except OSError:
            continue
    return mapping


def classify_failure(slurm_out: Path | None, run_dir: Path) -> tuple[str, str]:
    """Return (reason, evidence_snippet) by scanning the slurm out tail."""
    if slurm_out is None or not slurm_out.exists():
        # Fall back to the run's own log if any
        return ("NO_SLURM_OUT", "")
    try:
        with open(slurm_out, errors="replace") as f:
            text = f.read()
    except OSError as e:
        return ("READ_ERROR", str(e))

    tail = text[-8000:]  # last 8 KB usually contains the death rattle
    if TIMELIMIT_PAT.search(tail):
        m = TIMELIMIT_PAT.search(tail)
        return ("TIMELIMIT", m.group(0))
    if OOM_PAT.search(tail):
        m = OOM_PAT.search(tail)
        return ("OOM", m.group(0))
    if NAN_PAT.search(tail):
        return ("NAN", _grab_line(tail, NAN_PAT))
    if CANCELLED_PAT.search(tail):
        return ("CANCELLED", _grab_line(tail, CANCELLED_PAT))
    if (run_dir / "model_last.pt").exists():
        return ("NO_MODEL_PT_BUT_LAST", "")  # mid-training, slurm out doesn't say why
    return ("UNKNOWN", "")


def _grab_line(text: str, pat: re.Pattern) -> str:
    for line in text.splitlines():
        if pat.search(line):
            return line.strip()[:300]
    return ""


def archive_run(run_dir: Path, study: str, dry_run: bool) -> Path:
    dst = ARCHIVE_ROOT / study / run_dir.name
    if dry_run:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(run_dir), str(dst))
    return dst


def process_study(study: str, skip_runids: set, dry_run: bool, log_rows: list) -> tuple[int, int]:
    study_dir = EXP_ROOT / study
    if not study_dir.is_dir():
        return (0, 0)
    slurm_index = index_slurm_outputs(study)
    n_seen = 0
    n_archived = 0
    for run_dir in sorted(study_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        n_seen += 1
        st = status(run_dir)
        if st == "COMPLETE":
            continue
        if run_dir.name in skip_runids:
            print(f"  SKIP   {run_dir.name}  (in --skip-runid)")
            continue
        slurm_out = slurm_index.get(run_dir.name)
        reason, evidence = classify_failure(slurm_out, run_dir)
        dst = archive_run(run_dir, study, dry_run)
        n_archived += 1
        verb = "WOULD MOVE" if dry_run else "MOVED"
        print(f"  {verb:11s}  [{st}/{reason}]  {run_dir.name}")
        if evidence:
            print(f"               > {evidence}")
        log_rows.append({
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "study": study,
            "run_id": run_dir.name,
            "status": st,
            "reason": reason,
            "evidence": evidence,
            "slurm_out": str(slurm_out) if slurm_out else "",
            "archived_to": str(dst),
            "dry_run": dry_run,
        })
    return n_seen, n_archived


def write_log(rows: list, dry_run: bool):
    if not rows:
        return
    if dry_run:
        print(f"\n(dry-run: would append {len(rows)} rows to {LOG_PATH})")
        return
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"\nAppended {len(rows)} rows to {LOG_PATH}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--study", help="Single study name (default: all *data_ablation* studies)")
    p.add_argument("--skip-runid", nargs="*", default=[],
                   help="Run IDs (folder basenames) to leave alone, e.g. live jobs.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.study:
        studies = [args.study]
    else:
        studies = sorted(d.name for d in EXP_ROOT.iterdir()
                         if d.is_dir() and "data_ablation" in d.name)

    skip = set(args.skip_runid)
    log_rows: list = []
    grand_seen = grand_archived = 0
    for study in studies:
        print(f"\n=== {study} ===")
        seen, archived = process_study(study, skip, args.dry_run, log_rows)
        grand_seen += seen
        grand_archived += archived
        if archived == 0:
            print("  (nothing to archive)")

    print(f"\nTotal: scanned {grand_seen} run dirs, archived {grand_archived}.")
    write_log(log_rows, args.dry_run)


if __name__ == "__main__":
    main()
