#!/usr/bin/env python3
"""
Generic SLURM sweep launcher for theta-lab training runs.

Supports two sweep modes, both via the same YAML spec:

  grid mode  — cartesian product of param lists (hyperparameter search)
  runs mode  — explicit list of run dicts (architecture comparison)

Runs are batched into SLURM jobs. Runs with different conda environments
(e.g. mamba_env vs thesis_env) are automatically split into separate jobs.

Usage:
    python launch_sweep.py sweep_specs/lstm_mof6.yaml
    python launch_sweep.py sweep_specs/arch_comparison_mof6.yaml --dry-run
    python launch_sweep.py sweep_specs/lstm_mof6.yaml --out-root experiments/test
    python launch_sweep.py sweep_specs/lstm_mof6.yaml --no-compare

────────────────────────────────────────────────────────────
SPEC KEYS (all optional unless marked required)
────────────────────────────────────────────────────────────
study         [required] folder/study name
base_config   base YAML config path; can be overridden per-run
dataset_path  passed as --set (can be overridden per-run)
scaffold      passed as --set (can be overridden per-run)
model_class   passed as --set (can be overridden per-run)
exp_name      Python f-string template for the run name, e.g.
              "lstm_h{hidden}_l{num_layers}"  (grid mode)
              or set per-run in runs mode

fixed         dict of param->value added to every run via --set
no_plot       if true, passes --no-plot to train.py (default: true)

# Grid mode (cartesian product):
grid          dict of param -> list of values
derived       dict of param -> Python expression evaluated with
              grid param values in scope, e.g.
                dropout: "0.0 if num_layers == 1 else 0.1"

# Runs mode (explicit list):
runs          list of dicts; each dict is one run.
              Per-run keys:
                base_config  overrides top-level base_config
                env          conda env (default: thesis_env)
                exp_name     overrides top-level exp_name template
                time         per-run SLURM time override
                <anything>   passed as --set <key>=<val>

# Batching / SLURM:
runs_per_job  max runs to stack per SLURM job (default: 3)
              runs with different envs are never mixed in one job
time          wall-clock limit per batch job (default: "01:30:00")
compare_time  wall-clock limit for the compare job (default: "01:00:00")
partition     SLURM partition (default: "gpu_a100")
gpus          GPUs per job (default: 1)
ntasks        (default: 1)
cpus_per_task (default: 9)
"""
from __future__ import annotations

import argparse
import itertools
import subprocess
from collections import defaultdict
from pathlib import Path

import yaml


RESERVED_RUN_KEYS = {"base_config", "env", "exp_name", "time", "train_script"}
DEFAULT_ENV = "thesis_env"
TRAIN_CMD = "python -u last-layer-ode/train.py"
SACCT_CMD = "sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,MaxRSS,State -n"


# ── spec loading ──────────────────────────────────────────────────────────────

def load_spec(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── run generation ────────────────────────────────────────────────────────────

def expand_grid(spec: dict) -> list[dict]:
    """Cartesian product of spec['grid'], with derived params applied."""
    grid = spec.get("grid", {})
    if not grid:
        return [{}]
    keys = list(grid.keys())
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*[grid[k] for k in keys])]
    derived = spec.get("derived", {})
    result = []
    for combo in combos:
        for key, expr in derived.items():
            combo[key] = eval(expr, {}, dict(combo))  # noqa: S307 — local tool
        result.append(combo)
    return result


def get_runs(spec: dict) -> list[dict]:
    """Return a flat list of run dicts regardless of grid vs runs mode."""
    if "runs" in spec:
        return list(spec["runs"])
    runs = expand_grid(spec)
    if "env" in spec:
        for run in runs:
            run.setdefault("env", spec["env"])
    return runs


# ── command building ──────────────────────────────────────────────────────────

def resolve_exp_name(spec: dict, run: dict) -> str:
    template = run.get("exp_name") or spec.get("exp_name", "run")
    ctx = {**spec, **run}
    try:
        return template.format(**{k: v for k, v in ctx.items() if isinstance(v, (str, int, float, bool))})
    except KeyError:
        return template


def build_train_cmd(spec: dict, run: dict, out_root: str) -> str:
    base_config = run.get("base_config") or spec.get("base_config", "configs/archs/gru.yaml")
    exp_name = resolve_exp_name(spec, run)
    no_plot = spec.get("no_plot", False)
    train_script = run.get("train_script") or spec.get("train_script") or TRAIN_CMD.split()[-1]
    cmd = f"python -u {train_script}"

    parts = [cmd]
    if no_plot:
        parts.append("--no-plot")
    parts += [
        f"--config {base_config}",
        f"--set study={spec['study']}",
        f"--set out_root={out_root}",
        f"--set exp_name={exp_name}",
    ]

    # Top-level spec defaults (dataset_path, scaffold, model_class)
    for key in ("dataset_path", "scaffold", "model_class"):
        if key in spec and key not in run:
            parts.append(f"--set {key}={spec[key]}")

    def _fmt(val):
        # Lists: render via yaml flow style (so floats keep "1.0e-08" form
        # which YAML parses as float, unlike JSON's "1e-08" which YAML reads as
        # string), strip internal whitespace, and single-quote so the shell
        # passes them through as one --set arg.
        if isinstance(val, (list, tuple)):
            s = yaml.safe_dump(list(val), default_flow_style=True).strip()
            return "'" + s.replace(" ", "") + "'"
        return str(val)

    # Fixed params from spec (not already in run)
    for key, val in spec.get("fixed", {}).items():
        if key in RESERVED_RUN_KEYS:
            continue
        if key not in run:
            parts.append(f"--set {key}={_fmt(val)}")

    # Run-specific params
    for key, val in run.items():
        if key not in RESERVED_RUN_KEYS:
            parts.append(f"--set {key}={_fmt(val)}")

    return " ".join(parts)


def exp_label(cmd: str) -> str:
    for token in cmd.split():
        if token.startswith("exp_name="):
            return token.split("=", 1)[1]
    return "?"


# ── batching ──────────────────────────────────────────────────────────────────

def make_batches(runs: list[dict], runs_per_job: int) -> list[list[dict]]:
    """Group runs by env, then chunk each group into batches of runs_per_job."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        groups[run.get("env", DEFAULT_ENV)].append(run)

    batches = []
    for group in groups.values():
        for i in range(0, len(group), runs_per_job):
            batches.append(group[i : i + runs_per_job])
    return batches


# ── SLURM submission ──────────────────────────────────────────────────────────

def make_preamble(env: str) -> str:
    return (
        "module purge && "
        "module load 2025 && "
        "module load Anaconda3/2025.06-1 && "
        f"source activate {env}"
    )


def submit_batch(
    cmds: list[str],
    env: str,
    spec: dict,
    batch_num: int,
    study: str,
    dry_run: bool,
) -> str:
    wrap = " && ".join([make_preamble(env), *cmds, SACCT_CMD])
    time = spec.get("time", "01:30:00")

    sbatch_args = [
        "sbatch",
        f"--job-name={study}_b{batch_num}",
        f"--time={time}",
        f"--partition={spec.get('partition', 'gpu_a100')}",
        f"--gpus={spec.get('gpus', 1)}",
        f"--ntasks={spec.get('ntasks', 1)}",
        f"--cpus-per-task={spec.get('cpus_per_task', 9)}",
        f"--output=slurm_outputs/{study}/%A_%x.out",
        f"--wrap={wrap}",
    ]

    if dry_run:
        print(f"  [dry-run] batch {batch_num}  env={env}  time={time}")
        for cmd in cmds:
            print(f"    {cmd}")
        return f"DRY{batch_num}"

    result = subprocess.run(sbatch_args, capture_output=True, text=True, check=True)
    return result.stdout.strip().split()[-1]


def submit_compare(job_ids: list[str], study: str, time: str, dry_run: bool, env: str = DEFAULT_ENV, endpoint_r2: bool = False) -> str:
    dep = ":".join(job_ids)
    export = f"ALL,STUDY={study},ENV={env}"
    if endpoint_r2:
        export += ",ENDPOINT_R2=1"
    args = [
        "sbatch",
        f"--job-name={study}_compare",
        f"--dependency=afterany:{dep}",
        f"--time={time}",
        f"--output=slurm_outputs/{study}/%A_%x.out",
        f"--export={export}",
        "slurm_jobs/compare.job",
    ]

    if dry_run:
        print(f"  [dry-run] compare job  dep=afterany:{dep}")
        return "DRY_CMP"

    result = subprocess.run(args, capture_output=True, text=True, check=True)
    return result.stdout.strip().split()[-1]


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Submit a batched SLURM hyperparameter/architecture sweep.")
    parser.add_argument("spec", help="Path to sweep spec YAML")
    parser.add_argument("--out-root", default="experiments")
    parser.add_argument("--dry-run", action="store_true", help="Print without submitting")
    parser.add_argument("--no-compare", action="store_true", help="Skip the NRMSE compare job")
    args = parser.parse_args()

    spec = load_spec(args.spec)
    study = spec["study"]
    runs_per_job = int(spec.get("runs_per_job", 3))

    runs = get_runs(spec)
    batches = make_batches(runs, runs_per_job)

    n_runs = sum(len(b) for b in batches)
    print(f"Study      : {study}")
    print(f"Spec       : {args.spec}")
    print(f"Total runs : {n_runs}  →  {len(batches)} job(s)  ({runs_per_job} per job max)")
    print()

    Path(f"slurm_outputs/{study}").mkdir(parents=True, exist_ok=True)

    job_ids: list[str] = []
    sweep_env = DEFAULT_ENV
    for batch_num, batch in enumerate(batches):
        env = batch[0].get("env", DEFAULT_ENV)
        sweep_env = env
        cmds = [build_train_cmd(spec, run, args.out_root) for run in batch]
        labels = [exp_label(c) for c in cmds]
        jid = submit_batch(cmds, env, spec, batch_num, study, args.dry_run)
        job_ids.append(jid)
        print(f"Batch {batch_num}  env={env}  [{', '.join(labels)}]  → job {jid}")

    if not args.no_compare:
        print()
        # endpoint_r2 may live at top level or under fixed:
        has_endpoint_r2 = bool(
            spec.get("endpoint_r2") or spec.get("fixed", {}).get("endpoint_r2")
        )
        cjid = submit_compare(job_ids, study, spec.get("compare_time", "01:00:00"), args.dry_run, env=sweep_env, endpoint_r2=has_endpoint_r2)
        print(f"Compare    : job {cjid}")

    print()
    print(f"Results → {args.out_root}/{study}")
    print(f"Inspect:   python last-layer-ode/metrics/compare_runs.py {args.out_root}/{study} --csv results/{study}.csv")


if __name__ == "__main__":
    main()
