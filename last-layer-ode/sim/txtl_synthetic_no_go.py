"""TXTL synthetic no-go / kill-condition data generator.

Ported verbatim from next_steps/generate_fake_data_txtl_synthetic_no_go_tabs.py
with two non-mechanistic changes:

  1. The Spyder-friendly file-path defaults are removed. Callers pass paths
     explicitly (or use the CLI `main()`).
  2. The module exposes `generate_synthetic`, `write_outputs`, `read_workbook`,
     and the dataclasses so that other code (e.g. the npz build script) can
     drive it programmatically rather than only via the CLI.

The biological / no-go rule set, the cumulative-nL Excel format, the
randomization knobs, and the metadata schema are unchanged. See the original
file (and its commit history in next_steps/) for context.

Output format (matches the original):
    - one synthetic experiment per Excel sheet/tab
    - each synthetic sheet contains ONLY the original columns from the input workbook
    - metadata/summary written separately as CSV/JSON (optional)
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Defaults (kept editable here for the script entry point only)
# -----------------------------------------------------------------------------

DEFAULT_SEED = 7
DEFAULT_MODE = "randomized"  # choices: deterministic, randomized, both
DEFAULT_N_RANDOM_PER_TEMPLATE = 3


# -----------------------------------------------------------------------------
# Biological / data settings (unchanged from the original generator)
# -----------------------------------------------------------------------------

INPUT_COLS = [
    "3PGA",
    "AA",
    "DNA",
    "FB",
    "K-Glut",
    "Lysate 2%PEG",
    "Maltose",
    "Mg-Glut",
    "NTPs",
    "PEG8000",
    "T7RNAP",
    "water",
]
TIME_COL = "Time_seconds"

MRNA_RFU_COL = "Broccoli [RFU]"
PROTEIN_RFU_COL = "mCherry [RFU]"
OUTPUT_COLS = [MRNA_RFU_COL, PROTEIN_RFU_COL]

STOCKS = {
    "K-Glut": {"stock": 2000.0, "unit": "mM"},
    "Mg-Glut": {"stock": 200.0, "unit": "mM"},
    "PEG8000": {"stock": 16.0, "unit": "percent_wv"},
    "Maltose": {"stock": 500.0, "unit": "mM"},
    "T7RNAP": {"stock": 83.3, "unit": "U_per_uL"},
}

SOFT_KILL_TARGETS = {
    "Mg-Glut": {"target_final": 30.0, "unit": "mM"},
    "K-Glut": {"target_final": 300.0, "unit": "mM"},
    "PEG8000": {"target_final": 8.0, "unit": "percent_wv"},
}

FB_OVERDOSE_FACTOR = 2.0

NOISE_SD_MULTIPLIER = 1.0
MIN_BASELINE_SD = 1.0

RANDOMIZE_BACKGROUND_INPUTS = True
FIT_RANDOM_BACKGROUND_TO_VOLUME = True
MIN_EVENTS_PER_BACKGROUND_CHANNEL = 1
MAX_EVENTS_PER_BACKGROUND_CHANNEL = 3
MIN_EVENTS_PER_OVERDOSE_CHANNEL = 2
MAX_EVENTS_PER_OVERDOSE_CHANNEL = 3
MAX_RANDOM_EVENT_TIME_SECONDS = 2 * 60 * 60

NARROW_RANDOM_CHANNELS = {"Lysate 2%PEG", "DNA", "T7RNAP"}

GENERATE_FROM_ALL_SHEETS = True

MAX_EXCEL_SHEET_NAME_LEN = 31


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass
class WorkbookData:
    sheets: Dict[str, pd.DataFrame]
    reference_columns: List[str]
    raw_column_lookup: Dict[str, str]


@dataclass
class SyntheticRule:
    name: str
    short_name: str
    class_label: str
    output_mode: str
    description: str
    confidence: str
    zero_columns: Tuple[str, ...] = ()
    set_final_targets: Optional[Dict[str, Tuple[float, str]]] = None
    multiply_columns: Optional[Dict[str, float]] = None


@dataclass
class SyntheticExperiment:
    experiment_id: str
    sheet_name: str
    dataframe: pd.DataFrame
    metadata: Dict[str, object]


# -----------------------------------------------------------------------------
# Column and workbook helpers
# -----------------------------------------------------------------------------

def stripped_col_name(col: object) -> str:
    return str(col).strip()


def clean_columns_keep_lookup(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    raw_columns = [str(c) for c in df.columns]
    stripped_columns = [stripped_col_name(c) for c in raw_columns]

    seen: Dict[str, int] = {}
    unique_stripped = []
    for c in stripped_columns:
        if c not in seen:
            seen[c] = 0
            unique_stripped.append(c)
        else:
            seen[c] += 1
            unique_stripped.append(f"{c}__dup{seen[c]}")

    raw_lookup = {clean: raw for clean, raw in zip(unique_stripped, raw_columns)}
    out = df.copy()
    out.columns = unique_stripped
    return out, raw_lookup, unique_stripped


def read_workbook(path: Path) -> WorkbookData:
    path = Path(path)
    xls = pd.ExcelFile(path)
    sheets: Dict[str, pd.DataFrame] = {}
    reference_columns: Optional[List[str]] = None
    raw_column_lookup: Optional[Dict[str, str]] = None

    required = set(INPUT_COLS + [TIME_COL] + OUTPUT_COLS)

    for sheet in xls.sheet_names:
        raw_df = pd.read_excel(path, sheet_name=sheet)
        df, lookup, clean_cols = clean_columns_keep_lookup(raw_df)

        # Backfill missing REAGENT columns with 0 (e.g. 2025-11-26 K9 has no
        # 'AA' column — that's a genuine no-AA experiment, not a corrupt sheet,
        # and the trace confirms it: flat mCherry, no expression). Only the
        # time and output columns are non-recoverable; skip if those are gone.
        missing = sorted(required.difference(df.columns))
        non_recoverable = [c for c in missing if c == TIME_COL or c in OUTPUT_COLS]
        if non_recoverable:
            print(f"  skip {path.name}!{sheet}: missing non-recoverable cols {non_recoverable}")
            continue
        if missing:
            print(f"  backfill {path.name}!{sheet}: missing reagent cols {missing} → 0")
            for c in missing:
                df[c] = 0.0
                # Also add to the clean column set so the output writer keeps
                # the same per-workbook column schema.
                if c not in clean_cols:
                    clean_cols = list(clean_cols) + [c]
                    lookup[c] = c

        if reference_columns is None:
            reference_columns = clean_cols
            raw_column_lookup = lookup

        for c in INPUT_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
        df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")
        for c in OUTPUT_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        sheets[sheet] = df

    if reference_columns is None or raw_column_lookup is None:
        raise ValueError(f"No sheets found in {path}")

    return WorkbookData(
        sheets=sheets,
        reference_columns=reference_columns,
        raw_column_lookup=raw_column_lookup,
    )


def restore_original_output_format(
    df: pd.DataFrame,
    reference_columns: Sequence[str],
    raw_column_lookup: Dict[str, str],
) -> pd.DataFrame:
    missing = [c for c in reference_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot restore output format; missing columns: {missing}")
    out = df.loc[:, list(reference_columns)].copy()
    out = out.rename(columns={c: raw_column_lookup.get(c, c) for c in reference_columns})
    return out


def final_volume_nl(df: pd.DataFrame) -> float:
    return float(pd.to_numeric(df[INPUT_COLS].iloc[-1], errors="coerce").sum())


def infer_template_final_volume(sheets: Dict[str, pd.DataFrame]) -> float:
    vols = np.array([final_volume_nl(df) for df in sheets.values()], dtype=float)
    return float(np.nanmedian(vols))


def target_final_to_nl(component: str, target_final: float, final_volume_nl_: float) -> float:
    if component not in STOCKS:
        raise KeyError(f"No stock concentration configured for {component!r}")
    stock = float(STOCKS[component]["stock"])
    if stock <= 0:
        raise ValueError(f"Invalid stock concentration for {component}: {stock}")
    return float(target_final * final_volume_nl_ / stock)


def final_concentration(component: str, volume_nl: float, final_volume_nl_: float) -> Optional[float]:
    if component not in STOCKS or final_volume_nl_ <= 0:
        return None
    return float(STOCKS[component]["stock"]) * float(volume_nl) / float(final_volume_nl_)


def compute_baseline_stats(sheets: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
    first_rows = []
    for _, df in sheets.items():
        t0_idx = df[TIME_COL].idxmin()
        first_rows.append(df.loc[t0_idx, OUTPUT_COLS])

    base = pd.DataFrame(first_rows)
    stats: Dict[str, Dict[str, float]] = {}
    for col in OUTPUT_COLS:
        vals = pd.to_numeric(base[col], errors="coerce").dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            vals = np.array([0.0])
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else MIN_BASELINE_SD
        stats[col] = {
            "mean": float(np.mean(vals)),
            "sd": max(sd, MIN_BASELINE_SD),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
    return stats


def make_flat_trace(
    n: int,
    col: str,
    baseline_stats: Dict[str, Dict[str, float]],
    rng: np.random.Generator,
    noise_sd_multiplier: float = NOISE_SD_MULTIPLIER,
) -> np.ndarray:
    st = baseline_stats[col]
    mean = st["mean"]
    sd = max(st["sd"] * noise_sd_multiplier, MIN_BASELINE_SD)
    trace = rng.normal(loc=mean, scale=sd, size=n)
    lo = max(0.0, st["min"] - 3 * sd)
    hi = st["max"] + 3 * sd
    return np.rint(np.clip(trace, lo, hi)).astype(int)


# -----------------------------------------------------------------------------
# Rule definitions
# -----------------------------------------------------------------------------

def build_rules() -> List[SyntheticRule]:
    soft_set_targets = {
        comp: (cfg["target_final"], cfg["unit"])
        for comp, cfg in SOFT_KILL_TARGETS.items()
    }

    return [
        SyntheticRule(
            name="hard_no_T7RNAP",
            short_name="noT7",
            class_label="hard_zero",
            output_mode="flat_both",
            zero_columns=("T7RNAP",),
            description="No T7 RNA polymerase; T7-promoter transcription should not start.",
            confidence="high",
        ),
        SyntheticRule(
            name="hard_no_DNA",
            short_name="noDNA",
            class_label="hard_zero",
            output_mode="flat_both",
            zero_columns=("DNA",),
            description="No DNA template; no reporter mRNA or protein can be produced.",
            confidence="high",
        ),
        SyntheticRule(
            name="hard_no_FB",
            short_name="noFB",
            class_label="hard_zero",
            output_mode="flat_both",
            zero_columns=("FB",),
            description="No feeding buffer proxy; assumes this removes NTP/energy/cofactor supply.",
            confidence="medium",
        ),
        SyntheticRule(
            name="hard_no_added_Mg",
            short_name="noMg",
            class_label="hard_zero",
            output_mode="flat_both",
            zero_columns=("Mg-Glut",),
            description="No added Mg-glutamate; high-confidence only if residual Mg is insufficient.",
            confidence="medium",
        ),
        SyntheticRule(
            name="translation_off_no_lysate",
            short_name="noLys",
            class_label="translation_off",
            output_mode="flat_protein_only",
            zero_columns=("Lysate 2%PEG",),
            description="No lysate removes ribosomes/tRNAs/translation machinery; T7 transcription may still occur.",
            confidence="high_for_translation_only",
        ),
        SyntheticRule(
            name="soft_high_Mg_30mM_final",
            short_name="hiMg",
            class_label="soft_overdose",
            output_mode="flat_both",
            set_final_targets={"Mg-Glut": soft_set_targets["Mg-Glut"]},
            description="Very high Mg-glutamate boundary; off-manifold no-go prior.",
            confidence="soft",
        ),
        SyntheticRule(
            name="soft_high_PEG8000_8pct_final",
            short_name="hiPEG",
            class_label="soft_overdose",
            output_mode="flat_both",
            set_final_targets={"PEG8000": soft_set_targets["PEG8000"]},
            description="Very high PEG8000 crowding/viscosity boundary; off-manifold no-go prior.",
            confidence="soft",
        ),
        SyntheticRule(
            name="soft_high_FB_2x_volume",
            short_name="hiFB",
            class_label="soft_overdose",
            output_mode="flat_protein_only",
            multiply_columns={"FB": FB_OVERDOSE_FACTOR},
            description="High composite feeding-buffer/NTP proxy; best treated as translation-inhibitory.",
            confidence="soft_translation_only",
        ),
    ]


# -----------------------------------------------------------------------------
# Schedule helpers
# -----------------------------------------------------------------------------

def sorted_time_values(df: pd.DataFrame) -> np.ndarray:
    times = pd.to_numeric(df[TIME_COL], errors="coerce").dropna().to_numpy(dtype=float)
    times = np.unique(times)
    times.sort()
    return times


def choose_event_times(
    df: pd.DataFrame,
    n_events: int,
    rng: np.random.Generator,
    allow_t0: bool = True,
    max_time_s: float = MAX_RANDOM_EVENT_TIME_SECONDS,
) -> List[float]:
    times = sorted_time_values(df)
    if len(times) == 0:
        raise ValueError("No valid time values found")

    t0 = float(times[0])
    candidates = times[times <= max_time_s]
    if len(candidates) == 0:
        candidates = times

    if not allow_t0:
        candidates = candidates[candidates > t0]
        if len(candidates) == 0:
            candidates = times

    n = min(int(n_events), len(candidates))
    chosen = rng.choice(candidates, size=n, replace=False)
    chosen = sorted(float(x) for x in chosen)
    return chosen


def split_total_volume(total_nl: float, n_events: int, rng: np.random.Generator) -> np.ndarray:
    total = max(0.0, float(total_nl))
    if total == 0.0 or n_events <= 0:
        return np.zeros(max(n_events, 1), dtype=float)
    weights = rng.dirichlet(np.ones(n_events))
    return weights * total


def set_bolus_schedule(
    df: pd.DataFrame,
    col: str,
    events: Sequence[Tuple[float, float]],
) -> None:
    times = pd.to_numeric(df[TIME_COL], errors="coerce").to_numpy(dtype=float)
    cumulative = np.zeros(len(df), dtype=float)

    for event_time, delta in sorted(events, key=lambda x: x[0]):
        cumulative += np.where(times >= float(event_time), float(delta), 0.0)

    cumulative[np.abs(cumulative) < 1e-12] = 0.0
    df[col] = cumulative


def set_constant_cumulative_after_t0(df: pd.DataFrame, col: str, value: float) -> None:
    times = sorted_time_values(df)
    if len(times) == 0:
        df[col] = float(value)
        return
    t0 = float(times[0])
    df[col] = np.where(df[TIME_COL].astype(float) > t0, float(value), 0.0)


def empirical_final_input_table(sheets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for sheet, df in sheets.items():
        row = {"sheet": sheet}
        for col in INPUT_COLS:
            row[col] = float(pd.to_numeric(df[col].iloc[-1], errors="coerce"))
        rows.append(row)
    return pd.DataFrame(rows)


def sample_background_final_volume(
    col: str,
    empirical_finals: pd.DataFrame,
    template_df: pd.DataFrame,
    rng: np.random.Generator,
) -> float:
    vals = pd.to_numeric(empirical_finals[col], errors="coerce").dropna().to_numpy(dtype=float)
    vals = vals[vals >= 0]

    template_val = float(pd.to_numeric(template_df[col].iloc[-1], errors="coerce"))

    if len(vals) == 0:
        base = template_val
    else:
        nonzero = vals[vals > 0]
        pool = nonzero if len(nonzero) > 0 else vals
        base = float(rng.choice(pool))

    if col in NARROW_RANDOM_CHANNELS:
        factor = float(rng.uniform(0.75, 1.25))
    else:
        factor = float(rng.uniform(0.50, 1.50))

    sampled = max(0.0, base * factor)

    if len(vals) > 0:
        hi = float(np.nanpercentile(vals, 95)) * (1.25 if col not in NARROW_RANDOM_CHANNELS else 1.10)
        hi = max(hi, template_val, 1.0)
        sampled = min(sampled, hi)

    return sampled


def make_events_for_total(
    df: pd.DataFrame,
    total_nl: float,
    rng: np.random.Generator,
    min_events: int,
    max_events: int,
    allow_t0: bool = True,
) -> List[Tuple[float, float]]:
    total = max(0.0, float(total_nl))
    if total <= 0.0:
        return []

    n_events = int(rng.integers(min_events, max_events + 1))
    times = choose_event_times(df, n_events, rng, allow_t0=allow_t0)
    deltas = split_total_volume(total, len(times), rng)
    return [(float(t), float(v)) for t, v in zip(times, deltas)]


def final_totals_from_rule(
    template_df: pd.DataFrame,
    rule: SyntheticRule,
    target_final_volume: float,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    totals: Dict[str, float] = {}
    applied_targets: Dict[str, object] = {}

    for col in rule.zero_columns:
        totals[col] = 0.0
        applied_targets[col] = {"forced_final_volume_nl": 0.0}

    if rule.set_final_targets:
        for col, (target_final, unit) in rule.set_final_targets.items():
            val_nl = target_final_to_nl(col, float(target_final), target_final_volume)
            totals[col] = val_nl
            applied_targets[col] = {
                "target_final": float(target_final),
                "target_unit": unit,
                "set_volume_nl": val_nl,
                "stock": STOCKS.get(col, {}),
            }

    if rule.multiply_columns:
        for col, factor in rule.multiply_columns.items():
            current = float(pd.to_numeric(template_df[col].iloc[-1], errors="coerce"))
            val_nl = current * float(factor)
            totals[col] = val_nl
            applied_targets[col] = {
                "volume_factor": float(factor),
                "template_final_volume_nl": current,
                "set_volume_nl": val_nl,
            }

    return totals, applied_targets


def fit_final_totals_to_volume(
    final_totals: Dict[str, float],
    fixed_cols: Sequence[str],
    target_final_volume: float,
) -> Tuple[Dict[str, float], float, bool, Dict[str, object]]:
    totals = {c: float(v) for c, v in final_totals.items()}
    fixed = set(fixed_cols)

    nonwater_cols = [c for c in INPUT_COLS if c != "water"]
    fixed_sum = sum(totals.get(c, 0.0) for c in nonwater_cols if c in fixed)
    scalable_cols = [c for c in nonwater_cols if c not in fixed]
    scalable_sum = sum(totals.get(c, 0.0) for c in scalable_cols)
    nonwater_sum = fixed_sum + scalable_sum

    diagnostics = {
        "target_final_volume_nl": float(target_final_volume),
        "fixed_sum_before_scaling_nl": float(fixed_sum),
        "scalable_sum_before_scaling_nl": float(scalable_sum),
        "nonwater_sum_before_scaling_nl": float(nonwater_sum),
        "background_scale_factor": 1.0,
    }

    feasible = True
    scale = 1.0

    if nonwater_sum > target_final_volume and FIT_RANDOM_BACKGROUND_TO_VOLUME:
        available_for_scalable = target_final_volume - fixed_sum
        if available_for_scalable < 0:
            scale = 0.0
            feasible = False
        elif scalable_sum > 0:
            scale = max(0.0, min(1.0, available_for_scalable / scalable_sum))
        else:
            scale = 1.0
            feasible = fixed_sum <= target_final_volume

        for col in scalable_cols:
            totals[col] = totals.get(col, 0.0) * scale

    nonwater_after = sum(totals.get(c, 0.0) for c in nonwater_cols)
    water_nl = max(0.0, target_final_volume - nonwater_after)
    totals["water"] = water_nl

    if nonwater_after > target_final_volume + 1e-6:
        feasible = False

    diagnostics.update(
        {
            "background_scale_factor": float(scale),
            "nonwater_sum_after_scaling_nl": float(nonwater_after),
            "water_final_nl": float(water_nl),
            "feasible_constant_volume": bool(feasible),
        }
    )

    return totals, water_nl, feasible, diagnostics


# -----------------------------------------------------------------------------
# Synthetic experiment generation
# -----------------------------------------------------------------------------

def apply_outputs(
    df: pd.DataFrame,
    template_df: pd.DataFrame,
    rule: SyntheticRule,
    baseline_stats: Dict[str, Dict[str, float]],
    rng: np.random.Generator,
) -> None:
    if rule.output_mode == "flat_both":
        for col in OUTPUT_COLS:
            df[col] = make_flat_trace(len(df), col, baseline_stats, rng)
    elif rule.output_mode == "flat_protein_only":
        df[MRNA_RFU_COL] = template_df[MRNA_RFU_COL].to_numpy()
        df[PROTEIN_RFU_COL] = make_flat_trace(len(df), PROTEIN_RFU_COL, baseline_stats, rng)
    else:
        raise ValueError(f"Unknown output_mode: {rule.output_mode!r}")


def apply_rule_deterministic(
    template_sheet: str,
    template_df: pd.DataFrame,
    rule: SyntheticRule,
    baseline_stats: Dict[str, Dict[str, float]],
    rng: np.random.Generator,
    target_final_volume: float,
    replicate_id: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    df = template_df.copy(deep=True)

    fixed_totals, applied_targets = final_totals_from_rule(template_df, rule, target_final_volume)

    for col in rule.zero_columns:
        df[col] = 0.0

    for col, final_nl in fixed_totals.items():
        if col in rule.zero_columns or col == "water":
            continue
        set_constant_cumulative_after_t0(df, col, final_nl)

    nonwater_cols = [c for c in INPUT_COLS if c != "water"]
    nonwater_sum = float(pd.to_numeric(df[nonwater_cols].iloc[-1], errors="coerce").sum())
    water_nl = max(0.0, target_final_volume - nonwater_sum)
    feasible = nonwater_sum <= target_final_volume + 1e-6
    set_constant_cumulative_after_t0(df, "water", water_nl)

    apply_outputs(df, template_df, rule, baseline_stats, rng)

    final_input_volumes = {c: float(pd.to_numeric(df[c].iloc[-1], errors="coerce")) for c in INPUT_COLS}
    metadata = {
        **asdict(rule),
        "generation_mode": "deterministic",
        "template_sheet": template_sheet,
        "replicate_id": replicate_id,
        "target_final_volume_nl": float(target_final_volume),
        "actual_final_volume_nl": float(final_volume_nl(df)),
        "balanced_water_nl": float(water_nl),
        "feasible_constant_volume": bool(feasible),
        "applied_targets_json": json.dumps(applied_targets),
        "event_schedule_json": json.dumps({}),
        "final_input_volumes_json": json.dumps(final_input_volumes),
        "randomization_diagnostics_json": json.dumps({}),
        "stocks_json": json.dumps(STOCKS),
    }
    return df, metadata


def apply_rule_randomized(
    template_sheet: str,
    template_df: pd.DataFrame,
    rule: SyntheticRule,
    baseline_stats: Dict[str, Dict[str, float]],
    empirical_finals: pd.DataFrame,
    rng: np.random.Generator,
    target_final_volume: float,
    replicate_id: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    df = template_df.copy(deep=True)

    for col in INPUT_COLS:
        df[col] = 0.0

    fixed_totals, applied_targets = final_totals_from_rule(template_df, rule, target_final_volume)
    fixed_cols = set(fixed_totals.keys())

    final_totals: Dict[str, float] = {}
    for col in INPUT_COLS:
        if col == "water":
            continue
        if col in fixed_totals:
            final_totals[col] = float(fixed_totals[col])
        else:
            final_totals[col] = sample_background_final_volume(col, empirical_finals, template_df, rng)

    final_totals, water_nl, feasible, diagnostics = fit_final_totals_to_volume(
        final_totals=final_totals,
        fixed_cols=list(fixed_cols),
        target_final_volume=target_final_volume,
    )

    event_schedule: Dict[str, List[Dict[str, float]]] = {}
    overdose_cols = set((rule.set_final_targets or {}).keys()).union(set((rule.multiply_columns or {}).keys()))

    for col in INPUT_COLS:
        total = float(final_totals.get(col, 0.0))

        if col == "water":
            if total > 0:
                t0 = float(sorted_time_values(df)[0])
                events = [(t0, total)]
            else:
                events = []
        elif total <= 0.0:
            events = []
        elif col in overdose_cols:
            events = make_events_for_total(
                df, total, rng,
                min_events=MIN_EVENTS_PER_OVERDOSE_CHANNEL,
                max_events=MAX_EVENTS_PER_OVERDOSE_CHANNEL,
                allow_t0=True,
            )
        else:
            events = make_events_for_total(
                df, total, rng,
                min_events=MIN_EVENTS_PER_BACKGROUND_CHANNEL,
                max_events=MAX_EVENTS_PER_BACKGROUND_CHANNEL,
                allow_t0=True,
            )

        set_bolus_schedule(df, col, events)
        event_schedule[col] = [
            {"time_seconds": float(t), "delta_nl": float(v)} for t, v in events
        ]

    apply_outputs(df, template_df, rule, baseline_stats, rng)

    final_input_volumes = {c: float(pd.to_numeric(df[c].iloc[-1], errors="coerce")) for c in INPUT_COLS}
    metadata = {
        **asdict(rule),
        "generation_mode": "randomized",
        "template_sheet": template_sheet,
        "replicate_id": replicate_id,
        "target_final_volume_nl": float(target_final_volume),
        "actual_final_volume_nl": float(final_volume_nl(df)),
        "balanced_water_nl": float(water_nl),
        "feasible_constant_volume": bool(feasible),
        "applied_targets_json": json.dumps(applied_targets),
        "event_schedule_json": json.dumps(event_schedule),
        "final_input_volumes_json": json.dumps(final_input_volumes),
        "randomization_diagnostics_json": json.dumps(diagnostics),
        "stocks_json": json.dumps(STOCKS),
    }
    return df, metadata


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------

def sanitize_sheet_name(name: str) -> str:
    clean = re.sub(r"[\\/\?\*\[\]:]", "_", str(name))
    clean = re.sub(r"\s+", "_", clean).strip("_'")
    if not clean:
        clean = "Sheet"
    return clean[:MAX_EXCEL_SHEET_NAME_LEN]


def make_unique_sheet_name(base: str, used: set) -> str:
    base = sanitize_sheet_name(base)
    if base not in used:
        used.add(base)
        return base

    for i in range(1, 10000):
        suffix = f"_{i}"
        candidate = base[: MAX_EXCEL_SHEET_NAME_LEN - len(suffix)] + suffix
        if candidate not in used:
            used.add(candidate)
            return candidate

    raise RuntimeError("Could not create a unique Excel sheet name")


def make_experiment_id(template_sheet: str, rule: SyntheticRule, mode: str, replicate_id: int) -> str:
    return f"{template_sheet}__{mode}__{rule.name}__r{replicate_id:02d}"


def make_base_sheet_name(counter: int, template_sheet: str, rule: SyntheticRule, mode: str, replicate_id: int) -> str:
    mode_short = "det" if mode == "deterministic" else "rnd"
    return f"S{counter:03d}_{template_sheet}_{rule.short_name}_{mode_short}{replicate_id:02d}"


def generate_synthetic(
    workbook: WorkbookData,
    seed: int = DEFAULT_SEED,
    mode: str = DEFAULT_MODE,
    n_random_per_template: int = DEFAULT_N_RANDOM_PER_TEMPLATE,
) -> Tuple[List[SyntheticExperiment], pd.DataFrame, Dict[str, object]]:
    if mode not in {"deterministic", "randomized", "both"}:
        raise ValueError("mode must be one of: deterministic, randomized, both")

    rng = np.random.default_rng(seed)
    baseline_stats = compute_baseline_stats(workbook.sheets)
    target_final_volume = infer_template_final_volume(workbook.sheets)
    empirical_finals = empirical_final_input_table(workbook.sheets)
    rules = build_rules()

    if GENERATE_FROM_ALL_SHEETS:
        template_items: Iterable[Tuple[str, pd.DataFrame]] = workbook.sheets.items()
    else:
        first_sheet = next(iter(workbook.sheets))
        template_items = [(first_sheet, workbook.sheets[first_sheet])]

    generation_modes = [mode] if mode != "both" else ["deterministic", "randomized"]

    experiments: List[SyntheticExperiment] = []
    metadata_rows: List[Dict[str, object]] = []
    used_sheet_names: set = set()
    counter = 1

    for template_sheet, template_df in template_items:
        for rule in rules:
            for generation_mode in generation_modes:
                n_reps = 1 if generation_mode == "deterministic" else int(n_random_per_template)
                for replicate_id in range(1, n_reps + 1):
                    if generation_mode == "deterministic":
                        synthetic_df, meta = apply_rule_deterministic(
                            template_sheet=template_sheet,
                            template_df=template_df,
                            rule=rule,
                            baseline_stats=baseline_stats,
                            rng=rng,
                            target_final_volume=target_final_volume,
                            replicate_id=replicate_id,
                        )
                    else:
                        synthetic_df, meta = apply_rule_randomized(
                            template_sheet=template_sheet,
                            template_df=template_df,
                            rule=rule,
                            baseline_stats=baseline_stats,
                            empirical_finals=empirical_finals,
                            rng=rng,
                            target_final_volume=target_final_volume,
                            replicate_id=replicate_id,
                        )

                    experiment_id = make_experiment_id(template_sheet, rule, generation_mode, replicate_id)
                    base_sheet = make_base_sheet_name(counter, template_sheet, rule, generation_mode, replicate_id)
                    sheet_name = make_unique_sheet_name(base_sheet, used_sheet_names)

                    out_df = restore_original_output_format(
                        synthetic_df,
                        reference_columns=workbook.reference_columns,
                        raw_column_lookup=workbook.raw_column_lookup,
                    )

                    meta.update(
                        {
                            "experiment_id": experiment_id,
                            "output_sheet_name": sheet_name,
                            "n_rows": int(len(out_df)),
                        }
                    )
                    metadata_rows.append(meta)

                    experiments.append(
                        SyntheticExperiment(
                            experiment_id=experiment_id,
                            sheet_name=sheet_name,
                            dataframe=out_df,
                            metadata=meta,
                        )
                    )
                    counter += 1

    metadata = pd.DataFrame(metadata_rows)
    summary = {
        "n_real_templates": len(workbook.sheets),
        "n_synthetic_experiments": len(experiments),
        "target_final_volume_nl_median": float(target_final_volume),
        "baseline_stats": baseline_stats,
        "soft_kill_targets": SOFT_KILL_TARGETS,
        "fb_overdose_factor": FB_OVERDOSE_FACTOR,
        "mode": mode,
        "n_random_per_template": int(n_random_per_template),
        "output_format": "one synthetic experiment per Excel tab; experiment tabs contain only original input workbook columns",
        "reference_columns_internal": list(workbook.reference_columns),
        "reference_columns_output": [workbook.raw_column_lookup.get(c, c) for c in workbook.reference_columns],
    }

    return experiments, metadata, summary


def write_outputs(
    experiments: List[SyntheticExperiment],
    metadata: pd.DataFrame,
    summary: Dict[str, object],
    output_path: Path,
    csv_prefix: Optional[Path] = None,
    write_long_csv: bool = False,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_prefix is not None:
        csv_prefix = Path(csv_prefix)
        csv_prefix.parent.mkdir(parents=True, exist_ok=True)
        metadata.to_csv(csv_prefix.with_name(csv_prefix.name + "_metadata.csv"), index=False)
        with open(csv_prefix.with_name(csv_prefix.name + "_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        if write_long_csv:
            long_parts = []
            for exp in experiments:
                tmp = exp.dataframe.copy()
                tmp.insert(0, "synthetic_sheet", exp.sheet_name)
                tmp.insert(0, "experiment_id", exp.experiment_id)
                long_parts.append(tmp)
            pd.concat(long_parts, ignore_index=True).to_csv(
                csv_prefix.with_name(csv_prefix.name + "_synthetic_long.csv"),
                index=False,
            )

    try:
        engine = "xlsxwriter"
        __import__("xlsxwriter")
    except Exception:
        engine = "openpyxl"

    with pd.ExcelWriter(output_path, engine=engine) as writer:
        for exp in experiments:
            exp.dataframe.to_excel(writer, index=False, sheet_name=exp.sheet_name)
            if engine == "xlsxwriter":
                worksheet = writer.sheets[exp.sheet_name]
                for idx, col in enumerate(exp.dataframe.columns):
                    width = min(max(len(str(col)) + 2, 12), 24)
                    worksheet.set_column(idx, idx, width)
                worksheet.freeze_panes(1, 0)


# -----------------------------------------------------------------------------
# CLI entry point (kept for direct use; programmatic callers should use the
# functions above rather than going through this).
# -----------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate deterministic/randomized synthetic TXTL no-go datasets with one experiment per Excel tab."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input Excel workbook of real experiments.")
    parser.add_argument("--output", required=True, type=Path, help="Output Excel workbook of synthetic experiments.")
    parser.add_argument(
        "--csv-prefix",
        default=None,
        type=Path,
        help="Optional prefix for separate metadata/summary CSV/JSON outputs.",
    )
    parser.add_argument("--seed", default=DEFAULT_SEED, type=int)
    parser.add_argument(
        "--mode", default=DEFAULT_MODE,
        choices=["deterministic", "randomized", "both"],
    )
    parser.add_argument("--n-random-per-template", default=DEFAULT_N_RANDOM_PER_TEMPLATE, type=int)
    parser.add_argument("--write-long-csv", action="store_true")
    args, _ = parser.parse_known_args(argv)

    if not args.input.exists():
        raise FileNotFoundError(f"Input workbook not found: {args.input}")

    workbook = read_workbook(args.input)
    experiments, metadata, summary = generate_synthetic(
        workbook,
        seed=args.seed,
        mode=args.mode,
        n_random_per_template=args.n_random_per_template,
    )

    write_outputs(
        experiments=experiments,
        metadata=metadata,
        summary=summary,
        output_path=args.output,
        csv_prefix=args.csv_prefix,
        write_long_csv=args.write_long_csv,
    )

    print(json.dumps(summary, indent=2))
    print(f"Wrote Excel workbook: {args.output}")


if __name__ == "__main__":
    main()
