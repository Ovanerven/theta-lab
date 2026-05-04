
"""
experiment_parser.py

Utilities to read a folder full of Excel workbooks where **each sheet
represents one experiment**, then assemble:

* a tidy inputs DataFrame (all reagent columns + Time_seconds)
* a tidy outputs DataFrame (Broccoli RFU, mCherry RFU + Time_seconds)
* a helper plot that shows every input versus time.

Adjust the INPUT_COLS / OUTPUT_COLS lists below if your
column names differ.
"""
import os
import ast
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from bisect import bisect_left
from collections import defaultdict
from typing import Dict, List
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Mapping, MutableMapping, Any, Optional, Hashable, Sequence, Dict, List
import pandas as pd
from sklearn.preprocessing import StandardScaler
import copy

###############################################################################
# Key <-> path helpers
###############################################################################

Key = Tuple[str, str]  # (workbook_filename, sheet_name)



def _map_inputs_to_coarse(a: pd.DataFrame,
                         b: pd.DataFrame,
                         time_col = 'Time_seconds') -> pd.DataFrame:
    """
    Map sparse inputs from DataFrame b onto the coarse time steps of DataFrame a.

    Rules:
    1. Skip rows in b where all values in input_cols are zero.
    2. For b rows with any non-zero inputs, find the index in a with the next time >= b[time_col]
       (ceiling); if beyond the last time in a, this and subsequent b rows cannot be mapped (time runs out).
    3. Ensure each b row with input is mapped to a unique row in a. If the target index is already used,
       map to the next available a index > target. If none remain, stop mapping (time runs out).

    Parameters:
    - a: DataFrame with coarse time steps and zeroed input columns.
    - b: DataFrame with fine time steps and sparse non-zero inputs.
    - time_col: Name of the time column in both DataFrames.
    - input_cols: List of input column names to map from b to a.

    Returns:
    - a_copy: A new DataFrame based on a with mapped inputs from b.
    """
    print('mapping inputs')
    # Sort both DataFrames by time
    a_sorted = a.sort_values(time_col).reset_index(drop=True)
    b_sorted = b.sort_values(time_col).reset_index(drop=True)
    times = a_sorted[time_col].values

    input_cols = [i for i in b.columns if i != time_col]
    # Prepare result DataFrame
    a_copy = a_sorted.copy()

    # Track used indices in a
    used = set()

    # Iterate through b
    for _, row in b_sorted.iterrows():
        # Skip if all zero inputs
        if row[input_cols].abs().sum() == 0:
            continue

        # Find ceiling index
        t = row[time_col]
        idx = np.searchsorted(times, t, side='left')

        # If beyond last a time, cannot map further
        if idx >= len(times):
            break

        # If already used, find next free index
        while idx < len(times) and idx in used:
            idx += 1

        # If no free indices left, time runs out
        if idx >= len(times):
            break

        # Map inputs
        for col in input_cols:
            a_copy.at[idx, col] = row[col]

        used.add(idx)

    return a_copy



def fill_df_time(   dfs_in: dict,
                    dfs_out: dict,
                    time_col: str = "Time_seconds",
                    step: int = 60) -> dict:
    
        # ---------- 1.  minute grid ----------
        t_max        = dfs_out[time_col].iloc[-1]          # last timestamp
        max_slot     = t_max // step                   # e.g. 1333
        cap          = max_slot * step                 # e.g. 79 980 s
        time        = range(0,max_slot * step, step)  # 60, 120, …, cap

        # make a zero-filled template, keeping dtypes
        zero_row     = {c: dfs_in[c].dtype.type(0) for c in dfs_in.columns}
        new_df       = pd.DataFrame([zero_row] * len(time))
        new_df[time_col] = time
        new_df = _map_inputs_to_coarse(new_df, dfs_in)
        return new_df
        
# ----- apply to a whole dict -----
def remap_dict_fill(dfs: dict,
                    dfs_out: dict,
                    time_col: str = "Time_seconds",
                    step: int = 60) -> dict:
    
    data = {}
    for k,v in dfs.items():
        data[k] = fill_df_time(v,dfs_out[k], time_col, step) 
    return data

def trim_to_match_time(df_ref: pd.DataFrame,
                       df_other: pd.DataFrame,
                       time_col: str = "Time_seconds") -> pd.DataFrame:
    """
    Keep only the rows in *df_other* whose *time_col* value lies between
    the minimum and maximum times present in *df_ref*.
    """
    t_min, t_max = df_ref[time_col].agg(["min", "max"])
    mask = (df_other[time_col] >= t_min) & (df_other[time_col] <= t_max)
    return df_other.loc[mask].copy()

def _baseline_shift(series: pd.Series) -> pd.Series:
    """
    Return a copy of *series* that starts at zero.

    • Subtracts the first element from every value  
    • Clips any negative results to 0

    Parameters
    ----------
    series : pd.Series
        Numeric column (e.g., a time‑series trajectory).

    Returns
    -------
    pd.Series
        Shifted series with no negatives.
    """
    shifted = series - series.iloc[0]
    return shifted.clip(lower=0)

def _ensure_column(df: pd.DataFrame, column: str, fill_value: int | float = 0) -> pd.DataFrame:
    """
    Return *df* with *column* present. If the column is missing, add it
    and fill every row with *fill_value* (default = 0).

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to modify (a copy is returned).
    column : str
        Name of the column to ensure.
    fill_value : int | float, optional
        Value used to fill the new column if it’s created.

    Returns
    -------
    pd.DataFrame
        The modified (or unmodified) DataFrame.
    """
    if column in df.columns:
        return df.copy()          # nothing to do
    new_df = df.copy()
    new_df[column] = fill_value
    return new_df

def _slugify_component(text: str) -> str:
    """Return a filesystem‑safe slug.

    Keeps alnum + dash + underscore; everything else -> underscore; collapse repeats.
    """
    import re

    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", text.strip())
    slug = re.sub(r"_+", "_", slug)
    if not slug:
        slug = "untitled"
    return slug


def key_to_stem(key: Key) -> str:
    """Deterministic file stem for a (filename, sheet_name) key.

    Example: ("2025-03-06_output.xlsx", "Sheet 1") -> "2025-03-06_output__Sheet_1".
    The original workbook extension is *kept* in the stem to avoid collisions.
    """
    fname, sheet = key
    return f"{_slugify_component(fname)}__{_slugify_component(sheet)}"


def stem_to_key(stem: str) -> Key:
    """Inverse of key_to_stem() when created by that function.

    Splits on the last "__" occurrence; if missing, the sheet becomes "Sheet1".
    """
    if "__" in stem:
        fname, sheet = stem.rsplit("__", 1)
    else:
        fname, sheet = stem, "Sheet1"
    return fname, sheet


def drop_tail(df: pd.DataFrame, delete_frac: float) -> pd.DataFrame:
    """
    Keep only the first (1 – delete_frac) fraction of `df`.

    Parameters
    ----------
    df          : Input DataFrame.
    delete_frac : Fraction (0 ≤ f < 1) of rows to discard **from the end**.
                  *Example* – `delete_frac = 0.9` keeps the first 10 %.

    Returns
    -------
    trimmed_df  : The retained top-block, index reset.
    """
    if not (0.0 <= delete_frac < 1.0):
        raise ValueError("delete_frac must be in the half-open interval [0, 1).")

    keep_n = int(round(len(df) * (1.0 - delete_frac)))
    if keep_n < 1:
        raise ValueError("delete_frac too large – would leave an empty DataFrame.")

    return df.iloc[:keep_n].reset_index(drop=True)

def to_incremental_inputs(
    inputs_dict: Dict[str, pd.DataFrame],
    time_label: str = "Time_seconds",
    first_row_policy: str = "keep",          # "keep" | "zero"
) -> Dict[str, pd.DataFrame]:
    """
    Parameters
    ----------
    inputs_dict : dict[label → DataFrame]
        Your current input sheets with cumulative values.
    time_label  : str
        Column that stores time (left unchanged).
    first_row_policy : str
        • "keep" – the very first row keeps its original values
          (interpreted as the initial addition at t = 0).
        • "zero" – set the first row to 0 (no addition before t₀).

    Returns
    -------
    inc_inputs_dict : dict[label → DataFrame]
        Same structure as `inputs_dict`, but every non‑time column holds
        ΔX = Xₙ − Xₙ₋₁ instead of cumulative X.
    """
    if first_row_policy not in {"keep", "zero"}:
        raise ValueError('first_row_policy must be "keep" or "zero"')

    inc_dict = {}

    for label, df in inputs_dict.items():
        df = df.copy()                             # don’t mutate original
        num_cols = [c for c in df.columns if c != time_label]

        # vectorised diff on all numeric columns at once
        diffs = df[num_cols].diff()

        if first_row_policy == "keep":
            diffs.iloc[0] = df[num_cols].iloc[0]   # initial addition at t=0
        else:                                      # "zero"
            diffs = diffs.fillna(0)

        # overwrite the numeric columns with the per‑interval additions
        df[num_cols] = diffs

        # (optional) tiny negative values from float rounding → clamp to 0
        inc_dict[label] = df

    return inc_dict
def outputs_to_grid_nearest(
    outputs_dict: Dict[str, pd.DataFrame],
    target_dt: float,
    time_label: str = "Time_seconds",
) -> Dict[str, pd.DataFrame]:
    """
    Map every experiment’s outputs onto a fixed Δt grid by **nearest neighbour**.

    Parameters
    ----------
    outputs_dict : dict[label → DataFrame]
        Original per‑experiment outputs (arbitrary Δt, cumulative or not).
    target_dt    : float (seconds)
        Desired grid spacing – e.g. 2, 4, 6.  Must be > 0.
    time_label   : str
        Column that stores time.

    Returns
    -------
    out_grid_dict : dict[label → DataFrame]
        One DataFrame per experiment, rows at
        0, Δt, 2·Δt, …, same keys as `outputs_dict`.
    """
    if target_dt <= 0:
        raise ValueError("target_dt must be positive")

    out_grid_dict = {}

    for label, df in outputs_dict.items():

        # 1) sort & unique times -------------------------------------------------
        df = (
            df.sort_values(time_label)
              .drop_duplicates(time_label, keep="last")
        )
        times = df[time_label].to_numpy()
        payload = df.drop(columns=time_label).to_numpy()

        # 2) build uniform grid --------------------------------------------------
        t_max = np.ceil(times[-1] / target_dt) * target_dt
        grid_times = np.arange(0, t_max + 25 * target_dt, target_dt)

        # 3) nearest‑neighbour indices (vectorised) ------------------------------
        idx = np.searchsorted(times, grid_times, side="left")
        idx = np.clip(idx, 0, len(times) - 1)          # clamp to valid range

        # choose nearer of idx‑1 and idx
        left_idx = np.maximum(idx - 1, 0)
        take_left = (np.abs(grid_times - times[left_idx])
                     < np.abs(grid_times - times[idx]))
        idx[take_left] = left_idx[take_left]

        # 4) assemble grid DataFrame ---------------------------------------------
        grid_df = pd.DataFrame(payload[idx], columns=df.columns.drop(time_label))
        grid_df.insert(0, time_label, grid_times)

        out_grid_dict[label] = grid_df

    return out_grid_dict

def align_inputs_to_grid(
    inputs_dict: Dict[str, pd.DataFrame],
    target_dt: float,
    time_label: str = "Time_seconds",
) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """
    Map every experiment’s *inputs* onto a fixed Δt grid (0, Δt, 2·Δt, …).

    • Newly created rows are zero‑filled.
    • Each original row is placed in the nearest grid slot; if that slot
      is taken we move forward until a free one is found.
    • If the last slot is exceeded, the whole experiment is skipped.

    Returns
    -------
    aligned_inputs : dict[label → DataFrame]
    skipped_labels : list[str]
    """
    if target_dt <= 0:
        raise ValueError("target_dt must be positive seconds")

    aligned, skipped = {}, []

    for label, df_orig in inputs_dict.items():
        # ------------------- build zero‑filled target grid --------------------
        t_max = np.ceil(df_orig[time_label].max() / target_dt) * target_dt
        grid_times = np.arange(0, t_max + 25 * target_dt, target_dt)
        n_grid = len(grid_times)

        data_mat = np.zeros((n_grid, df_orig.shape[1]-1), dtype=np.float32)
        col_names = [c for c in df_orig.columns if c != time_label]
        skipped  = []
        dest_idx = []
        # ------------------- find free slots for each event ------------------
        for t in df_orig[time_label].to_numpy():
            idx = int(round(t / target_dt))
            # bump forward until free
            dest_idx.append(idx)
        if len(dest_idx) != len(set(dest_idx)):
            skipped.append(label)
            print('Mismatch between dt and dataset')
        if label in skipped:
            print(f"⚠  Skipped “{label}” – too many collisions for Δt={target_dt}")
            continue

        # ------------------- bulk‑copy the payload columns -------------------
        payload = df_orig[col_names].to_numpy(dtype=np.float32)
        dest_idx = np.asarray(dest_idx, dtype=int)
        data_mat[dest_idx] = payload

        # ------------------- assemble final DataFrame ------------------------
        grid_df = pd.DataFrame({time_label: grid_times})
        for i, name in enumerate(col_names):
            grid_df[name] = data_mat[:, i]

        aligned[label] = grid_df
        
    return aligned


def standard_scale_dict(
    exp_dict: Mapping[Hashable, pd.DataFrame],
    *,
    exclude: Sequence[str] = ("Time_seconds",),
    inplace: bool = False,
) -> Dict[Hashable, pd.DataFrame]:
    """
    Fit a single StandardScaler on the concatenation of every DataFrame in
    *exp_dict*, then scale each DataFrame so the transformation is identical.

    Accepts either of these structures::

        {"exp_1": df,           "exp_2": df2,           ...}
        {"exp_1": [df],         "exp_2": [df2],         ...}
        {"exp_1": (df,),        "exp_2": (df2,),        ...}

    Parameters
    ----------
    exp_dict : mapping
        Experiment dict as described above.
    exclude : sequence of str, optional
        Columns to leave untouched (default ``("Time_seconds",)``).
    inplace : bool, optional
        • False (default) → return **new** DataFrames; originals unchanged.  
        • True            → modify the DataFrames already inside *exp_dict*.

    Returns
    -------
    dict
        Same keys / structure as *exp_dict*, with scaled DataFrames.
        The fitted scaler is also exposed as ``result["__scaler__"]``.
    """
    # ------------------------------------------------------------------ #
    # 1. Collect common numeric columns present in every DataFrame
    # ------------------------------------------------------------------ #
    common_numeric = None
    frames_for_fit = []

    def _unwrap(val) -> pd.DataFrame:
        """Return the DataFrame, regardless of wrapping list/tuple."""
        if isinstance(val, pd.DataFrame):
            return val
        if isinstance(val, (list, tuple)) and len(val):
            return val[0]
        raise TypeError(f"Expected DataFrame or list/tuple[DataFrame], got {type(val)}")

    for val in exp_dict.values():
        df = _unwrap(val)
        num_cols = set(df.select_dtypes(include="number").columns) - set(exclude)
        common_numeric = num_cols if common_numeric is None else common_numeric & num_cols
        frames_for_fit.append(df)

    if not common_numeric:
        raise ValueError("No common numeric columns found to scale.")

    cols = sorted(common_numeric)

    # ------------------------------------------------------------------ #
    # 2. Fit scaler
    # ------------------------------------------------------------------ #
    concat = pd.concat([df[cols] for df in frames_for_fit], ignore_index=True)
    scaler = StandardScaler().fit(concat)

    # ------------------------------------------------------------------ #
    # 3. Transform each DataFrame
    # ------------------------------------------------------------------ #
    out_dict = exp_dict if inplace else {}

    for key, val in exp_dict.items():
        if isinstance(val, pd.DataFrame):
            df = val if inplace else val.copy()
            df[cols] = scaler.transform(df[cols])
            out_dict[key] = df
        else:  # list/tuple
            df0 = val[0] if inplace else val[0].copy()
            df0[cols] = scaler.transform(df0[cols])
            wrapped = type(val)([df0])  # preserve list vs tuple
            out_dict[key] = wrapped

    return out_dict, scaler

def standard_scale_list(
    array_list: List[np.ndarray],
    inplace: bool = False,
) -> List[np.ndarray]:
    """
    Apply one consistent StandardScaler to a list of numpy arrays.

    Parameters
    ----------
    array_list : list of np.ndarray
        Each array must have shape (N, D) — rows are samples, columns are features.
    inplace : bool, optional
        Whether to modify the arrays in-place (default = False).

    Returns
    -------
    list of np.ndarray
        Scaled arrays with the same shape as input. The fitted scaler is attached
        as `.scaler` attribute on the returned list.
    """
    if not array_list:
        return []

    # Check consistent number of columns
    n_features = array_list[0].shape[1]
    for arr in array_list:
        if arr.shape[1] != n_features:
            raise ValueError("All arrays must have the same number of columns.")

    # Fit scaler to concatenated data
    scaler = StandardScaler()
    stacked = np.vstack(array_list)
    scaler.fit(stacked)

    # Transform each array
    scaled = []
    for arr in array_list:
        transformed = scaler.transform(arr) if not inplace else scaler.transform(arr, copy=False)
        scaled.append(transformed)
    return scaled, scaler

def resample_io(
    inputs_df: pd.DataFrame,
    outputs_df: pd.DataFrame,
    *,
    time_label: str = "Time_seconds",
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Resample inputs and outputs to a uniform time grid.
    Inputs are forward-filled at each change, outputs are taken at the nearest time point.
    """

    # 1. Sort by time and remove duplicates
    inputs_df = (
        inputs_df.sort_values(time_label)
                 .drop_duplicates(subset=time_label, keep="last")
                 .reset_index(drop=True)
    )
    outputs_df = (
        outputs_df.sort_values(time_label)
                  .drop_duplicates(subset=time_label, keep="last")
                  .reset_index(drop=True)
    )

    # 2. Identify input columns
    input_cols = [c for c in inputs_df.columns if c != time_label]
    if not input_cols:
        raise ValueError("No input columns left after removing time label.")

    # 3. Determine minimal time step from input changes
    change_mask = inputs_df[input_cols].diff().fillna(0) != 0
    change_times = inputs_df.loc[change_mask.any(axis=1), time_label].values
    dt_candidates = (
        np.diff(inputs_df[time_label].values)
        if len(change_times) < 2
        else np.diff(change_times)
    )
    dt_candidates = dt_candidates[dt_candidates > 0]
    if dt_candidates.size == 0:
        raise ValueError("Cannot determine a positive Δt – time vector invalid?")
    base_dt = float(dt_candidates.min())

    # 4. Build uniform time grid
    t_start = float(inputs_df[time_label].iloc[0])
    t_end = float(inputs_df[time_label].iloc[-1])
    new_times = np.arange(t_start, t_end + base_dt, base_dt, dtype=float)
    skeleton = pd.DataFrame({time_label: new_times})
    skeleton[time_label] = skeleton[time_label].astype(inputs_df[time_label].dtype)

    # 5. Resample inputs by backward fill
    inputs_uniform = pd.merge_asof(
        skeleton,
        inputs_df[[time_label] + input_cols],
        on=time_label,
        direction="backward",
    )

    # 6. Resample outputs by nearest match
    output_cols = [c for c in outputs_df.columns if c != time_label]
    outputs_uniform = pd.merge_asof(
        skeleton,
        outputs_df[[time_label] + output_cols].sort_values(time_label),
        on=time_label,
        direction="nearest",
    )

    # # 7. Optionally drop tail to remove initial/final artifacts
    # inputs_uniform = drop_tail(inputs_uniform, delete_frac=0)
    # outputs_uniform = drop_tail(outputs_uniform, delete_frac=0.35)

    return inputs_uniform, outputs_uniform, base_dt


def parse_experiments(folder_path):
    """
    Parse all Excel files in the given folder. Each sheet in each file is treated as one experiment.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing Excel files (.xls, .xlsx).

    Returns
    -------
    inputs_dict : dict
        Mapping from (filename, sheet_name) to DataFrame of input variables and time.
    outputs_dict : dict
        Mapping from (filename, sheet_name) to DataFrame of outputs and time.
    """
    # Define expected output columns
    output_vars = ['Broccoli [RFU]', 'mCherry [RFU]']
    input_vars = ['3PGA', 'AA', 'DNA', 'FB', 'K-Glut', 'Lysate 2%PEG','Maltose', 'Mg-Glut', 'NTPs', 'PEG8000', 'T7RNAP', 'water']
    delete_vars = []
    time_col = 'Time_seconds'

    inputs_dict = {}
    outputs_dict = {}
    
    with open('/home/bob-van-sluijs/Desktop/groups/group_files_dict.txt', 'r') as f:
        group_files_dict = ast.literal_eval(f.read())


    for fname in os.listdir(folder_path):
        base, ext = os.path.splitext(fname)
        if ext.lower() not in ['.xls', '.xlsx']:
            continue
        file_path = os.path.join(folder_path, fname)

        # Choose engine explicitly to avoid pandas config errors
        engine = 'openpyxl' if ext.lower() == '.xlsx' else 'xlrd'
        try:
            sheets = pd.read_excel(file_path, sheet_name=None, engine=engine)
        except Exception as e:
            print(f"Warning: could not read '{fname}' ({e})")
            continue

        for sheet_name, df in sheets.items():
            # Strip whitespace in column headers
            df.columns = df.columns.str.strip()
            cols = list(df.columns)
            key = (fname, sheet_name)

            if time_col not in cols:
                # No time column, skip sheet
                continue

            # Identify output columns present in this sheet
            present_outs = [c for c in output_vars if c in cols]

            # Identify input columns: everything except time and outputs
            present_inputs = [c for c in cols if c != time_col and c not in present_outs]

            # Build DataFrames
            if present_inputs:
                inp_df = df[[time_col] + present_inputs].copy()
                for var in input_vars:
                    inp_df = _ensure_column(inp_df, var)
                inputs_dict[key] = inp_df

            if present_outs:
                out_df = df[[time_col] + present_outs].copy()
                for obs in output_vars:    
                    out_df[obs] = _baseline_shift(out_df[obs])
                outputs_dict[key] = out_df

    for label in inputs_dict.keys():
        inputs_dict[label], outputs_dict[label] = inputs_dict[label].iloc[1:].reset_index(drop=True), outputs_dict[label].iloc[1:].reset_index(drop=True)
        t0_shift = inputs_dict[label][time_col].iloc[0]           # new first time
        inputs_dict[label][time_col]  -= t0_shift
        outputs_dict[label][time_col] -= t0_shift
        inputs_dict[label], outputs_dict[label], time_dt = resample_io(inputs_dict[label],outputs_dict[label])
        
    for k,df in inputs_dict.items():
        zero_row = pd.DataFrame([[0] * len(df.columns)], columns=df.columns)
        df = pd.concat([zero_row, df], ignore_index=False)
        inputs_dict[k] = copy.deepcopy(df)

    conc_series_DNA = {}
    for group, file_list in group_files_dict.items():
        # Load concentration table
        conc_path = f"/home/bob-van-sluijs/Desktop/groups/{group}.csv"
        if not os.path.exists(conc_path):
            print(f"⚠️ Missing concentration table for {group}, skipping.")
            continue
    
        conc_df = pd.read_csv(conc_path)
        conc_series = pd.Series(conc_df.value.values, index=conc_df.compound.values)
        conc_series = pd.to_numeric(conc_series, errors='coerce')
        conc_series_DNA[group] = conc_series['DNA']

    def _compute_DNA(df, multiplier):
        # get the DNA
        import copy
        df_temp = copy.deepcopy(df)
        del df_temp[time_col]
        df_cumsum = df_temp.cumsum(axis=0)
        # # Step 2: Total volume per row (after cumsum)
        row_totals = df_cumsum.sum(axis=1)
        return (df_cumsum['DNA']/ row_totals)*multiplier

    inputs_dict = to_incremental_inputs(inputs_dict, time_label = time_col)
    for g, N in group_files_dict.items():
        for i in N:
            for k in inputs_dict.keys():
                a,b = k
                if a == i:
                    DNA =  _compute_DNA(inputs_dict[(a,b)],conc_series_DNA[g]).fillna(0).diff()
                    inputs_dict[(a,b)]['DNA c'] = copy.deepcopy(DNA)
                    inputs_dict[(a,b)] = inputs_dict[(a,b)].iloc[1:]   
                    
    print(inputs_dict[(a,b)])   
    inputs_dict = remap_dict_fill(inputs_dict,outputs_dict, time_col = time_col, step = 15)
    outputs_dict = outputs_to_grid_nearest(
    outputs_dict, 15, time_label=time_col)
    outputs_dict = {k:trim_to_match_time(inputs_dict[k],outputs_dict[k]) for k in inputs_dict.keys()}
    
    for g in ['group 4','group 5','group 6']:
        for i in group_files_dict[g]:
            for label in inputs_dict.keys():
                a,b = label
                if i == a:
                    inputs_dict[label]['Mg-Glut'] /= 2.5
                    if g == 'group 6':
                        inputs_dict[label]['NTPs'] /= 2.5
                        print('corrected')
    return inputs_dict, outputs_dict


def store_parsed_io(
    inputs_dict: Mapping[Key, pd.DataFrame],
    outputs_dict: Mapping[Key, pd.DataFrame],
    dest_dir: str | Path,
    *,
    fmt: str = "parquet",  # 'parquet' | 'feather' | 'pickle'
    compression: Optional[str] = None,  # e.g. 'snappy', 'gzip' (for parquet)
    metadata: Optional[Mapping[Key, Mapping[str, Any]]] = None,
) -> Dict[Key, Dict[str, Any]]:
    """Persist parsed experiment I/O DataFrames to disk.

    Parameter
    ----------
    inputs_dict, outputs_dict
        Dicts keyed by ``(filename, sheet_name)`` -> DataFrame.  Must share keys.
    dest_dir : path‑like
        Target directory (created if needed).
    fmt : {'parquet','feather','pickle'}, default 'parquet'
        Storage backend.  Parquet recommended (typed, columnar, efficient).
    compression : str or None
        Passed to pandas writer when supported.
    metadata : optional mapping
        Extra per‑experiment metadata (e.g., base_dt) to include in the manifest.
        Structure: {key: {"base_dt": float, ...}}.

    Returns
    -------
    manifest : dict
        {key: {"inputs_path": <Path>, "outputs_path": <Path>, **metadata}}
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    if set(inputs_dict.keys()) != set(outputs_dict.keys()):
        missing_in_out = set(inputs_dict) - set(outputs_dict)
        missing_in_inp = set(outputs_dict) - set(inputs_dict)
        raise ValueError(
            "inputs_dict and outputs_dict must have identical keys. "
            f"Missing in outputs: {missing_in_out}. Missing in inputs: {missing_in_inp}."
        )

    manifest: Dict[Key, Dict[str, Any]] = {}

    for key in inputs_dict:
        stem = key_to_stem(key)
        if fmt == "parquet":
            in_path = dest / f"{stem}__inputs.parquet"
            out_path = dest / f"{stem}__outputs.parquet"
            inputs_dict[key].to_parquet(in_path, compression=compression, index=False)
            outputs_dict[key].to_parquet(out_path, compression=compression, index=False)
        elif fmt == "feather":
            in_path = dest / f"{stem}__inputs.feather"
            out_path = dest / f"{stem}__outputs.feather"
            inputs_dict[key].reset_index(drop=True).to_feather(in_path)
            outputs_dict[key].reset_index(drop=True).to_feather(out_path)
        elif fmt == "pickle":
            in_path = dest / f"{stem}__inputs.pkl"
            out_path = dest / f"{stem}__outputs.pkl"
            inputs_dict[key].to_pickle(in_path)
            outputs_dict[key].to_pickle(out_path)
        else:
            raise ValueError(f"Unsupported fmt='{fmt}'.")

        # base metadata
        entry = {
            "inputs_path": in_path.name,
            "outputs_path": out_path.name,
        }
        if metadata and key in metadata:
            entry.update(metadata[key])
        manifest[key] = entry

    # write manifest.json (keys serialized as "fname|||sheet")
    manifest_json = {}
    for (fname, sheet), entry in manifest.items():
        manifest_json[f"{fname}|||{sheet}"] = entry

    with open(dest / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest_json, fh, indent=2)

    return manifest


def load_parsed_io(
    src_dir: str | Path,
    *,
    expect_fmt: Optional[str] = None,
) -> Tuple[Dict[Key, pd.DataFrame], Dict[Key, pd.DataFrame], Dict[Key, Dict[str, Any]]]:
    """Load previously stored experiment I/O DataFrames.

    Reads ``manifest.json`` written by :func:`store_parsed_io` and reconstructs
    the keyed dictionaries.

    Parameters
    ----------
    src_dir : path‑like
        Directory containing ``manifest.json`` + saved data files.
    expect_fmt : optional
        If given, verify that the discovered file extensions match expectation.

    Returns
    -------
    inputs_dict, outputs_dict, metadata_dict
        The first two mirror the structure returned by your ``parse_experiments``.
        The third contains the per‑experiment metadata entries from the manifest.
    """
    src = Path(src_dir)
    manifest_path = src / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No manifest.json in {src}.")

    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest_json = json.load(fh)

    inputs_dict: Dict[Key, pd.DataFrame] = {}
    outputs_dict: Dict[Key, pd.DataFrame] = {}
    metadata_dict: Dict[Key, Dict[str, Any]] = {}

    for key_str, entry in manifest_json.items():
        try:
            fname, sheet = key_str.split("|||", 1)
            key: Key = (fname, sheet)
    
            in_name = entry["inputs_path"]
            out_name = entry["outputs_path"]
            in_path = src / in_name
            out_path = src / out_name
    
            # infer fmt from extension
            ext = in_path.suffix.lower()
            if expect_fmt is not None:
                # crude check: ensure extension consistent
                if expect_fmt == "parquet" and ext != ".parquet":
                    raise ValueError(f"Expected parquet but found {ext} for {in_path}.")
                if expect_fmt == "feather" and ext != ".feather":
                    raise ValueError(f"Expected feather but found {ext} for {in_path}.")
                if expect_fmt == "pickle" and ext != ".pkl":
                    raise ValueError(f"Expected pickle but found {ext} for {in_path}.")
    
            if ext == ".parquet":
                inp_df = pd.read_parquet(in_path)
                out_df = pd.read_parquet(out_path)
            elif ext == ".feather":
                inp_df = pd.read_feather(in_path)
                out_df = pd.read_feather(out_path)
            elif ext == ".pkl":
                inp_df = pd.read_pickle(in_path)
                out_df = pd.read_pickle(out_path)
            else:
                raise ValueError(f"Unsupported extension '{ext}' in manifest for {in_path}.")
                
    
            inputs_dict[key] = inp_df
            outputs_dict[key] = out_df
    
            # copy metadata minus the path keys
            meta = {k: v for k, v in entry.items() if k not in {"inputs_path", "outputs_path"}}
            metadata_dict[key] = meta
        except FileNotFoundError:
            pass

    return inputs_dict, outputs_dict, metadata_dict
        

def plot_inputs(inputs_dict, input_vars=None, experiment_key=None):
    """
    Plot input variables over time for experiments.

    Parameters
    ----------
    inputs_dict : dict
        Mapping from (filename, sheet_name) to input DataFrame.
    input_vars : list of str, optional
        Which input variables to plot. If None, plots all found inputs.
    experiment_key : tuple (filename, sheet_name), optional
        If provided, only plots that specific experiment.
    """
    # Determine which experiments to plot
    keys = [experiment_key] if experiment_key else list(inputs_dict.keys())

    plt.figure(figsize=(8, 6))
    for key in keys:
        df = inputs_dict.get(key)
        if df is None:
            continue
        to_plot = input_vars or [c for c in df.columns if c != 'Time_seconds']
        for var in to_plot:
            if var in df.columns:
                plt.plot(df['Time_seconds'], df[var], label=f"{key[0]} - {key[1]}: {var}")

    plt.xlabel('Time (seconds)')
    plt.ylabel('Input value')
    plt.title('Inputs vs Time')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_outputs(outputs_dict, output_var=None):
    """
    Plot all sheets' outputs for a given output category in a single plot.

    Parameters
    ----------
    outputs_dict : dict
        Mapping from (filename, sheet_name) to output DataFrame.
    output_var : str, optional
        Name of the output variable to plot (e.g. 'Broccoli [RFU]' or 'mCherry [RFU]').
        If None, plots all available output variables in separate calls.
    """
    # Determine list of all variables if not specified
    if output_var:
        vars_to_plot = [output_var]
    else:
        # collect unique output columns across all sheets
        vars_to_plot = set()
        for df in outputs_dict.values():
            vars_to_plot.update([c for c in df.columns if c != 'Time_seconds'])
        vars_to_plot = list(vars_to_plot)

    for var in vars_to_plot:
        plt.figure(figsize=(8, 6))
        for key, df in outputs_dict.items():
            if var in df.columns:
                plt.plot(df['Time_seconds'], df[var], label=f"{key[0]} - {key[1]}")

        plt.xlabel('Time (seconds)')
        plt.ylabel(var)
        plt.title(f"All sheets: {var}")
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
        
def plot_all_inputs_by_category(inputs_dict):
    """
    Plot each input category across all sheets in a separate plot.

    Parameters
    ----------
    inputs_dict : dict
        Mapping from (filename, sheet_name) to input DataFrame.
    """
# plot_inputs(inputs_df, ("2025-03-06_output.xlsx", "Sheet1"))
# plot_outputs(outputs_df)
# load_parsed_io("C:\\Users\\bobva\\Desktop\\Normalized datasets\\")


