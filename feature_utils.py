"""Shared preprocessing and validation for DATA DIVERGENCE Track 2."""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd

TARGET = "next_interval_Usage_kWh"
DATE = "date"
REQUIRED_INPUT_COLUMNS = [
    DATE, "Usage_kWh",
    "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "CO2(tCO2)",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
    "NSM", "WeekStatus", "Day_of_week", "Load_Type",
]
CATEGORICAL_HINTS = ["WeekStatus", "Day_of_week", "Load_Type"]
NUMERIC_INPUT_COLUMNS = [
    "Usage_kWh", "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh", "CO2(tCO2)",
    "Lagging_Current_Power_Factor", "Leading_Current_Power_Factor", "NSM",
]

def parse_dates(series: pd.Series) -> pd.Series:
    """Parse the dataset's day-first timestamps, with a safe fallback."""
    raw = series.astype("string")
    dates = pd.to_datetime(raw, format="%d/%m/%Y %H:%M", errors="coerce")
    bad = dates.isna() & raw.notna() & raw.str.strip().ne("")
    if bad.any():
        fallback = pd.to_datetime(raw.loc[bad], errors="coerce", dayfirst=True)
        dates.loc[bad] = fallback
    return dates

def validate_input_frame(df: pd.DataFrame, *, training: bool = False) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("CSV contains headers but no data rows.")
    # Strip accidental spaces from headers, but detect collisions.
    cleaned = [str(c).strip() for c in df.columns]
    if len(cleaned) != len(set(cleaned)):
        raise ValueError("CSV has duplicate column names after trimming spaces.")
    df = df.copy()
    df.columns = cleaned
    required = REQUIRED_INPUT_COLUMNS + ([TARGET] if training else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")
    if df[DATE].isna().any() or df[DATE].astype("string").str.strip().eq("").any():
        rows = (df.index[df[DATE].isna() | df[DATE].astype("string").str.strip().eq("")] + 2).tolist()[:10]
        raise ValueError(f"Missing timestamp(s) at CSV row(s) {rows}.")
    parsed = parse_dates(df[DATE])
    if parsed.isna().any():
        rows = (df.index[parsed.isna()] + 2).tolist()[:10]
        raise ValueError(f"Invalid timestamp(s) at CSV row(s) {rows}; expected day/month/year hour:minute.")
    if parsed.duplicated().any():
        rows = (df.index[parsed.duplicated(keep=False)] + 2).tolist()[:10]
        raise ValueError(f"Duplicate timestamps found at CSV row(s) {rows}.")
    df[DATE] = parsed
    # Numeric predictor fields: invalid text becomes NaN and is reported, not silently ignored.
    for col in NUMERIC_INPUT_COLUMNS:
        original = df[col]
        converted = pd.to_numeric(original, errors="coerce")
        invalid_text = original.notna() & converted.isna()
        if invalid_text.any():
            rows = (df.index[invalid_text] + 2).tolist()[:10]
            warnings.warn(f"{col}: non-numeric value(s) at CSV row(s) {rows}; treating as missing.")
        converted = converted.replace([np.inf, -np.inf], np.nan)
        if converted.isna().any():
            warnings.warn(f"{col}: {int(converted.isna().sum())} missing/invalid value(s); model will receive missing values.")
        df[col] = converted
    for col in CATEGORICAL_HINTS:
        # Empty categorical values are explicit missing values.
        df[col] = df[col].astype("string").str.strip().replace("", pd.NA)
        if df[col].isna().any():
            warnings.warn(f"{col}: {int(df[col].isna().sum())} missing value(s); using __MISSING__.")
        df[col] = df[col].fillna("__MISSING__").astype(str)
    # Sort by time to ensure lag features are chronological.
    df = df.sort_values(DATE, kind="stable").reset_index(drop=True)
    if len(df) > 1:
        gaps = df[DATE].diff().dropna()
        irregular = gaps.ne(pd.Timedelta(minutes=15))
        if irregular.any():
            warnings.warn(
                f"{int(irregular.sum())} timestamp gap(s)/irregular interval(s) detected. "
                "Lag features are row-based, so check whether history is complete."
            )
    return df

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep this function identical for training and inference."""
    x = df.copy()
    if DATE in x.columns:
        dt = pd.to_datetime(x[DATE], errors="coerce")
        x["year"] = dt.dt.year
        x["month"] = dt.dt.month
        x["day"] = dt.dt.day
        x["hour"] = dt.dt.hour
        x["minute"] = dt.dt.minute
        x["dayofyear"] = dt.dt.dayofyear
        x["is_month_start"] = dt.dt.is_month_start.astype(float)
        x["is_month_end"] = dt.dt.is_month_end.astype(float)
        x["date_dayofweek"] = dt.dt.dayofweek
        minutes = dt.dt.hour.fillna(0) * 60 + dt.dt.minute.fillna(0)
        x["time_sin"] = np.sin(2 * np.pi * minutes / 1440)
        x["time_cos"] = np.cos(2 * np.pi * minutes / 1440)
        x = x.drop(columns=[DATE])
    if "Usage_kWh" in x.columns:
        usage = pd.to_numeric(x["Usage_kWh"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        for lag in [1, 2, 4, 96]:
            x[f"usage_lag_{lag}"] = usage.shift(lag)
        x["usage_roll_mean_4"] = usage.rolling(4, min_periods=1).mean()
        x["usage_roll_mean_96"] = usage.rolling(96, min_periods=1).mean()
        x["usage_roll_std_4"] = usage.rolling(4, min_periods=2).std()
    for col in x.columns:
        if x[col].dtype == "object" or isinstance(x[col].dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(x[col].dtype):
            x[col] = x[col].where(x[col].notna(), "__MISSING__").astype(str)
        else:
            x[col] = pd.to_numeric(x[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return x
