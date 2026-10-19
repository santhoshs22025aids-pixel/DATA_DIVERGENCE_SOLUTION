import sys, warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from feature_utils import TARGET, DATE, CATEGORICAL_HINTS, validate_input_frame, prepare_features

def predict(input_csv, output_csv, model_path=None):
    input_path = Path(input_csv)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path.resolve()}")
    if model_path is None:
        model_path = Path(__file__).with_name("model_pipeline.joblib")
    else:
        model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path.resolve()}. Train the model first.")
    try:
        raw = pd.read_csv(input_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("Input CSV is empty or has no header.") from exc
    if raw.empty:
        raise ValueError("Input CSV contains headers but no data rows.")
    # Keep original row identity so output order matches the input file.
    raw = raw.copy()
    raw["_original_row_order_internal_"] = np.arange(len(raw))
    df = validate_input_frame(raw.drop(columns=["_original_row_order_internal_"]), training=False)
    # validate_input_frame sorts chronologically; align original row order by a stable timestamp mapping.
    # Duplicate timestamps are rejected, so timestamp is a safe join key.
    original_order = pd.DataFrame({
        DATE: pd.to_datetime(raw[DATE].values, format="%d/%m/%Y %H:%M", errors="coerce"),
        "_original_order": np.arange(len(raw))
    })
    bad = original_order[DATE].isna()
    if bad.any():
        original_order.loc[bad, DATE] = pd.to_datetime(raw.loc[bad, DATE], errors="coerce", dayfirst=True)
    X = prepare_features(df)
    try:
        bundle = joblib.load(model_path)
    except Exception as exc:
        raise RuntimeError(f"Could not load model bundle '{model_path}': {exc}") from exc
    required_keys = {"model", "feature_columns", "categorical_columns"}
    if not isinstance(bundle, dict) or not required_keys.issubset(bundle):
        raise ValueError("Model bundle is invalid: expected model, feature_columns, and categorical_columns.")
    feature_columns = bundle["feature_columns"]
    categorical_columns = bundle["categorical_columns"]
    missing_model_features = [c for c in feature_columns if c not in X.columns]
    if missing_model_features:
        # Feature-engineering columns can be absent for a single-row input (e.g., lags); these are expected.
        allowed_generated = [c for c in missing_model_features if c.startswith("usage_lag_") or c.startswith("usage_roll_")]
        unexpected = [c for c in missing_model_features if c not in allowed_generated]
        if unexpected:
            raise ValueError(f"Internal feature mismatch; missing feature(s): {unexpected}")
        for c in allowed_generated:
            X[c] = np.nan
    X = X.reindex(columns=feature_columns)
    for col in categorical_columns:
        if col in X.columns:
            X[col] = X[col].where(X[col].notna(), "__MISSING__").astype(str)
    for col in X.columns:
        if col not in categorical_columns:
            X[col] = pd.to_numeric(X[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if X.empty:
        raise ValueError("No valid rows available for prediction.")
    # Report critical missing inputs; model can handle some NaNs but predictions may be less reliable.
    critical = ["Usage_kWh"]
    for col in critical:
        if col in X.columns and X[col].isna().any():
            rows = (X.index[X[col].isna()] + 2).tolist()[:10]
            raise ValueError(f"Critical input '{col}' is missing/non-numeric at CSV row(s) {rows}.")
    unknown_categories = {}
    for col in categorical_columns:
        if col in df.columns:
            known = set(str(v) for v in getattr(bundle["model"], "classes_", [])) if False else None
            values = set(df[col].astype(str).unique())
            expected = {
                "WeekStatus": {"Weekday", "Weekend"},
                "Day_of_week": {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"},
                "Load_Type": {"Light_Load", "Medium_Load", "Maximum_Load"},
            }.get(col)
            if expected and not values.issubset(expected):
                unknown_categories[col] = sorted(values - expected)
    if unknown_categories:
        warnings.warn(f"Unrecognized categorical value(s) detected: {unknown_categories}. Predictions will be attempted, but verify labels.")
    predictions = np.asarray(bundle["model"].predict(X), dtype=float).reshape(-1)
    if len(predictions) != len(df):
        raise RuntimeError(f"Prediction count mismatch: expected {len(df)}, got {len(predictions)}.")
    if not np.isfinite(predictions).all():
        bad_rows = (np.where(~np.isfinite(predictions))[0] + 2).tolist()[:10]
        raise RuntimeError(f"Model produced non-finite prediction(s) at sorted row(s) {bad_rows}.")
    # Keep original input columns/values and row order; map predictions back by timestamp.
    # Duplicate timestamps were already rejected, so timestamp mapping is unambiguous.
    prediction_map = dict(zip(df[DATE], predictions))
    output = raw.drop(columns=["_original_row_order_internal_"]).copy()
    parsed_original_dates = pd.to_datetime(
        output[DATE], format="%d/%m/%Y %H:%M", errors="coerce"
    )
    bad_original_dates = parsed_original_dates.isna()
    if bad_original_dates.any():
        parsed_original_dates.loc[bad_original_dates] = pd.to_datetime(
            output.loc[bad_original_dates, DATE], errors="coerce", dayfirst=True
        )
    output[TARGET] = parsed_original_dates.map(prediction_map).to_numpy(dtype=float)
    if output[TARGET].isna().any() or not np.isfinite(output[TARGET].to_numpy(dtype=float)).all():
        raise RuntimeError("Could not align every prediction back to the original input rows.")
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        output.to_csv(out_path, index=False)
    except PermissionError as exc:
        raise PermissionError(f"Cannot write '{out_path}'. Close it in Excel or choose another output filename.") from exc
    except OSError as exc:
        raise OSError(f"Could not write output CSV '{out_path}': {exc}") from exc
    print(f"Predictions generated: {len(output)}")
    print(f"Output file: {out_path.resolve()}")
    print("First five predictions:")
    print(output[[DATE, TARGET]].head().to_string(index=False))
    return output

if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print("Usage: python predict.py input.csv output.csv [model_pipeline.joblib]", file=sys.stderr)
        sys.exit(2)
    try:
        predict(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) == 4 else None)
    except Exception as exc:
        print(f"\nPREDICTION FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
