import sys, json, warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from feature_utils import TARGET, DATE, CATEGORICAL_HINTS, validate_input_frame, prepare_features

RANDOM_SEED = 42
TEST_SIZE = 0.20

def main(train_csv="train.csv", output="model_pipeline.joblib"):
    train_path = Path(train_csv)
    if not train_path.is_file():
        raise FileNotFoundError(f"Training CSV not found: {train_path.resolve()}")
    try:
        raw = pd.read_csv(train_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("Training CSV is empty.") from exc
    df = validate_input_frame(raw, training=True)
    y_raw = pd.to_numeric(df[TARGET], errors="coerce").replace([np.inf, -np.inf], np.nan)
    invalid_target = y_raw.isna()
    if invalid_target.any():
        warnings.warn(f"Dropping {int(invalid_target.sum())} row(s) with missing/non-numeric/non-finite target.")
        df = df.loc[~invalid_target].reset_index(drop=True)
        y = y_raw.loc[~invalid_target].reset_index(drop=True)
    else:
        y = y_raw.reset_index(drop=True)
    if len(df) < 20:
        raise ValueError(f"Only {len(df)} valid rows remain; at least 20 are required for a useful chronological split.")
    if (y < 0).any():
        warnings.warn("Negative target values detected. Confirm these are valid for your energy dataset.")
    X = prepare_features(df.drop(columns=[TARGET]))
    categorical_columns = [c for c in X.columns if X[c].dtype == "object" or pd.api.types.is_string_dtype(X[c].dtype)]
    for c in categorical_columns:
        X[c] = X[c].where(X[c].notna(), "__MISSING__").astype(str)
    for c in X.columns:
        if c not in categorical_columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    split = int(len(X) * (1 - TEST_SIZE))
    if split < 10 or len(X) - split < 2:
        raise ValueError("Not enough rows for chronological training/validation split.")
    X_train, X_valid = X.iloc[:split].copy(), X.iloc[split:].copy()
    y_train, y_valid = y.iloc[:split].copy(), y.iloc[split:].copy()
    print(f"Rows: {len(df)} | Features: {len(X.columns)} | Train: {len(X_train)} | Validation: {len(X_valid)}")
    print("Categorical columns:", categorical_columns)
    model = CatBoostRegressor(
        iterations=1000, depth=8, learning_rate=0.03,
        loss_function="RMSE", eval_metric="RMSE",
        random_seed=RANDOM_SEED, l2_leaf_reg=5,
        verbose=False, allow_writing_files=False, thread_count=4
    )
    model.fit(X_train, y_train, cat_features=categorical_columns,
              eval_set=(X_valid, y_valid), early_stopping_rounds=120, verbose=False)
    val_pred = np.asarray(model.predict(X_valid), dtype=float)
    if not np.isfinite(val_pred).all():
        raise RuntimeError("Model generated non-finite validation predictions.")
    mae = float(mean_absolute_error(y_valid, val_pred))
    rmse = float(np.sqrt(mean_squared_error(y_valid, val_pred)))
    baseline = X_valid["Usage_kWh"].to_numpy(dtype=float)
    baseline_mask = np.isfinite(baseline) & np.isfinite(y_valid.to_numpy(dtype=float))
    baseline_mae = float(mean_absolute_error(y_valid.to_numpy()[baseline_mask], baseline[baseline_mask])) if baseline_mask.any() else None
    baseline_rmse = float(np.sqrt(mean_squared_error(y_valid.to_numpy()[baseline_mask], baseline[baseline_mask]))) if baseline_mask.any() else None
    best = model.get_best_iteration()
    iterations = max(1, best + 1) if best is not None and best >= 0 else int(model.tree_count_)
    final_model = CatBoostRegressor(
        iterations=iterations, depth=8, learning_rate=0.03,
        loss_function="RMSE", random_seed=RANDOM_SEED, l2_leaf_reg=5,
        verbose=False, allow_writing_files=False, thread_count=4
    )
    final_model.fit(X, y, cat_features=categorical_columns, verbose=False)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": final_model, "feature_columns": list(X.columns),
        "categorical_columns": categorical_columns, "target": TARGET,
        "date_column": DATE, "validation_mae": mae, "validation_rmse": rmse,
        "baseline_mae": baseline_mae, "baseline_rmse": baseline_rmse,
        "iterations": int(iterations), "model_version": 2
    }
    joblib.dump(bundle, out_path)
    metrics = {
        "dataset_rows": int(len(df)), "dataset_columns": int(raw.shape[1]),
        "target": TARGET, "validation_method": "chronological 80/20 holdout",
        "validation_rows": int(len(X_valid)), "training_rows": int(len(X_train)),
        "features_used": list(X.columns), "categorical_features": categorical_columns,
        "validation_MAE": mae, "validation_RMSE": rmse,
        "persistence_baseline_MAE": baseline_mae, "persistence_baseline_RMSE": baseline_rmse,
        "final_model_iterations": int(iterations)
    }
    metrics_path = out_path.parent / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Validation MAE: {mae:.4f} kWh | RMSE: {rmse:.4f} kWh")
    print(f"Baseline MAE: {baseline_mae} | RMSE: {baseline_rmse}")
    print("Saved model:", out_path.resolve())
    print("Saved metrics:", metrics_path.resolve())

if __name__ == "__main__":
    try:
        main(sys.argv[1] if len(sys.argv) > 1 else "train.csv",
             sys.argv[2] if len(sys.argv) > 2 else "model_pipeline.joblib")
    except Exception as exc:
        print(f"\nTRAINING FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
