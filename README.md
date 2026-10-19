# DATA DIVERGENCE — Track 2: Steel Plant Energy Consumption Forecasting

## Files
- `train.py`: validates data, trains CatBoost, evaluates a chronological holdout, saves model and metrics.
- `predict.py`: validates new input and writes `next_interval_Usage_kWh` predictions.
- `feature_utils.py`: shared input validation and feature engineering used by both scripts.
- `test_pipeline.py`: automated normal/error-case tests.
- `train.csv`: supplied training data.
- `sample_input.csv`: sample inference input (target column is intentionally absent).
- `model_pipeline.joblib`: supplied trained model; regenerate it with `train.py` if needed.
- `metrics.json`: validation metrics.
- `report.md`: project report.
- `requirements.txt`: dependencies.

## Setup
Install the dependencies in the Python environment used by VS Code:
```bash
pip install -r requirements.txt
```

## Train
Keep `train.csv` in this folder and run:
```bash
python train.py
```
Or specify paths:
```bash
python train.py train.csv model_pipeline.joblib
```
Training saves `model_pipeline.joblib` and `metrics.json`.

## Predict
```bash
python predict.py sample_input.csv predictions.csv
```
An optional model path is supported:
```bash
python predict.py sample_input.csv predictions.csv model_pipeline.joblib
```
The prediction input must contain all required predictor columns but does not need the target. The output contains the input columns and predicted `next_interval_Usage_kWh`. Rows in output are restored to the same order as the input.

## Run automated tests
```bash
python -m unittest -v test_pipeline.py
```
The tests cover normal prediction, missing columns, invalid timestamps, empty input, duplicate timestamps, infinite values, invalid numeric values, and target-free inference.

## Notes and limitations
- The model is a regression model. Predictions are estimates, not guaranteed exact values.
- Validation uses the first 80% of the chronologically sorted dataset for training and the final 20% for validation. It is not the competition's hidden test score.
- Missing numeric values other than `Usage_kWh` are converted to missing values with warnings; CatBoost can handle missing numeric values, but too many missing inputs can reduce reliability.
- Input is sorted chronologically for feature engineering and output order is restored afterward.
- The 96-row lag assumes roughly 15-minute sampling and is only meaningful when preceding history is included. Gaps are warned about.
- Keep a copy of the original dataset and model before experimenting.
