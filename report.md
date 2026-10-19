# DATA DIVERGENCE — Final ML Solution Report
## Track 2: Steel Plant Energy Consumption Forecasting

### Objective
Forecast electricity usage (`next_interval_Usage_kWh`) for the next 15-minute interval using the current and historical plant telemetry. This matches the Track 2 task in the supplied problem statement.

### Dataset and findings
- Training file: `train(1).csv`; **28,031 rows** and **12 columns**.
- Target: `next_interval_Usage_kWh` (continuous value), so this is a regression task.
- Timestamp: `date`, parsed in day/month/year hour:minute format and sorted chronologically before validation.
- Missing values in the source data: **0 total**.
- Predictor groups include current usage, reactive power, power factor, CO₂, seconds since midnight, weekday/weekend, day of week and load type.

### Methodology
1. Parse `date` and create year, month, day, hour, minute, day-of-year, day-of-week and cyclical time-of-day features.
2. Create history features from `Usage_kWh`: lags of 1, 2, 4 and 96 rows, rolling means over 4 and 96 rows, and a 4-row rolling standard deviation. These use current/past usage only.
3. Use **CatBoostRegressor**, which supports categorical variables and missing numeric values. Categorical variables are passed natively; missing categorical values receive a dedicated label.
4. Validate chronologically: first 80% of rows for training, final 20% for validation. This better reflects forecasting future periods than a random split. Early stopping selects a suitable iteration count.
5. Refit on all labeled data and save `model_pipeline.joblib` for inference.

### Validation results
| Metric | Result |
|---|---:|
| RMSE | 8.2220 kWh |
| MAE | 3.9874 kWh |
| Persistence baseline RMSE (predict current usage) | 11.9469 kWh |
| Persistence baseline MAE | 5.2798 kWh |

RMSE penalizes larger forecast errors more strongly; MAE is the average absolute error in kWh. Compare the model with the persistence baseline to understand whether it improves on simply carrying current usage forward.

### Practical business value
A reliable 15-minute forecast can help plant operators anticipate demand peaks, schedule flexible loads, and reduce the risk of exceeding contracted power thresholds. Predictions should be monitored over time and retrained when equipment, production schedules or tariffs change.

### Files and reproduction
- `train.py`: reproducible training and chronological validation.
- `predict.py`: load the saved model and predict from a CSV with the same predictor schema.
- `model_pipeline.joblib`: final trained model and feature metadata.
- `metrics.json`: machine-readable validation details.

Install dependencies with `pip install -r requirements.txt`. Train with `python train.py train.csv model_pipeline.joblib`. Predict with `python predict.py test.csv predictions.csv`.

**Important inference note:** Keep input rows chronological so lag features are meaningful. For the first rows of a standalone test file, history lags may be missing unless the preceding history is included; the model handles these missing values. The target column is not required in test input.
