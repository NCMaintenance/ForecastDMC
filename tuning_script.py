import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error
import json
from datetime import timedelta
import warnings

# Suppress specific warnings if necessary, e.g., from sklearn or lightgbm
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")
# Attempt to use the correct LightGBMWarning or a more general one if specific path fails
try:
    warnings.filterwarnings("ignore", category=lgb.LightGBMWarning)
except AttributeError:
    print("Warning: lgb.LightGBMWarning not found, specific LightGBM warnings may not be suppressed.")
    # As a fallback, could try to suppress a more general warning if needed,
    # but often it's fine to let them pass if they are not disruptive.


print("Script started: Hyperparameter Tuning and MAE Re-evaluation")

# --- forecast_with_lags function (Adapted) ---
def forecast_with_lags(model, historical_data_df, future_df_features_only, features_for_model, target_column_name):
    """
    Generates forecasts iteratively, updating lag and rolling features
    with each new prediction.
    historical_data_df: DataFrame containing historical data, including the target_column_name.
    future_df_features_only: DataFrame with features for the future period (lags/rolling of target will be dynamic).
    features_for_model: List of all feature names the model expects.
    target_column_name: The name of the target variable.
    """

    # Ensure historical_data_df is sorted by Datetime if not already
    historical_data_df = historical_data_df.sort_values(by='Datetime')

    # This list will store the target values, extended with predictions.
    # Start with actual historical values of the target.
    temp_history_for_target = list(historical_data_df[target_column_name].values)

    predictions = []

    for i in range(len(future_df_features_only)):
        current_feature_values_map = {} # Use a map for clarity before ordering

        # Get the row from future_df_features_only which contains pre-computed non-target-related features
        current_future_row = future_df_features_only.iloc[i]

        # 1. Populate non-lag, non-rolling features from current_future_row
        for feature_name in features_for_model:
            if not (feature_name.startswith(f'Lag_{target_column_name}_') or \
                    feature_name.startswith(f'Rolling_Mean_3_{target_column_name}') or \
                    feature_name.startswith(f'Rolling_Mean_7_{target_column_name}')):
                if feature_name in current_future_row:
                    current_feature_values_map[feature_name] = current_future_row[feature_name]
                else:
                    # This case should ideally not be hit if future_df_features_only is prepared correctly
                    current_feature_values_map[feature_name] = 0
                    # print(f"Warning: Feature {feature_name} not found in future_df row {i}. Defaulting to 0.")


        # 2. Populate dynamic lag features for the target column
        for lag_order in range(1, 8): # Assuming max lag of 7
            lag_feature_name = f'Lag_{target_column_name}_{lag_order}'
            if lag_feature_name in features_for_model:
                if lag_order <= len(temp_history_for_target):
                    current_feature_values_map[lag_feature_name] = temp_history_for_target[-lag_order]
                else: # Not enough history for this lag (e.g. first few predictions)
                    current_feature_values_map[lag_feature_name] = 0 # Or some other fill like mean of available history

        # 3. Populate dynamic rolling mean features for the target column
        rolling_mean_3_name = f'Rolling_Mean_3_{target_column_name}'
        if rolling_mean_3_name in features_for_model:
            if len(temp_history_for_target) >= 3:
                current_feature_values_map[rolling_mean_3_name] = np.mean(temp_history_for_target[-3:])
            elif len(temp_history_for_target) > 0:
                current_feature_values_map[rolling_mean_3_name] = np.mean(temp_history_for_target)
            else:
                current_feature_values_map[rolling_mean_3_name] = 0

        rolling_mean_7_name = f'Rolling_Mean_7_{target_column_name}'
        if rolling_mean_7_name in features_for_model:
            if len(temp_history_for_target) >= 7:
                current_feature_values_map[rolling_mean_7_name] = np.mean(temp_history_for_target[-7:])
            elif len(temp_history_for_target) > 0:
                current_feature_values_map[rolling_mean_7_name] = np.mean(temp_history_for_target)
            else:
                current_feature_values_map[rolling_mean_7_name] = 0

        # 4. Ensure features are in the correct order as expected by the model
        ordered_feature_values = [current_feature_values_map.get(f_name, 0) for f_name in features_for_model]

        feature_vector = np.array(ordered_feature_values).reshape(1, -1)
        pred = model.predict(feature_vector)[0]
        pred = max(0, pred)  # Ensure predictions are non-negative
        predictions.append(pred)

        # Add current prediction to temp_history_for_target to be used for next step's lags/rolling means
        temp_history_for_target.append(pred)

    return predictions


# --- Main Script Logic ---
try:
    df = pd.read_csv('processed_may_data.csv')
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    print(f"Data loaded. Shape: {df.shape}")
except FileNotFoundError:
    print("Error: processed_may_data.csv not found. Exiting.")
    exit()

hospitals = df['Hospital'].unique()
targets = ['ED Beds', 'Trolleys', 'Capacity']

# Define base features (common across models, ensure they exist in df)
base_features_for_model = [
    'Hour_sin', 'Hour_cos', 'Day_sin', 'Day_cos', 'Month_sin', 'Month_cos',
    'IsWeekend', 'IsMonday', 'IsFriday', 'IsHoliday',
    'IsSummer', 'IsWinter', 'IsPeakHour', 'IsLowHour', 'Hospital_Code',
    'Hour', 'DayOfWeek', 'DayOfMonth', 'Month', 'Quarter', 'WeekOfYear'
]
base_features_for_model = [f for f in base_features_for_model if f in df.columns]


param_grid = {
    'n_estimators': [100, 200], # Further reduced for subtask speed
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5],
    'num_leaves': [20, 31], # Reduced options
    'subsample': [0.8],
    'colsample_bytree': [0.8],
    'reg_alpha': [0.1],
    'reg_lambda': [0.1],
    'random_state': [42],
    'force_col_wise': [True],
    'verbose': [-1]
}

all_mae_scores = []
tuned_hyperparameters_all = {}
forecast_days = 7 # Fixed for 7-day test period

for hospital in hospitals:
    hospital_df = df[df['Hospital'] == hospital].sort_values(by='Datetime').reset_index(drop=True)
    if hospital_df.empty:
        print(f"No data for hospital: {hospital}. Skipping.")
        continue

    tuned_hyperparameters_all[hospital] = {}

    # Determine split point for 7-day test set (forecast_days * 3 records if 3 records/day)
    # A more robust way is by date:
    if len(hospital_df) < (forecast_days * 3 * 2): # Need at least 2x test set size for dev
         print(f"Skipping {hospital} due to overall insufficient data: {len(hospital_df)} records")
         continue

    last_datetime = hospital_df['Datetime'].max()
    split_datetime_test = last_datetime - timedelta(days=forecast_days -1) # Start of the 7 full days period

    dev_df = hospital_df[hospital_df['Datetime'] < split_datetime_test].copy()
    test_df = hospital_df[hospital_df['Datetime'] >= split_datetime_test].copy()


    if dev_df.empty or len(dev_df) < 20:
        print(f"Skipping {hospital} due to insufficient development data: {len(dev_df)} records (need min 20 for CV).")
        continue
    if test_df.empty or len(test_df) < forecast_days * 3 * 0.9 : # check if test_df has roughly 21 records
        print(f"Skipping {hospital} due to insufficient test data: {len(test_df)} records (expected ~{forecast_days*3}).")
        continue

    print(f"\nProcessing Hospital: {hospital}, Dev records: {len(dev_df)}, Test records: {len(test_df)}")

    for target in targets:
        print(f"  Tuning for Target: {target}")

        current_model_base_features = list(base_features_for_model) # Make a copy
        if target in ['ED Beds', 'Trolleys']:
            if 'Capacity' in df.columns and 'Capacity' not in current_model_base_features:
                current_model_base_features.append('Capacity')
        elif target == 'Capacity': # If Capacity is target, it shouldn't be a feature
            if 'Capacity' in current_model_base_features:
                current_model_base_features.remove('Capacity')

        target_specific_lag_roll_features = [
            col for col in df.columns if
            (col.startswith(f'Lag_{target}_') or col.startswith(f'Rolling_Mean_3_{target}') or col.startswith(f'Rolling_Mean_7_{target}'))
             and target in col
        ]

        final_features_for_model = sorted(list(set(current_model_base_features + target_specific_lag_roll_features)))
        final_features_for_model = [f for f in final_features_for_model if f in dev_df.columns and f != target]

        # Drop rows with NaNs from lag/rolling features for tuning & ensure target is not NaN
        dev_df_cleaned = dev_df.dropna(subset=[target] + final_features_for_model).copy()

        if len(dev_df_cleaned) < 15:
            print(f"    Skipping {target} for {hospital} due to insufficient cleaned dev data for CV: {len(dev_df_cleaned)}")
            tuned_hyperparameters_all[hospital][target] = "Skipped - Insufficient data for CV"
            continue

        X_dev = dev_df_cleaned[final_features_for_model]
        y_dev = dev_df_cleaned[target]

        # Configure TimeSeriesSplit for GridSearchCV
        # test_size ensures that the validation set is not excessively small. Max 2 splits for speed.
        n_splits_cv = min(2, len(X_dev) // (forecast_days*3) if len(X_dev) // (forecast_days*3) > 1 else 2)
        cv_test_size = forecast_days * 3 # Validate on a period similar to final test set
        if len(X_dev) <= cv_test_size * (n_splits_cv) : # Ensure enough data for splits
            n_splits_cv = 2 # Fallback if data is very scarce
            if len(X_dev) <= cv_test_size *2 : # if still not enough for 2 splits of cv_test_size
                 cv_test_size = len(X_dev)//3 if len(X_dev)//3 > 0 else 1 # make test_size smaller

        if cv_test_size == 0:
            print(f"    Skipping {target} for {hospital}: cv_test_size is 0.")
            continue

        tscv = TimeSeriesSplit(n_splits=n_splits_cv, test_size=cv_test_size)

        print(f"    GridSearchCV for {target} with {n_splits_cv} splits, test_size={cv_test_size}...")
        gsearch = GridSearchCV(estimator=lgb.LGBMRegressor(), param_grid=param_grid, scoring='neg_mean_absolute_error', cv=tscv, verbose=0, n_jobs=1) # n_jobs=1 for stability in subtask

        try:
            gsearch.fit(X_dev, y_dev)
            best_params = gsearch.best_params_
            tuned_hyperparameters_all[hospital][target] = best_params
            print(f"    Best params for {target}: {best_params}")
        except Exception as e:
            print(f"    Error during GridSearchCV for {target} at {hospital}: {e}")
            tuned_hyperparameters_all[hospital][target] = f"Error during tuning: {str(e)}"
            continue # Skip to next target or hospital if tuning fails

        # Retrain on full dev_df_cleaned using best_params for MAE evaluation
        model_final = lgb.LGBMRegressor(**best_params)
        model_final.fit(X_dev, y_dev) # X_dev and y_dev are from dev_df_cleaned

        # Prepare features for the test_df period.
        # X_test_features_only should contain all features EXCEPT the target-specific lags/rolling means,
        # as these will be generated by forecast_with_lags.
        # Start with all features, then forecast_with_lags will use what it needs.
        X_test_for_predictions = test_df[final_features_for_model].copy()

        # We use dev_df_cleaned as historical_data_df because model was trained on it.
        # forecast_with_lags will use the tail of dev_df_cleaned's target to start predictions.
        predictions = forecast_with_lags(model_final, dev_df_cleaned, X_test_for_predictions, final_features_for_model, target)

        actuals = test_df[target].values[:len(predictions)]

        if len(actuals) == 0 or len(predictions) == 0 or len(actuals) != len(predictions):
            print(f"    Could not make predictions or align actuals for {target} at {hospital}. Actuals len: {len(actuals)}, Preds len: {len(predictions)}. Skipping MAE.")
            all_mae_scores.append({'hospital': hospital, 'target': target, 'mae': np.nan, 'params': best_params if 'best_params' in locals() else "Error in tuning"})
            continue

        mae = mean_absolute_error(actuals, predictions)
        print(f"    {hospital} - {target} - Tuned Test MAE: {mae:.2f}")
        all_mae_scores.append({'hospital': hospital, 'target': target, 'mae': mae, 'params': best_params})


# Save tuned hyperparameters
with open('tuned_hyperparameters_per_hospital_metric.json', 'w') as f:
    json.dump(tuned_hyperparameters_all, f, indent=4)
print("\nTuned hyperparameters saved to tuned_hyperparameters_per_hospital_metric.json")

# Save MAE scores
avg_mae_overall = np.nanmean([score['mae'] for score in all_mae_scores if 'mae' in score])
with open('tuned_mae_scores.txt', 'w') as f:
    f.write(f"Average MAE across all series after tuning: {avg_mae_overall:.4f}\n\n")
    for score in all_mae_scores:
        f.write(f"Hospital: {score['hospital']}, Target: {score['target']}, MAE: {score.get('mae', 'N/A'):.4f}, Params: {score.get('params', 'N/A')}\n")
print(f"Average MAE after tuning: {avg_mae_overall:.4f}")

print("Script finished: Hyperparameter Tuning and MAE Re-evaluation")
