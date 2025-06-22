import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error
import json
from datetime import datetime, timedelta
import warnings
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday
from pandas.tseries.offsets import DateOffset
from dateutil.rrule import MO

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")
try:
    warnings.filterwarnings("ignore", category=lgb.LightGBMWarning)
except AttributeError:
    print("Warning: lgb.LightGBMWarning not found.")

print("Script started: Simplified Selective Tuning (Iter 4 - No New Features)")

# --- IrishBankHolidays Class ---
# (Keep for consistency, though not used for new feature gen in this version)
class IrishBankHolidays(AbstractHolidayCalendar):
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=nearest_workday),
        Holiday("St. Brigid's Day", month=2, day=1, observance=nearest_workday),
        Holiday("St. Patrick's Day", month=3, day=17, observance=nearest_workday),
        Holiday("Easter Monday", month=1, day=1, offset=[pd.DateOffset(weekday=MO(1), weeks=1), pd.offsets.Easter()]),
        Holiday("May Day", month=5, day=1, offset=DateOffset(weekday=MO(1))),
        Holiday("June Bank Holiday", month=6, day=1, offset=DateOffset(weekday=MO(1))),
        Holiday("August Bank Holiday", month=8, day=1, offset=DateOffset(weekday=MO(1))),
        Holiday("October Bank Holiday", month=10, day=1, offset=DateOffset(weekday=MO(-1))),
        Holiday("Christmas Day", month=12, day=25),
        Holiday("St. Stephen's Day", month=12, day=26),
    ]

# --- forecast_with_lags function ---
def forecast_with_lags(model, historical_data_df, future_df_features_only, features_for_model, target_column_name):
    historical_data_df = historical_data_df.sort_values(by='Datetime')
    temp_history_for_target = list(historical_data_df[target_column_name].values)
    predictions = []
    for i in range(len(future_df_features_only)):
        current_feature_values_map = {}
        current_future_row = future_df_features_only.iloc[i]
        for feature_name in features_for_model:
            is_target_lag_or_roll = feature_name.startswith(f'Lag_{target_column_name}_') or \
                                   feature_name.startswith(f'Rolling_Mean_3_{target_column_name}') or \
                                   feature_name.startswith(f'Rolling_Mean_7_{target_column_name}')
            # Note: Iter3 ED Beds specific features are NOT part of this simplified run's dynamic features.
            # They would have been pre-calculated if this script used an augmented CSV.
            # Since we load original CSV, they are not used.

            if not is_target_lag_or_roll: # Base features, interactions from original CSV, etc.
                if feature_name in current_future_row:
                    current_feature_values_map[feature_name] = current_future_row[feature_name]
                else:
                    current_feature_values_map[feature_name] = 0
            else: # Dynamic target-specific lags/rolling means
                for lag_order in range(1, 8):
                    if feature_name == f'Lag_{target_column_name}_{lag_order}':
                        if lag_order <= len(temp_history_for_target): current_feature_values_map[feature_name] = temp_history_for_target[-lag_order]
                        else: current_feature_values_map[feature_name] = 0
                if feature_name == f'Rolling_Mean_3_{target_column_name}':
                    if len(temp_history_for_target) >= 3: current_feature_values_map[feature_name] = np.mean(temp_history_for_target[-3:])
                    elif len(temp_history_for_target) > 0: current_feature_values_map[feature_name] = np.mean(temp_history_for_target)
                    else: current_feature_values_map[feature_name] = 0
                if feature_name == f'Rolling_Mean_7_{target_column_name}':
                    if len(temp_history_for_target) >= 7: current_feature_values_map[feature_name] = np.mean(temp_history_for_target[-7:])
                    elif len(temp_history_for_target) > 0: current_feature_values_map[feature_name] = np.mean(temp_history_for_target)
                    else: current_feature_values_map[feature_name] = 0

        ordered_feature_values = [current_feature_values_map.get(f_name, 0) for f_name in features_for_model]
        feature_vector = np.array(ordered_feature_values).reshape(1, -1)
        pred = model.predict(feature_vector)[0]
        pred = max(0, pred)
        predictions.append(pred)
        temp_history_for_target.append(pred)
    return predictions

# --- Load Data (Original processed_may_data.csv) ---
try:
    df = pd.read_csv('processed_may_data.csv')
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    print(f"Original processed_may_data.csv loaded. Shape: {df.shape}")
except FileNotFoundError:
    print("Error: processed_may_data.csv not found. Exiting.")
    exit()

# NO NEW FEATURE ENGINEERING IN THIS SIMPLIFIED RUN
print("Skipping Iteration 2 and 3 feature engineering for this run due to timeouts.")
print(f"Using features from original processed_may_data.csv. Shape: {df.shape}")


# --- Main Script Logic ---
hospitals = df['Hospital'].unique()
targets = ['ED Beds', 'Trolleys', 'Capacity']

# Define feature sets based on original processed_may_data.csv columns
# These are features from the very first data processing script (run_processing.py)
original_features = [col for col in df.columns if col not in ['Hospital Group Name', 'Hospital', 'Date', 'Time', 'Datetime', 'DayGAR', 'ED Beds', 'Trolleys', 'Capacity']]

# Load hyperparameters from previous successful tuning runs (iter2 for T/C, iter3 for ED Beds)
try:
    with open('tuned_hyperparameters_iter2.json', 'r') as f:
        params_iter2 = json.load(f)
except FileNotFoundError: print("Warning: tuned_hyperparameters_iter2.json not found."); params_iter2 = {}
try:
    with open('tuned_hyperparameters_iter3.json', 'r') as f:
        params_iter3 = json.load(f)
except FileNotFoundError: print("Warning: tuned_hyperparameters_iter3.json not found."); params_iter3 = {}

all_mae_scores_iter4 = []
tuned_hyperparameters_iter4 = {}
forecast_days = 7

for hospital in hospitals:
    hospital_df = df[df['Hospital'] == hospital].sort_values(by='Datetime').reset_index(drop=True)
    if hospital_df.empty: continue
    tuned_hyperparameters_iter4[hospital] = {}

    if len(hospital_df) < (forecast_days * 3 * 2): # Min data for dev/test split
         print(f"Skipping {hospital} due to overall insufficient data: {len(hospital_df)} records")
         continue
    last_datetime = hospital_df['Datetime'].max()
    split_datetime_test = last_datetime - timedelta(days=forecast_days -1)
    dev_df = hospital_df[hospital_df['Datetime'] < split_datetime_test].copy()
    test_df = hospital_df[hospital_df['Datetime'] >= split_datetime_test].copy()

    if dev_df.empty or len(dev_df) < 15 or test_df.empty or len(test_df) < forecast_days*3*0.9 : # Min for training and reliable test
        print(f"Skipping {hospital} due to insufficient dev/test data after split (Dev: {len(dev_df)}, Test: {len(test_df)}).")
        continue

    print(f"\nProcessing Hospital: {hospital}, Dev records: {len(dev_df)}, Test records: {len(test_df)}")
    for target in targets:
        print(f"  Processing Target: {target} (Iter 4 - Simplified Features)")

        # Use all available original features, excluding other targets
        current_model_features = [f for f in original_features if f != 'ED Beds' and f != 'Trolleys' and f != 'Capacity']
        # Add 'Capacity' as a feature if it's not the target and it's an original feature
        if target in ['ED Beds', 'Trolleys'] and 'Capacity' in original_features and 'Capacity' not in current_model_features:
            current_model_features.append('Capacity')

        # Add target-specific lag/rolling features that exist in original_features
        target_specific_lag_roll_features = [col for col in original_features if (col.startswith(f'Lag_{target}_') or \
            col.startswith(f'Rolling_Mean_3_{target}') or col.startswith(f'Rolling_Mean_7_{target}')) and target in col]

        final_features_for_model = sorted(list(set(current_model_features + target_specific_lag_roll_features)))
        final_features_for_model = [f for f in final_features_for_model if f in dev_df.columns and f != target] # Ensure they exist

        dev_df_cleaned = dev_df.dropna(subset=[target] + final_features_for_model).copy()
        if len(dev_df_cleaned) < 15:
            print(f"    Skipping {target} for {hospital} (Iter 4) due to insufficient cleaned dev data: {len(dev_df_cleaned)}")
            tuned_hyperparameters_iter4[hospital][target] = "Skipped - Insufficient data"
            continue
        X_dev = dev_df_cleaned[final_features_for_model]
        y_dev = dev_df_cleaned[target]

        best_params_for_target = {}
        # NO GridSearchCV in this simplified run. Use previously tuned params.
        if target == 'ED Beds':
            print(f"    Using Iter3 params for ED Beds at {hospital}")
            best_params_for_target = params_iter3.get(hospital, {}).get(target, {})
            if not isinstance(best_params_for_target, dict) or not best_params_for_target:
                 best_params_for_target = {'n_estimators': 800, 'learning_rate': 0.03, 'max_depth': 7, 'num_leaves': 50, 'random_state': 42, 'verbose': -1, 'force_col_wise': True} # Generic fallback for ED Beds
        else: # Trolleys and Capacity
            print(f"    Using Iter2 params for {target} at {hospital}")
            best_params_for_target = params_iter2.get(hospital, {}).get(target, {})
            if not isinstance(best_params_for_target, dict) or not best_params_for_target:
                best_params_for_target = {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 5, 'num_leaves': 31, 'random_state': 42, 'verbose': -1, 'force_col_wise': True} # Generic fallback for T/C

        tuned_hyperparameters_iter4[hospital][target] = best_params_for_target
        model_final = lgb.LGBMRegressor(**best_params_for_target)
        try:
            model_final.fit(X_dev, y_dev)
            X_test_for_predictions = test_df[final_features_for_model].copy()
            predictions = forecast_with_lags(model_final, dev_df_cleaned, X_test_for_predictions, final_features_for_model, target)
            actuals = test_df[target].values[:len(predictions)]

            if len(actuals) == 0 or len(predictions) == 0 or len(actuals) != len(predictions):
                print(f"    Could not make predictions or align actuals for {target} (Iter 4) at {hospital}. Preds: {len(predictions)}, Actuals: {len(actuals)}. Skipping MAE.")
                all_mae_scores_iter4.append({'hospital': hospital, 'target': target, 'mae': np.nan, 'params': best_params_for_target})
            else:
                mae = mean_absolute_error(actuals, predictions)
                print(f"    {hospital} - {target} - Test MAE (Iter 4 - Simplified Features): {mae:.2f}")
                all_mae_scores_iter4.append({'hospital': hospital, 'target': target, 'mae': mae, 'params': best_params_for_target})
        except Exception as e_model_fit_pred:
            print(f"    Error during final model fit/prediction for {target} at {hospital}: {e_model_fit_pred}")
            all_mae_scores_iter4.append({'hospital': hospital, 'target': target, 'mae': np.nan, 'params': best_params_for_target, 'error': str(e_model_fit_pred)})


# Save results
with open('tuned_hyperparameters_iter4.json', 'w') as f:
    json.dump(tuned_hyperparameters_iter4, f, indent=4)
print("\nTuned hyperparameters (Iter 4) saved to tuned_hyperparameters_iter4.json")

avg_mae_overall_iter4 = np.nanmean([score['mae'] for score in all_mae_scores_iter4 if 'mae' in score])
with open('tuned_mae_scores_iter4.txt', 'w') as f:
    f.write(f"Average MAE across all series after tuning (Iter 4): {avg_mae_overall_iter4:.4f}\n\n")
    for score in all_mae_scores_iter4:
        f.write(f"Hospital: {score['hospital']}, Target: {score['target']}, MAE: {score.get('mae', 'N/A'):.4f}, Params: {score.get('params', 'N/A')}\n")
print(f"Average MAE after tuning (Iter 4): {avg_mae_overall_iter4:.4f}")

print("Script finished: Selective Feature/Hyperparameter Tuning (Iter 4)")
