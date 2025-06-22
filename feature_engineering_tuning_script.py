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
from dateutil.rrule import MO # For MO (Monday) in holiday rules

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")
try:
    warnings.filterwarnings("ignore", category=lgb.LightGBMWarning)
except AttributeError:
    print("Warning: lgb.LightGBMWarning not found, specific LightGBM warnings may not be suppressed.")

print("Script started: New Feature Engineering, Hyperparameter Tuning (Iter 2), and MAE Re-evaluation")

# --- IrishBankHolidays Class (Copied from DMC.py) ---
class IrishBankHolidays(AbstractHolidayCalendar):
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=nearest_workday),
        Holiday("St. Brigid's Day", month=2, day=1, observance=nearest_workday), # First Monday in February
        Holiday("St. Patrick's Day", month=3, day=17, observance=nearest_workday),
        Holiday("Easter Monday", month=1, day=1, offset=[pd.DateOffset(weekday=MO(1), weeks=1), pd.offsets.Easter()]), # Needs careful construction
        Holiday("May Day", month=5, day=1, offset=DateOffset(weekday=MO(1))), # First Monday in May
        Holiday("June Bank Holiday", month=6, day=1, offset=DateOffset(weekday=MO(1))), # First Monday in June
        Holiday("August Bank Holiday", month=8, day=1, offset=DateOffset(weekday=MO(1))), # First Monday in August
        Holiday("October Bank Holiday", month=10, day=1, offset=DateOffset(weekday=MO(-1))), # Last Monday in October
        Holiday("Christmas Day", month=12, day=25),
        Holiday("St. Stephen's Day", month=12, day=26),
    ]
    # Adjust Easter Monday - it's complex. For simplicity, will use a fixed list or approximate.
    # A proper Easter calculation is non-trivial.
    # For this script, we'll rely on the 'IsHoliday' feature already in processed_may_data.csv for direct holiday impact.
    # The proximity features will be calculated based on a list of holidays derived for the data range.

# --- forecast_with_lags function (Adapted from previous tuning_script.py) ---
def forecast_with_lags(model, historical_data_df, future_df_features_only, features_for_model, target_column_name):
    historical_data_df = historical_data_df.sort_values(by='Datetime')
    temp_history_for_target = list(historical_data_df[target_column_name].values)
    predictions = []
    for i in range(len(future_df_features_only)):
        current_feature_values_map = {}
        current_future_row = future_df_features_only.iloc[i]
        for feature_name in features_for_model:
            if not (feature_name.startswith(f'Lag_{target_column_name}_') or \
                    feature_name.startswith(f'Rolling_Mean_3_{target_column_name}') or \
                    feature_name.startswith(f'Rolling_Mean_7_{target_column_name}')):
                if feature_name in current_future_row:
                    current_feature_values_map[feature_name] = current_future_row[feature_name]
                else:
                    current_feature_values_map[feature_name] = 0
            for lag_order in range(1, 8):
                lag_feature_name = f'Lag_{target_column_name}_{lag_order}'
                if lag_feature_name == feature_name : # Check if current feature_name is this specific lag
                    if lag_order <= len(temp_history_for_target):
                        current_feature_values_map[lag_feature_name] = temp_history_for_target[-lag_order]
                    else:
                        current_feature_values_map[lag_feature_name] = 0
            rolling_mean_3_name = f'Rolling_Mean_3_{target_column_name}'
            if rolling_mean_3_name == feature_name:
                if len(temp_history_for_target) >= 3: current_feature_values_map[rolling_mean_3_name] = np.mean(temp_history_for_target[-3:])
                elif len(temp_history_for_target) > 0: current_feature_values_map[rolling_mean_3_name] = np.mean(temp_history_for_target)
                else: current_feature_values_map[rolling_mean_3_name] = 0
            rolling_mean_7_name = f'Rolling_Mean_7_{target_column_name}'
            if rolling_mean_7_name == feature_name:
                if len(temp_history_for_target) >= 7: current_feature_values_map[rolling_mean_7_name] = np.mean(temp_history_for_target[-7:])
                elif len(temp_history_for_target) > 0: current_feature_values_map[rolling_mean_7_name] = np.mean(temp_history_for_target)
                else: current_feature_values_map[rolling_mean_7_name] = 0
        ordered_feature_values = [current_feature_values_map.get(f_name, 0) for f_name in features_for_model]
        feature_vector = np.array(ordered_feature_values).reshape(1, -1)
        pred = model.predict(feature_vector)[0]
        pred = max(0, pred)
        predictions.append(pred)
        temp_history_for_target.append(pred)
    return predictions

# --- Load Data ---
try:
    df = pd.read_csv('processed_may_data.csv')
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    print(f"Data loaded. Shape: {df.shape}")
except FileNotFoundError:
    print("Error: processed_may_data.csv not found. Exiting.")
    exit()

# --- Advanced Feature Engineering ---
print("Starting Advanced Feature Engineering...")

# 1. Interaction Features
df['Hour_DayOfWeek'] = df['Hour'].astype(str) + "_" + df['DayOfWeek'].astype(str)
df['Hour_DayOfWeek'] = df['Hour_DayOfWeek'].astype('category').cat.codes

df['Hospital_Code_Hour'] = df['Hospital_Code'].astype(str) + "_" + df['Hour'].astype(str)
df['Hospital_Code_Hour'] = df['Hospital_Code_Hour'].astype('category').cat.codes

# Numeric interaction - ensure Capacity and Hour are appropriate for multiplication
if 'Capacity' in df.columns and 'Hour' in df.columns:
    df['Capacity_Hour_Interaction'] = df['Capacity'] * df['Hour']
else:
    print("Warning: 'Capacity' or 'Hour' not found for numeric interaction. Skipping 'Capacity_Hour_Interaction'.")

# 2. Holiday Proximity Features
# Generate a list of actual holiday dates within the data's range and a bit beyond for next holiday calculation
actual_holidays = []
if not df.empty:
    calendar = IrishBankHolidays()
    min_date = df['Datetime'].min()
    max_date = df['Datetime'].max()
    # Extend range slightly for 'DaysToNextHoliday'
    actual_holidays = calendar.holidays(start=min_date - timedelta(days=60), end=max_date + timedelta(days=60)).normalize()
    actual_holidays = sorted(list(set(actual_holidays))) # Ensure unique and sorted

if not actual_holidays:
    print("Warning: No holidays generated. Proximity features will be 0 or NaN.")
    df['DaysToNextHoliday'] = 0
    df['DaysFromLastHoliday'] = 0
else:
    df['DaysToNextHoliday'] = df['Datetime'].apply(lambda x: min([ (h - x.normalize()).days for h in actual_holidays if h >= x.normalize() ], default=365) )
    df['DaysFromLastHoliday'] = df['Datetime'].apply(lambda x: min([ (x.normalize() - h).days for h in actual_holidays if h <= x.normalize() ], default=365) )
    # Fill default (e.g. 365) for dates far from any holiday or if list is empty before/after
    df['DaysToNextHoliday'] = df['DaysToNextHoliday'].fillna(365) # If no next holiday found (e.g. at end of list)
    df['DaysFromLastHoliday'] = df['DaysFromLastHoliday'].fillna(365) # If no prev holiday (e.g. at start of list)


print(f"Data shape after new features: {df.shape}")
print(f"New features created: Hour_DayOfWeek, Hospital_Code_Hour, Capacity_Hour_Interaction (if applicable), DaysToNextHoliday, DaysFromLastHoliday")
# --- End of Feature Engineering ---


hospitals = df['Hospital'].unique()
targets = ['ED Beds', 'Trolleys', 'Capacity']

base_features_for_model = [
    'Hour_sin', 'Hour_cos', 'Day_sin', 'Day_cos', 'Month_sin', 'Month_cos',
    'IsWeekend', 'IsMonday', 'IsFriday', 'IsHoliday',
    'IsSummer', 'IsWinter', 'IsPeakHour', 'IsLowHour', 'Hospital_Code',
    'Hour', 'DayOfWeek', 'DayOfMonth', 'Month', 'Quarter', 'WeekOfYear',
    # New features added here
    'Hour_DayOfWeek', 'Hospital_Code_Hour', 'DaysToNextHoliday', 'DaysFromLastHoliday'
]
if 'Capacity_Hour_Interaction' in df.columns: # Add if it was created
    base_features_for_model.append('Capacity_Hour_Interaction')

base_features_for_model = [f for f in base_features_for_model if f in df.columns]


param_grid_iter2 = {
    'n_estimators': [200, 400], # Kept smaller for speed
    'learning_rate': [0.03, 0.05], # Try smaller rates
    'max_depth': [5, 7],
    'num_leaves': [31, 50], # Adjusted
    'subsample': [0.8], 'colsample_bytree': [0.8],
    'reg_alpha': [0.1], 'reg_lambda': [0.1],
    'random_state': [42], 'force_col_wise': [True], 'verbose': [-1]
}

all_mae_scores_iter2 = []
tuned_hyperparameters_iter2 = {}
forecast_days = 7

for hospital in hospitals:
    hospital_df = df[df['Hospital'] == hospital].sort_values(by='Datetime').reset_index(drop=True)
    if hospital_df.empty: continue
    tuned_hyperparameters_iter2[hospital] = {}

    if len(hospital_df) < (forecast_days * 3 * 2):
         print(f"Skipping {hospital} due to overall insufficient data: {len(hospital_df)} records")
         continue
    last_datetime = hospital_df['Datetime'].max()
    split_datetime_test = last_datetime - timedelta(days=forecast_days -1)
    dev_df = hospital_df[hospital_df['Datetime'] < split_datetime_test].copy()
    test_df = hospital_df[hospital_df['Datetime'] >= split_datetime_test].copy()

    if dev_df.empty or len(dev_df) < 20:
        print(f"Skipping {hospital} due to insufficient development data: {len(dev_df)} records.")
        continue
    if test_df.empty or len(test_df) < forecast_days * 3 * 0.9:
        print(f"Skipping {hospital} due to insufficient test data: {len(test_df)} records.")
        continue

    print(f"\nProcessing Hospital: {hospital}, Dev records: {len(dev_df)}, Test records: {len(test_df)}")
    for target in targets:
        print(f"  Tuning for Target: {target} with new features")
        current_model_base_features = list(base_features_for_model)
        if target in ['ED Beds', 'Trolleys']:
            if 'Capacity' in df.columns and 'Capacity' not in current_model_base_features:
                current_model_base_features.append('Capacity')
        elif target == 'Capacity':
            if 'Capacity' in current_model_base_features:
                current_model_base_features.remove('Capacity')

        target_specific_lag_roll_features = [
            col for col in df.columns if
            (col.startswith(f'Lag_{target}_') or col.startswith(f'Rolling_Mean_3_{target}') or col.startswith(f'Rolling_Mean_7_{target}'))
             and target in col
        ]
        final_features_for_model = sorted(list(set(current_model_base_features + target_specific_lag_roll_features)))
        final_features_for_model = [f for f in final_features_for_model if f in dev_df.columns and f != target]

        dev_df_cleaned = dev_df.dropna(subset=[target] + final_features_for_model).copy()
        if len(dev_df_cleaned) < 15:
            print(f"    Skipping {target} for {hospital} (new features) due to insufficient cleaned dev data for CV: {len(dev_df_cleaned)}")
            tuned_hyperparameters_iter2[hospital][target] = "Skipped - Insufficient data for CV"
            continue

        X_dev = dev_df_cleaned[final_features_for_model]
        y_dev = dev_df_cleaned[target]

        n_splits_cv = min(2, len(X_dev) // (forecast_days*3) if len(X_dev) // (forecast_days*3) > 1 else 2)
        cv_test_size = forecast_days * 3
        if len(X_dev) <= cv_test_size * (n_splits_cv):
            n_splits_cv = 2
            if len(X_dev) <= cv_test_size *2 :
                 cv_test_size = len(X_dev)//3 if len(X_dev)//3 > 0 else 1
        if cv_test_size == 0:
            print(f"    Skipping {target} for {hospital} (new features): cv_test_size is 0.")
            continue

        tscv = TimeSeriesSplit(n_splits=n_splits_cv, test_size=cv_test_size)
        print(f"    GridSearchCV for {target} (new features) with {n_splits_cv} splits, test_size={cv_test_size}...")
        gsearch = GridSearchCV(estimator=lgb.LGBMRegressor(), param_grid=param_grid_iter2, scoring='neg_mean_absolute_error', cv=tscv, verbose=0, n_jobs=1)

        try:
            gsearch.fit(X_dev, y_dev)
            best_params = gsearch.best_params_
            tuned_hyperparameters_iter2[hospital][target] = best_params
            print(f"    Best params for {target} (new features): {best_params}")
        except Exception as e:
            print(f"    Error during GridSearchCV for {target} (new features) at {hospital}: {e}")
            tuned_hyperparameters_iter2[hospital][target] = f"Error during tuning: {str(e)}"
            continue

        model_final = lgb.LGBMRegressor(**best_params)
        model_final.fit(X_dev, y_dev)

        X_test_for_predictions = test_df[final_features_for_model].copy()
        predictions = forecast_with_lags(model_final, dev_df_cleaned, X_test_for_predictions, final_features_for_model, target)
        actuals = test_df[target].values[:len(predictions)]

        if len(actuals) == 0 or len(predictions) == 0 or len(actuals) != len(predictions):
            print(f"    Could not make predictions or align actuals for {target} (new features) at {hospital}. Skipping MAE.")
            all_mae_scores_iter2.append({'hospital': hospital, 'target': target, 'mae': np.nan, 'params': best_params if 'best_params' in locals() else "Error in tuning"})
            continue

        mae = mean_absolute_error(actuals, predictions)
        print(f"    {hospital} - {target} - Tuned Test MAE (new features): {mae:.2f}")
        all_mae_scores_iter2.append({'hospital': hospital, 'target': target, 'mae': mae, 'params': best_params})

# Save results
with open('tuned_hyperparameters_iter2.json', 'w') as f:
    json.dump(tuned_hyperparameters_iter2, f, indent=4)
print("\nTuned hyperparameters (iter 2) saved to tuned_hyperparameters_iter2.json")

avg_mae_overall_iter2 = np.nanmean([score['mae'] for score in all_mae_scores_iter2 if 'mae' in score])
with open('tuned_mae_scores_iter2.txt', 'w') as f:
    f.write(f"Average MAE across all series after tuning (iter 2 with new features): {avg_mae_overall_iter2:.4f}\n\n")
    for score in all_mae_scores_iter2:
        f.write(f"Hospital: {score['hospital']}, Target: {score['target']}, MAE: {score.get('mae', 'N/A'):.4f}, Params: {score.get('params', 'N/A')}\n")
print(f"Average MAE after tuning (iter 2 with new features): {avg_mae_overall_iter2:.4f}")

print("Script finished: New Feature Engineering, Hyperparameter Tuning (Iter 2), and MAE Re-evaluation")
