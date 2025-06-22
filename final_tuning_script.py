import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error
import json
from datetime import datetime, timedelta
import warnings
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday # Keep for consistency if any part uses it
from pandas.tseries.offsets import DateOffset # Keep for consistency
from dateutil.rrule import MO # Keep for consistency

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")
try:
    warnings.filterwarnings("ignore", category=lgb.LightGBMWarning)
except AttributeError:
    print("Warning: lgb.LightGBMWarning not found.")

print("Script started: Final Feature Refinement, Focused Tuning (Iter 3), and MAE Re-evaluation")

# --- IrishBankHolidays Class (from previous script, for consistency if needed by any helper) ---
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

# --- forecast_with_lags function (Adapted from previous tuning_script.py) ---
def forecast_with_lags(model, historical_data_df, future_df_features_only, features_for_model, target_column_name):
    historical_data_df = historical_data_df.sort_values(by='Datetime')
    temp_history_for_target = list(historical_data_df[target_column_name].values)
    predictions = []
    for i in range(len(future_df_features_only)):
        current_feature_values_map = {}
        current_future_row = future_df_features_only.iloc[i]
        for feature_name in features_for_model:
            # Handle non-lag/rolling features
            if not (feature_name.startswith(f'Lag_{target_column_name}_') or \
                    feature_name.startswith(f'Rolling_Mean_3_{target_column_name}') or \
                    feature_name.startswith(f'Rolling_Mean_7_{target_column_name}') or \
                    feature_name.startswith(f'Rolling_Std_7_{target_column_name}') or \
                    feature_name.startswith(f'Lag1_Diff_{target_column_name}') or \
                    feature_name.startswith(f'Lag_1_minus_7_{target_column_name}')):
                if feature_name in current_future_row:
                    current_feature_values_map[feature_name] = current_future_row[feature_name]
                else:
                    current_feature_values_map[feature_name] = 0 # Default for missing base/interaction features

            # Handle dynamic features (lags, rolling means, new diffs, new stds for target)
            for lag_order in range(1, 8): # Max lag 7
                if feature_name == f'Lag_{target_column_name}_{lag_order}':
                    if lag_order <= len(temp_history_for_target):
                        current_feature_values_map[feature_name] = temp_history_for_target[-lag_order]
                    else:
                        current_feature_values_map[feature_name] = 0

            if feature_name == f'Rolling_Mean_3_{target_column_name}':
                if len(temp_history_for_target) >= 3: current_feature_values_map[feature_name] = np.mean(temp_history_for_target[-3:])
                elif len(temp_history_for_target) > 0: current_feature_values_map[feature_name] = np.mean(temp_history_for_target)
                else: current_feature_values_map[feature_name] = 0

            if feature_name == f'Rolling_Mean_7_{target_column_name}':
                if len(temp_history_for_target) >= 7: current_feature_values_map[feature_name] = np.mean(temp_history_for_target[-7:])
                elif len(temp_history_for_target) > 0: current_feature_values_map[feature_name] = np.mean(temp_history_for_target)
                else: current_feature_values_map[feature_name] = 0

            # New dynamic features for ED Beds (or other targets if generalized)
            if feature_name == f'Rolling_Std_7_{target_column_name}':
                if len(temp_history_for_target) >= 7: current_feature_values_map[feature_name] = np.std(temp_history_for_target[-7:])
                elif len(temp_history_for_target) > 1: current_feature_values_map[feature_name] = np.std(temp_history_for_target)
                else: current_feature_values_map[feature_name] = 0

            if feature_name == f'Lag1_Diff_{target_column_name}': # Lag_1 - Lag_2
                if len(temp_history_for_target) >= 2: current_feature_values_map[feature_name] = temp_history_for_target[-1] - temp_history_for_target[-2]
                else: current_feature_values_map[feature_name] = 0

            if feature_name == f'Lag_1_minus_7_{target_column_name}': # Lag_1 - Lag_7
                if len(temp_history_for_target) >= 7: current_feature_values_map[feature_name] = temp_history_for_target[-1] - temp_history_for_target[-7]
                else: current_feature_values_map[feature_name] = 0

        ordered_feature_values = [current_feature_values_map.get(f_name, 0) for f_name in features_for_model]
        feature_vector = np.array(ordered_feature_values).reshape(1, -1)
        pred = model.predict(feature_vector)[0]
        pred = max(0, pred)
        predictions.append(pred)
        temp_history_for_target.append(pred)
    return predictions

# --- Load Data ---
try:
    df_full = pd.read_csv('processed_may_data.csv') # Load original preprocessed data
    df_full['Datetime'] = pd.to_datetime(df_full['Datetime'])
    print(f"Base data loaded. Shape: {df_full.shape}")

    # Load features from iter2 (these were added to df in feature_engineering_tuning_script.py)
    # For this script, we assume processed_may_data.csv is the one *after* iter2's feature engineering.
    # If not, we'd need to re-create them. The prompt implies loading the output of iter2.
    # Let's assume 'processed_may_data.csv' is the one from step 1, and we re-apply iter2 features + new ones.

    # Re-apply Iter2 features for self-containment and clarity
    print("Re-applying Iteration 2 features...")
    df_full['Hour_DayOfWeek'] = df_full['Hour'].astype(str) + "_" + df_full['DayOfWeek'].astype(str)
    df_full['Hour_DayOfWeek'] = df_full['Hour_DayOfWeek'].astype('category').cat.codes
    df_full['Hospital_Code_Hour'] = df_full['Hospital_Code'].astype(str) + "_" + df_full['Hour'].astype(str)
    df_full['Hospital_Code_Hour'] = df_full['Hospital_Code_Hour'].astype('category').cat.codes
    if 'Capacity' in df_full.columns and 'Hour' in df_full.columns:
        df_full['Capacity_Hour_Interaction'] = df_full['Capacity'] * df_full['Hour']

    actual_holidays = []
    if not df_full.empty:
        calendar = IrishBankHolidays()
        min_date = df_full['Datetime'].min(); max_date = df_full['Datetime'].max()
        actual_holidays = calendar.holidays(start=min_date - timedelta(days=60), end=max_date + timedelta(days=60)).normalize()
        actual_holidays = sorted(list(set(actual_holidays)))
    if not actual_holidays:
        df_full['DaysToNextHoliday'] = 365; df_full['DaysFromLastHoliday'] = 365
    else:
        df_full['DaysToNextHoliday'] = df_full['Datetime'].apply(lambda x: min([ (h - x.normalize()).days for h in actual_holidays if h >= x.normalize() ], default=365) )
        df_full['DaysFromLastHoliday'] = df_full['Datetime'].apply(lambda x: min([ (x.normalize() - h).days for h in actual_holidays if h <= x.normalize() ], default=365) )
        df_full['DaysToNextHoliday'] = df_full['DaysToNextHoliday'].fillna(365)
        df_full['DaysFromLastHoliday'] = df_full['DaysFromLastHoliday'].fillna(365)
    print(f"Data shape after re-applying Iter2 features: {df_full.shape}")

except FileNotFoundError:
    print("Error: processed_may_data.csv not found. Exiting.")
    exit()


# --- Targeted Feature Engineering for 'ED Beds' (Iter 3) ---
print("Starting Targeted Feature Engineering (Iter 3) for ED Beds...")
df_list_iter3 = []
for hospital_code in df_full['Hospital_Code'].unique():
    hospital_data = df_full[df_full['Hospital_Code'] == hospital_code].copy()
    hospital_data = hospital_data.sort_values(by='Datetime')

    # Ensure lag features exist (they should from initial processing)
    if 'Lag_ED Beds_1' in hospital_data.columns and 'Lag_ED Beds_2' in hospital_data.columns:
        hospital_data[f'Lag1_Diff_ED_Beds'] = hospital_data['Lag_ED Beds_1'] - hospital_data['Lag_ED Beds_2']
    else:
        hospital_data[f'Lag1_Diff_ED_Beds'] = 0
        print(f"Warning: Lag_ED Beds_1 or _2 missing for hospital {hospital_code}, Lag1_Diff_ED_Beds set to 0.")

    if 'Lag_ED Beds_1' in hospital_data.columns and 'Lag_ED Beds_7' in hospital_data.columns:
        hospital_data[f'Lag_1_minus_7_ED_Beds'] = hospital_data['Lag_ED Beds_1'] - hospital_data['Lag_ED Beds_7']
    else:
        hospital_data[f'Lag_1_minus_7_ED_Beds'] = 0
        print(f"Warning: Lag_ED Beds_1 or _7 missing for hospital {hospital_code}, Lag_1_minus_7_ED_Beds set to 0.")

    hospital_data[f'Rolling_Std_7_ED_Beds'] = hospital_data['ED Beds'].rolling(window=7, min_periods=1).std()

    # Fill NaNs that might arise from rolling/diff operations, especially at the start of each group
    for col in [f'Lag1_Diff_ED_Beds', f'Lag_1_minus_7_ED_Beds', f'Rolling_Std_7_ED_Beds']:
        hospital_data[col] = hospital_data[col].fillna(method='bfill').fillna(method='ffill').fillna(0)

    df_list_iter3.append(hospital_data)

if df_list_iter3:
    df = pd.concat(df_list_iter3).reset_index(drop=True)
    print(f"Data shape after Iter3 features: {df.shape}")
else:
    print("Error: No data processed after hospital grouping for Iter3 features. Exiting.")
    exit()

# --- Main Script Logic ---
hospitals = df['Hospital'].unique()
targets = ['ED Beds', 'Trolleys', 'Capacity']

# Base features from iter2 + new ED Bed features
base_features_iter3 = [
    'Hour_sin', 'Hour_cos', 'Day_sin', 'Day_cos', 'Month_sin', 'Month_cos',
    'IsWeekend', 'IsMonday', 'IsFriday', 'IsHoliday',
    'IsSummer', 'IsWinter', 'IsPeakHour', 'IsLowHour', 'Hospital_Code',
    'Hour', 'DayOfWeek', 'DayOfMonth', 'Month', 'Quarter', 'WeekOfYear',
    'Hour_DayOfWeek', 'Hospital_Code_Hour', 'DaysToNextHoliday', 'DaysFromLastHoliday',
    'Lag1_Diff_ED_Beds', 'Lag_1_minus_7_ED_Beds', 'Rolling_Std_7_ED_Beds' # New ED features
]
if 'Capacity_Hour_Interaction' in df.columns: base_features_iter3.append('Capacity_Hour_Interaction')
base_features_iter3 = [f for f in base_features_iter3 if f in df.columns]


# Load best params from iter2 as a starting point for grid definitions
try:
    with open('tuned_hyperparameters_iter2.json', 'r') as f:
        prev_best_params_all_models = json.load(f)
except FileNotFoundError:
    print("Warning: tuned_hyperparameters_iter2.json not found. Using a default broad grid.")
    prev_best_params_all_models = {}

all_mae_scores_iter3 = []
tuned_hyperparameters_iter3 = {}
forecast_days = 7

for hospital in hospitals:
    hospital_df = df[df['Hospital'] == hospital].sort_values(by='Datetime').reset_index(drop=True)
    if hospital_df.empty: continue
    tuned_hyperparameters_iter3[hospital] = {}

    if len(hospital_df) < (forecast_days * 3 * 2):
         print(f"Skipping {hospital} due to overall insufficient data: {len(hospital_df)} records")
         continue
    last_datetime = hospital_df['Datetime'].max()
    split_datetime_test = last_datetime - timedelta(days=forecast_days -1)
    dev_df = hospital_df[hospital_df['Datetime'] < split_datetime_test].copy()
    test_df = hospital_df[hospital_df['Datetime'] >= split_datetime_test].copy()

    if dev_df.empty or len(dev_df) < 20 or test_df.empty or len(test_df) < forecast_days*3*0.9:
        print(f"Skipping {hospital} due to insufficient dev/test data after split.")
        continue

    print(f"\nProcessing Hospital: {hospital}, Dev records: {len(dev_df)}, Test records: {len(test_df)}")
    for target in targets:
        print(f"  Tuning for Target: {target} (Iter 3)")

        prev_params_hospital_target = prev_best_params_all_models.get(hospital, {}).get(target, {})
        if not isinstance(prev_params_hospital_target, dict): prev_params_hospital_target = {} # Handle "Skipped" case

        # Define focused param grid
        param_grid_iter3 = {
            'n_estimators': [prev_params_hospital_target.get('n_estimators', 400),
                             prev_params_hospital_target.get('n_estimators', 400) + 200, 1000], # [400, 600, 800, 1000]
            'learning_rate': [0.01, 0.03, prev_params_hospital_target.get('learning_rate', 0.05)], # [0.01, 0.03, 0.05]
            'max_depth': [prev_params_hospital_target.get('max_depth', 7)], # Keep from iter2 or expand slightly: [5, 7, 9]
            'num_leaves': [prev_params_hospital_target.get('num_leaves', 50) -10 if prev_params_hospital_target.get('num_leaves', 50) > 20 else 20 ,
                           prev_params_hospital_target.get('num_leaves', 50),
                           prev_params_hospital_target.get('num_leaves', 50) +10 ], # [prev-10, prev, prev+10]
            'subsample': [0.8], 'colsample_bytree': [0.8],
            'reg_alpha': [0.1], 'reg_lambda': [0.1],
            'random_state': [42], 'force_col_wise': [True], 'verbose': [-1]
        }
        # Ensure num_leaves is reasonable
        param_grid_iter3['num_leaves'] = [max(2, nl) for nl in param_grid_iter3['num_leaves']] # min 2 leaves
        param_grid_iter3['n_estimators'] = sorted(list(set(param_grid_iter3['n_estimators']))) # Unique sorted values
        param_grid_iter3['learning_rate'] = sorted(list(set(param_grid_iter3['learning_rate'])))
        param_grid_iter3['num_leaves'] = sorted(list(set(param_grid_iter3['num_leaves'])))


        current_model_base_features = list(base_features_iter3)
        if target in ['ED Beds', 'Trolleys']:
            if 'Capacity' in df.columns and 'Capacity' not in current_model_base_features:
                current_model_base_features.append('Capacity')
        elif target == 'Capacity':
            if 'Capacity' in current_model_base_features: current_model_base_features.remove('Capacity')

        target_specific_lag_roll_features = [col for col in df.columns if (col.startswith(f'Lag_{target}_') or \
            col.startswith(f'Rolling_Mean_3_{target}') or col.startswith(f'Rolling_Mean_7_{target}')) and target in col]

        # For ED Beds, include its new specific features. For other targets, exclude them.
        ed_beds_specific_iter3_features = ['Lag1_Diff_ED_Beds', 'Lag_1_minus_7_ED_Beds', 'Rolling_Std_7_ED_Beds']

        final_features_for_model = current_model_base_features + target_specific_lag_roll_features
        if target == 'ED Beds':
            final_features_for_model.extend(ed_beds_specific_iter3_features)
        else: # Exclude ED-specific features if not predicting ED Beds
            final_features_for_model = [f for f in final_features_for_model if f not in ed_beds_specific_iter3_features]

        final_features_for_model = sorted(list(set(f for f in final_features_for_model if f in dev_df.columns and f != target)))

        dev_df_cleaned = dev_df.dropna(subset=[target] + final_features_for_model).copy()
        if len(dev_df_cleaned) < 15:
            print(f"    Skipping {target} for {hospital} (Iter 3) due to insufficient cleaned dev data: {len(dev_df_cleaned)}")
            tuned_hyperparameters_iter3[hospital][target] = "Skipped - Insufficient data"
            continue

        X_dev = dev_df_cleaned[final_features_for_model]
        y_dev = dev_df_cleaned[target]

        n_splits_cv = 2; cv_test_size = forecast_days * 3
        if len(X_dev) <= cv_test_size *2 : cv_test_size = max(1, len(X_dev)//3)
        tscv = TimeSeriesSplit(n_splits=n_splits_cv, test_size=cv_test_size)

        print(f"    GridSearchCV for {target} (Iter 3) with {n_splits_cv} splits, test_size={cv_test_size}...")
        gsearch = GridSearchCV(estimator=lgb.LGBMRegressor(), param_grid=param_grid_iter3, scoring='neg_mean_absolute_error', cv=tscv, verbose=0, n_jobs=1)

        try:
            gsearch.fit(X_dev, y_dev)
            best_params = gsearch.best_params_
            tuned_hyperparameters_iter3[hospital][target] = best_params
            print(f"    Best params for {target} (Iter 3): {best_params}")
        except Exception as e:
            print(f"    Error during GridSearchCV for {target} (Iter 3) at {hospital}: {e}")
            tuned_hyperparameters_iter3[hospital][target] = f"Error: {str(e)}"
            continue

        model_final = lgb.LGBMRegressor(**best_params)
        model_final.fit(X_dev, y_dev)

        X_test_for_predictions = test_df[final_features_for_model].copy()
        predictions = forecast_with_lags(model_final, dev_df_cleaned, X_test_for_predictions, final_features_for_model, target)
        actuals = test_df[target].values[:len(predictions)]

        if len(actuals) == 0 or len(predictions) == 0 or len(actuals) != len(predictions):
            print(f"    Could not make predictions or align actuals for {target} (Iter 3) at {hospital}. Skipping MAE.")
            all_mae_scores_iter3.append({'hospital': hospital, 'target': target, 'mae': np.nan, 'params': best_params if 'best_params' in locals() else "Error"})
            continue

        mae = mean_absolute_error(actuals, predictions)
        print(f"    {hospital} - {target} - Tuned Test MAE (Iter 3): {mae:.2f}")
        all_mae_scores_iter3.append({'hospital': hospital, 'target': target, 'mae': mae, 'params': best_params})

# Save results
with open('tuned_hyperparameters_iter3.json', 'w') as f:
    json.dump(tuned_hyperparameters_iter3, f, indent=4)
print("\nTuned hyperparameters (Iter 3) saved to tuned_hyperparameters_iter3.json")

avg_mae_overall_iter3 = np.nanmean([score['mae'] for score in all_mae_scores_iter3 if 'mae' in score])
with open('tuned_mae_scores_iter3.txt', 'w') as f:
    f.write(f"Average MAE across all series after tuning (Iter 3 with new ED Bed features): {avg_mae_overall_iter3:.4f}\n\n")
    for score in all_mae_scores_iter3:
        f.write(f"Hospital: {score['hospital']}, Target: {score['target']}, MAE: {score.get('mae', 'N/A'):.4f}, Params: {score.get('params', 'N/A')}\n")
print(f"Average MAE after tuning (Iter 3): {avg_mae_overall_iter3:.4f}")

print("Script finished: Final Feature Refinement, Focused Tuning (Iter 3), and MAE Re-evaluation")
