import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error
import json
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday
from pandas.tseries.offsets import DateOffset
from dateutil.rrule import MO

# --- Define Irish Bank Holidays (Copied from DMC.py) ---
# This is needed if create_future_dates uses it.
class IrishBankHolidays(AbstractHolidayCalendar):
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=nearest_workday),
        Holiday("St. Brigid's Day", month=2, day=1, observance=nearest_workday),
        Holiday("St. Patrick's Day", month=3, day=17, observance=nearest_workday),
        Holiday("May Day", month=5, day=1, offset=DateOffset(weekday=MO(1))),
        Holiday("June Bank Holiday", month=6, day=1, offset=DateOffset(weekday=MO(1))),
        Holiday("August Bank Holiday", month=8, day=1, offset=DateOffset(weekday=MO(1))),
        Holiday("October Bank Holiday", month=10, day=1, offset=DateOffset(weekday=MO(-1))),
        Holiday("Christmas Day", month=12, day=25),
        Holiday("St. Stephen's Day", month=12, day=26),
    ]

# --- create_future_dates function (Copied from DMC.py) ---
def create_future_dates(last_date, hospital, hospital_code, additional_capacity, days=7):
    future_dates_list = []
    times = ['08:00', '14:00', '20:00']
    for day_offset in range(1, days + 1):
        future_date_val = last_date + timedelta(days=day_offset)
        for time_str in times:
            future_datetime = pd.to_datetime(f"{future_date_val.date()} {time_str}")
            hour = future_datetime.hour
            day_of_week = future_datetime.dayofweek
            day_of_month = future_datetime.day
            month = future_datetime.month
            quarter = (month - 1) // 3 + 1
            week_of_year = int(future_datetime.isocalendar().week)
            is_weekend = int(day_of_week in [5, 6])
            is_monday = int(day_of_week == 0)
            is_friday = int(day_of_week == 4)
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            day_sin = np.sin(2 * np.pi * day_of_week / 7)
            day_cos = np.cos(2 * np.pi * day_of_week / 7)
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)
            is_holiday = 0
            try:
                calendar = IrishBankHolidays()
                check_date = future_datetime.normalize()
                generated_holidays = calendar.holidays(start=check_date, end=check_date)
                is_holiday = int(check_date in generated_holidays)
            except Exception:
                pass
            is_summer = int(month in [6, 7, 8])
            is_winter = int(month in [12, 1, 2])
            is_peak_hour = int(hour == 20)
            is_low_hour = int(hour == 8)
            future_dates_list.append({
                'Date': future_datetime.date(),
                'Time': time_str,
                'Datetime': future_datetime,
                'Hospital': hospital, 'Hour': hour, 'DayOfWeek': day_of_week,
                'DayOfMonth': day_of_month, 'Month': month, 'Quarter': quarter,
                'WeekOfYear': week_of_year, 'IsWeekend': is_weekend, 'IsMonday': is_monday,
                'IsFriday': is_friday, 'Hour_sin': hour_sin, 'Hour_cos': hour_cos,
                'Day_sin': day_sin, 'Day_cos': day_cos, 'Month_sin': month_sin, 'Month_cos': month_cos,
                'IsHoliday': is_holiday, 'IsSummer': is_summer, 'IsWinter': is_winter,
                'IsPeakHour': is_peak_hour, 'IsLowHour': is_low_hour,
                'Hospital_Code': hospital_code, 'Capacity': additional_capacity
            })
    return pd.DataFrame(future_dates_list)

# --- forecast_with_lags function (Copied from DMC.py) ---
def forecast_with_lags(model, historical_data, future_df_features_only, features, target_column):
    predictions = []
    temp_history = historical_data[target_column].tolist()

    for i in range(len(future_df_features_only)):
        feature_values_for_step = {}
        current_row_features = future_df_features_only.iloc[i]

        for f_name in features:
            if not (f_name.startswith(f'Lag_{target_column}_') or f_name.startswith(f'Rolling_Mean_')):
                if f_name in current_row_features:
                    feature_values_for_step[f_name] = current_row_features[f_name]
                else:
                    feature_values_for_step[f_name] = 0 # Should ideally not happen

        for lag_num in range(1, 8): # Max lag 7
            lag_feature_name = f'Lag_{target_column}_{lag_num}'
            if lag_feature_name in features:
                if lag_num <= len(temp_history):
                    feature_values_for_step[lag_feature_name] = temp_history[-lag_num]
                else:
                    feature_values_for_step[lag_feature_name] = 0

        roll_mean_3_name = f'Rolling_Mean_3_{target_column}'
        if roll_mean_3_name in features:
            if len(temp_history) >= 3:
                feature_values_for_step[roll_mean_3_name] = np.mean(temp_history[-3:])
            elif len(temp_history) > 0:
                feature_values_for_step[roll_mean_3_name] = np.mean(temp_history)
            else:
                feature_values_for_step[roll_mean_3_name] = 0

        roll_mean_7_name = f'Rolling_Mean_7_{target_column}'
        if roll_mean_7_name in features:
            if len(temp_history) >= 7:
                feature_values_for_step[roll_mean_7_name] = np.mean(temp_history[-7:])
            elif len(temp_history) > 0:
                feature_values_for_step[roll_mean_7_name] = np.mean(temp_history)
            else:
                feature_values_for_step[roll_mean_7_name] = 0

        ordered_feature_values = [feature_values_for_step[f_name] for f_name in features]
        feature_vector = np.array(ordered_feature_values).reshape(1, -1)
        pred = model.predict(feature_vector)[0]
        pred = max(0, pred)
        predictions.append(pred)
        temp_history.append(pred)
    return predictions

def run_core_logic():
    print("Starting core forecasting logic...")
    forecast_days = 7
    hyperparameters_to_save = {}
    all_mae_scores = []

    try:
        df_processed = pd.read_csv("processed_may_data.csv")
        df_processed['Datetime'] = pd.to_datetime(df_processed['Datetime'])
        print(f"Data loaded: {len(df_processed)} records.")
    except FileNotFoundError:
        print("Error: `processed_may_data.csv` not found.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    hospitals = sorted(df_processed['Hospital'].unique())
    selected_hospitals = hospitals # Process all hospitals

    all_cols = df_processed.columns.tolist()
    potential_features = [
        col for col in all_cols if col not in [
            'Hospital Group Name', 'Hospital', 'Date', 'DayGAR', 'Time', 'Datetime',
            'ED Beds', 'Trolleys', 'Capacity'
        ]
    ]

    for hospital in selected_hospitals:
        print(f"\nProcessing Hospital: {hospital}")
        hospital_data_full = df_processed[df_processed['Hospital'] == hospital].copy()
        hospital_data_full = hospital_data_full.sort_values(by='Datetime').reset_index(drop=True)

        if hospital_data_full.empty:
            print(f"No data for {hospital}.")
            continue

        split_point = len(hospital_data_full) - (forecast_days * 3)
        if split_point <= 0:
            print(f"Insufficient data for {hospital} for 7-day test set ({len(hospital_data_full)} records). Skipping.")
            continue

        train_df = hospital_data_full.iloc[:split_point].copy()
        test_df = hospital_data_full.iloc[split_point:].copy()
        print(f"Train records: {len(train_df)}, Test records: {len(test_df)}")

        for target_col_name in ['ED Beds', 'Trolleys', 'Capacity']:
            print(f"  Target: {target_col_name}")
            if train_df[target_col_name].isnull().all() or len(train_df) < 10:
                print(f"  Insufficient training data for {target_col_name} ({len(train_df)} records or all null). Skipping.")
                continue

            current_target_lag_roll_features = [
                col for col in potential_features if (col.startswith(f'Lag_{target_col_name}_') or col.startswith(f'Rolling_Mean_')) and target_col_name in col
            ]

            defined_base_features = [
                'Hour', 'DayOfWeek', 'DayOfMonth', 'Month', 'Quarter', 'WeekOfYear',
                'IsWeekend', 'IsMonday', 'IsFriday', 'Hour_sin', 'Hour_cos', 'Day_sin', 'Day_cos',
                'Month_sin', 'Month_cos', 'IsHoliday', 'IsSummer', 'IsWinter',
                'IsPeakHour', 'IsLowHour', 'Hospital_Code'
            ]
            model_base_features = defined_base_features[:]
            if target_col_name != 'Capacity' and 'Capacity' in df_processed.columns:
                if 'Capacity' not in model_base_features:
                     model_base_features.append('Capacity') # This is the general capacity feature
            elif target_col_name == 'Capacity' and 'Capacity' in model_base_features:
                model_base_features.remove('Capacity') # Remove if 'Capacity' is the target

            all_features_for_model = model_base_features + current_target_lag_roll_features
            available_features = [f for f in all_features_for_model if f in train_df.columns and f != target_col_name]

            X_train = train_df[available_features].dropna()
            y_train = train_df.loc[X_train.index, target_col_name]

            if len(X_train) < 5:
                print(f"  Insufficient data after feature selection for {target_col_name} ({len(X_train)} records). Skipping.")
                continue

            lgbm_params = {
                'n_estimators': min(600, len(X_train) * 2), 'learning_rate': 0.05,
                'max_depth': min(5, len(available_features) + 1 if len(available_features) > 0 else 2), # Ensure max_depth > 0
                'num_leaves': min(50, 2 ** min(10, (len(available_features) + 1 if len(available_features) > 0 else 2)) - 1), # Ensure num_leaves is reasonable
                'subsample': 0.8, 'colsample_bytree': 0.8,
                'reg_alpha': 0.15, 'reg_lambda': 0.15,
                'verbose': -1, 'random_state': 42, 'force_col_wise': True
            }
            if lgbm_params['num_leaves'] < 2: lgbm_params['num_leaves'] = 2 # Minimum for LightGBM
            if lgbm_params['max_depth'] < 1: lgbm_params['max_depth'] = 1


            if not hyperparameters_to_save:
                hyperparameters_to_save = {
                    'description': 'LightGBM Regressor parameters from run_dmc_core.py',
                    'parameters': lgbm_params
                }

            model = lgb.LGBMRegressor(**lgbm_params)
            model.fit(X_train, y_train)

            # Prepare features for the test period.
            # `create_future_dates` can generate the non-lag/roll features for the test period.
            # Then `forecast_with_lags` will use train_df history to build lags for these future dates.

            last_train_date = train_df['Datetime'].max().date()
            hospital_code_val = train_df['Hospital_Code'].iloc[0]
            # Use mean capacity from TRAINING data for creating future features, if Capacity is not the target
            # If Capacity IS the target, this value is less critical for that specific model's feature generation
            # but create_future_dates expects it.
            mean_train_capacity = train_df['Capacity'].mean()


            future_df_test_period = create_future_dates(
                last_date=pd.to_datetime(last_train_date), # Ensure it's datetime obj
                hospital=hospital,
                hospital_code=hospital_code_val,
                additional_capacity=mean_train_capacity,
                days=forecast_days
            )

            # Align `future_df_test_period` with `test_df`'s actual Datetime stamps for direct comparison
            # And ensure it only contains features expected by `forecast_with_lags`
            future_df_test_period_aligned = future_df_test_period[future_df_test_period['Datetime'].isin(test_df['Datetime'])].reset_index(drop=True)

            # We need to ensure the features in 'future_df_test_period_aligned' (passed as future_df_features_only to forecast_with_lags)
            # are the same 'available_features' used for training, excluding lags/rolling of the target.
            # forecast_with_lags will build the lags/rolling features dynamically.
            X_test_features_for_forecast = future_df_test_period_aligned[model_base_features].copy() # Start with base features
             # If 'Capacity' was added to model_base_features and it's not the target, ensure it's there.
            if 'Capacity' in model_base_features and 'Capacity' not in X_test_features_for_forecast.columns and 'Capacity' in future_df_test_period_aligned.columns:
                 X_test_features_for_forecast['Capacity'] = future_df_test_period_aligned['Capacity']


            test_predictions = forecast_with_lags(model, train_df, X_test_features_for_forecast, available_features, target_col_name)
            y_test_actual = test_df[test_df['Datetime'].isin(future_df_test_period_aligned['Datetime'])][target_col_name].values

            if len(test_predictions) == len(y_test_actual) and len(y_test_actual) > 0 :
                mae = mean_absolute_error(y_test_actual, test_predictions)
                all_mae_scores.append(mae)
                print(f"  MAE for {target_col_name}: {mae:.2f}")
            elif len(y_test_actual) == 0:
                 print(f"  No actual values for MAE calculation for {target_col_name} (likely date alignment issue). Skipping MAE.")
            else:
                print(f"  Prediction/actual length mismatch for {target_col_name}. Pred: {len(test_predictions)}, Actual: {len(y_test_actual)}. Skipping MAE.")

    if all_mae_scores:
        avg_mae_all = np.mean(all_mae_scores)
        print(f"\nAverage MAE (All Processed Series): {avg_mae_all:.2f}")
    else:
        print("\nNo MAE scores were calculated.")

    if hyperparameters_to_save:
        try:
            with open("best_hyperparameters.json", "w") as f:
                json.dump(hyperparameters_to_save, f, indent=4)
            print("Hyperparameters saved to best_hyperparameters.json")
        except Exception as e:
            print(f"Error saving hyperparameters: {e}")
    else:
        print("No hyperparameters were set to be saved.")

if __name__ == "__main__":
    run_core_logic()

print("Script finished.")
