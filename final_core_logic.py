import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error
import json
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday
from pandas.tseries.offsets import DateOffset
from dateutil.rrule import MO
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")
try:
    warnings.filterwarnings("ignore", category=lgb.LightGBMWarning)
except AttributeError:
    print("Warning: lgb.LightGBMWarning not found.")

# Global storage for category mappings
category_mappings = {}

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

def add_iteration2_features(df, calendar_obj):
    global category_mappings # Use the global mapping
    print("Adding Iteration 2 features with robust categorical encoding...")
    df_eng = df.copy()

    # Hour_DayOfWeek
    df_eng['Hour_DayOfWeek_str'] = df_eng['Hour'].astype(str) + "_" + df_eng['DayOfWeek'].astype(str)
    # Fit dtype only once based on all unique string values in the current dataset
    if 'Hour_DayOfWeek' not in category_mappings:
        # Sort categories to ensure consistent code assignment if data order changes slightly in future re-runs (though unique() doesn't guarantee order)
        # A more robust way is to sort the unique values before creating dtype.
        unique_cats = sorted(df_eng['Hour_DayOfWeek_str'].unique())
        category_mappings['Hour_DayOfWeek'] = pd.CategoricalDtype(categories=unique_cats, ordered=False)
    df_eng['Hour_DayOfWeek'] = df_eng['Hour_DayOfWeek_str'].astype(category_mappings['Hour_DayOfWeek']).cat.codes

    # Hospital_Code_Hour
    df_eng['Hospital_Code_Hour_str'] = df_eng['Hospital_Code'].astype(str) + "_" + df_eng['Hour'].astype(str)
    if 'Hospital_Code_Hour' not in category_mappings:
        unique_cats_hc_hr = sorted(df_eng['Hospital_Code_Hour_str'].unique())
        category_mappings['Hospital_Code_Hour'] = pd.CategoricalDtype(categories=unique_cats_hc_hr, ordered=False)
    df_eng['Hospital_Code_Hour'] = df_eng['Hospital_Code_Hour_str'].astype(category_mappings['Hospital_Code_Hour']).cat.codes

    if 'Capacity' in df_eng.columns and 'Hour' in df_eng.columns:
        df_eng['Capacity_Hour_Interaction'] = df_eng['Capacity'] * df_eng['Hour']

    actual_holidays = []
    if not df_eng.empty:
        min_date = df_eng['Datetime'].min(); max_date = df_eng['Datetime'].max()
        try:
            actual_holidays = calendar_obj.holidays(start=min_date - timedelta(days=45), end=max_date + timedelta(days=45)).normalize()
            actual_holidays = sorted(list(set(actual_holidays)))
        except Exception as e: print(f"Warning: Holiday generation failed for proximity features: {e}"); actual_holidays = []

    if not actual_holidays:
        df_eng['DaysToNextHoliday'] = 365; df_eng['DaysFromLastHoliday'] = 365
    else:
        df_eng['DaysToNextHoliday'] = df_eng['Datetime'].apply(lambda x: min([ (h - x.normalize()).days for h in actual_holidays if h >= x.normalize() ], default=365))
        df_eng['DaysFromLastHoliday'] = df_eng['Datetime'].apply(lambda x: min([ (x.normalize() - h).days for h in actual_holidays if h <= x.normalize() ], default=365))
    df_eng['DaysToNextHoliday'] = df_eng['DaysToNextHoliday'].fillna(365)
    df_eng['DaysFromLastHoliday'] = df_eng['DaysFromLastHoliday'].fillna(365)

    print(f"Shape after Iteration 2 features: {df_eng.shape}")
    # Drop intermediate string columns if no longer needed
    df_eng = df_eng.drop(columns=['Hour_DayOfWeek_str', 'Hospital_Code_Hour_str'], errors='ignore')
    return df_eng

def create_future_dates(last_date, hospital, hospital_code, current_capacity_value, days, calendar_obj, existing_df_for_holiday_range):
    global category_mappings # Use the global mapping
    future_dates_list = []
    times = ['08:00', '14:00', '20:00']
    actual_holidays = []
    if not existing_df_for_holiday_range.empty:
        min_hist_date = existing_df_for_holiday_range['Datetime'].min(); max_hist_date = existing_df_for_holiday_range['Datetime'].max()
        try:
            actual_holidays = calendar_obj.holidays(start=min_hist_date - timedelta(days=45), end=max_hist_date + timedelta(days=days + 45)).normalize()
            actual_holidays = sorted(list(set(actual_holidays)))
        except Exception as e: print(f"Warning: Holiday generation for future dates failed: {e}"); actual_holidays = []

    for day_offset in range(1, days + 1):
        future_date_val = pd.to_datetime(last_date) + timedelta(days=day_offset)
        for time_str in times:
            fut_dt = pd.to_datetime(f"{future_date_val.date()} {time_str}")
            h = fut_dt.hour; dow = fut_dt.dayofweek; dom = fut_dt.day; m = fut_dt.month; q = (m-1)//3+1; woy = int(fut_dt.isocalendar().week)
            is_hol = 0
            if actual_holidays: is_hol = int(fut_dt.normalize() in actual_holidays)

            # Categorical interaction features using stored mappings
            hour_dayofweek_str_future = f"{h}_{dow}"
            hour_dayofweek_code = pd.Series([hour_dayofweek_str_future]).astype(category_mappings.get('Hour_DayOfWeek', 'category')).cat.codes[0]

            hospital_code_hour_str_future = f"{hospital_code}_{h}"
            hospital_code_hour_code = pd.Series([hospital_code_hour_str_future]).astype(category_mappings.get('Hospital_Code_Hour', 'category')).cat.codes[0]

            cap_hour_inter = current_capacity_value * h
            days_to_next = 365; days_from_last = 365
            if actual_holidays:
                days_to_next = min([(hol_date - fut_dt.normalize()).days for hol_date in actual_holidays if hol_date >= fut_dt.normalize()], default=365)
                days_from_last = min([(fut_dt.normalize() - hol_date).days for hol_date in actual_holidays if hol_date <= fut_dt.normalize()], default=365)

            future_dates_list.append({
                'Datetime': fut_dt, 'Hospital': hospital, 'Hour': h, 'DayOfWeek': dow, 'DayOfMonth': dom, 'Month': m, 'Quarter': q, 'WeekOfYear': woy,
                'IsWeekend': int(dow in [5,6]), 'IsMonday': int(dow==0), 'IsFriday': int(dow==4),
                'Hour_sin': np.sin(2*np.pi*h/24), 'Hour_cos': np.cos(2*np.pi*h/24), 'Day_sin': np.sin(2*np.pi*dow/7), 'Day_cos': np.cos(2*np.pi*dow/7),
                'Month_sin': np.sin(2*np.pi*m/12), 'Month_cos': np.cos(2*np.pi*m/12), 'IsHoliday': is_hol,
                'IsSummer': int(m in [6,7,8]), 'IsWinter': int(m in [12,1,2]), 'IsPeakHour': int(h==20), 'IsLowHour': int(h==8),
                'Hospital_Code': hospital_code, 'Capacity': current_capacity_value,
                'Hour_DayOfWeek': hour_dayofweek_code, 'Hospital_Code_Hour': hospital_code_hour_code,
                'Capacity_Hour_Interaction': cap_hour_inter, 'DaysToNextHoliday': days_to_next, 'DaysFromLastHoliday': days_from_last
            })
    return pd.DataFrame(future_dates_list)

def forecast_with_lags(model, historical_data_df, future_df, features_for_model, target_column_name):
    temp_target_history = historical_data_df[target_column_name].tolist()
    predictions = []
    for i in range(len(future_df)):
        current_feature_values_map = {}
        current_future_row = future_df.iloc[i]
        for f_name in features_for_model:
            if not (f_name.startswith(f'Lag_{target_column_name}_') or f_name.startswith(f'Rolling_Mean_')):
                if f_name in current_future_row: current_feature_values_map[f_name] = current_future_row[f_name]
                else: current_feature_values_map[f_name] = 0
            else:
                for lag_order in range(1, 8):
                    if f_name == f'Lag_{target_column_name}_{lag_order}':
                        if lag_order <= len(temp_target_history): current_feature_values_map[f_name] = temp_target_history[-lag_order]
                        else: current_feature_values_map[f_name] = 0
                if f_name == f'Rolling_Mean_3_{target_column_name}':
                    if len(temp_target_history) >= 3: current_feature_values_map[f_name] = np.mean(temp_target_history[-3:])
                    elif len(temp_target_history) > 0: current_feature_values_map[f_name] = np.mean(temp_target_history)
                    else: current_feature_values_map[f_name] = 0
                if f_name == f'Rolling_Mean_7_{target_column_name}':
                    if len(temp_target_history) >= 7: current_feature_values_map[f_name] = np.mean(temp_target_history[-7:])
                    elif len(temp_target_history) > 0: current_feature_values_map[f_name] = np.mean(temp_target_history)
                    else: current_feature_values_map[f_name] = 0
        ordered_feature_values = [current_feature_values_map.get(f_name, 0) for f_name in features_for_model]
        feature_vector = np.array(ordered_feature_values).reshape(1, -1)
        pred = model.predict(feature_vector)[0]
        pred = max(0, pred)
        predictions.append(pred)
        temp_target_history.append(pred)
    return predictions

def direct_run_mae_calculation():
    print("Running final_core_logic.py for MAE calculation (Iter2 Features & HPs with corrected encoding)...")
    forecast_days_direct = 7 ; final_mae_scores_direct = []
    try:
        df_initial_direct = pd.read_csv("processed_may_data.csv"); df_initial_direct['Datetime'] = pd.to_datetime(df_initial_direct['Datetime'])
        print(f"Direct run: Loaded initial data, shape {df_initial_direct.shape}")
        calendar_direct = IrishBankHolidays()
        # Call add_iteration2_features which now stores mappings globally
        df_featured_direct = add_iteration2_features(df_initial_direct, calendar_direct)
        print(f"Direct run: Applied Iter2 features, shape {df_featured_direct.shape}")

        hps_iter2_direct = {}
        try:
            with open('tuned_hyperparameters_iter2.json', 'r') as f_hp_direct: hps_iter2_direct = json.load(f_hp_direct)
            print("Direct run: Loaded tuned_hyperparameters_iter2.json")
        except Exception as e: print(f"ERROR (Direct run) loading tuned_hyperparameters_iter2.json: {e}. Using fallback HPs."); hps_iter2_direct = {hosp: {t: {'n_estimators': 400, 'learning_rate':0.05, 'max_depth':7, 'num_leaves':31, 'random_state': 42, 'verbose': -1, 'force_col_wise':True} for t in ['ED Beds', 'Trolleys', 'Capacity']} for hosp in df_featured_direct['Hospital'].unique()}

        selected_hospitals_direct = sorted(df_featured_direct['Hospital'].unique())
        base_original_features_direct = ['Hour', 'DayOfWeek', 'DayOfMonth', 'Month', 'Quarter', 'WeekOfYear', 'IsWeekend', 'IsMonday', 'IsFriday', 'Hour_sin', 'Hour_cos', 'Day_sin', 'Day_cos', 'Month_sin', 'Month_cos', 'IsHoliday', 'IsSummer', 'IsWinter', 'IsPeakHour', 'IsLowHour', 'Hospital_Code']
        iter2_added_features_direct = ['Hour_DayOfWeek', 'Hospital_Code_Hour', 'DaysToNextHoliday', 'DaysFromLastHoliday']
        if 'Capacity_Hour_Interaction' in df_featured_direct.columns: iter2_added_features_direct.append('Capacity_Hour_Interaction')
        iter2_full_feature_set_direct = [f for f in (base_original_features_direct + iter2_added_features_direct) if f in df_featured_direct.columns]

        for hospital_name_direct in selected_hospitals_direct:
            hospital_data_full_direct = df_featured_direct[df_featured_direct['Hospital'] == hospital_name_direct].copy().sort_values(by='Datetime').reset_index(drop=True)
            if hospital_data_full_direct.empty: print(f"No data for {hospital_name_direct}, skipping."); continue
            split_point_direct = len(hospital_data_full_direct) - (forecast_days_direct * 3)
            if split_point_direct <= 0: print(f"Insufficient data for {hospital_name_direct} for 7-day test split. Skipping."); continue
            train_df_direct = hospital_data_full_direct.iloc[:split_point_direct].copy(); test_df_direct = hospital_data_full_direct.iloc[split_point_direct:].copy()
            if train_df_direct.empty or test_df_direct.empty or len(train_df_direct) < 15: print(f"Insufficient train/test data for {hospital_name_direct} after split. Skipping."); continue
            hospital_code_direct = train_df_direct['Hospital_Code'].iloc[0]; mean_capacity_direct = train_df_direct['Capacity'].fillna(0).mean()

            for target_direct in ['ED Beds', 'Trolleys', 'Capacity']:
                print(f"  Direct Run - Processing {hospital_name_direct} - {target_direct}")
                model_features_current_direct = list(iter2_full_feature_set_direct)
                if target_direct != 'Capacity' and 'Capacity' in df_featured_direct.columns:
                    if 'Capacity' not in model_features_current_direct: model_features_current_direct.append('Capacity')
                elif target_direct == 'Capacity':
                    if 'Capacity' in model_features_current_direct: model_features_current_direct.remove('Capacity')
                target_specific_lags_direct = [c for c in df_featured_direct.columns if (c.startswith(f'Lag_{target_direct}_') or c.startswith(f'Rolling_Mean_')) and target_direct in c]
                final_model_features_direct = model_features_current_direct + target_specific_lags_direct
                final_model_features_direct = sorted(list(set(f for f in final_model_features_direct if f in train_df_direct.columns and f != target_direct)))
                X_train_direct = train_df_direct.dropna(subset=[target_direct] + final_model_features_direct)[final_model_features_direct]
                y_train_direct = train_df_direct.loc[X_train_direct.index, target_direct]
                if X_train_direct.empty or len(X_train_direct) < 5: print(f"    Skipping {target_direct} for {hospital_name_direct} due to insufficient data after NA drop."); continue

                hps_current_direct = hps_iter2_direct.get(hospital_name_direct, {}).get(target_direct, {})
                if not hps_current_direct or not isinstance(hps_current_direct, dict) :
                    print(f"Warning: Using default HPs for {hospital_name_direct} - {target_direct}")
                    hps_current_direct = {'n_estimators': 400, 'learning_rate':0.05, 'max_depth':7, 'num_leaves':31, 'random_state': 42, 'verbose': -1, 'force_col_wise':True}

                model_direct = lgb.LGBMRegressor(**hps_current_direct); model_direct.fit(X_train_direct, y_train_direct)
                last_train_dt_direct = train_df_direct['Datetime'].max()
                # Pass df_featured_direct for holiday range context
                future_df_test_direct = create_future_dates(last_train_dt_direct.date(), hospital_name_direct, hospital_code_direct, mean_capacity_direct, forecast_days_direct, calendar_direct, df_featured_direct)
                future_df_test_direct = future_df_test_direct[future_df_test_direct['Datetime'].isin(test_df_direct['Datetime'])].reset_index(drop=True)
                if future_df_test_direct.empty: print(f"    Warning: Future dates empty for {target_direct}. Skipping MAE."); continue

                preds_direct = forecast_with_lags(model_direct, train_df_direct, future_df_test_direct, final_model_features_direct, target_direct)
                actuals_direct = test_df_direct[test_df_direct['Datetime'].isin(future_df_test_direct['Datetime'])][target_direct].values
                if len(preds_direct) == len(actuals_direct) and len(actuals_direct) > 0:
                    mae_direct = mean_absolute_error(actuals_direct, preds_direct)
                    final_mae_scores_direct.append({'hospital': hospital_name_direct, 'target': target_direct, 'mae': mae_direct})
                    print(f"    MAE (Direct Run) for {hospital_name_direct} - {target_direct}: {mae_direct:.2f}")
                else: print(f"    Length mismatch (Direct Run) for {target_direct}. Preds: {len(preds_direct)}, Actuals: {len(actuals_direct)}")

        if final_mae_scores_direct:
            avg_mae_direct_run = np.nanmean([s['mae'] for s in final_mae_scores_direct if 'mae' in s])
            print(f"\nFinal Average MAE (Direct Run with Iter2 Feats & HPs, fixed encoding): {avg_mae_direct_run:.4f}")
            with open('final_dmc_mae_scores_iter5.txt', 'w') as f_mae_direct: # Save to new file iter5
                f_mae_direct.write(f"Final Average MAE (Iter2 Features & HPs, fixed encoding): {avg_mae_direct_run:.4f}\n\n")
                for score_info_direct in final_mae_scores_direct: f_mae_direct.write(f"Hospital: {score_info_direct['hospital']}, Target: {score_info_direct['target']}, MAE: {score_info_direct.get('mae', 'N/A'):.4f}\n")
            print("Final MAE scores (iter5) from direct run saved to `final_dmc_mae_scores_iter5.txt`")
        else: print("No MAE scores were calculated in direct run.")
    except Exception as e_direct: print(f"Error during direct run of final_core_logic.py: {e_direct}"); import traceback; print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    direct_run_mae_calculation()
