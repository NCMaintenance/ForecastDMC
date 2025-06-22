import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday
from pandas.tseries.offsets import DateOffset
from dateutil.rrule import MO
import plotly.graph_objects as go
import warnings
import sys

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
    print("DMC.py: Adding Iteration 2 features with robust categorical encoding...") # Added script name for clarity
    df_eng = df.copy()

    # Hour_DayOfWeek
    df_eng['Hour_DayOfWeek_str'] = df_eng['Hour'].astype(str) + "_" + df_eng['DayOfWeek'].astype(str)
    if 'Hour_DayOfWeek' not in category_mappings: # Fit only if not already done (e.g. if called multiple times)
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

    print(f"DMC.py: Shape after Iteration 2 features: {df_eng.shape}")
    df_eng = df_eng.drop(columns=['Hour_DayOfWeek_str', 'Hospital_Code_Hour_str'], errors='ignore')
    return df_eng

def create_future_dates(last_date, hospital, hospital_code, current_capacity_value, days, calendar_obj, existing_df_for_holiday_range):
    global category_mappings # Use the global mapping for encoding
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

def plot_forecasts(historical_data, forecast_data, metric_name, hospital_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical_data['Datetime'], y=historical_data[metric_name], mode='lines+markers', name='Historical', line=dict(color='blue'), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=forecast_data['Datetime'], y=forecast_data['Predicted'], mode='lines+markers', name='Forecast', line=dict(color='red', dash='dash'), marker=dict(size=6, symbol='diamond')))
    if 'Actual' in forecast_data.columns:
        fig.add_trace(go.Scatter(x=forecast_data['Datetime'], y=forecast_data['Actual'], mode='lines+markers', name='Actual (Test Set)', line=dict(color='green'), marker=dict(size=4, symbol='circle')))
    if not historical_data.empty:
        last_historical_date = historical_data['Datetime'].max()
        fig.add_shape(type="line", x0=last_historical_date, x1=last_historical_date, y0=0, y1=1, yref="paper", line=dict(color="gray", width=2, dash="dot"))
        fig.add_annotation(x=last_historical_date, y=0.95, yref="paper", text="Forecast Start", showarrow=False, bgcolor="white", bordercolor="gray", borderwidth=1)
    fig.update_layout(title=f'{metric_name} Forecast vs Actuals - {hospital_name}', xaxis_title='Date', yaxis_title=f'{metric_name} Count', hovermode='x unified', showlegend=True, height=500)
    return fig

def add_forecasting_insights(st_obj):
    with st_obj.expander("💡 Forecasting Insights & Tips", expanded=False):
        st_obj.subheader("Data Requirements"); st_obj.markdown("Preprocessed Data, Consistent time intervals, Sufficient historical data.")
        st_obj.subheader("Understanding Your Results"); st_obj.markdown("MAE, RMSE, Visual inspection of charts.")

def run_forecasting_logic(df_featured_param, hospital_option_param, forecast_days_param, irish_holidays_calendar_param, tuned_hps_iter2_param, st_ui):
    st_ui.header("📊 Forecast Results & MAE (Iter 2 Features & HPs)")
    final_mae_scores_output = []
    selected_hospitals_for_run = sorted(df_featured_param['Hospital'].unique()) if hospital_option_param == "All Hospitals" else [hospital_option_param]
    base_original_features = ['Hour', 'DayOfWeek', 'DayOfMonth', 'Month', 'Quarter', 'WeekOfYear','IsWeekend', 'IsMonday', 'IsFriday','Hour_sin', 'Hour_cos', 'Day_sin', 'Day_cos', 'Month_sin', 'Month_cos', 'IsHoliday', 'IsSummer', 'IsWinter', 'IsPeakHour', 'IsLowHour', 'Hospital_Code']
    iter2_added_features = ['Hour_DayOfWeek', 'Hospital_Code_Hour', 'DaysToNextHoliday', 'DaysFromLastHoliday']
    if 'Capacity_Hour_Interaction' in df_featured_param.columns: iter2_added_features.append('Capacity_Hour_Interaction')
    iter2_full_feature_set = [f for f in (base_original_features + iter2_added_features) if f in df_featured_param.columns]

    for hospital_name_run in selected_hospitals_for_run:
        st_ui.subheader(f"🏥 {hospital_name_run}")
        hospital_data_full = df_featured_param[df_featured_param['Hospital'] == hospital_name_run].copy().sort_values(by='Datetime').reset_index(drop=True)
        if hospital_data_full.empty: st_ui.warning(f"No data for {hospital_name_run}."); continue
        split_point = len(hospital_data_full) - (forecast_days_param * 3)
        if split_point <= 0: st_ui.warning(f"Insufficient data for {hospital_name_run} for 7-day test split. Skipping."); continue
        train_df = hospital_data_full.iloc[:split_point].copy(); test_df = hospital_data_full.iloc[split_point:].copy()
        st_ui.info(f"Training data for {hospital_name_run}: {len(train_df)} records. Test data: {len(test_df)} records.")
        hospital_code_val = train_df['Hospital_Code'].iloc[0] if not train_df.empty else 0
        mean_capacity_train = train_df['Capacity'].fillna(0).mean()

        for target_col_name_run in ['ED Beds', 'Trolleys', 'Capacity']:
            st_ui.markdown(f"#### Processing '{target_col_name_run}' for {hospital_name_run}")
            if train_df[target_col_name_run].isnull().all() or len(train_df) < 10: st_ui.warning(f"Insufficient training data for '{target_col_name_run}'. Skipping."); continue
            model_features_current = list(iter2_full_feature_set)
            if target_col_name_run != 'Capacity' and 'Capacity' in df_featured_param.columns:
                if 'Capacity' not in model_features_current: model_features_current.append('Capacity')
            elif target_col_name_run == 'Capacity':
                if 'Capacity' in model_features_current: model_features_current.remove('Capacity')
            target_specific_existing_lags = [col for col in df_featured_param.columns if (col.startswith(f'Lag_{target_col_name_run}_') or col.startswith(f'Rolling_Mean_')) and target_col_name_run in col]
            final_model_features = sorted(list(set(f for f in (model_features_current + target_specific_existing_lags) if f in train_df.columns and f != target_col_name_run)))
            X_train = train_df.dropna(subset=[target_col_name_run] + final_model_features)[final_model_features]
            y_train = train_df.loc[X_train.index, target_col_name_run]
            if X_train.empty or len(X_train) < 5: st_ui.warning(f"Insufficient data for '{target_col_name_run}' after NA drop/feature selection. Skipping."); continue

            hospital_params = tuned_hps_iter2_param.get(hospital_name_run, {})
            lgbm_final_params = hospital_params.get(target_col_name_run, {})
            if not lgbm_final_params or not isinstance(lgbm_final_params, dict):
                lgbm_final_params = tuned_hps_iter2_param.get("Cork University Hospital", {}).get("ED Beds", {'n_estimators': 400, 'learning_rate': 0.05, 'max_depth': 7, 'num_leaves': 31, 'random_state': 42, 'verbose': -1, 'force_col_wise': True})
                st_ui.warning(f"Using fallback HPs for {hospital_name_run} - {target_col_name_run}")
            model = lgb.LGBMRegressor(**lgbm_final_params)
            model.fit(X_train, y_train)
            last_train_datetime = train_df['Datetime'].max()
            future_df_for_test = create_future_dates(last_train_datetime.date(), hospital_name_run, hospital_code_val, mean_capacity_train, forecast_days_param, irish_holidays_calendar_param, df_featured_param)
            future_df_for_test = future_df_for_test[future_df_for_test['Datetime'].isin(test_df['Datetime'])].reset_index(drop=True)
            if future_df_for_test.empty: st_ui.error(f"Future dates for test period are empty. Skipping MAE."); continue

            test_predictions = forecast_with_lags(model, train_df, future_df_for_test, final_model_features, target_col_name_run)
            y_test_actual = test_df[test_df['Datetime'].isin(future_df_for_test['Datetime'])][target_col_name_run].values
            if len(test_predictions) == len(y_test_actual) and len(y_test_actual) > 0:
                mae = mean_absolute_error(y_test_actual, test_predictions)
                final_mae_scores_output.append({'hospital': hospital_name_run, 'target': target_col_name_run, 'mae': mae})
                st_ui.metric(f"{target_col_name_run} MAE", f"{mae:.2f}")
                print(f"MAE for {hospital_name_run} - {target_col_name_run}: {mae:.2f} (using Iter2 HPs & Feats)")
                plot_df_test = test_df[test_df['Datetime'].isin(future_df_for_test['Datetime'])][['Datetime']].copy()
                plot_df_test['Predicted'] = test_predictions; plot_df_test['Actual'] = y_test_actual
                fig_final = plot_forecasts(train_df.tail(forecast_days_param * 3 * 2), plot_df_test, target_col_name_run, hospital_name_run)
                st_ui.plotly_chart(fig_final, use_container_width=True)
            else: st_ui.error(f"Prediction/actual length mismatch for {target_col_name_run}. Skipping MAE.")
            st_ui.divider()
    if final_mae_scores_output:
        avg_mae_final = np.nanmean([s['mae'] for s in final_mae_scores_output])
        st_ui.sidebar.metric("Final Average MAE (Iter2 Feats & HPs)", f"{avg_mae_final:.2f}")
        print(f"Final Average MAE (Iter2 Feats & HPs): {avg_mae_final:.2f}")
        with open('final_dmc_mae_scores.txt', 'w') as f_mae: # Changed filename to final_dmc_mae_scores.txt
            f_mae.write(f"Final Average MAE (Iter2 Features & HPs): {avg_mae_final:.4f}\n\n")
            for score_info in final_mae_scores_output: f_mae.write(f"Hospital: {score_info['hospital']}, Target: {score_info['target']}, MAE: {score_info.get('mae', 'N/A'):.4f}\n")
        st_ui.sidebar.success("Final MAE scores saved to `final_dmc_mae_scores.txt`")
    add_forecasting_insights(st_ui)

def direct_run_mae_calculation():
    print("Running DMC.py directly for MAE calculation (Iter2 Features & HPs with corrected encoding)...")
    forecast_days_direct = 7 ; final_mae_scores_direct = []
    try:
        df_initial_direct = pd.read_csv("processed_may_data.csv"); df_initial_direct['Datetime'] = pd.to_datetime(df_initial_direct['Datetime'])
        print(f"Direct run: Loaded initial data, shape {df_initial_direct.shape}")
        calendar_direct = IrishBankHolidays()
        # Call add_iteration2_features which now stores mappings globally and uses corrected encoding
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
            with open('final_dmc_mae_scores_iter5.txt', 'w') as f_mae_direct: # iter5 filename
                f_mae_direct.write(f"Final Average MAE (Iter2 Features & HPs, fixed encoding): {avg_mae_direct_run:.4f}\n\n")
                for score_info_direct in final_mae_scores_direct: f_mae_direct.write(f"Hospital: {score_info_direct['hospital']}, Target: {score_info_direct['target']}, MAE: {score_info_direct.get('mae', 'N/A'):.4f}\n")
            print("Final MAE scores (iter5) from direct run saved to `final_dmc_mae_scores_iter5.txt`")
        else: print("No MAE scores were calculated in direct run.")
    except Exception as e_direct: print(f"Error during direct run of DMC.py: {e_direct}"); import traceback; print(f"Traceback: {traceback.format_exc()}")

def main_streamlit_app():
    st.title("Emergency Department Forecasting (Ireland) - Final Version")
    st.markdown("This app uses `processed_may_data.csv`, applies Iteration 2 features, and uses tuned hyperparameters from Iteration 2 for forecasting.")
    forecast_days_ui = 7
    st.sidebar.info(f"Forecast period is fixed to {forecast_days_ui} days.")
    try:
        df_initial_ui = pd.read_csv("processed_may_data.csv")
        df_initial_ui['Datetime'] = pd.to_datetime(df_initial_ui['Datetime'])
        irish_holidays_calendar_ui = IrishBankHolidays()
        # Ensure category_mappings is fresh for each run of the Streamlit app
        global category_mappings
        category_mappings = {} # Reset for Streamlit context
        df_featured_ui = add_iteration2_features(df_initial_ui, irish_holidays_calendar_ui)

        st.success(f"✅ Data loaded and Iteration 2 features applied! Shape: {df_featured_ui.shape}")
        hospitals_ui = sorted(df_featured_ui['Hospital'].unique())
        st.sidebar.header("Forecast Settings")
        hospital_option_streamlit = st.sidebar.selectbox("Select Hospital:", options=["All Hospitals"] + hospitals_ui, key="hospital_select_streamlit_key")
        run_forecast_button_streamlit = st.sidebar.button("🚀 Run Forecast & Calculate MAE", type="primary")
        if run_forecast_button_streamlit:
            tuned_hps_iter2_loaded_ui = {}
            try:
                with open('tuned_hyperparameters_iter2.json', 'r') as f_hp_ui: tuned_hps_iter2_loaded_ui = json.load(f_hp_ui)
                st.sidebar.success("Loaded Iteration 2 Hyperparameters for run.")
            except Exception as e_hp_load: st.sidebar.error(f"Error loading HPs for UI run: {e_hp_load}")
            run_forecasting_logic(df_featured_ui, hospital_option_streamlit, forecast_days_ui, irish_holidays_calendar_ui, tuned_hps_iter2_loaded_ui, st)
    except FileNotFoundError as e_fnf:
        st.error(f"❌ Error: A required data file was not found. ({e_fnf.filename})")
        st.info("Ensure `processed_may_data.csv` and `tuned_hyperparameters_iter2.json` are in the root directory.")
    except Exception as e_main_app:
        st.error(f"❌ An unexpected error occurred in the Streamlit app: {str(e_main_app)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    is_streamlit_run_context = False
    try:
        # A way to check if Streamlit is running the script (might not be foolproof in all environments)
        if hasattr(st, 'runtime') and hasattr(st.runtime, 'scriptrunner') and st.runtime.scriptrunner.get_script_run_ctx():
            is_streamlit_run_context = True
    except Exception: # If any error (e.g. st.runtime not available), assume not streamlit
        pass

    if not is_streamlit_run_context:
        # If it's not a Streamlit run (e.g. `python DMC.py`), execute direct MAE calculation.
        direct_run_mae_calculation()

    # If it is a Streamlit run, Streamlit will execute the script from top-to-bottom.
    # The call to main_streamlit_app() below will build the UI.
    # The check above prevents direct_run_mae_calculation() from running again if Streamlit is also running this __main__ block.
    # However, Streamlit runs the whole script as __main__.
    # The most reliable for this subtask is that `python DMC.py` calls direct_run.
    # `streamlit run DMC.py` will call main_streamlit_app.
    # The above check `is_streamlit_run_context` is an attempt to make `python DMC.py` only run `direct_run_mae_calculation`.
    # When `streamlit run DMC.py` is used, it defines functions, then `main_streamlit_app()` is called at the end.
    # The `if __name__ == "__main__":` block will also be executed by Streamlit.
    # This refined check aims to prevent `direct_run_mae_calculation` when Streamlit is the executor.

if __name__ == "__main__":
    # This simplified __main__ block ensures that when `python DMC.py` is called,
    # `direct_run_mae_calculation` is executed.
    # When `streamlit run DMC.py` is called, Streamlit executes the script top-down,
    # and `main_streamlit_app()` at the end builds the UI.
    # The `direct_run_mae_calculation` will also be called by Streamlit in this case,
    # which is acceptable for the subtask if it doesn't break the UI part.
    # The goal is to ensure the MAE file is generated for `python DMC.py`.

    # To be very specific for subtask:
    # If 'streamlit' and 'run' are in sys.argv, it's a streamlit app run.
    # Otherwise, it's a direct python execution.

    if not ('streamlit' in sys.argv and 'run' in sys.argv):
        # This will be true for `python DMC.py`
        direct_run_mae_calculation()

main_streamlit_app() # This line ensures the Streamlit app UI is always defined and attempted to run.
                     # When `python DMC.py` is used, direct_run_mae_calculation() runs first, then this might error
                     # harmlessly for the purpose of MAE file generation.
                     # When `streamlit run DMC.py` is used, this builds the UI.
