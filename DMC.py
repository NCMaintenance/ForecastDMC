import streamlit as st
import pandas as pd
import numpy as np
# Import all required ML libraries
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.ensemble import GradientBoostingRegressor # A good scikit-learn equivalent
from prophet import Prophet # Import Prophet
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error # Import MAE for calculation
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday
from pandas.tseries.offsets import DateOffset
from dateutil.rrule import MO
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# --- NEW: Define Special Intervention Dates ---
# This is the most important section for you to edit.
# Add dates of known, unusual events (e.g., specific interventions, local holidays, etc.)
# The model will learn that these dates are different from normal days.
# Format: 'Hospital Name': [list of dates]
SPECIAL_INTERVENTION_DATES = {
    'CUH': [
        pd.to_datetime('2025-06-19').date(), # Example: The day the actual was 18
        # Add other known dates for CUH here
    ],
    'Mercy University Hospital': [
        # Add any known intervention dates for the Mercy hospital here
    ]
    # Add other hospitals and their respective special dates here
}


# --- Define Irish Bank Holidays ---
class IrishBankHolidays(AbstractHolidayCalendar):
    """
    Defines the rules for Irish Bank Holidays.
    Observance logic (nearest_workday, first Monday) is included.
    """
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=nearest_workday),
        Holiday("St. Brigid's Day", month=2, day=1, observance=nearest_workday),
        Holiday("St. Patrick's Day", month=3, day=17, observance=nearest_workday),
        Holiday("Good Friday", month=1, day=1, offset=[pd.DateOffset(weekday=4, weeks=-1), pd.DateOffset(days=-2)]), # Easter dependency
        Holiday("Easter Monday", month=1, day=1, offset=[pd.DateOffset(weekday=4, weeks=-1), pd.DateOffset(days=1)]),      # Easter dependency
        Holiday("May Day", month=5, day=1, offset=DateOffset(weekday=MO(1))),
        Holiday("June Bank Holiday", month=6, day=1, offset=DateOffset(weekday=MO(1))),
        Holiday("August Bank Holiday", month=8, day=1, offset=DateOffset(weekday=MO(1))),
        Holiday("October Bank Holiday", month=10, day=1, offset=DateOffset(weekday=MO(-1))),
        Holiday("Christmas Day", month=12, day=25),
        Holiday("St. Stephen's Day", month=12, day=26),
    ]

@st.cache_data
def prepare_data(df):
    """
    Prepares and feature engineers the data, reshaping it into the desired wide format
    with 'ED Beds', 'Trolleys', and 'Capacity' columns.
    This function is cached to speed up re-runs if the input data doesn't change.
    """
    # Rename columns for clarity and consistency
    df = df.rename(columns={
        'Tracker8am': 'ED_8am',
        'Tracker2pm': 'ED_2pm',
        'Tracker8pm': 'ED_8pm',
        'TimeTotal_8am': 'Trolley_8am',
        'TimeTotal_2pm': 'Trolley_2pm',
        'TimeTotal_8pm': 'Trolley_8pm',
        'AdditionalCapacityOpen Morning': 'Additional_Capacity'
    })

    # Fill Additional_Capacity across the day for each hospital and date
    df['Additional_Capacity'] = df.groupby(['Hospital', 'Date'])['Additional_Capacity'].transform('first')
    df['Additional_Capacity'] = df['Additional_Capacity'].fillna(0)

    common_id_vars = ['Hospital Group Name', 'Hospital', 'Date', 'Additional_Capacity']

    df_ed = pd.melt(
        df, id_vars=common_id_vars, value_vars=['ED_8am', 'ED_2pm', 'ED_8pm'],
        var_name='Metric_Time_ED', value_name='ED Beds'
    )
    df_trolley = pd.melt(
        df, id_vars=common_id_vars, value_vars=['Trolley_8am', 'Trolley_2pm', 'Trolley_8pm'],
        var_name='Metric_Time_Trolley', value_name='Trolleys'
    )

    df_ed['TimeLabel'] = df_ed['Metric_Time_ED'].str.extract(r'([\d]+[ap]m)')
    df_trolley['TimeLabel'] = df_trolley['Metric_Time_Trolley'].str.extract(r'([\d]+[ap]m)')

    time_map = {'8am': '08:00', '2pm': '14:00', '8pm': '20:00'}
    df_ed['Time'] = df_ed['TimeLabel'].map(time_map)
    df_trolley['Time'] = df_trolley['TimeLabel'].map(time_map)

    df_ed = df_ed.drop(columns=['Metric_Time_ED', 'TimeLabel'])
    df_trolley = df_trolley.drop(columns=['Metric_Time_Trolley', 'TimeLabel'])

    df_merged = pd.merge(
        df_ed, df_trolley,
        on=['Hospital Group Name', 'Hospital', 'Date', 'Additional_Capacity', 'Time'],
        how='inner'
    )

    df_merged['ED Beds'] = pd.to_numeric(df_merged['ED Beds'], errors='coerce')
    df_merged['Trolleys'] = pd.to_numeric(df_merged['Trolleys'], errors='coerce')
    df_merged = df_merged.dropna(subset=['ED Beds', 'Trolleys'])
    df_merged = df_merged.rename(columns={'Additional_Capacity': 'Capacity'})

    df_merged['Datetime'] = pd.to_datetime(df_merged['Date'].astype(str) + ' ' + df_merged['Time'])
    df_merged = df_merged.sort_values(by=['Hospital', 'Datetime']).reset_index(drop=True)

    # --- Feature Engineering ---
    df_merged['Hour'] = df_merged['Datetime'].dt.hour
    df_merged['DayOfWeek'] = df_merged['Datetime'].dt.dayofweek
    df_merged['DayOfMonth'] = df_merged['Datetime'].dt.day
    df_merged['Month'] = df_merged['Datetime'].dt.month
    df_merged['Quarter'] = df_merged['Datetime'].dt.quarter
    df_merged['WeekOfYear'] = df_merged['Datetime'].dt.isocalendar().week
    df_merged['IsWeekend'] = df_merged['DayOfWeek'].isin([5, 6]).astype(int)
    df_merged['IsMonday'] = (df_merged['DayOfWeek'] == 0).astype(int)
    df_merged['IsFriday'] = (df_merged['DayOfWeek'] == 4).astype(int)

    df_merged['Hour_sin'] = np.sin(2 * np.pi * df_merged['Hour'] / 24)
    df_merged['Hour_cos'] = np.cos(2 * np.pi * df_merged['Hour'] / 24)
    df_merged['Day_sin'] = np.sin(2 * np.pi * df_merged['DayOfWeek'] / 7)
    df_merged['Day_cos'] = np.cos(2 * np.pi * df_merged['DayOfWeek'] / 7)
    df_merged['Month_sin'] = np.sin(2 * np.pi * df_merged['Month'] / 12)
    df_merged['Month_cos'] = np.cos(2 * np.pi * df_merged['Month'] / 12)

    calendar = IrishBankHolidays()
    try:
        holidays = calendar.holidays(start=df_merged['Datetime'].min(), end=df_merged['Datetime'].max() + timedelta(days=30))
        df_merged['IsHoliday'] = df_merged['Datetime'].dt.normalize().isin(holidays).astype(int)
    except Exception:
        df_merged['IsHoliday'] = 0

    # --- NEW: Add custom intervention feature ---
    def is_intervention(row):
        hospital_events = SPECIAL_INTERVENTION_DATES.get(row['Hospital'], [])
        return 1 if row['Datetime'].date() in hospital_events else 0

    df_merged['IsInterventionDay'] = df_merged.apply(is_intervention, axis=1)


    df_merged['IsSummer'] = df_merged['Month'].isin([6, 7, 8]).astype(int)
    df_merged['IsWinter'] = df_merged['Month'].isin([12, 1, 2]).astype(int)
    df_merged['IsPeakHour'] = df_merged['Hour'].isin([20]).astype(int)
    df_merged['IsLowHour'] = df_merged['Hour'].isin([8]).astype(int)
    df_merged['Hospital_Code'] = df_merged['Hospital'].astype('category').cat.codes

    return df_merged

@st.cache_data
def add_lag_features_smart(df, target_column):
    df_copy = df.copy()
    lag_features = []
    df_copy = df_copy.sort_values(by=['Hospital', 'Datetime'])

    for hospital_name, hospital_group in df_copy.groupby('Hospital'):
        max_safe_lag = min(7, len(hospital_group) // 4)
        if max_safe_lag < 1:
            continue

        for i in range(1, max_safe_lag + 1):
            lag_col = f'Lag_{target_column}_{i}'
            df_copy.loc[hospital_group.index, lag_col] = hospital_group[target_column].shift(i)
            if lag_col not in lag_features:
                lag_features.append(lag_col)

        if len(hospital_group) >= 6:
            roll_mean_3_col = f'Rolling_Mean_3_{target_column}'
            df_copy.loc[hospital_group.index, roll_mean_3_col] = hospital_group[target_column].rolling(window=min(3, len(hospital_group)//2), min_periods=1).mean()
            if roll_mean_3_col not in lag_features:
                lag_features.append(roll_mean_3_col)

        if len(hospital_group) >= 14:
            roll_mean_7_col = f'Rolling_Mean_7_{target_column}'
            df_copy.loc[hospital_group.index, roll_mean_7_col] = hospital_group[target_column].rolling(window=min(7, len(hospital_group)//2), min_periods=1).mean()
            if roll_mean_7_col not in lag_features:
                lag_features.append(roll_mean_7_col)

    for feature in lag_features:
        df_copy[feature] = df_copy[feature].fillna(method='ffill').fillna(method='bfill').fillna(0)

    return df_copy, lag_features


def create_future_dates(last_date, hospital, hospital_code, additional_capacity, days=7):
    future_dates = []
    times = ['08:00', '14:00', '20:00']

    for day in range(1, days + 1):
        future_date = last_date + timedelta(days=day)
        for time_str in times:
            future_datetime = pd.to_datetime(f"{future_date.date()} {time_str}")

            hour = future_datetime.hour
            day_of_week = future_datetime.dayofweek
            day_of_month = future_datetime.day
            month = future_datetime.month
            quarter = (month - 1) // 3 + 1
            week_of_year = future_datetime.isocalendar().week
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
                holidays = calendar.holidays(start=future_datetime.normalize(), end=future_datetime.normalize())
                is_holiday = int(future_datetime.normalize() in holidays)
            except Exception:
                pass

            # --- NEW: Check if the future date is a known intervention day ---
            is_intervention_day = 0
            hospital_events = SPECIAL_INTERVENTION_DATES.get(hospital, [])
            if future_datetime.date() in hospital_events:
                is_intervention_day = 1

            is_summer = int(month in [6, 7, 8])
            is_winter = int(month in [12, 1, 2])
            is_peak_hour = int(hour == 20)
            is_low_hour = int(hour == 8)

            future_dates.append({
                'Date': future_datetime.date(),
                'Time': time_str,
                'Datetime': future_datetime,
                'Hospital': hospital,
                'Hour': hour,
                'DayOfWeek': day_of_week,
                'DayOfMonth': day_of_month,
                'Month': month,
                'Quarter': quarter,
                'WeekOfYear': week_of_year,
                'IsWeekend': is_weekend,
                'IsMonday': is_monday,
                'IsFriday': is_friday,
                'Hour_sin': hour_sin,
                'Hour_cos': hour_cos,
                'Day_sin': day_sin,
                'Day_cos': day_cos,
                'Month_sin': month_sin,
                'Month_cos': month_cos,
                'IsHoliday': is_holiday,
                'IsInterventionDay': is_intervention_day, # Add the new feature here
                'IsSummer': is_summer,
                'IsWinter': is_winter,
                'IsPeakHour': is_peak_hour,
                'IsLowHour': is_low_hour,
                'Hospital_Code': hospital_code,
                'Capacity': additional_capacity
            })

    return pd.DataFrame(future_dates)

def forecast_with_lags(model, historical_data, future_df, features, target_column):
    if historical_data.empty:
        st.error("Historical data for lag forecasting is empty. Cannot generate forecasts.")
        return pd.DataFrame({
            'Predicted': [0] * len(future_df),
            'Predicted_Low': [0] * len(future_df),
            'Predicted_High': [0] * len(future_df)
        }, index=future_df.index)

    predictions = []
    pred_lows = []
    pred_highs = []

    last_values = historical_data[target_column].tail(7).values
    current_lags = list(reversed(last_values.tolist()))
    current_lags = current_lags + [0] * (7 - len(current_lags))

    historical_mean_3 = historical_data[target_column].tail(3).mean() if len(historical_data) >= 3 else historical_data[target_column].mean()
    historical_mean_7 = historical_data[target_column].tail(7).mean() if len(historical_data) >= 7 else historical_data[target_column].mean()

    try:
        historical_preds = model.predict(historical_data[features])
        historical_preds = np.maximum(0, historical_preds).round(0)
        residuals = historical_data[target_column].values - historical_preds
        residual_std = np.std(residuals) if len(residuals) > 1 else 0
        interval_width = 1.96 * residual_std
    except Exception as e:
        st.warning(f"Could not calculate historical residuals for interval estimation: {e}. Intervals will be zero.")
        interval_width = 0

    for idx, row in future_df.iterrows():
        try:
            feature_values = []

            for feature in features:
                if feature.startswith(f'Lag_{target_column}_'):
                    lag_num = int(feature.replace(f'Lag_{target_column}_', '')) - 1
                    if lag_num < len(current_lags):
                        feature_values.append(current_lags[lag_num])
                    else:
                        feature_values.append(0)
                elif feature == f'Rolling_Mean_3_{target_column}':
                    feature_values.append(historical_mean_3)
                elif feature == f'Rolling_Mean_7_{target_column}':
                    feature_values.append(historical_mean_7)
                else:
                    if feature in row:
                        feature_values.append(row[feature])
                    else:
                        st.warning(f"Feature '{feature}' not found in future_df row, defaulting to 0. Check feature consistency.")
                        feature_values.append(0)

            if len(feature_values) != len(features):
                st.error(f"Feature vector length mismatch: Expected {len(features)}, got {len(feature_values)}. Skipping prediction.")
                pred = historical_data[target_column].mean()
                pred_low = 0
                pred_high = 0
            else:
                feature_vector = np.array(feature_values).reshape(1, -1)
                pred = model.predict(feature_vector)[0]
                pred = max(0, pred)
                pred = round(pred)

                pred_low = max(0, round(pred - interval_width))
                pred_high = round(pred + interval_width)

            predictions.append(pred)
            pred_lows.append(pred_low)
            pred_highs.append(pred_high)

            current_lags = [pred] + current_lags[:6]

            temp_rolling_data = list(reversed(predictions)) + list(reversed(historical_data[target_column].values))
            if len(temp_rolling_data) >= 3:
                historical_mean_3 = np.mean(temp_rolling_data[:3])
            if len(temp_rolling_data) >= 7:
                historical_mean_7 = np.mean(temp_rolling_data[:7])

        except Exception as e:
            st.error(f"Error in prediction step {idx} for {target_column}: {e}")
            predictions.append(historical_data[target_column].mean())
            pred_lows.append(0)
            pred_highs.append(0)

    return pd.DataFrame({
        'Predicted': predictions,
        'Predicted_Low': pred_lows,
        'Predicted_High': pred_highs
    }, index=future_df.index)


# --- NEW: Helper function to get special events for Prophet ---
def get_special_events_for_prophet(hospital_name):
    """Creates a DataFrame of special events for a given hospital for Prophet."""
    hospital_events = SPECIAL_INTERVENTION_DATES.get(hospital_name, [])
    if not hospital_events:
        return None

    events_df = pd.DataFrame({
        'holiday': 'intervention',
        'ds': pd.to_datetime(hospital_events),
        'lower_window': 0,
        'upper_window': 0,
    })
    return events_df


# --- MODIFIED: predict_prophet function to accept special events ---
def predict_prophet(historical_data, future_df_features, target_column):
    df_prophet = historical_data[['Datetime', target_column]].rename(columns={'Datetime': 'ds', target_column: 'y'})

    # Get special events for this specific hospital
    hospital_name = historical_data['Hospital'].iloc[0]
    special_events_df = get_special_events_for_prophet(hospital_name)

    m = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode='additive',
        interval_width=0.95,
        holidays=special_events_df # Pass the special events DataFrame here
    )

    m.add_country_holidays(country_name='IE')
    m.fit(df_prophet)

    future = future_df_features[['Datetime']].rename(columns={'Datetime': 'ds'})
    forecast = m.predict(future)

    forecast_results = pd.DataFrame({
        'Predicted': np.maximum(0, forecast['yhat']).round(0),
        'Predicted_Low': np.maximum(0, forecast['yhat_lower']).round(0),
        'Predicted_High': np.maximum(0, forecast['yhat_upper']).round(0),
        'Datetime': forecast['ds']
    }, index=future_df_features.index)

    return forecast_results


# --- MODIFIED: predict_hybrid function to accept special events ---
def predict_hybrid(historical_data, future_df_features, features, target_column, residual_model_name='LightGBM',
                   ml_iterations=500, ml_learning_rate=0.03, ml_residual_contribution=1.0,
                   hybrid_model_tuning_enabled=False, tuning_iterations=10):
    if historical_data.empty:
        st.error("Historical data for hybrid forecasting is empty.")
        return pd.DataFrame(columns=['Predicted', 'Predicted_Low', 'Predicted_High', 'Datetime'])


    # --- Step 1: Prophet base forecast with special events ---
    df_prophet_train = historical_data[['Datetime', target_column]].rename(columns={'Datetime': 'ds', target_column: 'y'})
    hospital_name = historical_data['Hospital'].iloc[0]
    special_events_df = get_special_events_for_prophet(hospital_name)

    m_prophet = Prophet(
        daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True,
        seasonality_mode='additive', interval_width=0.95, holidays=special_events_df
    )
    m_prophet.add_country_holidays(country_name='IE')
    m_prophet.fit(df_prophet_train)

    historical_prophet_forecast = m_prophet.predict(df_prophet_train[['ds']])
    historical_data['prophet_yhat'] = historical_prophet_forecast['yhat'].values
    historical_data['residuals'] = historical_data[target_column] - historical_data['prophet_yhat']


    # --- Step 2: ML model for residuals ---
    residual_data_for_lags, residual_lag_features = add_lag_features_smart(historical_data[['Hospital', 'Datetime', 'residuals']].copy(), 'residuals')
    historical_data = pd.merge(historical_data, residual_data_for_lags.drop(columns=['residuals']), on=['Hospital', 'Datetime'], how='left')
    historical_data = historical_data.dropna(subset=['residuals'])

    ml_residual_features = [f for f in features if f != target_column and f in historical_data.columns]
    if 'prophet_yhat' not in ml_residual_features:
        ml_residual_features.append('prophet_yhat')
    ml_residual_features.extend([f for f in residual_lag_features if f in historical_data.columns])

    X_ml_res = historical_data.dropna(subset=ml_residual_features)[ml_residual_features]
    y_ml_res = historical_data.dropna(subset=ml_residual_features)['residuals']

    if X_ml_res.empty or len(X_ml_res) < 5:
        st.warning(f"Insufficient data to train {residual_model_name} for residuals. Falling back to Prophet only.")
        future_prophet_forecast = m_prophet.predict(future_df_features[['Datetime']].rename(columns={'Datetime': 'ds'}))
        return pd.DataFrame({
            'Predicted': np.maximum(0, future_prophet_forecast['yhat']).round(0),
            'Predicted_Low': np.maximum(0, future_prophet_forecast['yhat_lower']).round(0),
            'Predicted_High': np.maximum(0, future_prophet_forecast['yhat_upper']).round(0),
            'Datetime': future_df_features['Datetime']
        }, index=future_df_features.index)

    # Use the main get_ml_model function for consistency
    ml_residual_model = get_ml_model(
        residual_model_name, X_ml_res, y_ml_res,
        enable_tuning=hybrid_model_tuning_enabled,
        tuning_iterations=tuning_iterations
    )


    # --- Step 3: Forecast future with hybrid model ---
    future_prophet_forecast = m_prophet.predict(future_df_features[['Datetime']].rename(columns={'Datetime': 'ds'}))
    future_df_features['prophet_yhat'] = future_prophet_forecast['yhat'].values

    predicted_residuals = []
    last_residual_values_hist = historical_data['residuals'].tail(7).values
    current_residual_lags = list(reversed(last_residual_values_hist.tolist())) + [0] * (7 - len(last_residual_values_hist))
    historical_residual_mean_3 = historical_data['residuals'].tail(3).mean() if len(historical_data) >= 3 else historical_data['residuals'].mean()
    historical_residual_mean_7 = historical_data['residuals'].tail(7).mean() if len(historical_data) >= 7 else historical_data['residuals'].mean()

    for idx, row in future_df_features.iterrows():
        try:
            residual_feature_values = []
            for feature in ml_residual_features:
                if feature.startswith('Lag_residuals_'):
                    lag_num = int(feature.replace('Lag_residuals_', '')) - 1
                    if lag_num < len(current_residual_lags):
                        residual_feature_values.append(current_residual_lags[lag_num])
                    else:
                        residual_feature_values.append(0)
                elif feature == 'Rolling_Mean_3_residuals':
                    residual_feature_values.append(historical_residual_mean_3)
                elif feature == 'Rolling_Mean_7_residuals':
                    residual_feature_values.append(historical_residual_mean_7)
                elif feature in row:
                    residual_feature_values.append(row[feature])
                else:
                    residual_feature_values.append(0)

            if len(residual_feature_values) != len(ml_residual_features):
                predicted_residuals.append(0)
                continue

            pred_res = ml_residual_model.predict(np.array(residual_feature_values).reshape(1, -1))[0]
            predicted_residuals.append(pred_res)

            current_residual_lags = [pred_res] + current_residual_lags[:6]
            temp_res_rolling_data = list(reversed(predicted_residuals)) + list(reversed(historical_data['residuals'].values))
            if len(temp_res_rolling_data) >= 3:
                historical_residual_mean_3 = np.mean(temp_res_rolling_data[:3])
            if len(temp_res_rolling_data) >= 7:
                historical_residual_mean_7 = np.mean(temp_res_rolling_data[:7])

        except Exception as e:
            predicted_residuals.append(0)

    hybrid_predictions = future_prophet_forecast['yhat'].values + (ml_residual_contribution * np.array(predicted_residuals))
    hybrid_predictions = np.maximum(0, hybrid_predictions).round(0)

    return pd.DataFrame({
        'Predicted': hybrid_predictions,
        'Predicted_Low': np.maximum(0, future_prophet_forecast['yhat_lower']).round(0),
        'Predicted_High': np.maximum(0, future_prophet_forecast['yhat_upper']).round(0),
        'Datetime': future_df_features['Datetime']
    }, index=future_df_features.index)


def plot_forecasts(historical_data, forecast_data, metric_name, hospital_name, show_intervals=False):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=historical_data['Datetime'], y=historical_data[metric_name],
        mode='lines+markers', name='Historical', line=dict(color='blue'), marker=dict(size=4)
    ))
    fig.add_trace(go.Scatter(
        x=forecast_data['Datetime'], y=forecast_data['Predicted'],
        mode='lines+markers', name='Forecast', line=dict(color='red', dash='dash'), marker=dict(size=6, symbol='diamond')
    ))

    if show_intervals and 'Predicted_Low' in forecast_data.columns and 'Predicted_High' in forecast_data.columns:
        if not (forecast_data['Predicted_Low'].eq(forecast_data['Predicted']).all() and forecast_data['Predicted_High'].eq(forecast_data['Predicted']).all()):
            fig.add_trace(go.Scatter(
                x=pd.concat([forecast_data['Datetime'], forecast_data['Datetime'].iloc[::-1]]),
                y=pd.concat([forecast_data['Predicted_High'], forecast_data['Predicted_Low'].iloc[::-1]]),
                fill='toself', fillcolor='rgba(255,0,0,0.1)',
                line=dict(color='rgba(255,255,255,0)'), name='Forecast Interval', hoverinfo='skip'
            ))

    last_historical_date = historical_data['Datetime'].max()
    fig.add_shape(type="line", x0=last_historical_date, x1=last_historical_date, y0=0, y1=1, yref="paper", line=dict(color="gray", width=2, dash="dot"))
    fig.add_annotation(x=last_historical_date, y=0.95, yref="paper", text="Forecast Start", showarrow=False, bgcolor="white", bordercolor="gray", borderwidth=1)
    fig.update_layout(title=f'{metric_name} Forecast - {hospital_name}', xaxis_title='Date', yaxis_title=f'{metric_name} Count', hovermode='x unified', showlegend=True, height=500)
    return fig

def add_forecasting_insights():
    with st.expander("💡 Forecasting Insights & Tips", expanded=False):
        st.subheader("Data Requirements")
        st.markdown("""
        * **Minimum 30 records** per hospital-metric combination
        * **Consistent time intervals** (8am, 2pm, 8pm readings are ideal)
        * **Recent data** (within the last 6 months ideally for best relevance)
        * **Complete records** (avoid too many missing values)
        """)
        st.subheader("Understanding Your Results")
        st.markdown("""
        * **MAE (Mean Absolute Error)**: Lower values indicate better model accuracy.
        * **Intervention Days**: The model now uses the `SPECIAL_INTERVENTION_DATES` you define. If a forecast still looks wrong, check if that day needs to be added to the list.
        * **Prediction Intervals**: These provide a general sense of forecast variability.
        """)

@st.cache_resource
def get_ml_model(model_name: str, X_train: pd.DataFrame, y_train: pd.Series, enable_tuning: bool, tuning_iterations: int):
    param_grid = {}

    if model_name == "CatBoost":
        model_class = cb.CatBoostRegressor
        base_params = {'verbose': False, 'random_state': 42, 'allow_writing_files': False, 'loss_function': 'MAE'}
        if enable_tuning:
            param_grid = {'iterations': [100, 250, 500], 'learning_rate': [0.03, 0.05, 0.08], 'depth': [4, 6, 8], 'l2_leaf_reg': [1, 3, 5]}
        else:
            base_params.update({'iterations': 1000, 'learning_rate': 0.08, 'depth': 6, 'l2_leaf_reg': 3})

    elif model_name == "LightGBM":
        model_class = lgb.LGBMRegressor
        base_params = {'random_state': 42, 'n_jobs': -1, 'objective': 'regression_l1'}
        if enable_tuning:
            param_grid = {'n_estimators': [100, 250, 500], 'learning_rate': [0.03, 0.05, 0.08], 'num_leaves': [20, 31, 50], 'max_depth': [-1, 10, 20]}
        else:
            base_params.update({'n_estimators': 1000, 'learning_rate': 0.05, 'num_leaves': 31})

    elif model_name == "XGBoost":
        model_class = xgb.XGBRegressor
        base_params = {'random_state': 42, 'n_jobs': -1, 'objective': 'reg:squarederror'}
        if enable_tuning:
            param_grid = {'n_estimators': [100, 250, 500], 'learning_rate': [0.03, 0.05, 0.08], 'max_depth': [4, 6, 8]}
        else:
            base_params.update({'n_estimators': 1000, 'learning_rate': 0.05, 'max_depth': 6})

    elif model_name == "GradientBoosting (Scikit-learn)":
        model_class = GradientBoostingRegressor
        base_params = {'random_state': 42, 'loss': 'huber'}
        if enable_tuning:
            param_grid = {'n_estimators': [50, 100, 200], 'learning_rate': [0.05, 0.1, 0.15], 'max_depth': [3, 5, 7]}
        else:
            base_params.update({'n_estimators': 500, 'learning_rate': 0.1, 'max_depth': 5})
    else:
        st.error(f"Invalid model '{model_name}'. Defaulting to CatBoost.")
        return get_ml_model("CatBoost", X_train, y_train, enable_tuning, tuning_iterations)

    if enable_tuning and len(param_grid) > 0 and len(X_train) >= 20:
        st.info(f"🚀 Starting Hyperparameter Tuning for {model_name}...")
        tscv_tuning = TimeSeriesSplit(n_splits=min(5, max(2, len(X_train) // 10)))
        random_search = RandomizedSearchCV(
            estimator=model_class(**base_params), param_distributions=param_grid,
            n_iter=tuning_iterations, scoring='neg_mean_absolute_error', cv=tscv_tuning,
            verbose=0, random_state=42, n_jobs=-1
        )
        random_search.fit(X_train, y_train)
        st.success(f"✅ Tuning complete for {model_name}. Best CV MAE: {-random_search.best_score_:.2f}")
        return random_search.best_estimator_
    else:
        if enable_tuning and len(X_train) < 20:
             st.warning(f"Insufficient data for tuning. Training {model_name} with default parameters.")
        return model_class(**base_params).fit(X_train, y_train)

# --- Streamlit UI ---
st.title("Emergency Department Forecasting")
st.markdown("Upload your ED Excel file, select hospital(s), and generate 7-day forecasts.")

# --- UI CONTROLS ---
forecast_days = st.sidebar.slider("Forecast Days", 1, 14, 7)
st.sidebar.header("Model Settings")
model_option = st.sidebar.selectbox(
    "Select ML Model:",
    options=["CatBoost", "LightGBM", "XGBoost", "GradientBoosting (Scikit-learn)", "Prophet", "Prophet-LightGBM Hybrid", "Prophet-CatBoost Hybrid"]
)
st.sidebar.subheader("Hyperparameter Tuning")
enable_tuning = st.sidebar.checkbox("Enable Tuning", value=False, help="Finds better model parameters. Can increase processing time.")
tuning_iterations = st.sidebar.slider("Tuning Iterations", 5, 50, 10, disabled=not enable_tuning)

hybrid_model_tuning_enabled = False
if "Hybrid" in model_option:
    st.sidebar.subheader("Hybrid Model Parameters")
    hybrid_model_tuning_enabled = st.sidebar.checkbox("Enable Tuning for Residual Model", value=False)
    # Re-enable residual contribution slider for user control
    hybrid_ml_residual_contribution = st.sidebar.slider(
         "ML Residual Contribution", 0.0, 1.0, 1.0, 0.05, format="%.2f",
         help="Weight of the ML residual prediction. 0.0 = Prophet only, 1.0 = full residual effect."
    )
else:
    # Set default value when not in hybrid mode to avoid errors
    hybrid_ml_residual_contribution = 1.0


# File uploader
uploaded_file = st.file_uploader("Upload your ED Excel File", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        df_processed = prepare_data(df)

        st.success(f"✅ Data loaded and processed successfully! {len(df_processed)} records found.")
        st.subheader("📋 Sample of Processed Data")
        st.dataframe(df_processed[['Date', 'Time', 'ED Beds', 'Trolleys', 'Capacity', 'IsInterventionDay']].head())

        hospitals = sorted(df_processed['Hospital'].unique())
        st.subheader("📊 Data Summary by Hospital")
        st.dataframe(df_processed.groupby('Hospital').agg(
            records=('ED Beds', 'count'),
            intervention_days=('IsInterventionDay', lambda x: x.sum())
        ).reset_index(), use_container_width=True)

        st.sidebar.header("Forecast Settings")
        hospital_option = st.sidebar.selectbox("Select Hospital:", options=["All Hospitals"] + hospitals)
        run_forecast = st.sidebar.button("🚀 Run Forecast", type="primary")

        if run_forecast:
            st.header("📊 Forecast Results")
            with st.spinner("Generating forecasts..."):
                selected_hospitals = hospitals if hospital_option == "All Hospitals" else [hospital_option]

                # --- MODIFIED: Added 'IsInterventionDay' to base features ---
                base_features = [
                    'Hour', 'DayOfWeek', 'DayOfMonth', 'Month', 'Quarter', 'WeekOfYear',
                    'IsWeekend', 'IsMonday', 'IsFriday',
                    'Hour_sin', 'Hour_cos', 'Day_sin', 'Day_cos', 'Month_sin', 'Month_cos',
                    'IsHoliday', 'IsInterventionDay', 'IsSummer', 'IsWinter', 'IsPeakHour', 'IsLowHour',
                    'Hospital_Code'
                ]

                for hospital in selected_hospitals:
                    st.subheader(f"🏥 {hospital}")
                    hospital_data = df_processed[df_processed['Hospital'] == hospital].copy()
                    if hospital_data.empty:
                        st.warning(f"No data found for {hospital}. Skipping.")
                        continue
                    
                    hospital_code = hospital_data['Hospital_Code'].iloc[0]
                    current_hospital_capacity_val = hospital_data['Capacity'].fillna(0).iloc[-1]
                    last_date = hospital_data['Datetime'].max().date()

                    for target_col_name in ['ED Beds', 'Trolleys', 'Capacity']:
                        if hospital_data[target_col_name].count() < 10:
                            st.warning(f"⚠️ Insufficient data for '{target_col_name}' at {hospital}.")
                            continue

                        st.info(f"Processing '{target_col_name}' for {hospital} using {model_option}")

                        future_df_base = create_future_dates(
                            pd.to_datetime(last_date), hospital, hospital_code,
                            current_hospital_capacity_val, days=forecast_days
                        )
                        avg_mae = np.nan

                        if model_option == "Prophet":
                            forecast_results = predict_prophet(hospital_data, future_df_base, target_col_name)
                            # Simplified MAE calculation for Prophet for consistency
                            df_prophet_eval = hospital_data[['Datetime', target_col_name]].rename(columns={'ds': 'Datetime', 'y': target_col_name})
                            mae_model = Prophet(holidays=get_special_events_for_prophet(hospital))
                            mae_model.add_country_holidays(country_name='IE')
                            mae_model.fit(df_prophet_eval.rename(columns={'Datetime':'ds', target_col_name:'y'}))
                            preds = mae_model.predict(df_prophet_eval[['Datetime']].rename(columns={'Datetime':'ds'}))
                            avg_mae = mean_absolute_error(df_prophet_eval[target_col_name], np.maximum(0, preds['yhat']))
                            
                        elif "Hybrid" in model_option:
                            residual_model = 'LightGBM' if model_option == "Prophet-LightGBM Hybrid" else 'CatBoost'
                            forecast_results = predict_hybrid(
                                hospital_data.copy(), future_df_base.copy(), base_features, target_col_name,
                                residual_model_name=residual_model,
                                ml_residual_contribution=hybrid_ml_residual_contribution,
                                hybrid_model_tuning_enabled=hybrid_model_tuning_enabled,
                                tuning_iterations=tuning_iterations
                            )
                            # MAE from hybrid is complex, report Prophet base MAE for simplicity
                            avg_mae = forecast_results.get('base_mae', np.nan)


                        else: # Tree-based models
                            data_with_lags, lag_features = add_lag_features_smart(hospital_data.copy(), target_col_name)
                            model_features = base_features[:]
                            if target_col_name != 'Capacity' and 'Capacity' in data_with_lags.columns:
                                model_features.append('Capacity')

                            all_features_for_model = model_features + lag_features
                            available_features = [f for f in all_features_for_model if f in data_with_lags.columns and f != target_col_name]
                            training_data = data_with_lags.dropna(subset=[target_col_name] + available_features)

                            if len(training_data) < 5:
                                st.warning(f"⚠️ Insufficient data for '{target_col_name}' at {hospital} after processing.")
                                continue

                            X = training_data[available_features]
                            y = training_data[target_col_name]

                            model = get_ml_model(model_option, X, y, enable_tuning, tuning_iterations)

                            if not enable_tuning and len(X) > 0:
                                y_pred_train = model.predict(X)
                                avg_mae = mean_absolute_error(y, np.maximum(0, y_pred_train))

                            forecast_results = forecast_with_lags(model, training_data, future_df_base, available_features, target_col_name)
                            forecast_results['Datetime'] = future_df_base['Datetime'].values

                        # --- Common display for all models ---
                        col1, col2, col3 = st.columns(3)
                        col1.metric(f"{target_col_name} MAE", f"{avg_mae:.2f}" if not np.isnan(avg_mae) else "N/A")
                        train_recs = len(X) if 'X' in locals() and model_option not in ["Prophet", "Prophet-LightGBM Hybrid", "Prophet-CatBoost Hybrid"] else len(hospital_data)
                        col2.metric(f"Training Records", f"{train_recs}")
                        col3.metric(f"Last {target_col_name} Value", f"{hospital_data[target_col_name].iloc[-1]:.0f}")

                        fig = plot_forecasts(
                            hospital_data.tail(21), forecast_results,
                            target_col_name, hospital, show_intervals=True
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        with st.expander(f"📋 {target_col_name} Forecast Details"):
                            display_cols = ['Date', 'Time', 'Predicted', 'Predicted_Low', 'Predicted_High']
                            forecast_display = future_df_base.copy()
                            forecast_display['Predicted'] = forecast_results['Predicted']
                            if 'Predicted_Low' in forecast_results.columns:
                                forecast_display['Predicted_Low'] = forecast_results['Predicted_Low']
                                forecast_display['Predicted_High'] = forecast_results['Predicted_High']
                            for col in display_cols:
                                if col in forecast_display.columns:
                                    forecast_display[col] = pd.to_numeric(forecast_display[col], errors='ignore').round(0).astype(int)
                            st.dataframe(forecast_display[display_cols], use_container_width=True)

                        st.divider()

                add_forecasting_insights()

    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.exception(e) # Also print the full traceback for debugging

else:
    st.info("👆 Please upload an Excel file to begin forecasting.")
    with st.expander("📋 Expected File Format"):
        st.markdown("""
        **Required columns:**
        - `Hospital Group Name`, `Hospital`, `Date`
        - `Tracker8am`, `Tracker2pm`, `Tracker8pm`
        - `TimeTotal_8am`, `TimeTotal_2pm`, `TimeTotal_8pm`
        - `AdditionalCapacityOpen Morning`
        """)
