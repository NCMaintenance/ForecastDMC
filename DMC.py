import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import xgboost as xgb
import holidays
from sklearn.metrics import mean_absolute_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import logging

# Suppress verbose logging
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

st.set_page_config(layout="wide", page_title="ED Metrics Hybrid Forecasting")
st.title("🏥 Emergency Department Metrics Hybrid Forecasting (Prophet + XGBoost)")

st.sidebar.title("Upload & Filter")
uploaded_file = st.sidebar.file_uploader("Upload Excel file", type=["xlsx", "xls"])

hospital_filter = None
if uploaded_file:
    df_temp = pd.read_excel(uploaded_file)
    df_temp.columns = df_temp.columns.str.replace(" ", "")  # Remove spaces from all columns
    
    if 'Hospital' in df_temp.columns:
        hospitals = sorted(df_temp['Hospital'].dropna().unique())
        hospital_filter = st.sidebar.selectbox("Select Hospital", ["All"] + hospitals)
    else:
        st.sidebar.warning("No 'Hospital' column found.")

run_button = st.sidebar.button("Run Forecast")

# Define target metrics and their time suffixes (ensure no spaces)
target_metrics_info = [
    ('Tracker8am', '08:00:00'),
    ('Tracker2pm', '14:00:00'),
    ('Tracker8pm', '20:00:00'),
    ('AdditionalCapacityOpenMorning', '08:00:00'),
    ('TimeTotal_8am', '08:00:00'),
    ('TimeTotal_2pm', '14:00:00'),
    ('TimeTotal_8pm', '20:00:00'),
]

# Setup Ireland holidays including custom Feb 2025 bank holiday on 18th Feb 2025
ie_holidays = holidays.Ireland(years=[2024, 2025])
ie_holidays.append({'2025-02-18': "New Feb Bank Holiday"})

def add_features(df):
    df['year'] = df['ds'].dt.year
    df['month'] = df['ds'].dt.month
    df['day'] = df['ds'].dt.day
    df['dayofweek'] = df['ds'].dt.dayofweek
    df['hour'] = df['ds'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['is_holiday'] = df['ds'].dt.date.map(lambda x: 1 if x in ie_holidays else 0)
    df['day_after_holiday'] = df['is_holiday'].shift(1).fillna(0)
    return df

if uploaded_file and run_button:
    try:
        df = pd.read_excel(uploaded_file)
        df.columns = df.columns.str.replace(" ", "")
        
        if 'Date' not in df.columns:
            st.error("Missing 'Date' column in data.")
            st.stop()
        
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Filter by hospital if selected
        if hospital_filter and hospital_filter != "All":
            df = df[df['Hospital'] == hospital_filter]

        if df.empty:
            st.warning("No data for selected hospital.")
            st.stop()

        st.subheader("Filtered Data Preview")
        st.dataframe(df.head())

        # Prepare dataframe for forecasts
        forecast_horizon_days = 7
        last_date = df['Date'].max()
        
        # Create future dataframe with all 3 times per day for next 7 days
        future_dates = []
        for day_offset in range(1, forecast_horizon_days + 1):
            day = last_date + pd.Timedelta(days=day_offset)
            for hour_str in ['08:00:00', '14:00:00', '20:00:00']:
                dt_str = day.strftime('%Y-%m-%d') + ' ' + hour_str
                future_dates.append(pd.to_datetime(dt_str))
        future_df_base = pd.DataFrame({'ds': future_dates})
        future_df_base = add_features(future_df_base)

        all_forecasts = []
        all_metrics = []

        # Forecast loop per metric
        for metric_name, time_suffix in target_metrics_info:
            st.markdown(f"## Forecast: {metric_name}")

            # Prepare historical data for this metric at specified time
            temp = df.copy()
            temp['ds'] = pd.to_datetime(temp['Date'].dt.strftime('%Y-%m-%d') + ' ' + time_suffix)
            if metric_name not in temp.columns:
                st.warning(f"Metric '{metric_name}' not found in data, skipping.")
                continue
            hist_df = temp[['ds', metric_name]].dropna()
            hist_df = hist_df.rename(columns={metric_name: 'y'}).sort_values('ds')

            if hist_df.empty or hist_df['y'].isna().all():
                st.warning(f"No historical data for metric {metric_name}, skipping.")
                continue

            # Add features for Prophet + XGBoost
            hist_df = add_features(hist_df)
            hist_df['y_lag1'] = hist_df['y'].shift(1)
            hist_df['y_rolling_mean_3'] = hist_df['y'].rolling(window=3).mean().shift(1)
            hist_df = hist_df.dropna()

            # Prophet model
            m = Prophet()
            regressors = ['year', 'month', 'day', 'dayofweek', 'hour', 
                          'hour_sin', 'hour_cos', 'is_holiday', 'day_after_holiday',
                          'y_lag1', 'y_rolling_mean_3']
            for reg in regressors:
                m.add_regressor(reg)
            m.fit(hist_df.rename(columns={'ds': 'ds', 'y': 'y'})[ ['ds', 'y'] + regressors ])

            # Prepare future data for this time suffix only
            future = future_df_base[future_df_base['ds'].dt.strftime('%H:%M:%S') == time_suffix].copy()

            # We set lag features for future from last known historical values:
            last_y = hist_df['y'].iloc[-1]
            last_y_lag1 = hist_df['y_lag1'].iloc[-1]
            last_roll = hist_df['y_rolling_mean_3'].iloc[-1]

            future['y_lag1'] = last_y
            future['y_rolling_mean_3'] = last_roll

            forecast_prophet = m.predict(future)

            # XGBoost model
            # Train on historical data
            X_train = hist_df[regressors]
            y_train = hist_df['y']

            xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
            xgb_model.fit(X_train, y_train)

            # Predict XGBoost on future
            X_future = future[regressors]
            xgb_pred = xgb_model.predict(X_future)

            # Hybrid forecast: average Prophet & XGBoost predictions
            hybrid_pred = (forecast_prophet['yhat'].values + xgb_pred) / 2
            # Round and clip negatives to zero
            hybrid_pred = np.clip(np.round(hybrid_pred), 0, None).astype(int)

            # Build output DataFrame for forecast
            forecast_df = future[['ds']].copy()
            forecast_df['Prophet'] = np.round(forecast_prophet['yhat'], 0).astype(int)
            forecast_df['XGBoost'] = np.round(xgb_pred, 0).astype(int)
            forecast_df['Hybrid_Forecast'] = hybrid_pred

            # Evaluate MAE on training set by predicting training data for XGBoost & Prophet
            prophet_train_pred = m.predict(hist_df)[ 'yhat' ]
            xgb_train_pred = xgb_model.predict(X_train)
            hybrid_train_pred = (prophet_train_pred + xgb_train_pred) / 2

            mae_prophet = mean_absolute_error(hist_df['y'], prophet_train_pred)
            mae_xgb = mean_absolute_error(hist_df['y'], xgb_train_pred)
            mae_hybrid = mean_absolute_error(hist_df['y'], hybrid_train_pred)

            st.markdown(f"**MAE on training data:**")
            st.write(f"Prophet: {mae_prophet:.2f}, XGBoost: {mae_xgb:.2f}, Hybrid: {mae_hybrid:.2f}")

            # Plot results: last 30 days + forecast
            recent_hist = hist_df[['ds', 'y']].copy()
            recent_hist = recent_hist[recent_hist['ds'] >= (last_date - pd.Timedelta(days=30))]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=recent_hist['ds'], y=recent_hist['y'],
                                     mode='lines+markers', name='Historical'))

            fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['Hybrid_Forecast'],
                                     mode='lines+markers', name='Hybrid Forecast'))

            fig.update_layout(title=f"{metric_name} - Last 30 Days + 7-Day Forecast",
                              xaxis_title='Date',
                              yaxis_title='Beds',
                              height=400)

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Forecast Table")
            st.dataframe(forecast_df.rename(columns={'ds': 'Date'}).set_index('Date'))

            all_forecasts.append(forecast_df.assign(Metric=metric_name))
            all_metrics.append({
                'Metric': metric_name,
                'MAE_Prophet': mae_prophet,
                'MAE_XGBoost': mae_xgb,
                'MAE_Hybrid': mae_hybrid
            })

        if all_metrics:
            st.markdown("## Summary of MAE for all metrics")
            st.table(pd.DataFrame(all_metrics).set_index('Metric'))

    except Exception as e:
        st.error(f"Error processing file: {e}")
