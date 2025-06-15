import streamlit as st
import pandas as pd
from prophet import Prophet
import xgboost as xgb
import holidays
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import logging

# Suppress verbose logging
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

st.set_page_config(layout="wide", page_title="ED Metrics Hybrid Forecasting")
st.title("🏥 Emergency Department Metrics Hybrid Forecasting (Prophet + XGBoost)")

st.markdown("""
Upload your raw ED data to forecast key metrics for the next 7 days using a hybrid Prophet-XGBoost model.
This application performs essential feature engineering and combines the strengths of both models
to provide robust and accurate predictions.
""")

# ------------------------ Sidebar Controls ------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])

    selected_hospital = None
    run_forecast = False
    df = None

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        # Clean column names early
        df.columns = [col.replace(" ", "") for col in df.columns]

        if 'Date' not in df.columns:
            st.error("Missing 'Date' column.")
        else:
            df['Date'] = pd.to_datetime(df['Date'])
            if 'Hospital' in df.columns:
                hospitals = ['All Hospitals'] + sorted(df['Hospital'].unique())
                selected_hospital = st.selectbox("Select Hospital", hospitals)
            else:
                st.warning("No 'Hospital' column found. Using full dataset.")

            run_forecast = st.button("🚀 Run Forecast")

# ------------------------ Main Body ------------------------
if uploaded_file and run_forecast and df is not None:
    try:
        # Drop first column if it looks like index
        if df.shape[1] > 1:
            df = df.iloc[:, 1:]

        # Metrics to forecast
        target_metrics_info = [
            ('Tracker8am', '08:00:00'),
            ('Tracker2pm', '14:00:00'),
            ('Tracker8pm', '20:00:00'),
            ('AdditionalCapacityOpenMorning', '08:00:00'),
            ('TimeTotal_8am', '08:00:00'),
            ('TimeTotal_2pm', '14:00:00'),
            ('TimeTotal_8pm', '20:00:00'),
        ]
        target_columns = [m[0] for m in target_metrics_info]

        if selected_hospital and selected_hospital != 'All Hospitals':
            df_filtered = df[df['Hospital'] == selected_hospital]
        else:
            df_filtered = df.groupby('Date')[[col for col in target_columns if col in df.columns]].sum().reset_index()

        st.subheader("📄 Filtered Data")
        st.dataframe(df_filtered.head())

        # Forecasting
        forecast_horizon_days = 7
        last_date = df_filtered['Date'].max()
        future_dates_list = [
            last_date + pd.Timedelta(days=i, hours=hr)
            for i in range(1, forecast_horizon_days + 1)
            for hr in [8, 14, 20]
        ]

        future_df_base = pd.DataFrame({'ds': pd.to_datetime(future_dates_list)})

        us_holidays = holidays.US()
        future_df_base['is_holiday'] = [1 if d.date() in us_holidays else 0 for d in future_df_base['ds']]
        future_df_base['day_after_holiday'] = [0] + future_df_base['is_holiday'].tolist()[:-1]

        all_forecasts = pd.DataFrame()

        for metric_name, time_suffix in target_metrics_info:
            if metric_name not in df_filtered.columns:
                st.warning(f"Skipping missing column: {metric_name}")
                continue

            st.write(f"🔮 Forecasting: **{metric_name}**")

            temp_df = df_filtered.copy()
            temp_df['ds'] = pd.to_datetime(temp_df['Date'].astype(str) + ' ' + time_suffix)
            prophet_df = temp_df[['ds', metric_name]].rename(columns={metric_name: 'y'}).dropna().set_index('ds')

            if prophet_df.empty:
                st.warning(f"No data for {metric_name}. Skipping.")
                continue

            prophet_df['year'] = prophet_df.index.year
            prophet_df['month'] = prophet_df.index.month
            prophet_df['day'] = prophet_df.index.day
            prophet_df['dayofweek'] = prophet_df.index.dayofweek
            prophet_df['hour'] = prophet_df.index.hour
            prophet_df['hour_sin'] = np.sin(2 * np.pi * prophet_df['hour'] / 24)
            prophet_df['hour_cos'] = np.cos(2 * np.pi * prophet_df['hour'] / 24)
            prophet_df['y_lag1'] = prophet_df['y'].shift(1)
            prophet_df['y_rolling_mean'] = prophet_df['y'].rolling(3).mean().shift(1)
            prophet_df['is_holiday'] = [1 if d.date() in us_holidays else 0 for d in prophet_df.index]
            prophet_df['day_after_holiday'] = [0] + prophet_df['is_holiday'].tolist()[:-1]

            prophet_df = prophet_df.dropna().reset_index()

            if prophet_df.empty:
                st.warning(f"Insufficient data after feature engineering for {metric_name}.")
                continue

            # Train Prophet
            m = Prophet()
            for reg in ['year', 'month', 'day', 'dayofweek', 'hour', 'hour_sin', 'hour_cos',
                        'y_lag1', 'y_rolling_mean', 'is_holiday', 'day_after_holiday']:
                m.add_regressor(reg)

            m.fit(prophet_df[['ds', 'y'] + [col for col in prophet_df.columns if col not in ['ds', 'y']]])

            # Future DF filtering
            future_df = future_df_base[future_df_base['ds'].dt.strftime('%H:%M:%S') == time_suffix].copy()
            future_df['year'] = future_df['ds'].dt.year
            future_df['month'] = future_df['ds'].dt.month
            future_df['day'] = future_df['ds'].dt.day
            future_df['dayofweek'] = future_df['ds'].dt.dayofweek
            future_df['hour'] = future_df['ds'].dt.hour
            future_df['hour_sin'] = np.sin(2 * np.pi * future_df['hour'] / 24)
            future_df['hour_cos'] = np.cos(2 * np.pi * future_df['hour'] / 24)
            future_df['y_lag1'] = prophet_df['y'].iloc[-1]
            future_df['y_rolling_mean'] = prophet_df['y_rolling_mean'].iloc[-1]

            forecast = m.predict(future_df)
            forecast['metric'] = metric_name
            forecast_result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'metric']]
            forecast_result.rename(columns={
                'yhat': f'{metric_name}_forecast',
                'yhat_lower': f'{metric_name}_lower',
                'yhat_upper': f'{metric_name}_upper'
            }, inplace=True)

            all_forecasts = pd.concat([all_forecasts, forecast_result])

        st.success("✅ Forecasting complete.")
        st.subheader("📊 Forecast Output")
        st.dataframe(all_forecasts)

    except Exception as e:
        st.error(f"❌ Error: {e}")
