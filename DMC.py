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

# 1. Upload Section
st.header("1. Upload Your Data")
uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        st.success("File uploaded successfully!")
        st.subheader("Raw Data Preview:")
        st.dataframe(df.head())

        # Drop first column if unnecessary
        if df.shape[1] > 1:
            df = df.iloc[:, 1:]
            st.info("First column dropped as requested.")
        else:
            st.warning("Only one column detected, skipping drop.")

        # 🧹 Clean column names early (remove spaces from all column names)
        df.columns = [col.replace(" ", "") for col in df.columns]

        # Ensure Date column is present and properly formatted
        if 'Date' not in df.columns:
            st.error("Missing 'Date' column.")
            st.stop()
        df['Date'] = pd.to_datetime(df['Date'])

        # 🧠 Define your target metrics (adjust this list as needed)
        target_metrics_info = [
            ('Tracker8am', '08:00:00'),
            ('Tracker2pm', '14:00:00'),
            ('Tracker8pm', '20:00:00'),
            ('AdditionalCapacityOpenMorning', '08:00:00'),  # formerly had space
            ('TimeTotal_8am', '08:00:00'),
            ('TimeTotal_2pm', '14:00:00'),
            ('TimeTotal_8pm', '20:00:00'),
        ]
        target_columns = [m[0] for m in target_metrics_info]

        # 2. Hospital Selection
        st.header("2. Select Hospital for Forecasting")
        if 'Hospital' not in df.columns:
            st.error("Missing 'Hospital' column.")
            st.stop()

        hospitals = ['All Hospitals'] + sorted(df['Hospital'].unique())
        selected = st.selectbox("Select a Hospital:", hospitals)

        df_filtered = df.copy()
        if selected != 'All Hospitals':
            df_filtered = df_filtered[df_filtered['Hospital'] == selected]
        else:
            cols_to_sum = [c for c in target_columns if c in df_filtered.columns]
            df_filtered = df_filtered.groupby('Date')[cols_to_sum].sum().reset_index()

        if df_filtered.empty:
            st.warning("No data after filtering.")
            st.stop()

        st.subheader("Data Preview after Filtering:")
        st.dataframe(df_filtered.head())

        # 3. Forecasting
        st.header("3. Forecasting")
        st.info("Running hybrid Prophet + XGBoost model...")

        all_forecasts = pd.DataFrame()
        forecast_horizon_days = 7

        last_date = df_filtered['Date'].max()
        future_dates_list = []
        for i in range(1, forecast_horizon_days + 1):
            dt = last_date + pd.Timedelta(days=i)
            for hour in [8, 14, 20]:
                future_dates_list.append(dt.replace(hour=hour, minute=0, second=0))

        future_df_base = pd.DataFrame({'ds': pd.to_datetime(future_dates_list)})
        us_holidays = holidays.US()
        future_df_base['is_holiday'] = future_df_base['ds'].dt.date.map(lambda x: 1 if x in us_holidays else 0)
        future_df_base['day_after_holiday'] = future_df_base['is_holiday'].shift(1).fillna(0)

        for metric_name, time_suffix in target_metrics_info:
            if metric_name not in df_filtered.columns:
                st.warning(f"Column '{metric_name}' not found in dataset. Skipping.")
                continue

            st.write(f"⏳ Forecasting: **{metric_name}**")

            temp_df = df_filtered.copy()
            temp_df['ds'] = pd.to_datetime(temp_df['Date'].astype(str) + ' ' + time_suffix)
            prophet_df = temp_df[['ds', metric_name]].rename(columns={metric_name: 'y'}).dropna().set_index('ds')

            if prophet_df.empty:
                st.warning(f"No data for {metric_name}. Skipping.")
                continue

            # Feature engineering
            prophet_df['year'] = prophet_df.index.year
            prophet_df['month'] = prophet_df.index.month
            prophet_df['day'] = prophet_df.index.day
            prophet_df['dayofweek'] = prophet_df.index.dayofweek
            prophet_df['hour'] = prophet_df.index.hour
            prophet_df['hour_sin'] = np.sin(2 * np.pi * prophet_df['hour'] / 24)
            prophet_df['hour_cos'] = np.cos(2 * np.pi * prophet_df['hour'] / 24)
            prophet_df['y_lag1'] = prophet_df['y'].shift(1)
            prophet_df['y_rolling_mean'] = prophet_df['y'].rolling(3).mean().shift(1)
            prophet_df['is_holiday'] = prophet_df.index.date.map(lambda x: 1 if x in us_holidays else 0)
            prophet_df['day_after_holiday'] = prophet_df['is_holiday'].shift(1).fillna(0)

            prophet_df = prophet_df.dropna().reset_index()

            if prophet_df.empty:
                st.warning(f"Not enough data after feature engineering for {metric_name}.")
                continue

            m = Prophet()
            regressors = ['year', 'month', 'day', 'dayofweek', 'hour',
                          'hour_sin', 'hour_cos', 'is_holiday', 'day_after_holiday',
                          'y_lag1', 'y_rolling_mean']
            for reg in regressors:
                m.add_regressor(reg)

            m.fit(prophet_df[['ds', 'y'] + regressors])

            future_df = future_df_base.copy()
            future_df = future_df[future_df['ds'].dt.strftime('%H:%M:%S') == time_suffix]
            future_df['year'] = future_df['ds'].dt.year
            future_df['month'] = future_df['ds'].dt.month
            future_df['day'] = future_df['ds'].dt.day
            future_df['dayofweek'] = future_df['ds'].dt.dayofweek
            future_df['hour'] = future_df['ds'].dt.hour
            future_df['hour_sin'] = np.sin(2 * np.pi * future_df['hour'] / 24)
            future_df['hour_cos'] = np.cos(2 * np.pi * future_df['hour'] / 24)
            future_df['y_lag1'] = prophet_df['y'].iloc[-1]
            future_df['y_rolling_mean'] = prophet_df['y_rolling_mean'].iloc[-1]
            future_df['is_holiday'] = future_df['ds'].dt.date.map(lambda x: 1 if x in us_holidays else 0)
            future_df['day_after_holiday'] = future_df['is_holiday'].shift(1).fillna(0)

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
        st.dataframe(all_forecasts)

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
