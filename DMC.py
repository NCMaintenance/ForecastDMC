import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import xgboost as xgb
import holidays
from sklearn.metrics import mean_absolute_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------- CACHING FUNCTIONS ---------------

@st.cache_data
def load_data(file):
    df = pd.read_excel(file)
    return df

@st.cache_data
def get_ireland_holidays(years):
    ir_holidays = holidays.Ireland(years=years)
    # Add custom Feb holiday for example (Feb 15)
    for year in years:
        ir_holidays.append({f'{year}-02-15': "Custom Feb Bank Holiday"})
    return ir_holidays

@st.cache_data
def filter_hospital(df, hospital, target_cols):
    if hospital == "All Hospitals":
        filtered = df.groupby('Date')[target_cols].sum().reset_index()
    else:
        filtered = df[df['Hospital'] == hospital]
    return filtered

@st.cache_data
def train_prophet_model(df, metric, time_suffix, ir_holidays):
    temp_df = df.copy()
    temp_df['ds'] = pd.to_datetime(temp_df['Date'].astype(str) + ' ' + time_suffix)
    prophet_df = temp_df[['ds', metric]].rename(columns={metric: 'y'}).dropna().set_index('ds')

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
    prophet_df['is_holiday'] = prophet_df.index.date.map(lambda x: 1 if x in ir_holidays else 0)
    prophet_df['day_after_holiday'] = prophet_df['is_holiday'].shift(1).fillna(0)
    prophet_df = prophet_df.dropna().reset_index()

    m = Prophet()
    for reg in ['year', 'month', 'day', 'dayofweek', 'hour',
                'hour_sin', 'hour_cos', 'is_holiday', 'day_after_holiday',
                'y_lag1', 'y_rolling_mean']:
        m.add_regressor(reg)

    m.fit(prophet_df[['ds', 'y'] + [c for c in prophet_df.columns if c not in ['ds', 'y']]])

    return m, prophet_df

@st.cache_data
def train_xgboost_model(prophet_df):
    features = ['year', 'month', 'day', 'dayofweek', 'hour',
                'hour_sin', 'hour_cos', 'is_holiday', 'day_after_holiday',
                'y_lag1', 'y_rolling_mean']
    X = prophet_df[features]
    y = prophet_df['y']

    model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, features

# ------------- MAIN APP -------------------

st.set_page_config(layout="wide", page_title="ED Metrics Hybrid Forecasting")

st.title("🏥 Emergency Department Metrics Hybrid Forecasting (Prophet + XGBoost)")

# Sidebar: Upload + Hospital Select + Run button
st.sidebar.header("Upload & Settings")

uploaded_file = st.sidebar.file_uploader("Upload Excel file", type=['xlsx', 'xls'])
if uploaded_file:
    df_raw = load_data(uploaded_file)

    # Drop first col if multiple columns
    if df_raw.shape[1] > 1:
        df_raw = df_raw.iloc[:, 1:]

    if 'Date' not in df_raw.columns or 'Hospital' not in df_raw.columns:
        st.sidebar.error("Excel must include 'Date' and 'Hospital' columns.")
        st.stop()

    df_raw['Date'] = pd.to_datetime(df_raw['Date'])

    # Metrics - You can edit these to your actual metric names and times
    target_metrics_info = [
        ('Tracker8am', '08:00:00'),
        ('Tracker2pm', '14:00:00'),
        ('Tracker8pm', '20:00:00'),
        ('AdditionalCapacityOpenMorning', '08:00:00'),  # space removed as requested
        ('TimeTotal_8am', '08:00:00'),
        ('TimeTotal_2pm', '14:00:00'),
        ('TimeTotal_8pm', '20:00:00'),
    ]
    target_cols = [m[0] for m in target_metrics_info]

    # Filter metrics present in data
    target_cols = [col for col in target_cols if col in df_raw.columns]

    hospitals = ['All Hospitals'] + sorted(df_raw['Hospital'].unique())
    selected_hospital = st.sidebar.selectbox("Select Hospital", hospitals)

    if st.sidebar.button("Run Forecast"):

        # Filter by hospital
        df_filtered = filter_hospital(df_raw, selected_hospital, target_cols)

        if df_filtered.empty:
            st.warning("No data available after filtering.")
            st.stop()

        # Ireland Holidays with custom Feb holiday
        years = list(df_filtered['Date'].dt.year.unique())
        ir_holidays = get_ireland_holidays(years)

        forecast_horizon_days = 7
        last_date = df_filtered['Date'].max()

        # Prepare future dates for each time suffix
        future_dates_list = []
        for i in range(1, forecast_horizon_days + 1):
            dt = last_date + pd.Timedelta(days=i)
            for hour in [8, 14, 20]:
                future_dates_list.append(dt.replace(hour=hour, minute=0, second=0))
        future_df_base = pd.DataFrame({'ds': pd.to_datetime(future_dates_list)})
        future_df_base['is_holiday'] = future_df_base['ds'].dt.date.map(lambda x: 1 if x in ir_holidays else 0)
        future_df_base['day_after_holiday'] = future_df_base['is_holiday'].shift(1).fillna(0)

        all_results = []

        st.header(f"Forecast Results for {selected_hospital}")

        for metric_name, time_suffix in target_metrics_info:
            if metric_name not in df_filtered.columns:
                st.warning(f"Metric {metric_name} not found in data. Skipping.")
                continue

            st.subheader(f"Metric: {metric_name}")

            # Train Prophet model
            prophet_model, prophet_df = train_prophet_model(df_filtered, metric_name, time_suffix, ir_holidays)

            # Train XGBoost model
            xgb_model, xgb_features = train_xgboost_model(prophet_df)

            # Prepare future dataframe for this time_suffix
            future_df = future_df_base[future_df_base['ds'].dt.strftime('%H:%M:%S') == time_suffix].copy()
            future_df['year'] = future_df['ds'].dt.year
            future_df['month'] = future_df['ds'].dt.month
            future_df['day'] = future_df['ds'].dt.day
            future_df['dayofweek'] = future_df['ds'].dt.dayofweek
            future_df['hour'] = future_df['ds'].dt.hour
            future_df['hour_sin'] = np.sin(2 * np.pi * future_df['hour'] / 24)
            future_df['hour_cos'] = np.cos(2 * np.pi * future_df['hour'] / 24)
            # Use last known lag and rolling mean from training data
            future_df['y_lag1'] = prophet_df['y'].iloc[-1]
            future_df['y_rolling_mean'] = prophet_df['y_rolling_mean'].iloc[-1]

            # Prophet predictions
            prophet_preds = prophet_model.predict(future_df)
            prophet_preds = prophet_preds[['ds', 'yhat']].copy()

            # XGBoost predictions
            xgb_preds = xgb_model.predict(future_df[xgb_features])
            xgb_preds = np.round(xgb_preds).clip(min=0)

            # Hybrid prediction (average)
            hybrid_preds = ((prophet_preds['yhat'] + xgb_preds) / 2).round().clip(min=0)

            # Prepare results df
            results_df = pd.DataFrame({
                'DateTime': future_df['ds'],
                'Prophet Forecast': prophet_preds['yhat'].round().clip(min=0),
                'XGBoost Forecast': xgb_preds,
                'Hybrid Forecast': hybrid_preds
            })

            # Calculate MAE on historical (using last 7 days as holdout)
            test_mask = (prophet_df['ds'] > (prophet_df['ds'].max() - pd.Timedelta(days=7)))
            y_true = prophet_df.loc[test_mask, 'y']
            # Align features for XGBoost MAE on historical
            X_test = prophet_df.loc[test_mask, xgb_features]
            y_pred_xgb = xgb_model.predict(X_test)
            y_pred_xgb = np.round(y_pred_xgb).clip(min=0)

            prophet_mae = mean_absolute_error(y_true, prophet_df.loc[test_mask, 'y'])
            xgb_mae = mean_absolute_error(y_true, y_pred_xgb)
            hybrid_mae = (prophet_mae + xgb_mae) / 2  # Simplistic average MAE for demo

            st.markdown(f"**MAE (last 7 days historical):** Prophet = {prophet_mae:.2f}, XGBoost = {xgb_mae:.2f}, Hybrid = {hybrid_mae:.2f}")

            # Plotly chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=results_df['DateTime'], y=results_df['Prophet Forecast'],
                                     mode='lines+markers', name='Prophet'))
            fig.add_trace(go.Scatter(x=results_df['DateTime'], y=results_df['XGBoost Forecast'],
                                     mode='lines+markers', name='XGBoost'))
            fig.add_trace(go.Scatter(x=results_df['DateTime'], y=results_df['Hybrid Forecast'],
                                     mode='lines+markers', name='Hybrid'))

            fig.update_layout(
                title=f"Forecasts for {metric_name}",
                xaxis_title="DateTime",
                yaxis_title="Forecasted Value (Beds)",
                yaxis=dict(range=[0, max(results_df['Hybrid Forecast'].max(), 10)+5])
            )

            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(results_df.set_index('DateTime'))

            all_results.append((metric_name, results_df))

else:
    st.info("Please upload an Excel file to begin.")
