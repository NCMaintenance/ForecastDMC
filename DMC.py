import pandas as pd
import numpy as np
import xgboost as xgb
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from utils import create_all_hospitals_features, create_advanced_features, load_metric_data, METRICS

st.set_page_config(layout="wide")
st.title("XGBoost-Only Forecast App")

@st.cache_data

def load_all_hospitals_features():
    all_features = {}
    for metric in METRICS:
        df = load_metric_data(metric)
        if df is not None:
            all_features[metric] = create_all_hospitals_features(df)
    return all_features

@st.cache_data

def train_xgboost_model(prophet_df, all_hospital_features, metric_name):
    base_features = ['year', 'month', 'day', 'dayofweek', 'hour', 'week_of_year']
    cyclical_features = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos']
    lag_features = [f'y_lag{i}' for i in [1, 2, 3, 7, 14]]
    rolling_features = [f'y_rolling_mean_{w}' for w in [3, 7, 14, 30]] + \
                      [f'y_rolling_std_{w}' for w in [3, 7, 14, 30]] + \
                      [f'y_rolling_max_{w}' for w in [3, 7, 14, 30]] + \
                      [f'y_rolling_min_{w}' for w in [3, 7, 14, 30]]
    pct_features = ['y_pct_change_1d', 'y_pct_change_7d', 'y_volatility_7d']
    holiday_features = ['is_holiday', 'days_since_holiday', 'days_until_holiday']
    pattern_features = ['is_weekend', 'is_monday', 'is_friday', 'is_summer', 'is_winter']
    interaction_features = ['weekend_evening', 'monday_morning']

    all_features = base_features + cyclical_features + lag_features + rolling_features + \
                   pct_features + holiday_features + pattern_features + interaction_features

    if metric_name in all_hospital_features:
        system_features = ['system_trend_7d', 'system_trend_30d', 'system_capacity_utilization']
        system_df = all_hospital_features[metric_name]
        prophet_df = prophet_df.merge(system_df[['ds'] + system_features], on='ds', how='left')
        prophet_df[system_features] = prophet_df[system_features].bfill().ffill()
        all_features.extend(system_features)

    available_features = [f for f in all_features if f in prophet_df.columns and prophet_df[f].notna().sum() > 0.8 * len(prophet_df)]
    feature_data = prophet_df[['ds', 'y'] + available_features].copy().dropna()

    if len(feature_data) < 20:
        raise ValueError("Too much data lost after dropping NaN values")

    X = feature_data[available_features]
    y = feature_data['y']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    train_size = int(0.8 * len(X_scaled))
    X_train, X_val = X_scaled[:train_size], X_scaled[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=20
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    return model, scaler, available_features, feature_data

def evaluate_xgboost_model(model, scaler, features, df, test_size=14):
    df = df.dropna()
    if len(df) < test_size:
        test_size = max(1, len(df) // 4)
    test_data = df.tail(test_size).copy()
    if test_data.empty:
        return {'XGBoost': {'MAE': 0, 'RMSE': 0, 'MAPE': 0}}

    X_test = test_data[features].copy().fillna(0)
    X_test_scaled = scaler.transform(X_test)
    y_true = test_data['y'].values
    y_pred = model.predict(X_test_scaled)

    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
    return {
        'XGBoost': {
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAPE': mape
        }
    }

# --- Sidebar and user input ---
metric_name = st.sidebar.selectbox("Select a metric", METRICS)
df = load_metric_data(metric_name)
if df is None:
    st.error("Failed to load data for selected metric.")
    st.stop()

hospital_id = st.sidebar.selectbox("Select a hospital", df['hospital_id'].unique())
df_filtered = df[df['hospital_id'] == hospital_id].copy()
prophet_df = create_advanced_features(df_filtered)

if st.sidebar.button("Run Enhanced Forecast"):
    all_hospital_features = load_all_hospitals_features()

    try:
        model, scaler, features, cleaned_data = train_xgboost_model(prophet_df, all_hospital_features, metric_name)
        metrics = evaluate_xgboost_model(model, scaler, features, cleaned_data)

        st.subheader("Forecast Accuracy")
        st.json(metrics)

    except Exception as e:
        st.error(f"Training failed: {e}")
