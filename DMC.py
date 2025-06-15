import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from datetime import timedelta

st.set_page_config(layout="wide")

# Define system-wide feature sets
all_hospital_features = {
    'IP': ['admissions', 'discharges', 'inpatient_beds', 'icu_beds'],
    'OP': ['op_visits', 'ed_visits', 'lab_tests']
}

def add_lag_features(df, column, lags):
    for lag in lags:
        df[f"{column}_lag_{lag}"] = df[column].shift(lag)
    return df

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values('ds')
    return df

def prepare_features(df):
    df = df.copy()
    df['dayofweek'] = df['ds'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['is_month_start'] = df['ds'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['ds'].dt.is_month_end.astype(int)
    df = add_lag_features(df, 'y', lags=[1, 2, 3, 7])
    return df

def train_xgb_model_only(prophet_df, all_hospital_features, metric_name):
    # Enhanced feature selection
    time_features = ['dayofweek', 'is_weekend', 'is_month_start', 'is_month_end']
    lag_features = ['y_lag_1', 'y_lag_2', 'y_lag_3', 'y_lag_7']
    all_features = time_features + lag_features

    # Add system-wide features
    if metric_name in all_hospital_features:
        system_features = all_hospital_features[metric_name]
        for feat in system_features:
            if feat in prophet_df.columns:
                all_features.append(feat)
                for lag in [1, 2, 3, 7]:
                    prophet_df[f"{feat}_lag_{lag}"] = prophet_df[feat].shift(lag)
                    all_features.append(f"{feat}_lag_{lag}")

    # Clean and align
    available_features = [f for f in all_features if f in prophet_df.columns and 
                          prophet_df[f].notna().mean() > 0.8]
    
    feature_data = prophet_df[['ds', 'y'] + available_features].dropna()
    if len(feature_data) < 20:
        raise ValueError("Too little clean data for training.")
    
    X = feature_data[available_features]
    y = feature_data['y']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=20
    )
    
    train_size = int(0.8 * len(X_scaled))
    X_train, X_val = X_scaled[:train_size], X_scaled[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    return xgb_model, scaler, available_features, feature_data

def forecast_xgb_model(xgb_model, scaler, feature_data, features, steps=14):
    last_data = feature_data.copy()
    forecasts = []
    for _ in range(steps):
        if len(last_data) == 0:
            break

        last_row = last_data.iloc[-1].copy()
        next_date = last_row['ds'] + timedelta(days=1)
        new_row = {'ds': next_date}

        # Date features
        new_row['dayofweek'] = next_date.dayofweek
        new_row['is_weekend'] = int(next_date.dayofweek in [5, 6])
        new_row['is_month_start'] = int(next_date.is_month_start)
        new_row['is_month_end'] = int(next_date.is_month_end)

        # Lag features
        for lag in [1, 2, 3, 7]:
            if len(forecasts) >= lag:
                new_row[f"y_lag_{lag}"] = forecasts[-lag]
            else:
                idx = -lag
                new_row[f"y_lag_{lag}"] = last_data.iloc[idx]['y'] if len(last_data) >= -idx else np.nan

        # System features
        for f in features:
            if '_lag_' in f:
                base = f.split('_lag_')[0]
                lag = int(f.split('_lag_')[1])
                if base in last_data.columns and len(last_data) >= lag:
                    new_row[f] = last_data.iloc[-lag][base]
                else:
                    new_row[f] = np.nan
            elif f not in new_row:
                if f in last_data.columns:
                    new_row[f] = last_data.iloc[-1][f]
                else:
                    new_row[f] = 0

        row_df = pd.DataFrame([new_row])
        X = row_df[features]
        X_scaled = scaler.transform(X)
        pred = xgb_model.predict(X_scaled)[0]
        new_row['y'] = pred
        forecasts.append(pred)
        last_data = pd.concat([last_data, pd.DataFrame([new_row])], ignore_index=True)
    return forecasts

def evaluate_xgb_model(xgb_model, scaler, features, prophet_df, test_size=14):
    test_data = prophet_df.tail(test_size).dropna()
    if len(test_data) == 0:
        return {'XGBoost': {'MAE': 0, 'RMSE': 0, 'MAPE': 0}}
    
    X_test = test_data[features]
    y_true = test_data['y']
    
    X_test_scaled = scaler.transform(X_test)
    xgb_forecast = xgb_model.predict(X_test_scaled)
    
    mape = np.mean(np.abs((y_true - xgb_forecast) / np.maximum(y_true, 1))) * 100
    
    return {
        'XGBoost': {
            'MAE': mean_absolute_error(y_true, xgb_forecast),
            'RMSE': np.sqrt(mean_squared_error(y_true, xgb_forecast)),
            'MAPE': mape
        }
    }

# Streamlit UI
st.title("Time Series Forecasting with Enhanced XGBoost")
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df = load_data(uploaded_file)
    metric_options = [col for col in df.columns if col not in ['ds']]
    metric_name = st.selectbox("Select Metric to Forecast (y):", metric_options)

    df = df.rename(columns={metric_name: 'y'})
    df = prepare_features(df)

    try:
        xgb_model, scaler, features, feature_data = train_xgb_model_only(df, all_hospital_features, metric_name)
        forecasts = forecast_xgb_model(xgb_model, scaler, feature_data, features, steps=14)
        metrics = evaluate_xgb_model(xgb_model, scaler, features, df)

        st.subheader("Forecasting Metrics (Last 14 Days)")
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"{metrics['XGBoost']['MAE']:.2f}")
        col2.metric("RMSE", f"{metrics['XGBoost']['RMSE']:.2f}")
        col3.metric("MAPE", f"{metrics['XGBoost']['MAPE']:.2f}%")

        st.subheader("Forecast Plot")
        fig, ax = plt.subplots(figsize=(12, 6))
        df['y'].plot(ax=ax, label='Actual')
        forecast_dates = pd.date_range(start=df['ds'].max() + timedelta(days=1), periods=len(forecasts))
        ax.plot(forecast_dates, forecasts, label='XGBoost Forecast', linestyle='--')
        ax.legend()
        st.pyplot(fig)

        st.subheader("Feature Importance (XGBoost)")
        importance = xgb_model.feature_importances_
        importance_df = pd.DataFrame({'Feature': features, 'Importance': importance})
        importance_df = importance_df.sort_values(by='Importance', ascending=False).head(15)
        st.dataframe(importance_df.reset_index(drop=True))
    except Exception as e:
        st.error(f"Training failed: {e}")
