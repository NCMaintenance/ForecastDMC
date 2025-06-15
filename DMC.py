import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from datetime import timedelta

st.set_page_config(layout="wide")

# Feature mappings for hospital systems
all_hospital_features = {
    'IP': ['admissions', 'discharges', 'inpatient_beds', 'icu_beds'],
    'OP': ['op_visits', 'ed_visits', 'lab_tests']
}

# Add lag features
def add_lag_features(df, column, lags):
    for lag in lags:
        df[f"{column}_lag_{lag}"] = df[column].shift(lag)
    return df

# Load file
@st.cache_data
def load_data(file):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
    df = df.dropna(subset=['ds'])
    df = df.sort_values('ds')
    return df

# Generate time-related and lag features
def prepare_features(df):
    df = df.copy()
    df['dayofweek'] = df['ds'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['is_month_start'] = df['ds'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['ds'].dt.is_month_end.astype(int)
    df = add_lag_features(df, 'y', lags=[1, 2, 3, 7])
    return df

# XGBoost training
def train_xgb_model_only(df, all_hospital_features, metric_name):
    time_features = ['dayofweek', 'is_weekend', 'is_month_start', 'is_month_end']
    lag_features = ['y_lag_1', 'y_lag_2', 'y_lag_3', 'y_lag_7']
    all_features = time_features + lag_features

    # Add hospital features + lags
    for category, features in all_hospital_features.items():
        if metric_name in features:
            for feat in features:
                if feat in df.columns:
                    all_features.append(feat)
                    for lag in [1, 2, 3, 7]:
                        df[f"{feat}_lag_{lag}"] = df[feat].shift(lag)
                        all_features.append(f"{feat}_lag_{lag}")

    # Drop NA and prepare train set
    usable_features = [f for f in all_features if f in df.columns and df[f].notna().mean() > 0.8]
    feature_data = df[['ds', 'y'] + usable_features].dropna()
    if len(feature_data) < 20:
        raise ValueError("Too little clean data for training.")

    X = feature_data[usable_features]
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
    return xgb_model, scaler, usable_features, feature_data

# Forecast future values
def forecast_xgb_model(xgb_model, scaler, feature_data, features, steps=14):
    last_data = feature_data.copy()
    forecasts = []

    for _ in range(steps):
        last_row = last_data.iloc[-1]
        next_date = last_row['ds'] + timedelta(days=1)
        new_row = {'ds': next_date}
        new_row['dayofweek'] = next_date.dayofweek
        new_row['is_weekend'] = int(next_date.dayofweek in [5, 6])
        new_row['is_month_start'] = int(next_date.is_month_start)
        new_row['is_month_end'] = int(next_date.is_month_end)

        for lag in [1, 2, 3, 7]:
            if len(forecasts) >= lag:
                new_row[f'y_lag_{lag}'] = forecasts[-lag]
            else:
                idx = -lag
                new_row[f'y_lag_{lag}'] = last_data.iloc[idx]['y'] if len(last_data) >= -idx else np.nan

        for f in features:
            if '_lag_' in f and f not in new_row:
                base = f.split('_lag_')[0]
                lag = int(f.split('_lag_')[1])
                if base in last_data.columns and len(last_data) >= lag:
                    new_row[f] = last_data.iloc[-lag][base]
                else:
                    new_row[f] = np.nan
            elif f not in new_row:
                new_row[f] = last_data[f].iloc[-1] if f in last_data.columns else 0

        row_df = pd.DataFrame([new_row])
        X = row_df[features]
        X_scaled = scaler.transform(X)
        pred = xgb_model.predict(X_scaled)[0]
        new_row['y'] = pred
        forecasts.append(pred)
        last_data = pd.concat([last_data, pd.DataFrame([new_row])], ignore_index=True)

    return forecasts

# Evaluate forecast accuracy
def evaluate_xgb_model(xgb_model, scaler, features, df, test_size=14):
    test_data = df.tail(test_size).dropna()
    if len(test_data) == 0:
        return {'XGBoost': {'MAE': 0, 'RMSE': 0, 'MAPE': 0}}

    X_test = test_data[features]
    y_true = test_data['y']
    X_scaled = scaler.transform(X_test)
    y_pred = xgb_model.predict(X_scaled)

    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
    return {
        'XGBoost': {
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAPE': mape
        }
    }

# UI section
st.title("📊 Time Series Forecasting with XGBoost")
uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file:
    df = load_data(uploaded_file)
    metric_options = [col for col in df.columns if col != 'ds' and pd.api.types.is_numeric_dtype(df[col])]
    metric_name = st.selectbox("Select Target Column (y)", metric_options)

    df = df.rename(columns={metric_name: 'y'})
    df = prepare_features(df)

    try:
        xgb_model, scaler, features, feature_data = train_xgb_model_only(df, all_hospital_features, metric_name)
        forecasts = forecast_xgb_model(xgb_model, scaler, feature_data, features)
        metrics = evaluate_xgb_model(xgb_model, scaler, features, df)

        st.subheader("📈 Forecast Metrics (last 14 actual points)")
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"{metrics['XGBoost']['MAE']:.2f}")
        col2.metric("RMSE", f"{metrics['XGBoost']['RMSE']:.2f}")
        col3.metric("MAPE", f"{metrics['XGBoost']['MAPE']:.2f}%")

        st.subheader("📅 Forecast Plot")
        fig, ax = plt.subplots(figsize=(12, 6))
        df['y'].plot(ax=ax, label='Actual')
        future_dates = pd.date_range(start=df['ds'].max() + timedelta(days=1), periods=len(forecasts))
        ax.plot(future_dates, forecasts, label='Forecast', linestyle='--')
        ax.legend()
        st.pyplot(fig)

        st.subheader("📊 Feature Importances")
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': xgb_model.feature_importances_
        }).sort_values('Importance', ascending=False).head(15)
        st.dataframe(importance_df.reset_index(drop=True))
    except Exception as e:
        st.error(f"❌ Forecasting failed: {e}")
else:
    st.info("📁 Please upload a dataset to begin.")
