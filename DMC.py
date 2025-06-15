import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import holidays
import xgboost as xgb
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="🏥 ED Hybrid Forecasting")

# Sidebar controls
with st.sidebar:
    st.title("🏥 ED Hybrid Forecasting")
    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])
    hospital_filter = None
    run_button = st.button("Run Forecast")

st.markdown("""
# Emergency Department Hybrid Forecasting (Prophet + XGBoost)

Upload your ED data, select hospital, then run the hybrid forecast for 7 days ahead.  
Forecasts are non-negative rounded bed counts.
""")

if uploaded_file and run_button:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.replace(" ", "")  # Remove spaces from columns

    if "Date" not in df.columns or "Hospital" not in df.columns:
        st.error("Your data must contain 'Date' and 'Hospital' columns.")
        st.stop()

    df["Date"] = pd.to_datetime(df["Date"])

    hospitals = df["Hospital"].unique().tolist()
    hospital_filter = st.sidebar.selectbox("Select Hospital", ["All"] + hospitals)

    if hospital_filter != "All":
        df = df[df["Hospital"] == hospital_filter]

    st.write("### Raw Data Sample")
    st.dataframe(df.head())

    # Define target metrics and their associated forecast times
    target_metrics_info = [
        ("Tracker8am", "08:00:00"),
        ("Tracker2pm", "14:00:00"),
        ("Tracker8pm", "20:00:00"),
        ("AdditionalCapacityOpenMorning", "09:00:00"),
        ("TimeTotal_8am", "08:00:00"),
        ("TimeTotal_2pm", "14:00:00"),
        ("TimeTotal_8pm", "20:00:00")
    ]

    # Setup Ireland holidays with St. Brigid's Day (first Monday in Feb)
    years = list(df["Date"].dt.year.unique()) + [datetime.now().year + 1]
    ie_holidays = holidays.country_holidays("IE", years=years)
    for y in years:
        feb1 = datetime(y, 2, 1)
        st_brigid = feb1 + timedelta(days=(7 - feb1.weekday()) % 7)  # First Monday in Feb
        ie_holidays[st_brigid] = "St. Brigid's Day"

    forecast_horizon = 7
    results = {}
    maes = {}

    for metric, time_str in target_metrics_info:
        if metric not in df.columns:
            st.warning(f"Column '{metric}' missing in data, skipping.")
            continue

        st.header(f"Forecast for {metric}")

        # Prepare the dataframe for this metric
        temp = df[["Date", metric]].copy()
        temp["ds"] = pd.to_datetime(temp["Date"].astype(str) + " " + time_str)
        temp = temp.dropna(subset=[metric])
        temp = temp.rename(columns={metric: "y"})
        temp["y"] = temp["y"].clip(lower=0).round()

        # Feature engineering
        temp["year"] = temp["ds"].dt.year
        temp["month"] = temp["ds"].dt.month
        temp["day"] = temp["ds"].dt.day
        temp["dayofweek"] = temp["ds"].dt.dayofweek
        temp["hour"] = temp["ds"].dt.hour
        temp["hour_sin"] = np.sin(2 * np.pi * temp["hour"] / 24)
        temp["hour_cos"] = np.cos(2 * np.pi * temp["hour"] / 24)
        temp["is_holiday"] = temp["ds"].dt.date.apply(lambda d: 1 if d in ie_holidays else 0)
        temp["day_after_holiday"] = temp["ds"].dt.date.apply(lambda d: 1 if (d - timedelta(days=1)) in ie_holidays else 0)
        temp["y_lag1"] = temp["y"].shift(1)
        temp["y_rolling_mean"] = temp["y"].rolling(3).mean().shift(1)
        temp = temp.dropna()

        if temp.shape[0] < 10:
            st.warning(f"Not enough data to forecast {metric} (need >= 10 rows after preprocessing).")
            continue

        # Define regressors/features
        regressors = ["year", "month", "day", "dayofweek", "hour",
                      "hour_sin", "hour_cos", "is_holiday", "day_after_holiday",
                      "y_lag1", "y_rolling_mean"]

        # --------------------------
        # 1. Fit Prophet model
        # --------------------------
        m = Prophet()
        for r in regressors:
            m.add_regressor(r)
        m.fit(temp[["ds", "y"] + regressors])

        # --------------------------
        # 2. Fit XGBoost model
        # --------------------------
        X = temp[regressors]
        y = temp["y"]
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            objective="reg:squarederror",
            random_state=42,
        )
        xgb_model.fit(X, y)

        # --------------------------
        # 3. Prepare future dataframe for 7 days
        # --------------------------
        last_date = temp["ds"].max()
        future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_horizon + 1)]
        future = pd.DataFrame({"ds": [d.replace(hour=int(time_str[:2]), minute=0, second=0) for d in future_dates]})

        future["year"] = future["ds"].dt.year
        future["month"] = future["ds"].dt.month
        future["day"] = future["ds"].dt.day
        future["dayofweek"] = future["ds"].dt.dayofweek
        future["hour"] = future["ds"].dt.hour
        future["hour_sin"] = np.sin(2 * np.pi * future["hour"] / 24)
        future["hour_cos"] = np.cos(2 * np.pi * future["hour"] / 24)
        future["is_holiday"] = future["ds"].dt.date.apply(lambda d: 1 if d in ie_holidays else 0)
        future["day_after_holiday"] = future["ds"].dt.date.apply(lambda d: 1 if (d - timedelta(days=1)) in ie_holidays else 0)

        # Use last known lag and rolling mean values for features
        future["y_lag1"] = temp["y"].iloc[-1]
        future["y_rolling_mean"] = temp["y_rolling_mean"].iloc[-1]

        # --------------------------
        # 4. Predict with Prophet
        # --------------------------
        prophet_forecast = m.predict(future)
        prophet_forecast["yhat"] = prophet_forecast["yhat"].clip(lower=0).round()

        # --------------------------
        # 5. Predict with XGBoost
        # --------------------------
        xgb_pred = xgb_model.predict(future[regressors])
        xgb_pred = np.clip(np.round(xgb_pred), 0, None)

        # --------------------------
        # 6. Hybrid forecast: weighted average
        # --------------------------
        w_prophet = 0.6
        w_xgb = 0.4
        hybrid_pred = w_prophet * prophet_forecast["yhat"] + w_xgb * xgb_pred
        hybrid_pred = np.clip(np.round(hybrid_pred), 0, None)

        # Prepare output DataFrame
        forecast_df = future[["ds"]].copy()
        forecast_df["Prophet"] = prophet_forecast["yhat"].values
        forecast_df["XGBoost"] = xgb_pred
        forecast_df["Hybrid"] = hybrid_pred.astype(int)

        # --------------------------
        # 7. Calculate MAE on last 7 days (backtest)
        # --------------------------
        if temp.shape[0] > 14:
            test_data = temp.iloc[-7:]
            test_X = test_data[regressors]
            test_y = test_data["y"]

            pred_prophet_back = m.predict(test_data[["ds"] + regressors])["yhat"].clip(lower=0).round()
            pred_xgb_back = xgb_model.predict(test_X)
            pred_xgb_back = np.clip(np.round(pred_xgb_back), 0, None)

            hybrid_back = w_prophet * pred_prophet_back + w_xgb * pred_xgb_back
            hybrid_back = np.clip(np.round(hybrid_back), 0, None)

            mae_prophet = np.mean(np.abs(test_y - pred_prophet_back))
            mae_xgb = np.mean(np.abs(test_y - pred_xgb_back))
            mae_hybrid = np.mean(np.abs(test_y - hybrid_back))

            maes[metric] = {
                "Prophet MAE": round(mae_prophet, 2),
                "XGBoost MAE": round(mae_xgb, 2),
                "Hybrid MAE": round(mae_hybrid, 2),
            }
        else:
            maes[metric] = {
                "Prophet MAE": "N/A",
                "XGBoost MAE": "N/A",
                "Hybrid MAE": "N/A",
            }

        results[metric] = forecast_df

        # --------------------------
        # 8. Plot results
        # --------------------------
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=forecast_df["ds"], y=forecast_df["Prophet"],
            mode="lines+markers", name="Prophet Forecast"
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df["ds"], y=forecast_df["XGBoost"],
            mode="lines+markers", name="XGBoost Forecast"
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df["ds"], y=forecast_df["Hybrid"],
            mode="lines+markers", name="Hybrid Forecast", line=dict(width=3)
        ))
        fig.update_layout(title=f"Forecast Comparison for {metric}",
                          xaxis_title="Date",
                          yaxis_title="Beds (Rounded, Non-negative)")

        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(forecast_df.rename(columns={"ds": "Date"}))
        st.write("MAE (last 7 days backtest):", maes[metric])

else:
    st.info("Upload your Excel file and press 'Run Forecast' to start.")
