import streamlit as st
import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
import plotly.graph_objs as go

# Page configuration
st.set_page_config(page_title="7-Day Forecasting", layout="wide")
st.title("📈 7-Day Rolling Forecasting App")
st.markdown("Upload your hospital tracker data to generate accurate 7-day forecasts with real rolling evaluation.")

# File upload
uploaded_file = st.file_uploader("📤 Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file:
    # Load file
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    if "Date" not in df.columns or "Hospital" not in df.columns:
        st.error("The file must contain at least 'Date' and 'Hospital' columns.")
        st.stop()

    # Parse date
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    hospitals = df["Hospital"].unique().tolist()

    # Hospital selector
    hospital_choice = st.selectbox("🏥 Select a Hospital", ["All"] + hospitals)
    if hospital_choice != "All":
        df = df[df["Hospital"] == hospital_choice]

    # Fill missing
    df["Tracker8pm"].fillna(method="ffill", inplace=True)

    # Targets to forecast
    targets = [
        'Tracker8am', 'Tracker2pm', 'Tracker8pm',
        'AdditionalCapacityOpen Morning',
        'TimeTotal_8am', 'TimeTotal_2pm', 'TimeTotal_8pm'
    ]
    df = df.dropna(subset=targets, how="all")

    # Add time-based features
    df["dayofweek"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["weekofyear"] = df["Date"].dt.isocalendar().week.astype(int)

    # Convert categoricals to category
    for col in ['Hospital', 'Hospital Group Name', 'DayGAR']:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Lag and rolling features
    lags = [1, 2, 3, 5, 7]
    windows = [3, 7]
    for col in targets:
        for lag in lags:
            df[f"{col}_lag_{lag}"] = df.groupby("Hospital")[col].shift(lag)
        for win in windows:
            df[f"{col}_rollmean_{win}"] = df.groupby("Hospital")[col].shift(1).rolling(win).mean()

    df = df.dropna()

    # One-hot encode categoricals
    df_encoded = pd.get_dummies(df, drop_first=True)

    # Define features AFTER encoding
    feature_cols = [c for c in df_encoded.columns if c not in targets + ['Date']]
    train = df_encoded.iloc[:-7]
    test = df_encoded.iloc[-7:]

    X_train = train[feature_cols]
    y_train = train[targets]
    X_test = test[feature_cols]
    y_test = test[targets]

    # Model setup
    model = MultiOutputRegressor(LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42))
    model.fit(X_train, y_train)

    # Rolling 1-day forecast for 7 days
    preds, actuals = [], []
    for i in range(7):
        x_row = X_test.iloc[[i]]
        pred = model.predict(x_row)[0]
        preds.append(pred)
        actuals.append(y_test.iloc[i].values)

    forecast_df = pd.DataFrame(preds, columns=targets, index=test["Date"])
    actual_df = pd.DataFrame(actuals, columns=targets, index=test["Date"])

    # MAE per variable
    mae = {col: mean_absolute_error(actual_df[col], forecast_df[col]) for col in targets}

    # Output section
    st.subheader("📋 Forecast Table")
    st.dataframe(forecast_df.style.format("{:.1f}"))

    st.subheader("📊 MAE Per Variable")
    st.table(pd.DataFrame(mae.items(), columns=["Target", "MAE"]).set_index("Target").style.format("{:.2f}"))

    st.subheader("📈 Forecast vs Actual (Plotly Charts)")
    for col in targets:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df[col], mode='lines+markers', name='Forecast'))
        fig.add_trace(go.Scatter(x=actual_df.index, y=actual_df[col], mode='lines+markers', name='Actual'))
        fig.update_layout(title=f"{col}", xaxis_title="Date", yaxis_title=col)
        st.plotly_chart(fig, use_container_width=True)
