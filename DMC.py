import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import os

# Try to import TimesFM (placeholder - update when actual package is known)
try:
    from timesfm import TimesFM  # <-- Replace if actual import path differs
    st.success("✅ `timesfm` library found! Attempting actual model inference.")
    ACTUAL_TIMESFM_AVAILABLE = True
except ImportError:
    st.warning("⚠️ `timesfm` library not found. Please install it (`pip install timesfm`) to run actual TimesFM. Falling back to simulated forecast.")
    ACTUAL_TIMESFM_AVAILABLE = False
except Exception as e:
    st.error(f"❌ Error importing `timesfm`: {e}")
    ACTUAL_TIMESFM_AVAILABLE = False

def run_timesfm_forecast(data: pd.Series, horizon: int) -> pd.DataFrame:
    if not ACTUAL_TIMESFM_AVAILABLE:
        st.error("TimesFM library not available.")
        return pd.DataFrame(columns=['Date', 'Forecasted Value'])

    st.info("🚀 Running TimesFM model... (This may take time)")
    try:
        input_data = data.values.reshape(1, -1)  # (1, seq_length)

        # --- Placeholder for actual TimesFM inference ---
        # Replace the following with actual model init and call
        st.warning("⚠️ Replace this section with real TimesFM inference code.")
        # Example: forecast_values = tsm.forecast(input_data)[0, :, 0]

        # --- Fallback to simulated forecast until TimesFM API is ready ---
        return simulate_forecast_fallback(data, horizon)

    except Exception as e:
        st.error(f"❌ Error during TimesFM inference: {e}")
        return simulate_forecast_fallback(data, horizon)

def simulate_forecast_fallback(data: pd.Series, horizon: int, seed: int = 42) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame(columns=['Date', 'Forecasted Value'])

    np.random.seed(seed)
    last_date = data.index[-1]
    freq = pd.infer_freq(data.index)

    if freq:
        try:
            if 'D' in freq:
                interval = timedelta(days=int(freq.replace('D', ''))) if freq != 'D' else timedelta(days=1)
            elif 'H' in freq:
                interval = timedelta(hours=int(freq.replace('H', ''))) if freq != 'H' else timedelta(hours=1)
            elif 'Min' in freq:
                interval = timedelta(minutes=int(freq.replace('Min', ''))) if freq != 'Min' else timedelta(minutes=1)
            elif 'M' in freq:
                interval = timedelta(days=30)
            else:
                interval = timedelta(days=1)
        except:
            interval = timedelta(days=1)
    else:
        if len(data) > 1:
            interval = data.index.to_series().diff().dropna().min()
            if pd.isnull(interval):
                interval = timedelta(days=1)
        else:
            interval = timedelta(days=1)

    forecast_dates = [last_date + (interval * (i + 1)) for i in range(horizon)]
    diff_mean = data.diff().mean() if len(data) > 1 else 0
    if pd.isna(diff_mean):
        diff_mean = 0

    last_value = data.iloc[-1]
    forecasted_values = [
        last_value + (i + 1) * diff_mean + np.random.normal(0, 0.05 * last_value)
        for i in range(horizon)
    ]

    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Forecasted Value': forecasted_values
    }).set_index('Date')
    return forecast_df

# --- Streamlit UI ---
st.set_page_config(layout="centered", page_title="TimesFM Forecasting App", page_icon="📈")
st.title("📈 Time Series Forecasting with TimesFM")

st.markdown("""
Upload a CSV with a datetime column and a numerical value column.  
We'll use the `TimesFM` model if available; otherwise, we simulate a forecast.
""")

# --- File Upload ---
st.header("1. Upload Your Data")
uploaded_file = st.file_uploader("Upload CSV", type="csv")

df = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ CSV uploaded successfully!")
        st.subheader("Preview")
        st.write(df.head())
    except Exception as e:
        st.error(f"Error reading file: {e}")

# --- Configuration ---
st.header("2. Configure Forecasting")
date_col = st.selectbox("Select Date Column", [''] + (df.columns.tolist() if df is not None else []))
value_col = st.selectbox("Select Value Column", [''] + (df.select_dtypes(include=np.number).columns.tolist() if df is not None else []))
horizon = st.number_input("Forecast Horizon (periods)", min_value=1, value=7, step=1)

# --- Forecasting Button ---
st.header("3. Forecast")
if st.button("Generate Forecast"):
    if df is None:
        st.warning("Please upload a CSV.")
    elif not date_col or not value_col:
        st.warning("Please select both date and value columns.")
    else:
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col, value_col])
            df = df.set_index(date_col).sort_index()
            df = df[~df.index.duplicated(keep='first')]

            series = pd.to_numeric(df[value_col], errors='coerce').dropna()

            if series.empty or len(series) < 2:
                st.error("Insufficient data after cleaning.")
            else:
                st.success("✅ Data ready for forecasting.")
                st.subheader("Historical Data")
                st.line_chart(series)

                st.subheader("Forecast")
                forecast_df = run_timesfm_forecast(series, horizon)
                if not forecast_df.empty:
                    st.write(forecast_df)
                    combined = pd.concat([series, forecast_df['Forecasted Value']])
                    st.line_chart(combined)
                else:
                    st.warning("Forecast could not be generated.")

        except Exception as e:
            st.error(f"Unexpected error: {e}")

# --- Footer Instructions ---
st.markdown("""
---
### How to Run Locally

1. **Save this script** as `timesfm_app.py`.
2. **Install dependencies**:
    ```bash
    pip install streamlit pandas numpy
    # Optional: pip install timesfm
    ```
3. **Run the app**:
    ```bash
    streamlit run timesfm_app.py
    ```

> ⚠️ Replace the TimesFM model section with the real API once it's available. See [TimesFM GitHub](https://github.com/google-research/timesfm) for updates.
""")
