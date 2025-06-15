import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta

# --- MUST be the very first Streamlit command ---
st.set_page_config(layout="centered", page_title="TimesFM Forecasting App", page_icon="📈")

# Try importing timesfm library
try:
    from timesfm import TimesFM
    ACTUAL_TIMESFM_AVAILABLE = True
except ImportError:
    ACTUAL_TIMESFM_AVAILABLE = False
except Exception:
    ACTUAL_TIMESFM_AVAILABLE = False

def simulate_forecast_fallback(data: pd.Series, horizon: int) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame(columns=['Date', 'Forecasted Value'])

    last_date = data.index[-1]
    freq = pd.infer_freq(data.index)
    if freq:
        if 'D' in freq:
            interval = timedelta(days=int(freq.replace('D',''))) if freq != 'D' else timedelta(days=1)
        elif 'H' in freq:
            interval = timedelta(hours=int(freq.replace('H',''))) if freq != 'H' else timedelta(hours=1)
        elif 'Min' in freq:
            interval = timedelta(minutes=int(freq.replace('Min',''))) if freq != 'Min' else timedelta(minutes=1)
        elif 'MS' in freq or 'M' in freq:
            interval = timedelta(days=30)
        else:
            interval = timedelta(days=1)
    else:
        interval = timedelta(days=(data.index[1] - data.index[0]).days) if len(data) > 1 else timedelta(days=1)
        if interval.days == 0:
            interval = data.index.to_series().diff().dropna().min()
            if pd.isna(interval):
                interval = timedelta(days=1)

    forecast_dates = [last_date + (interval * (i + 1)) for i in range(horizon)]

    diff_mean = data.diff().mean() if len(data) > 1 else 0
    if pd.isna(diff_mean):
        diff_mean = 0

    last_value = data.iloc[-1]
    forecasted_values = [last_value + (i + 1) * diff_mean + np.random.normal(0, 0.05 * last_value) for i in range(horizon)]

    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Forecasted Value': forecasted_values
    })
    forecast_df.set_index('Date', inplace=True)
    return forecast_df


def run_timesfm_forecast(data: pd.Series, horizon: int) -> pd.DataFrame:
    if not ACTUAL_TIMESFM_AVAILABLE:
        st.error("Cannot run actual TimesFM: Library not available.")
        return pd.DataFrame(columns=['Date', 'Forecasted Value'])

    st.info("Initializing and running actual TimesFM model... (This might take a moment)")
    try:
        # Placeholder for actual TimesFM initialization and forecast call
        # Replace this block with actual TimesFM usage based on their docs
        
        # For demonstration, fallback to simulation here
        st.info("TimesFM model would be loaded and invoked here. Replace this with real code.")
        return simulate_forecast_fallback(data, horizon)

    except Exception as e:
        st.error(f"Error during actual TimesFM forecasting: {e}. Falling back to simulation.")
        return simulate_forecast_fallback(data, horizon)


# Main app UI
st.title("📈 Time Series Forecasting with TimesFM")
st.markdown("""
Upload your CSV file, specify the date and value columns, and set a forecast horizon.
This app attempts to use the actual `timesfm` library for forecasting. If `timesfm` is not found,
it will fall back to a simulated forecast.
""")

# Show library import status message here (after page config)
if ACTUAL_TIMESFM_AVAILABLE:
    st.success("✅ `timesfm` library found! Attempting actual model inference.")
else:
    st.warning("⚠️ `timesfm` library not found. Please install it (`pip install timesfm`) to run actual TimesFM. Falling back to simulated forecast.")

# --- File Upload Section ---
st.header("1. Upload Your Data")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

df = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("CSV file uploaded successfully!")
        st.subheader("Preview of Your Data")
        st.write(df.head())

        st.subheader("Available Columns")
        st.write(df.columns.tolist())

    except Exception as e:
        st.error(f"Error reading CSV: {e}. Please ensure it's a valid CSV file.")

# --- Configuration Section ---
st.header("2. Configure Forecasting")

date_column_options = [''] + (df.columns.tolist() if df is not None else [])
date_col = st.selectbox(
    "Select the Date/Time Column",
    options=date_column_options,
    help="This column should contain your time series dates/timestamps."
)

value_column_options = [''] + (df.select_dtypes(include=np.number).columns.tolist() if df is not None else [])
value_col = st.selectbox(
    "Select the Value Column to Forecast",
    options=value_column_options,
    help="This column should contain the numerical values you want to predict."
)

forecast_horizon = st.number_input(
    "Enter Forecast Horizon (number of periods to predict)",
    min_value=1,
    value=7,
    step=1,
    help="How many future periods (e.g., days, months) do you want to forecast?"
)

# --- Perform Forecasting ---
st.header("3. Run Forecasting")

if st.button("Generate Forecast"):
    if df is None:
        st.warning("Please upload a CSV file first.")
    elif not date_col or not value_col:
        st.warning("Please select both a Date/Time Column and a Value Column.")
    else:
        try:
            # Preprocessing
            st.info("Preprocessing data...")
            temp_df = df.copy()
            temp_df[date_col] = pd.to_datetime(temp_df[date_col])
            temp_df.set_index(date_col, inplace=True)
            temp_df.sort_index(inplace=True)

            time_series_data = pd.to_numeric(temp_df[value_col], errors='coerce').dropna()

            if time_series_data.empty:
                st.error("The selected value column is empty or contains no valid numerical data after processing. Please check your data or column selection.")
            else:
                st.success("Data preprocessed successfully!")
                st.subheader("Historical Data Plot")
                st.line_chart(time_series_data)

                st.subheader("Forecasted Results")
                if ACTUAL_TIMESFM_AVAILABLE:
                    forecast_df_results = run_timesfm_forecast(time_series_data, forecast_horizon)
                else:
                    forecast_df_results = simulate_forecast_fallback(time_series_data, forecast_horizon)

                if not forecast_df_results.empty:
                    st.write(forecast_df_results)
                    st.subheader("Forecast Plot")
                    combined_data = pd.concat([time_series_data, forecast_df_results['Forecasted Value']])
                    st.line_chart(combined_data)
                else:
                    st.warning("No forecast could be generated. Please check your data or `timesfm` installation.")

        except KeyError:
            st.error("One of the selected columns was not found or accessible after processing. Please check column names.")
        except Exception as e:
            st.error(f"An unexpected error occurred during forecasting: {e}")

st.markdown("""
---
### How to Run This Application Locally:

1.  **Save the code:** Copy the entire code block above and save it as a Python file (e.g., `timesfm_app.py`) on your local machine.

2.  **Install necessary libraries:** Open your terminal or command prompt and run:
    ```bash
    pip install streamlit pandas numpy
    ```
    **For actual TimesFM functionality, you will also need the `timesfm` library.** Please refer to official documentation for exact installation instructions:
    ```bash
    pip install timesfm
    ```
    *(Note: TimesFM API usage may require downloading checkpoints or additional setup.)*

3.  **Run the Streamlit app:** In your terminal, navigate to the directory where you saved `timesfm_app.py` and run:
    ```bash
    streamlit run timesfm_app.py
    ```
    This will open the application in your web browser. Upload your CSV file, select your columns, and set the forecast horizon.
""")
