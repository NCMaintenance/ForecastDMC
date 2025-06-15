import streamlit as st
import pandas as pd
from datetime import timedelta

# Forecasting imports
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.utils.likelihood_models import GaussianLikelihood

# Plotting
import matplotlib.pyplot as plt

st.set_page_config(page_title="ED & Trolley Forecasting", layout="wide")
st.title("Hospital ED & Trolley Forecasting App")

# 1. Upload data
uploaded_file = st.file_uploader("Upload your Excel data file", type=["xlsx", "xls"])  
if uploaded_file is None:
    st.info("Please upload an Excel file with Date and the 7 metrics columns.")
    st.stop()

# 2. Load Data
@st.cache_data
def load_data(file) -> pd.DataFrame:
    df = pd.read_excel(file)
    df.columns = df.columns.str.strip()
    # Expect a 'Date' column
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df

df = load_data(uploaded_file)
st.write("#### Raw Data", df.head())

# 3. Select metrics to forecast
metrics = st.multiselect(
    "Select metrics to forecast (7 available)",
    options=[c for c in df.columns if c != 'Date'],
    default=[c for c in df.columns if c != 'Date']
)
if not metrics:
    st.warning("Please select at least one metric.")
    st.stop()

# 4. Prepare Darts TimeSeries
@st.cache_data
def prepare_timeseries(df: pd.DataFrame, metrics: list) -> TimeSeries:
    ts = TimeSeries.from_dataframe(df, time_col='Date', value_cols=metrics)
    return ts

ts = prepare_timeseries(df, metrics)

# 5. Model training parameters
input_chunk = st.sidebar.number_input(
    "Input sequence length (days)", min_value=7, max_value=60, value=14)
output_chunk = st.sidebar.number_input(
    "Forecast horizon (days)", min_value=1, max_value=30, value=7)
epochs = st.sidebar.number_input(
    "Training epochs", min_value=10, max_value=500, value=100, step=10)

# 6. Train/Test split
train_ts = ts[:-output_chunk]
test_ts  = ts[-output_chunk:]

# 7. Fit N-BEATS Model with probabilistic likelihood
@st.cache_resource
def train_model(train_ts, input_chunk, output_chunk, epochs):
    model = NBEATSModel(
        input_chunk_length=input_chunk,
        output_chunk_length=output_chunk,
        n_epochs=epochs,
        likelihood=GaussianLikelihood(),
        random_state=42
    )
    model.fit([train_ts], verbose=False)
    return model

with st.spinner("Training N-BEATS model..."):
    model = train_model(train_ts, input_chunk, output_chunk, epochs)

# 8. Forecast
with st.spinner(f"Forecasting next {output_chunk} days..."):
    # Generate probabilistic forecasts
    forecast_ts = model.predict(
        n=output_chunk,
        num_samples=200,
        series=train_ts
    )

# 9. Extract mean and intervals
forecast_df = forecast_ts.pd_dataframe().reset_index()
forecast_df.rename(columns={'index': 'Date'}, inplace=True)

# Use the quantile method to get intervals
quantiles = model.predict_interval(train_ts, n=output_chunk, num_samples=200)
q_lower = quantiles[0].pd_dataframe().reset_index().rename(columns={'index': 'Date'})
q_upper = quantiles[1].pd_dataframe().reset_index().rename(columns={'index': 'Date'})

# 10. Display results
st.write(f"## Forecast for next {output_chunk} days")
for metric in metrics:
    st.write(f"### {metric}")
    plt.figure(figsize=(10, 4))
    # Plot historical
    plt.plot(df['Date'], df[metric], label='Historical')
    # Plot forecast mean
    plt.plot(forecast_df['Date'], forecast_df[metric], label='Forecast')
    # Plot intervals
    plt.fill_between(
        q_lower['Date'],
        q_lower[metric],
        q_upper[metric],
        alpha=0.3,
        label='95% Prediction Interval'
    )
    plt.xlabel('Date')
    plt.ylabel(metric)
    plt.legend()
    st.pyplot(plt)

st.success("Forecasting complete!")

# End of Streamlit app
