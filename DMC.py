import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.utils.likelihood_models import GaussianLikelihood

st.set_page_config(page_title="ED & Trolley Forecasting", layout="wide")
st.title("Hospital ED & Trolley Forecasting App")

# Upload file
uploaded_file = st.file_uploader("Upload your Excel data file", type=["xlsx", "xls"])
if not uploaded_file:
    st.stop()

# Load and clean data
@st.cache_data
def load_data(file):
    df = pd.read_excel(file)
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df

df = load_data(uploaded_file)

# Select hospital
df["Hospital"] = df["Hospital"].astype(str).str.strip()
hospital_list = df["Hospital"].unique().tolist()
hospital_choice = st.selectbox("Select Hospital (or All)", options=["All"] + hospital_list)

if hospital_choice != "All":
    df = df[df["Hospital"] == hospital_choice]

# Coerce all non-Date columns to numeric
df_cleaned = df.copy()
for col in df_cleaned.columns:
    if col != "Date":
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors="coerce")

# Drop non-numeric columns from metric options
numeric_cols = df_cleaned.select_dtypes(include=["number"]).columns.tolist()
if not numeric_cols:
    st.error("No numeric columns available for forecasting.")
    st.stop()

metrics = st.multiselect(
    "Select metrics to forecast (numeric only)",
    options=numeric_cols,
    default=numeric_cols
)
if not metrics:
    st.warning("Select at least one metric.")
    st.stop()

# Create TimeSeries
@st.cache_data
def prepare_timeseries(df, metrics):
    return TimeSeries.from_dataframe(df, time_col="Date", value_cols=metrics)

ts = prepare_timeseries(df_cleaned, metrics)

# Sidebar: model params
input_chunk = st.sidebar.number_input("Input sequence length", min_value=7, max_value=60, value=14)
output_chunk = st.sidebar.number_input("Forecast horizon (days)", min_value=1, max_value=30, value=7)
epochs = st.sidebar.number_input("Training epochs", min_value=10, max_value=300, value=100, step=10)

# Split
train_ts = ts[:-output_chunk]
test_ts = ts[-output_chunk:]

# Train model
@st.cache_resource
def train_model(train_ts, input_chunk, output_chunk, epochs):
    model = NBEATSModel(
        input_chunk_length=input_chunk,
        output_chunk_length=output_chunk,
        n_epochs=epochs,
        likelihood=GaussianLikelihood(),
        random_state=42,
    )
    model.fit([train_ts], verbose=False)
    return model

with st.spinner("Training model..."):
    model = train_model(train_ts, input_chunk, output_chunk, epochs)

# Forecast
with st.spinner(f"Forecasting next {output_chunk} days..."):
    forecast = model.predict(n=output_chunk, num_samples=200, series=train_ts)

# Plot
st.subheader(f"Forecast for {hospital_choice if hospital_choice != 'All' else 'All Hospitals'}")

forecast_df = forecast.pd_dataframe().reset_index().rename(columns={'index': 'Date'})
train_df = train_ts.pd_dataframe().reset_index().rename(columns={'index': 'Date'})

for metric in metrics:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_df["Date"], train_df[metric], label="History")
    ax.plot(forecast_df["Date"], forecast_df[metric], label="Forecast")
    ax.set_title(f"{metric} Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel(metric)
    ax.legend()
    st.pyplot(fig)

st.success("Forecasting complete.")
