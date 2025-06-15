import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from darts import TimeSeries
from darts.models import InformerModel
from darts.metrics import mae
from darts.utils.timeseries_generation import datetime_attribute_timeseries

st.set_page_config(layout="wide")
st.title("🏥 Hospital Bed Forecasting using Transformers")

st.sidebar.header("Upload Excel File")
file = st.sidebar.file_uploader("Upload Excel file", type=["xlsx"])

if file:
    df = pd.read_excel(file)

    # Clean & preprocess
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Date'])

    # Hospital selection
    hospital = st.sidebar.selectbox("Select Hospital", df['Hospital'].unique())
    selected_df = df[df['Hospital'] == hospital].copy()
    selected_df = selected_df.sort_values('Date')

    # Target feature selection
    targets = [
        'Tracker8am',
        'Tracker2pm',
        'Tracker8pm',
        'AdditionalCapacityOpen Morning',
        'TimeTotal_8am',
        'TimeTotal_2pm',
        'TimeTotal_8pm'
    ]
    target_col = st.sidebar.selectbox("Select Variable to Forecast", targets)

    # Convert to TimeSeries
    series = TimeSeries.from_dataframe(selected_df, 'Date', target_col)

    # Create time covariates (day of week, month)
    day_of_week = datetime_attribute_timeseries(series, attribute="day_of_week", one_hot=True)
    month = datetime_attribute_timeseries(series, attribute="month", one_hot=True)
    covariates = day_of_week.stack(month)

    # Train/Test split
    split_date = selected_df['Date'].iloc[-30]
    train, test = series.split_after(split_date)
    cov_train, cov_test = covariates.split_after(split_date)

    st.subheader("📊 Historical Data")
    fig = px.line(selected_df, x='Date', y=target_col, title=f"{hospital}: {target_col} Over Time")
    st.plotly_chart(fig, use_container_width=True)

    st.sidebar.header("Model Parameters")
    input_chunk = st.sidebar.slider("Input Chunk Length", 14, 90, 30)
    output_chunk = st.sidebar.slider("Forecast Horizon (Days)", 7, 60, 14)

    st.info("⏳ Training Informer model... (may take a minute)")
    model = InformerModel(
        input_chunk_length=input_chunk,
        output_chunk_length=output_chunk,
        n_epochs=100,
        batch_size=32,
        model_name=f"informer_{target_col.replace(' ', '_')}",
        save_checkpoints=True,
        force_reset=True,
    )

    model.fit(train, future_covariates=cov_train, verbose=True)

    forecast = model.predict(output_chunk, future_covariates=cov_test)
    full_df = pd.DataFrame({
        "Date": forecast.time_index,
        "Forecast": forecast.values().flatten(),
        "Actual": test.values().flatten()[:output_chunk] if len(test) >= output_chunk else None
    })

    st.subheader("📈 Forecast vs Actual")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=full_df['Date'], y=full_df['Forecast'], name='Forecast'))
    if full_df['Actual'].notna().all():
        fig2.add_trace(go.Scatter(x=full_df['Date'], y=full_df['Actual'], name='Actual'))
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📋 Forecast Table")
    st.dataframe(full_df.round(2))

    if full_df['Actual'].notna().all():
        error = mae(TimeSeries.from_dataframe(full_df, 'Date', 'Forecast'),
                    TimeSeries.from_dataframe(full_df, 'Date', 'Actual'))
        st.metric(label=f"Mean Absolute Error (MAE) for {target_col}", value=f"{error:.2f}")
else:
    st.warning("Please upload an Excel file with hospital data to begin.")
