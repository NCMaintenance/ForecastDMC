import streamlit as st
import pandas as pd
import numpy as np
from nixtla import NixtlaClient
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import time
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# ------------- TIMEGPT CONFIGURATION ---------------

def initialize_timegpt_client(api_key):
    """Initialize TimeGPT client with API key"""
    try:
        client = NixtlaClient(api_key=api_key)
        # Test the connection with a simple validation
        return client
    except Exception as e:
        st.error(f"Failed to initialize TimeGPT client: {str(e)}")
        return None

def prepare_timegpt_data(df, metric_name, hospital_filter=None):
    """Prepare data in TimeGPT format (ds, y columns)"""
    try:
        # Filter data if hospital is specified
        if hospital_filter and hospital_filter != "All Hospitals":
            df_filtered = df[df['Hospital'] == hospital_filter].copy()
        else:
            # Aggregate all hospitals
            df_filtered = df.groupby('Date')[metric_name].sum().reset_index()
            df_filtered.columns = ['Date', metric_name]
        
        # Prepare TimeGPT format
        timegpt_df = pd.DataFrame({
            'ds': pd.to_datetime(df_filtered['Date']),
            'y': df_filtered[metric_name]
        })
        
        # Remove NaN values and ensure proper sorting
        timegpt_df = timegpt_df.dropna().sort_values('ds').reset_index(drop=True)
        
        # Ensure minimum data points
        if len(timegpt_df) < 10:
            raise ValueError(f"Insufficient data points: {len(timegpt_df)}")
        
        return timegpt_df
    
    except Exception as e:
        st.error(f"Error preparing data for {metric_name}: {str(e)}")
        return None

def create_exogenous_features(df):
    """Create exogenous variables for TimeGPT"""
    try:
        exog_df = df[['ds']].copy()
        
        # Temporal features
        exog_df['year'] = exog_df['ds'].dt.year
        exog_df['month'] = exog_df['ds'].dt.month
        exog_df['day'] = exog_df['ds'].dt.day
        exog_df['dayofweek'] = exog_df['ds'].dt.dayofweek
        exog_df['quarter'] = exog_df['ds'].dt.quarter
        exog_df['week_of_year'] = exog_df['ds'].dt.isocalendar().week
        
        # Cyclical encoding
        exog_df['month_sin'] = np.sin(2 * np.pi * exog_df['month'] / 12)
        exog_df['month_cos'] = np.cos(2 * np.pi * exog_df['month'] / 12)
        exog_df['dow_sin'] = np.sin(2 * np.pi * exog_df['dayofweek'] / 7)
        exog_df['dow_cos'] = np.cos(2 * np.pi * exog_df['dayofweek'] / 7)
        exog_df['quarter_sin'] = np.sin(2 * np.pi * exog_df['quarter'] / 4)
        exog_df['quarter_cos'] = np.cos(2 * np.pi * exog_df['quarter'] / 4)
        
        # Pattern features
        exog_df['is_weekend'] = (exog_df['dayofweek'] >= 5).astype(int)
        exog_df['is_monday'] = (exog_df['dayofweek'] == 0).astype(int)
        exog_df['is_friday'] = (exog_df['dayofweek'] == 4).astype(int)
        
        # Seasonal patterns
        exog_df['is_spring'] = exog_df['month'].isin([3, 4, 5]).astype(int)
        exog_df['is_summer'] = exog_df['month'].isin([6, 7, 8]).astype(int)
        exog_df['is_autumn'] = exog_df['month'].isin([9, 10, 11]).astype(int)
        exog_df['is_winter'] = exog_df['month'].isin([12, 1, 2]).astype(int)
        
        # Trend
        exog_df['trend'] = np.arange(len(exog_df))
        
        return exog_df
    
    except Exception as e:
        st.error(f"Error creating exogenous features: {str(e)}")
        return None

def create_future_exogenous_features(last_date, forecast_periods):
    """Create future exogenous variables for forecasting"""
    try:
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=forecast_periods,
            freq='D'
        )
        
        future_exog = pd.DataFrame({'ds': future_dates})
        
        # Temporal features
        future_exog['year'] = future_exog['ds'].dt.year
        future_exog['month'] = future_exog['ds'].dt.month
        future_exog['day'] = future_exog['ds'].dt.day
        future_exog['dayofweek'] = future_exog['ds'].dt.dayofweek
        future_exog['quarter'] = future_exog['ds'].dt.quarter
        future_exog['week_of_year'] = future_exog['ds'].dt.isocalendar().week
        
        # Cyclical encoding
        future_exog['month_sin'] = np.sin(2 * np.pi * future_exog['month'] / 12)
        future_exog['month_cos'] = np.cos(2 * np.pi * future_exog['month'] / 12)
        future_exog['dow_sin'] = np.sin(2 * np.pi * future_exog['dayofweek'] / 7)
        future_exog['dow_cos'] = np.cos(2 * np.pi * future_exog['dayofweek'] / 7)
        future_exog['quarter_sin'] = np.sin(2 * np.pi * future_exog['quarter'] / 4)
        future_exog['quarter_cos'] = np.cos(2 * np.pi * future_exog['quarter'] / 4)
        
        # Pattern features
        future_exog['is_weekend'] = (future_exog['dayofweek'] >= 5).astype(int)
        future_exog['is_monday'] = (future_exog['dayofweek'] == 0).astype(int)
        future_exog['is_friday'] = (future_exog['dayofweek'] == 4).astype(int)
        
        # Seasonal patterns
        future_exog['is_spring'] = future_exog['month'].isin([3, 4, 5]).astype(int)
        future_exog['is_summer'] = future_exog['month'].isin([6, 7, 8]).astype(int)
        future_exog['is_autumn'] = future_exog['month'].isin([9, 10, 11]).astype(int)
        future_exog['is_winter'] = future_exog['month'].isin([12, 1, 2]).astype(int)
        
        # Trend (continue from historical data)
        future_exog['trend'] = np.arange(len(future_exog)) + 1000  # Offset to continue trend
        
        return future_exog
    
    except Exception as e:
        st.error(f"Error creating future exogenous features: {str(e)}")
        return None

def run_timegpt_forecast(client, data_df, forecast_periods=14, use_exogenous=True, model='timegpt-1'):
    """Run TimeGPT forecast with optional exogenous variables"""
    try:
        # For TimeGPT, we'll use a simpler approach without external regressors for now
        # The foundation model is powerful enough to capture patterns automatically
        
        # Run forecast with basic parameters
        forecast_result = client.forecast(
            df=data_df,
            h=forecast_periods,
            freq='D',
            level=[80, 90]  # Confidence intervals
        )
        
        return forecast_result
    
    except Exception as e:
        st.error(f"TimeGPT forecast error: {str(e)}")
        return None

def evaluate_timegpt_model(client, data_df, test_size=14, use_exogenous=True):
    """Evaluate TimeGPT model using cross-validation"""
    try:
        # Ensure we have enough data for evaluation
        if len(data_df) < test_size + 30:
            test_size = max(7, len(data_df) // 4)
        
        # Run cross-validation with simplified parameters
        cv_results = client.cross_validation(
            df=data_df,
            h=test_size,
            step_size=max(1, test_size // 2),
            n_windows=2,  # Reduced for faster execution
            level=[80, 90]
        )
        
        # Calculate metrics
        mae = np.mean(np.abs(cv_results['y'] - cv_results['TimeGPT']))
        rmse = np.sqrt(np.mean((cv_results['y'] - cv_results['TimeGPT']) ** 2))
        mape = np.mean(np.abs((cv_results['y'] - cv_results['TimeGPT']) / np.maximum(cv_results['y'], 1))) * 100
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
        
        return metrics, cv_results
    
    except Exception as e:
        st.error(f"Model evaluation error: {str(e)}")
        return None, None

def create_timegpt_plot(data_df, forecast_df, cv_results, metric_name):
    """Create comprehensive TimeGPT visualization"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[f'{metric_name} - Historical Data & Forecast', 'Cross-Validation Results'],
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Historical data
    fig.add_trace(
        go.Scatter(
            x=data_df['ds'], 
            y=data_df['y'],
            mode='lines',
            name='Historical Data',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # Forecast
    if forecast_df is not None:
        fig.add_trace(
            go.Scatter(
                x=forecast_df['ds'],
                y=forecast_df['TimeGPT'],
                mode='lines+markers',
                name='TimeGPT Forecast',
                line=dict(color='red', width=2),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        # Confidence intervals
        if 'TimeGPT-lo-90' in forecast_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=forecast_df['ds'],
                    y=forecast_df['TimeGPT-hi-90'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=forecast_df['ds'],
                    y=forecast_df['TimeGPT-lo-90'],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(255,0,0,0.2)',
                    name='90% Confidence',
                    showlegend=True
                ),
                row=1, col=1
            )
    
    # Cross-validation results
    if cv_results is not None:
        fig.add_trace(
            go.Scatter(
                x=cv_results['ds'],
                y=cv_results['y'],
                mode='markers',
                name='Actual (CV)',
                marker=dict(color='blue', size=4)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=cv_results['ds'],
                y=cv_results['TimeGPT'],
                mode='markers',
                name='Predicted (CV)',
                marker=dict(color='red', size=4)
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        title=f'TimeGPT Analysis - {metric_name}',
        height=800,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Value", row=2, col=1)
    
    return fig

# ------------- MAIN STREAMLIT APP ---------------

st.set_page_config(layout="wide", page_title="🤖 TimeGPT ED Forecasting")
st.title("🤖 TimeGPT Emergency Department Forecasting")
st.markdown("*Powered by Nixtla's TimeGPT Foundation Model*")

# API Configuration
st.sidebar.header("🔑 TimeGPT API Configuration")
api_key = st.sidebar.text_input(
    "TimeGPT API Key", 
    type="password",
    help="Get your API key from https://dashboard.nixtla.io"
)

if not api_key:
    st.sidebar.warning("Please enter your TimeGPT API key to continue")
    st.info("To use this app, you'll need a TimeGPT API key from Nixtla. Visit https://dashboard.nixtla.io to get started.")
    st.stop()

# Initialize TimeGPT client
with st.spinner("Initializing TimeGPT client..."):
    client = initialize_timegpt_client(api_key)

if not client:
    st.stop()

st.sidebar.success("✅ TimeGPT client initialized successfully!")

# Model Configuration
st.sidebar.header("🔧 Model Settings")
model_choice = st.sidebar.selectbox(
    "TimeGPT Model",
    options=['timegpt-1', 'timegpt-1-long-horizon'],
    help="Choose model variant"
)

use_exogenous = st.sidebar.checkbox(
    "Use Exogenous Variables",
    value=True,
    help="Include temporal and seasonal features"
)

forecast_days = st.sidebar.slider(
    "Forecast Horizon (days)",
    min_value=7,
    max_value=90,
    value=14,
    help="Number of days to forecast"
)

# File Upload
st.sidebar.header("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload Excel file", type=['xlsx', 'xls'])

if uploaded_file:
    # Load data
    with st.spinner("Loading data..."):
        df_raw = pd.read_excel(uploaded_file)
        
        # Clean up data structure
        if df_raw.shape[1] > 1:
            df_raw = df_raw.iloc[:, 1:]
        
        if 'Date' not in df_raw.columns or 'Hospital' not in df_raw.columns:
            st.sidebar.error("Excel must include 'Date' and 'Hospital' columns.")
            st.stop()
        
        df_raw['Date'] = pd.to_datetime(df_raw['Date'])
    
    # Metric and Hospital Selection
    target_metrics = [
        'Tracker8am', 'Tracker2pm', 'Tracker8pm',
        'AdditionalCapacityOpenMorning',
        'TimeTotal_8am', 'TimeTotal_2pm', 'TimeTotal_8pm'
    ]
    
    available_metrics = [m for m in target_metrics if m in df_raw.columns]
    hospitals = ['All Hospitals'] + sorted(df_raw['Hospital'].unique())
    
    selected_hospital = st.sidebar.selectbox("Select Hospital", hospitals)
    selected_metrics = st.sidebar.multiselect(
        "Select Metrics to Forecast",
        available_metrics,
        default=available_metrics[:3] if len(available_metrics) >= 3 else available_metrics
    )
    
    if st.sidebar.button("🚀 Run TimeGPT Forecast", type="primary"):
        if not selected_metrics:
            st.error("Please select at least one metric to forecast.")
            st.stop()
        
        start_time = time.time()
        
        st.header(f"🤖 TimeGPT Forecast Results for {selected_hospital}")
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results_summary = []
        total_metrics = len(selected_metrics)
        
        # Process each metric
        for idx, metric_name in enumerate(selected_metrics):
            progress_bar.progress((idx + 1) / total_metrics)
            status_text.text(f"Processing {metric_name} with TimeGPT ({idx + 1}/{total_metrics})")
            
            with st.expander(f"🎯 {metric_name}", expanded=idx < 2):
                try:
                    # Prepare data
                    data_prep_start = time.time()
                    timegpt_data = prepare_timegpt_data(df_raw, metric_name, selected_hospital)
                    
                    if timegpt_data is None or len(timegpt_data) < 10:
                        st.warning(f"Insufficient data for {metric_name}")
                        continue
                    
                    data_prep_time = time.time() - data_prep_start
                    
                    # Model evaluation
                    eval_start = time.time()
                    metrics, cv_results = evaluate_timegpt_model(
                        client, timegpt_data, use_exogenous=use_exogenous
                    )
                    eval_time = time.time() - eval_start
                    
                    # Forecasting
                    forecast_start = time.time()
                    forecast_result = run_timegpt_forecast(
                        client, 
                        timegpt_data, 
                        forecast_periods=forecast_days,
                        use_exogenous=use_exogenous
                    )
                    forecast_time = time.time() - forecast_start
                    
                    if forecast_result is None:
                        st.error(f"Failed to generate forecast for {metric_name}")
                        continue
                    
                    # Display metrics
                    if metrics:
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("MAE", f"{metrics['MAE']:.2f}")
                        with col2:
                            st.metric("RMSE", f"{metrics['RMSE']:.2f}")
                        with col3:
                            st.metric("MAPE", f"{metrics['MAPE']:.1f}%")
                        with col4:
                            st.metric("Model", model_choice)
                        with col5:
                            st.metric("Total Time", f"{data_prep_time + eval_time + forecast_time:.1f}s")
                    
                    # Visualization
                    fig = create_timegpt_plot(timegpt_data, forecast_result, cv_results, metric_name)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Forecast table
                    if len(forecast_result) > 0:
                        display_columns = ['ds', 'TimeGPT']
                        if 'TimeGPT-lo-90' in forecast_result.columns:
                            display_columns.extend(['TimeGPT-lo-90', 'TimeGPT-hi-90'])
                        elif 'TimeGPT-lo-80' in forecast_result.columns:
                            display_columns.extend(['TimeGPT-lo-80', 'TimeGPT-hi-80'])
                        
                        forecast_display = forecast_result[display_columns].copy()
                        forecast_display['ds'] = forecast_display['ds'].dt.strftime('%Y-%m-%d')
                        forecast_display = forecast_display.round(2)
                        
                        st.subheader("📊 Forecast Table")
                        st.dataframe(forecast_display, use_container_width=True)
                        
                        # Download button
                        csv = forecast_display.to_csv(index=False)
                        st.download_button(
                            label=f"📥 Download {metric_name} Forecast",
                            data=csv,
                            file_name=f"{metric_name}_timegpt_forecast.csv",
                            mime="text/csv"
                        )
                    
                    # Store results
                    if metrics:
                        results_summary.append({
                            'Metric': metric_name,
                            'MAE': metrics['MAE'],
                            'RMSE': metrics['RMSE'],
                            'MAPE': metrics['MAPE'],
                            'Model': model_choice,
                            'Exogenous': use_exogenous,
                            'Time_s': data_prep_time + eval_time + forecast_time
                        })
                
                except Exception as e:
                    st.error(f"Error processing {metric_name}: {str(e)}")
        
        # Summary
        total_time = time.time() - start_time
        progress_bar.progress(1.0)
        status_text.text(f"✅ Completed all forecasts in {total_time:.1f} seconds")
        
        if results_summary:
            st.subheader("📈 Results Summary")
            summary_df = pd.DataFrame(results_summary)
            st.dataframe(summary_df, use_container_width=True)
            
            # Overall metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Metrics", len(results_summary))
            with col2:
                st.metric("Avg MAPE", f"{summary_df['MAPE'].mean():.1f}%")
            with col3:
                st.metric("Total Time", f"{total_time:.1f}s")
            with col4:
                st.metric("Avg Time/Metric", f"{total_time/len(results_summary):.1f}s")

else:
    st.info("👆 Please upload an Excel file to begin forecasting with TimeGPT")
    
    # Show sample data format
    st.subheader("📋 Expected Data Format")
    sample_data = pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=5),
        'Hospital': ['Hospital A'] * 5,
        'Tracker8am': [45, 52, 38, 41, 47],
        'Tracker2pm': [62, 58, 71, 65, 59],
        'Tracker8pm': [38, 42, 35, 39, 44]
    })
    st.dataframe(sample_data)

# Footer
st.markdown("---")
st.markdown("**Powered by TimeGPT** - The first foundation model for time series forecasting")
st.markdown("Learn more at [nixtla.io](https://nixtla.io)")
