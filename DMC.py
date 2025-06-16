import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import holidays
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from numba import jit, prange
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial
import time
warnings.filterwarnings('ignore')

# ------------- SPEED OPTIMIZATIONS ---------------

# Use Numba JIT compilation for numerical computations
@jit(nopython=True)
def fast_rolling_stats(values, window):
    """Fast rolling statistics using Numba"""
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    
    # Get the window of data
    start_idx = max(0, n - window)
    window_data = values[start_idx:n]
    
    if len(window_data) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    
    mean_val = np.mean(window_data)
    std_val = np.std(window_data) if len(window_data) > 1 else 0.0
    max_val = np.max(window_data)
    min_val = np.min(window_data)
    median_val = np.median(window_data)
    
    return mean_val, std_val, max_val, min_val, median_val

@jit(nopython=True)
def fast_ema(values, alpha):
    """Fast exponential moving average"""
    if len(values) == 0:
        return 0.0
    
    ema = values[0]
    for i in range(1, len(values)):
        ema = alpha * values[i] + (1 - alpha) * ema
    return ema

@jit(nopython=True)
def fast_pct_change(values, periods):
    """Fast percentage change calculation"""
    n = len(values)
    if n <= periods or values[n-periods-1] == 0:
        return 0.0
    return (values[n-1] - values[n-periods-1]) / values[n-periods-1]

@jit(nopython=True)
def fast_volatility(values, window):
    """Fast volatility calculation"""
    n = len(values)
    if n < window:
        window_data = values
    else:
        window_data = values[n-window:n]
    
    if len(window_data) <= 1:
        return 0.0
    
    mean_val = np.mean(window_data)
    if mean_val == 0:
        return 0.0
    
    std_val = np.std(window_data)
    return std_val / mean_val

# ------------- ENHANCED FEATURE ENGINEERING (OPTIMIZED) ---------------

@st.cache_data
def get_ireland_holidays(years):
    """Cached holiday calculation"""
    ir_holidays = holidays.Ireland(years=years)
    # Add custom Feb holiday for example (Feb 15)
    for year in years:
        ir_holidays[f'{year}-02-15'] = "Custom Feb Bank Holiday"
    return ir_holidays

def create_temporal_features(dates):
    """Vectorized temporal feature creation"""
    df = pd.DataFrame(index=dates)
    
    # Basic temporal features
    df['year'] = dates.year
    df['month'] = dates.month
    df['day'] = dates.day
    df['dayofweek'] = dates.dayofweek
    df['hour'] = dates.hour
    df['week_of_year'] = dates.isocalendar().week
    df['quarter'] = dates.quarter
    df['day_of_year'] = dates.dayofyear
    
    # Cyclical encoding (vectorized)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
    df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
    
    # Pattern features
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_monday'] = (df['dayofweek'] == 0).astype(int)
    df['is_friday'] = (df['dayofweek'] == 4).astype(int)
    
    # Seasonal patterns
    df['is_spring'] = df['month'].isin([3, 4, 5]).astype(int)
    df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
    df['is_autumn'] = df['month'].isin([9, 10, 11]).astype(int)
    df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
    
    # Interaction features
    df['weekend_evening'] = df['is_weekend'] * (df['hour'] >= 18).astype(int)
    df['friday_evening'] = df['is_friday'] * (df['hour'] >= 17).astype(int)
    
    return df

@st.cache_data
def create_optimized_features(df, metric, ir_holidays):
    """Optimized feature engineering with reduced feature set"""
    temp_df = df.copy()
    temp_df['ds'] = pd.to_datetime(temp_df['Date'])
    feature_df = temp_df[['ds', metric]].rename(columns={metric: 'y'}).dropna().set_index('ds')
    
    # Create temporal features (vectorized)
    temporal_features = create_temporal_features(feature_df.index)
    feature_df = pd.concat([feature_df, temporal_features], axis=1)
    
    # Reduced set of lag features (most important ones)
    important_lags = [1, 7, 14]  # Reduced from [1, 2, 3, 7, 14, 21, 30]
    for lag in important_lags:
        feature_df[f'y_lag{lag}'] = feature_df['y'].shift(lag)
    
    # Reduced rolling windows (most important ones)
    important_windows = [7, 30]  # Reduced from [3, 7, 14, 30, 60]
    for window in important_windows:
        feature_df[f'y_rolling_mean_{window}'] = feature_df['y'].rolling(window, min_periods=1).mean()
        feature_df[f'y_rolling_std_{window}'] = feature_df['y'].rolling(window, min_periods=1).std()
    
    # Essential EMA (reduced)
    feature_df['y_ema_0.3'] = feature_df['y'].ewm(alpha=0.3).mean()
    
    # Essential percentage changes
    feature_df['y_pct_change_1d'] = feature_df['y'].pct_change(1)
    feature_df['y_pct_change_7d'] = feature_df['y'].pct_change(7)
    
    # Holiday effects (simplified)
    holiday_dates = set(pd.to_datetime(list(ir_holidays.keys())).date)
    
    if hasattr(feature_df.index, 'date'):
        date_series = pd.Series(feature_df.index.date, index=feature_df.index)
    else:
        date_series = pd.Series([d.date() for d in feature_df.index], index=feature_df.index)
    
    feature_df['is_holiday'] = date_series.map(lambda x: 1 if x in holiday_dates else 0)
    
    # Trend features
    feature_df['trend'] = np.arange(len(feature_df))
    
    # Clean up NaN values efficiently
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    
    # Fast NaN filling
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col == 'y':
            continue
        if 'pct_change' in col:
            feature_df[col] = feature_df[col].fillna(0)
        else:
            feature_df[col] = feature_df[col].fillna(feature_df[col].median())
    
    return feature_df.reset_index()

def parallel_feature_creation(args):
    """Helper function for parallel processing"""
    df_filtered, metric_name, ir_holidays = args
    try:
        return metric_name, create_optimized_features(df_filtered, metric_name, ir_holidays)
    except Exception as e:
        return metric_name, None

@st.cache_data
def train_optimized_xgboost(feature_df, metric_name):
    """Optimized XGBoost training with reduced complexity"""
    
    # Reduced feature set for speed
    essential_features = [
        'year', 'month', 'dayofweek', 'hour', 'quarter',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
        'y_lag1', 'y_lag7', 'y_lag14',
        'y_rolling_mean_7', 'y_rolling_mean_30', 'y_rolling_std_7', 'y_rolling_std_30',
        'y_ema_0.3', 'y_pct_change_1d', 'y_pct_change_7d',
        'is_holiday', 'is_weekend', 'is_friday', 'weekend_evening', 'friday_evening',
        'is_spring', 'is_summer', 'is_autumn', 'is_winter',
        'trend'
    ]
    
    # Filter available features
    available_features = [f for f in essential_features if f in feature_df.columns]
    
    if len(feature_df) < 30:
        raise ValueError(f"Insufficient data: {len(feature_df)} rows")
    
    # Prepare data
    model_data = feature_df[['ds', 'y'] + available_features].copy()
    model_data = model_data.dropna()
    
    if len(model_data) < 20:
        raise ValueError("Too much data lost after cleaning")
    
    X = model_data[available_features]
    y = model_data['y']
    
    # Fast scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Optimized XGBoost parameters for speed
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,  # Reduced from 300
        max_depth=6,       # Reduced from 8
        learning_rate=0.1, # Increased for faster convergence
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,         # Use all CPU cores
        tree_method='hist', # Faster tree method
        early_stopping_rounds=10  # Reduced from 30
    )
    
    # Simple train/validation split for speed
    train_size = int(0.85 * len(X_scaled))  # Use more data for training
    X_train, X_val = X_scaled[:train_size], X_scaled[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # Train with early stopping
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    return xgb_model, scaler, available_features, model_data

# ------------- FAST ITERATIVE FORECASTING ---------------

def fast_iterative_forecast(xgb_model, scaler, features, model_data, forecast_periods=14):
    """Optimized iterative forecasting with proper feature updating"""
    
    # Pre-allocate arrays for speed
    forecasts = np.zeros(forecast_periods)
    forecast_dates = []
    
    # Convert to numpy for faster operations
    historical_values = model_data['y'].values
    last_date = pd.to_datetime(model_data['ds'].iloc[-1])
    
    # Create working array that we'll extend
    working_values = historical_values.copy()
    
    # Pre-calculate holiday dates for speed
    holiday_dates = set()  # Simplified - add holiday logic if needed
    
    for i in range(forecast_periods):
        # Calculate next date
        next_date = last_date + pd.Timedelta(days=1)
        
        # Fast temporal feature calculation
        next_features = {}
        next_features['year'] = next_date.year
        next_features['month'] = next_date.month
        next_features['dayofweek'] = next_date.dayofweek
        next_features['hour'] = next_date.hour
        next_features['quarter'] = next_date.quarter
        
        # Fast cyclical features
        next_features['hour_sin'] = np.sin(2 * np.pi * next_features['hour'] / 24)
        next_features['hour_cos'] = np.cos(2 * np.pi * next_features['hour'] / 24)
        next_features['dow_sin'] = np.sin(2 * np.pi * next_features['dayofweek'] / 7)
        next_features['dow_cos'] = np.cos(2 * np.pi * next_features['dayofweek'] / 7)
        next_features['month_sin'] = np.sin(2 * np.pi * next_features['month'] / 12)
        next_features['month_cos'] = np.cos(2 * np.pi * next_features['month'] / 12)
        
        # Fast lag features using numba-optimized functions
        if len(working_values) >= 1:
            next_features['y_lag1'] = working_values[-1]
        if len(working_values) >= 7:
            next_features['y_lag7'] = working_values[-7]
        if len(working_values) >= 14:
            next_features['y_lag14'] = working_values[-14]
        
        # Fast rolling statistics
        mean_7, std_7, _, _, _ = fast_rolling_stats(working_values, 7)
        mean_30, std_30, _, _, _ = fast_rolling_stats(working_values, 30)
        next_features['y_rolling_mean_7'] = mean_7
        next_features['y_rolling_std_7'] = std_7
        next_features['y_rolling_mean_30'] = mean_30
        next_features['y_rolling_std_30'] = std_30
        
        # Fast EMA
        next_features['y_ema_0.3'] = fast_ema(working_values, 0.3)
        
        # Fast percentage changes
        next_features['y_pct_change_1d'] = fast_pct_change(working_values, 1)
        next_features['y_pct_change_7d'] = fast_pct_change(working_values, 7)
        
        # Fast pattern features
        next_features['is_holiday'] = 0  # Simplified
        next_features['is_weekend'] = 1 if next_features['dayofweek'] >= 5 else 0
        next_features['is_friday'] = 1 if next_features['dayofweek'] == 4 else 0
        next_features['weekend_evening'] = next_features['is_weekend'] * (1 if next_features['hour'] >= 18 else 0)
        next_features['friday_evening'] = next_features['is_friday'] * (1 if next_features['hour'] >= 17 else 0)
        
        # Seasonal features
        next_features['is_spring'] = 1 if next_features['month'] in [3, 4, 5] else 0
        next_features['is_summer'] = 1 if next_features['month'] in [6, 7, 8] else 0
        next_features['is_autumn'] = 1 if next_features['month'] in [9, 10, 11] else 0
        next_features['is_winter'] = 1 if next_features['month'] in [12, 1, 2] else 0
        
        # Trend feature
        next_features['trend'] = len(working_values)
        
        # Fill missing features with defaults
        for feature in features:
            if feature not in next_features:
                next_features[feature] = 0
        
        # Create feature vector
        feature_vector = np.array([next_features.get(f, 0) for f in features]).reshape(1, -1)
        
        # Scale and predict
        feature_vector_scaled = scaler.transform(feature_vector)
        prediction = xgb_model.predict(feature_vector_scaled)[0]
        prediction = max(0, prediction)  # Ensure non-negative
        
        # Update working data
        working_values = np.append(working_values, prediction)
        
        # Store results
        forecasts[i] = prediction
        forecast_dates.append(next_date)
        last_date = next_date
    
    return forecasts.tolist(), forecast_dates

# ------------- EVALUATION AND VISUALIZATION ---------------

def fast_evaluate_model(xgb_model, scaler, features, model_data):
    """Fast model evaluation"""
    test_size = min(14, len(model_data) // 4)
    test_size = max(3, test_size)
    
    test_data = model_data.tail(test_size).copy()
    
    X_test = test_data[features].fillna(0)
    X_test_scaled = scaler.transform(X_test)
    y_true = test_data['y'].values
    y_pred = xgb_model.predict(X_test_scaled)
    
    # Fast metric calculation
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
    
    metrics = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
    return metrics, y_true, y_pred, test_data['ds'].values

def create_fast_plot(historical_dates, historical_values, test_pred_dates, test_pred_values, forecast_dates, forecast_values, metric_name):
    """Optimized plotting"""
    fig = go.Figure()
    
    # Plot full historical actuals
    fig.add_trace(go.Scatter(
        x=historical_dates, y=historical_values,
        mode='lines', name='Actual',
        line=dict(color='blue', width=2)
    ))
    
    # Plot predictions on the test set
    if len(test_pred_values) > 0:
        fig.add_trace(go.Scatter(
            x=test_pred_dates, y=test_pred_values,
            mode='lines', name='Predicted (Test Set)',
            line=dict(color='red', width=2, dash='dash')
        ))
    
    # Plot forecast
    if len(forecast_values) > 0:
        fig.add_trace(go.Scatter(
            x=forecast_dates, y=forecast_values,
            mode='lines+markers', name='Forecast',
            line=dict(color='green', width=2),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title=f'{metric_name} - Fast Forecast Results',
        xaxis_title='Date', yaxis_title='Value',
        hovermode='x unified', height=400
    )
    
    return fig

# ------------- MAIN STREAMLIT APP ---------------

st.set_page_config(layout="wide", page_title="⚡ Fast XGBoost ED Forecasting")
st.title("⚡ High-Speed XGBoost Emergency Department Forecasting")
st.markdown("*Optimized for speed with proper iterative forecasting*")

# Performance settings in sidebar
st.sidebar.header("⚙️ Performance Settings")
use_parallel = st.sidebar.checkbox("Parallel Processing", value=True, help="Use multiple CPU cores")
max_workers = st.sidebar.slider("Max Workers", 1, mp.cpu_count(), mp.cpu_count()//2)

st.sidebar.header("📁 Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Upload Excel file", type=['xlsx', 'xls'])

if uploaded_file:
    # Load and preprocess data
    with st.spinner("Loading data..."):
        df_raw = pd.read_excel(uploaded_file)
        
        if df_raw.shape[1] > 1:
            df_raw = df_raw.iloc[:, 1:]
        
        if 'Date' not in df_raw.columns or 'Hospital' not in df_raw.columns:
            st.sidebar.error("Excel must include 'Date' and 'Hospital' columns.")
            st.stop()
        
        df_raw['Date'] = pd.to_datetime(df_raw['Date'])
    
    # Configuration
    target_metrics_info = [
        ('Tracker8am', '08:00:00'),
        ('Tracker2pm', '14:00:00'),
        ('Tracker8pm', '20:00:00'),
        ('AdditionalCapacityOpenMorning', '08:00:00'),
        ('TimeTotal_8am', '08:00:00'),
        ('TimeTotal_2pm', '14:00:00'),
        ('TimeTotal_8pm', '20:00:00'),
    ]
    target_cols = [m[0] for m in target_metrics_info if m[0] in df_raw.columns]
    
    hospitals = ['All Hospitals'] + sorted(df_raw['Hospital'].unique())
    selected_hospital = st.sidebar.selectbox("Select Hospital", hospitals)
    
    forecast_days = st.sidebar.slider("Forecast Days", min_value=7, max_value=30, value=14)
    
    # Model selection
    selected_metrics = st.sidebar.multiselect(
        "Select Metrics to Forecast", 
        [m[0] for m in target_metrics_info if m[0] in df_raw.columns],
        default=[m[0] for m in target_metrics_info if m[0] in df_raw.columns][:3]  # Default to first 3
    )
    
    if st.sidebar.button("⚡ Run Fast Forecast", type="primary"):
        start_time = time.time()
        
        # Get holidays
        years = list(df_raw['Date'].dt.year.unique())
        ir_holidays = get_ireland_holidays(years)
        
        # Filter data
        if selected_hospital == "All Hospitals":
            df_filtered = df_raw.groupby('Date')[target_cols].sum().reset_index()
        else:
            df_filtered = df_raw[df_raw['Hospital'] == selected_hospital].copy()
        
        st.header(f"⚡ Fast Forecast Results for {selected_hospital}")
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results_summary = []
        total_metrics = len(selected_metrics)
        
        # Parallel processing option
        if use_parallel and total_metrics > 1:
            st.info(f"🚀 Using parallel processing with {max_workers} workers")
            
            # Prepare arguments for parallel processing
            args_list = [(df_filtered, metric_name, ir_holidays) for metric_name in selected_metrics]
            
            # Process features in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                feature_results = list(executor.map(parallel_feature_creation, args_list))
            
            # Process each metric
            for idx, (metric_name, feature_df) in enumerate(feature_results):
                if feature_df is None:
                    st.error(f"Failed to process {metric_name}")
                    continue
                
                progress_bar.progress((idx + 1) / total_metrics)
                status_text.text(f"Processing {metric_name} ({idx + 1}/{total_metrics})")
                
                with st.expander(f"📊 {metric_name}", expanded=idx < 2):  # Show first 2 expanded
                    try:
                        if len(feature_df) < 50:
                            st.warning(f"Insufficient data for {metric_name}. Need at least 50 rows, got {len(feature_df)}. Skipping.")
                            continue
                        
                        # Train model
                        model_start = time.time()
                        xgb_model, scaler, features, model_data = train_optimized_xgboost(feature_df, metric_name)
                        model_time = time.time() - model_start
                        
                        # Evaluate
                        metrics, y_true, y_pred, test_dates = fast_evaluate_model(xgb_model, scaler, features, model_data)
                        
                        # Forecast
                        forecast_start = time.time()
                        forecasts, forecast_dates = fast_iterative_forecast(
                            xgb_model, scaler, features, model_data, forecast_days
                        )
                        forecast_time = time.time() - forecast_start
                        
                        # Display results
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("MAE", f"{metrics['MAE']:.2f}")
                        with col2:
                            st.metric("RMSE", f"{metrics['RMSE']:.2f}")
                        with col3:
                            st.metric("MAPE", f"{metrics['MAPE']:.1f}%")
                        with col4:
                            st.metric("Features", len(features))
                        with col5:
                            st.metric("Speed", f"{model_time + forecast_time:.1f}s")
                        
                        # Plot
                        fig = create_fast_plot(model_data['ds'], model_data['y'], test_dates, y_pred, forecast_dates, forecasts, metric_name)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Forecast table (compact)
                        if len(forecasts) > 0:
                            forecast_df = pd.DataFrame({
                                'Date': forecast_dates,
                                'Forecast': [f"{f:.1f}" for f in forecasts]
                            })
                            st.dataframe(forecast_df.head(7), use_container_width=True)  # Show first week
                        
                        results_summary.append({
                            'Metric': metric_name,
                            'MAE': metrics['MAE'],
                            'RMSE': metrics['RMSE'],
                            'MAPE': metrics['MAPE'],
                            'Features': len(features),
                            'Time_s': model_time + forecast_time
                        })
                        
                    except Exception as e:
                        st.error(f"Error processing {metric_name}: {str(e)}")
        
        else:
            # Sequential processing
            for idx, metric_name in enumerate(selected_metrics):
                progress_bar.progress((idx + 1) / total_metrics)
                status_text.text(f"Processing {metric_name} ({idx + 1}/{total_metrics})")
                
                if metric_name not in df_filtered.columns:
                    st.warning(f"Metric '{metric_name}' not found in the data. Skipping.")
                    continue
                
                with st.expander(f"📊 {metric_name}", expanded=idx < 2):
                    try:
                        # Create features
                        feature_df = create_optimized_features(df_filtered, metric_name, ir_holidays)
                        
                        if len(feature_df) < 50:
                            st.warning(f"Insufficient data for {metric_name}. Need at least 50 rows, got {len(feature_df)}. Skipping.")
                            continue
                        
                        # Train model
                        model_start = time.time()
                        xgb_model, scaler, features, model_data = train_optimized_xgboost(feature_df, metric_name)
                        model_time = time.time() - model_start
                        
                        # Evaluate
                        metrics, y_true, y_pred, test_dates = fast_evaluate_model(xgb_model, scaler, features, model_data)
                        
                        # Forecast
                        forecast_start = time.time()
                        forecasts, forecast_dates = fast_iterative_forecast(
                            xgb_model, scaler, features, model_data, forecast_days
                        )
                        forecast_time = time.time() - forecast_start
                        
                        # Display results
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("MAE", f"{metrics['MAE']:.2f}")
                        with col2:
                            st.metric("RMSE", f"{metrics['RMSE']:.2f}")
                        with col3:
                            st.metric("MAPE", f"{metrics['MAPE']:.1f}%")
                        with col4:
                            st.metric("Features", len(features))
                        with col5:
                            st.metric("Speed", f"{model_time + forecast_time:.1f}s")
                        
                        # Plot
                        fig = create_fast_plot(model_data['ds'], model_data['y'], test_dates, y_pred, forecast_dates, forecasts, metric_name)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Forecast table
                        if len(forecasts) > 0:
                            forecast_df = pd.DataFrame({
                                'Date': forecast_dates,
                                'Forecast': [f"{f:.1f}" for f in forecasts]
                            })
                            st.dataframe(forecast_df.head(7), use_container_width=True)
                        
                        results_summary.append({
                            'Metric': metric_name,
                            'MAE': metrics['MAE'],
                            'RMSE': metrics['RMSE'],
                            'MAPE': metrics['MAPE'],
                            'Features': len(features),
                            'Time_s': model_time + forecast_time
                        })
                        
                    except Exception as e:
                        st.error(f"Error processing {metric_name}: {str(e)}")
        
        # Final Summary
        if results_summary:
            st.subheader("📊 Overall Forecast Summary")
            summary_df = pd.DataFrame(results_summary)
            st.dataframe(summary_df.set_index('Metric'), use_container_width=True)
            st.success(f"✅ Forecasting completed in {time.time() - start_time:.2f} seconds!")
        else:
            st.warning("No metrics were successfully processed for forecasting.")

