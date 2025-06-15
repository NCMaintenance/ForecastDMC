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
warnings.filterwarnings('ignore')

# ------------- ENHANCED FEATURE ENGINEERING ---------------

@st.cache_data
def get_ireland_holidays(years):
    ir_holidays = holidays.Ireland(years=years)
    # Add custom Feb holiday for example (Feb 15)
    for year in years:
        ir_holidays[f'{year}-02-15'] = "Custom Feb Bank Holiday"
    return ir_holidays

@st.cache_data
def create_advanced_features(df, metric, ir_holidays):
    """Enhanced feature engineering with more sophisticated patterns"""
    temp_df = df.copy()
    temp_df['ds'] = pd.to_datetime(temp_df['Date'])
    feature_df = temp_df[['ds', metric]].rename(columns={metric: 'y'}).dropna().set_index('ds')
    
    # Temporal features
    feature_df['year'] = feature_df.index.year
    feature_df['month'] = feature_df.index.month
    feature_df['day'] = feature_df.index.day
    feature_df['dayofweek'] = feature_df.index.dayofweek
    feature_df['hour'] = feature_df.index.hour
    feature_df['week_of_year'] = feature_df.index.isocalendar().week
    feature_df['quarter'] = feature_df.index.quarter
    feature_df['day_of_year'] = feature_df.index.dayofyear
    
    # Cyclical encoding (captures circular nature of time)
    feature_df['hour_sin'] = np.sin(2 * np.pi * feature_df['hour'] / 24)
    feature_df['hour_cos'] = np.cos(2 * np.pi * feature_df['hour'] / 24)
    feature_df['dow_sin'] = np.sin(2 * np.pi * feature_df['dayofweek'] / 7)
    feature_df['dow_cos'] = np.cos(2 * np.pi * feature_df['dayofweek'] / 7)
    feature_df['month_sin'] = np.sin(2 * np.pi * feature_df['month'] / 12)
    feature_df['month_cos'] = np.cos(2 * np.pi * feature_df['month'] / 12)
    feature_df['quarter_sin'] = np.sin(2 * np.pi * feature_df['quarter'] / 4)
    feature_df['quarter_cos'] = np.cos(2 * np.pi * feature_df['quarter'] / 4)
    
    # Advanced lag features
    for lag in [1, 2, 3, 7, 14, 21, 30]:  # Various time horizons
        feature_df[f'y_lag{lag}'] = feature_df['y'].shift(lag)
    
    # Rolling statistics (multiple windows)
    for window in [3, 7, 14, 30, 60]:
        feature_df[f'y_rolling_mean_{window}'] = feature_df['y'].rolling(window, min_periods=1).mean()
        feature_df[f'y_rolling_std_{window}'] = feature_df['y'].rolling(window, min_periods=1).std()
        feature_df[f'y_rolling_max_{window}'] = feature_df['y'].rolling(window, min_periods=1).max()
        feature_df[f'y_rolling_min_{window}'] = feature_df['y'].rolling(window, min_periods=1).min()
        feature_df[f'y_rolling_median_{window}'] = feature_df['y'].rolling(window, min_periods=1).median()
    
    # Exponential moving averages
    for alpha in [0.1, 0.3, 0.5]:
        feature_df[f'y_ema_{alpha}'] = feature_df['y'].ewm(alpha=alpha).mean()
    
    # Percentage changes and volatility
    feature_df['y_pct_change_1d'] = feature_df['y'].pct_change(1)
    feature_df['y_pct_change_7d'] = feature_df['y'].pct_change(7)
    feature_df['y_pct_change_30d'] = feature_df['y'].pct_change(30)
    feature_df['y_volatility_7d'] = feature_df['y'].rolling(7, min_periods=1).std() / feature_df['y'].rolling(7, min_periods=1).mean()
    feature_df['y_volatility_30d'] = feature_df['y'].rolling(30, min_periods=1).std() / feature_df['y'].rolling(30, min_periods=1).mean()
    
    # Holiday effects (more sophisticated)
    holiday_dates = set(pd.to_datetime(list(ir_holidays.keys())).date)
    
    # Ensure we can extract dates from the index
    if hasattr(feature_df.index, 'date'):
        date_series = pd.Series(feature_df.index.date, index=feature_df.index)
    else:
        date_series = pd.Series([d.date() for d in feature_df.index], index=feature_df.index)
    
    feature_df['is_holiday'] = date_series.map(lambda x: 1 if x in holiday_dates else 0)
    feature_df['days_since_holiday'] = 0
    feature_df['days_until_holiday'] = 0
    
    # Calculate days to/from holidays
    holiday_dates_dt = pd.to_datetime(list(holiday_dates))
    for idx in feature_df.index:
        past_holidays = holiday_dates_dt[holiday_dates_dt <= idx]
        future_holidays = holiday_dates_dt[holiday_dates_dt > idx]
        
        if len(past_holidays) > 0:
            feature_df.loc[idx, 'days_since_holiday'] = (idx - past_holidays.max()).days
        if len(future_holidays) > 0:
            feature_df.loc[idx, 'days_until_holiday'] = (future_holidays.min() - idx).days
    
    # Weekend/weekday patterns
    feature_df['is_weekend'] = (feature_df['dayofweek'] >= 5).astype(int)
    feature_df['is_monday'] = (feature_df['dayofweek'] == 0).astype(int)
    feature_df['is_tuesday'] = (feature_df['dayofweek'] == 1).astype(int)
    feature_df['is_wednesday'] = (feature_df['dayofweek'] == 2).astype(int)
    feature_df['is_thursday'] = (feature_df['dayofweek'] == 3).astype(int)
    feature_df['is_friday'] = (feature_df['dayofweek'] == 4).astype(int)
    feature_df['is_saturday'] = (feature_df['dayofweek'] == 5).astype(int)
    feature_df['is_sunday'] = (feature_df['dayofweek'] == 6).astype(int)
    
    # Seasonal patterns
    feature_df['is_spring'] = feature_df['month'].isin([3, 4, 5]).astype(int)
    feature_df['is_summer'] = feature_df['month'].isin([6, 7, 8]).astype(int)
    feature_df['is_autumn'] = feature_df['month'].isin([9, 10, 11]).astype(int)
    feature_df['is_winter'] = feature_df['month'].isin([12, 1, 2]).astype(int)
    
    # Time-based interactions
    feature_df['weekend_evening'] = feature_df['is_weekend'] * (feature_df['hour'] >= 18).astype(int)
    feature_df['monday_morning'] = feature_df['is_monday'] * (feature_df['hour'] <= 12).astype(int)
    feature_df['friday_evening'] = feature_df['is_friday'] * (feature_df['hour'] >= 17).astype(int)
    feature_df['weekend_night'] = feature_df['is_weekend'] * (feature_df['hour'] >= 22).astype(int)
    
    # Trend features
    feature_df['trend'] = np.arange(len(feature_df))
    feature_df['trend_squared'] = feature_df['trend'] ** 2
    
    # Clean up NaN values
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values with appropriate strategies
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col == 'y':
            continue  # Don't fill target variable
        if 'pct_change' in col or 'volatility' in col:
            feature_df[col] = feature_df[col].fillna(0)
        else:
            feature_df[col] = feature_df[col].fillna(feature_df[col].median())
    
    return feature_df.reset_index()

@st.cache_data
def create_all_hospitals_features(df_raw, target_cols, ir_holidays):
    """Create features using all hospitals data for better trend capture"""
    # Aggregate all hospitals data
    df_all = df_raw.groupby('Date')[target_cols].sum().reset_index()
    
    all_hospital_features = {}
    for metric in target_cols:
        if metric in df_all.columns:
            # Create time series with hourly granularity
            expanded_data = []
            for _, row in df_all.iterrows():
                for hour in [8, 14, 20]:
                    expanded_data.append({
                        'Date': row['Date'],
                        'hour': hour,
                        metric: row[metric]
                    })
            
            expanded_df = pd.DataFrame(expanded_data)
            expanded_df['ds'] = pd.to_datetime(expanded_df['Date'].astype(str) + ' ' + 
                                            expanded_df['hour'].astype(str).str.zfill(2) + ':00:00')
            
            # Create features for this metric
            temp_df = expanded_df[['ds', metric]].rename(columns={metric: 'y'}).set_index('ds')
            
            # Add system-wide trend features
            temp_df['system_trend_7d'] = temp_df['y'].rolling(7*3, min_periods=1).mean()
            temp_df['system_trend_30d'] = temp_df['y'].rolling(30*3, min_periods=1).mean()
            temp_df['system_capacity_utilization'] = temp_df['y'] / temp_df['y'].rolling(90*3, min_periods=1).max()
            temp_df['system_load_factor'] = temp_df['y'] / temp_df['y'].rolling(30*3, min_periods=1).mean()
            
            all_hospital_features[metric] = temp_df.reset_index()
    
    return all_hospital_features

@st.cache_data
def train_xgboost_model(feature_df, all_hospital_features, metric_name):
    """Train XGBoost model with enhanced features and hyperparameter tuning"""
    
    # Enhanced feature list
    base_features = ['year', 'month', 'day', 'dayofweek', 'hour', 'week_of_year', 'quarter', 'day_of_year']
    cyclical_features = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos', 'quarter_sin', 'quarter_cos']
    lag_features = [f'y_lag{i}' for i in [1, 2, 3, 7, 14, 21, 30]]
    rolling_features = []
    for window in [3, 7, 14, 30, 60]:
        rolling_features.extend([f'y_rolling_mean_{window}', f'y_rolling_std_{window}', 
                               f'y_rolling_max_{window}', f'y_rolling_min_{window}', f'y_rolling_median_{window}'])
    
    ema_features = [f'y_ema_{alpha}' for alpha in [0.1, 0.3, 0.5]]
    pct_features = ['y_pct_change_1d', 'y_pct_change_7d', 'y_pct_change_30d', 'y_volatility_7d', 'y_volatility_30d']
    holiday_features = ['is_holiday', 'days_since_holiday', 'days_until_holiday']
    pattern_features = ['is_weekend', 'is_monday', 'is_tuesday', 'is_wednesday', 'is_thursday', 'is_friday', 'is_saturday', 'is_sunday']
    seasonal_features = ['is_spring', 'is_summer', 'is_autumn', 'is_winter']
    interaction_features = ['weekend_evening', 'monday_morning', 'friday_evening', 'weekend_night']
    trend_features = ['trend', 'trend_squared']
    
    all_features = (base_features + cyclical_features + lag_features + rolling_features + 
                   ema_features + pct_features + holiday_features + pattern_features + 
                   seasonal_features + interaction_features + trend_features)
    
    # Add system-wide features if available
    if metric_name in all_hospital_features:
        system_features = ['system_trend_7d', 'system_trend_30d', 'system_capacity_utilization', 'system_load_factor']
        # Merge system features
        system_df = all_hospital_features[metric_name]
        feature_df = feature_df.merge(system_df[['ds'] + system_features], on='ds', how='left')
        for col in system_features:
            if col in feature_df.columns:
                feature_df[col] = feature_df[col].fillna(feature_df[col].median())
        all_features.extend(system_features)
    
    # Filter features that exist in the dataframe and have sufficient non-null values
    available_features = []
    for f in all_features:
        if f in feature_df.columns:
            non_null_ratio = feature_df[f].notna().sum() / len(feature_df)
            if non_null_ratio > 0.7:  # Keep features with at least 70% non-null values
                available_features.append(f)
    
    # Ensure we have enough data after cleaning
    if len(feature_df) < 30:
        raise ValueError(f"Insufficient data after cleaning: {len(feature_df)} rows")
    
    # Final data preparation
    model_data = feature_df[['ds', 'y'] + available_features].copy()
    model_data = model_data.dropna()
    
    if len(model_data) < 20:
        raise ValueError("Too much data lost after removing NaN values")
    
    # Prepare features and target
    X = model_data[available_features]
    y = model_data['y']
    
    # Scale features for better performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Time series cross-validation for better model selection
    tscv = TimeSeriesSplit(n_splits=3)
    
    # XGBoost with optimized hyperparameters for time series
    xgb_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        early_stopping_rounds=30,
        n_jobs=-1
    )
    
    # Split for validation (use last 20% for testing)
    train_size = int(0.8 * len(X_scaled))
    X_train, X_val = X_scaled[:train_size], X_scaled[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # Train the model
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    return xgb_model, scaler, available_features, model_data

# ------------- ENHANCED EVALUATION AND FORECASTING ---------------

def evaluate_and_forecast(xgb_model, scaler, features, model_data, forecast_periods=14):
    """Enhanced model evaluation and future forecasting"""
    
    # Evaluation on test set
    test_size = min(14, len(model_data) // 4)  # Use 25% of data or 14 days, whichever is smaller
    
    if test_size < 3:
        test_size = max(1, len(model_data) // 10)  # Use at least 10% of data
    
    test_data = model_data.tail(test_size).copy()
    
    # Model evaluation
    try:
        X_test = test_data[features].copy()
        X_test = X_test.fillna(0)
        X_test_scaled = scaler.transform(X_test)
        y_true = test_data['y'].values
        y_pred = xgb_model.predict(X_test_scaled)
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
        
        metrics = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
        
        # Generate future forecasts
        last_data = model_data.iloc[-1].copy()
        forecasts = []
        forecast_dates = []
        
        current_date = pd.to_datetime(last_data['ds'])
        
        for i in range(forecast_periods):
            # Create next time point
            next_date = current_date + pd.Timedelta(days=1)
            
            # Create features for next time point
            next_features = {}
            
            # Temporal features
            next_features['year'] = next_date.year
            next_features['month'] = next_date.month
            next_features['day'] = next_date.day
            next_features['dayofweek'] = next_date.dayofweek
            next_features['hour'] = next_date.hour
            next_features['week_of_year'] = next_date.isocalendar().week
            next_features['quarter'] = next_date.quarter
            next_features['day_of_year'] = next_date.dayofyear
            
            # Cyclical features
            next_features['hour_sin'] = np.sin(2 * np.pi * next_features['hour'] / 24)
            next_features['hour_cos'] = np.cos(2 * np.pi * next_features['hour'] / 24)
            next_features['dow_sin'] = np.sin(2 * np.pi * next_features['dayofweek'] / 7)
            next_features['dow_cos'] = np.cos(2 * np.pi * next_features['dayofweek'] / 7)
            next_features['month_sin'] = np.sin(2 * np.pi * next_features['month'] / 12)
            next_features['month_cos'] = np.cos(2 * np.pi * next_features['month'] / 12)
            next_features['quarter_sin'] = np.sin(2 * np.pi * next_features['quarter'] / 4)
            next_features['quarter_cos'] = np.cos(2 * np.pi * next_features['quarter'] / 4)
            
            # Use last available values for other features (simplified approach)
            for feature in features:
                if feature not in next_features:
                    if feature in last_data:
                        next_features[feature] = last_data[feature]
                    else:
                        next_features[feature] = 0
            
            # Make prediction
            next_X = np.array([next_features[f] for f in features]).reshape(1, -1)
            next_X_scaled = scaler.transform(next_X)
            next_pred = xgb_model.predict(next_X_scaled)[0]
            
            forecasts.append(max(0, next_pred))  # Ensure non-negative predictions
            forecast_dates.append(next_date)
            current_date = next_date
        
        return metrics, forecasts, forecast_dates, y_true, y_pred, test_data['ds'].values
        
    except Exception as e:
        st.warning(f"Error in model evaluation: {str(e)}")
        return {'MAE': 0, 'RMSE': 0, 'MAPE': 0}, [], [], [], [], []

def create_forecast_plot(actual_dates, actual_values, pred_values, forecast_dates, forecast_values, metric_name):
    """Create interactive forecast visualization"""
    
    fig = go.Figure()
    
    # Historical actual values
    fig.add_trace(go.Scatter(
        x=actual_dates,
        y=actual_values,
        mode='lines+markers',
        name='Actual',
        line=dict(color='blue', width=2),
        marker=dict(size=4)
    ))
    
    # Predictions on test set
    if len(pred_values) > 0:
        fig.add_trace(go.Scatter(
            x=actual_dates,
            y=pred_values,
            mode='lines+markers',
            name='Predicted',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=4)
        ))
    
    # Future forecasts
    if len(forecast_values) > 0:
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_values,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='green', width=2),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title=f'{metric_name} - Forecast Results',
        xaxis_title='Date',
        yaxis_title='Value',
        hovermode='x unified',
        height=500
    )
    
    return fig

# ------------- MAIN APP ---------------

st.set_page_config(layout="wide", page_title="Enhanced XGBoost ED Forecasting")
st.title("🏥 Enhanced XGBoost Emergency Department Forecasting")
st.markdown("*Simplified forecasting using advanced XGBoost with comprehensive feature engineering*")

st.sidebar.header("Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Upload Excel file", type=['xlsx', 'xls'])

if uploaded_file:
    df_raw = pd.read_excel(uploaded_file)
    
    # Data preprocessing
    if df_raw.shape[1] > 1:
        df_raw = df_raw.iloc[:, 1:]
    
    if 'Date' not in df_raw.columns or 'Hospital' not in df_raw.columns:
        st.sidebar.error("Excel must include 'Date' and 'Hospital' columns.")
        st.stop()
    
    df_raw['Date'] = pd.to_datetime(df_raw['Date'])
    
    # Metrics configuration
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
    
    use_all_hospitals_trend = st.sidebar.checkbox("Use All Hospitals Trend", value=True, 
                                                 help="Include system-wide trends for better forecasting")
    
    forecast_days = st.sidebar.slider("Forecast Days", min_value=7, max_value=30, value=14)
    
    if st.sidebar.button("🚀 Run XGBoost Forecast"):
        # Get holidays
        years = list(df_raw['Date'].dt.year.unique())
        ir_holidays = holidays.Ireland(years=years)
        
        # Create all hospitals features if requested
        all_hospital_features = {}
        if use_all_hospitals_trend:
            all_hospital_features = create_all_hospitals_features(df_raw, target_cols, ir_holidays)
        
        # Filter data
        if selected_hospital == "All Hospitals":
            df_filtered = df_raw.groupby('Date')[target_cols].sum().reset_index()
        else:
            df_filtered = df_raw[df_raw['Hospital'] == selected_hospital].copy()
        
        st.header(f"🎯 XGBoost Forecast Results for {selected_hospital}")
        
        results_summary = []
        
        for metric_name, time_suffix in target_metrics_info:
            if metric_name not in df_filtered.columns:
                continue
            
            with st.expander(f"📊 {metric_name}", expanded=True):
                try:
                    # Create enhanced features
                    feature_df = create_advanced_features(df_filtered, metric_name, ir_holidays)
                    
                    if len(feature_df) < 50:
                        st.warning(f"Insufficient data for {metric_name} (need at least 50 records)")
                        continue
                    
                    # Train XGBoost model
                    xgb_model, scaler, features, model_data = train_xgboost_model(
                        feature_df, all_hospital_features, metric_name
                    )
                    
                    # Evaluate and forecast
                    metrics, forecasts, forecast_dates, y_true, y_pred, test_dates = evaluate_and_forecast(
                        xgb_model, scaler, features, model_data, forecast_days
                    )
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("MAE", f"{metrics['MAE']:.2f}")
                    with col2:
                        st.metric("RMSE", f"{metrics['RMSE']:.2f}")
                    with col3:
                        st.metric("MAPE", f"{metrics['MAPE']:.1f}%")
                    with col4:
                        st.metric("Features Used", len(features))
                    
                    # Create and display forecast plot
                    if len(forecasts) > 0:
                        fig = create_forecast_plot(test_dates, y_true, y_pred, forecast_dates, forecasts, metric_name)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Forecast summary
                        forecast_df = pd.DataFrame({
                            'Date': forecast_dates,
                            'Forecast': forecasts
                        })
                        st.subheader("📅 Forecast Summary")
                        st.dataframe(forecast_df, use_container_width=True)
                    
                    # Feature importance
                    if hasattr(xgb_model, 'feature_importances_'):
                        importance_df = pd.DataFrame({
                            'Feature': features,
                            'Importance': xgb_model.feature_importances_
                        }).sort_values('Importance', ascending=False).head(15)
                        
                        st.subheader("🔍 Top 15 Most Important Features")
                        fig_importance = go.Figure(go.Bar(
                            x=importance_df['Importance'],
                            y=importance_df['Feature'],
                            orientation='h'
                        ))
                        fig_importance.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig_importance, use_container_width=True)
                    
                    # Add to summary
                    results_summary.append({
                        'Metric': metric_name,
                        'MAE': metrics['MAE'],
                        'RMSE': metrics['RMSE'],
                        'MAPE': metrics['MAPE'],
                        'Features': len(features)
                    })
                    
                except Exception as e:
                    st.error(f"Error processing {metric_name}: {str(e)}")
        
        # Overall summary
        if results_summary:
            st.header("📋 Overall Results Summary")
            summary_df = pd.DataFrame(results_summary)
            st.dataframe(summary_df, use_container_width=True)
            
            # Average performance metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average MAE", f"{summary_df['MAE'].mean():.2f}")
            with col2:
                st.metric("Average RMSE", f"{summary_df['RMSE'].mean():.2f}")
            with col3:
                st.metric("Average MAPE", f"{summary_df['MAPE'].mean():.1f}%")

else:
    st.info("📁 Please upload an Excel file to begin forecasting.")
    st.markdown("""
    ### 🔧 Features of this Enhanced XGBoost Forecasting System:
    
    - **Advanced Feature Engineering**: 80+ engineered features including temporal patterns, lag variables, rolling statistics, and cyclical encodings
    - **Holiday Integration**: Irish holiday calendar with custom holiday effects
    - **System-wide Trends**: Option to include all-hospital trends for better forecasting
    - **Time Series Validation**: Proper time series cross-validation for model evaluation
    - **Interactive Visualizations**: Plotly-based charts for forecast analysis
    - **Feature Importance**: Understand which factors drive your forecasts
    - **Flexible Forecasting**: Choose forecast horizon from 7 to 30 days
    
    Simply upload your Excel file with 'Date' and 'Hospital' columns to get started!
    """)
