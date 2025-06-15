import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import xgboost as xgb
import holidays
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
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
    prophet_df = temp_df[['ds', metric]].rename(columns={metric: 'y'}).dropna().set_index('ds')
    
    # Temporal features
    prophet_df['year'] = prophet_df.index.year
    prophet_df['month'] = prophet_df.index.month
    prophet_df['day'] = prophet_df.index.day
    prophet_df['dayofweek'] = prophet_df.index.dayofweek
    prophet_df['hour'] = prophet_df.index.hour
    prophet_df['week_of_year'] = prophet_df.index.isocalendar().week
    
    # Cyclical encoding (better than sine/cosine alone)
    prophet_df['hour_sin'] = np.sin(2 * np.pi * prophet_df['hour'] / 24)
    prophet_df['hour_cos'] = np.cos(2 * np.pi * prophet_df['hour'] / 24)
    prophet_df['dow_sin'] = np.sin(2 * np.pi * prophet_df['dayofweek'] / 7)
    prophet_df['dow_cos'] = np.cos(2 * np.pi * prophet_df['dayofweek'] / 7)
    prophet_df['month_sin'] = np.sin(2 * np.pi * prophet_df['month'] / 12)
    prophet_df['month_cos'] = np.cos(2 * np.pi * prophet_df['month'] / 12)
    
    # Advanced lag features
    for lag in [1, 2, 3, 7, 14]:  # 1-3 days, 1-2 weeks
        prophet_df[f'y_lag{lag}'] = prophet_df['y'].shift(lag)
    
    # Rolling statistics (multiple windows)
    for window in [3, 7, 14, 30]:
        prophet_df[f'y_rolling_mean_{window}'] = prophet_df['y'].rolling(window).mean().shift(1)
        prophet_df[f'y_rolling_std_{window}'] = prophet_df['y'].rolling(window).std().shift(1)
        prophet_df[f'y_rolling_max_{window}'] = prophet_df['y'].rolling(window).max().shift(1)
        prophet_df[f'y_rolling_min_{window}'] = prophet_df['y'].rolling(window).min().shift(1)
    
    # Percentage changes and volatility
    prophet_df['y_pct_change_1d'] = prophet_df['y'].pct_change(1)
    prophet_df['y_pct_change_7d'] = prophet_df['y'].pct_change(7)
    prophet_df['y_volatility_7d'] = prophet_df['y'].rolling(7).std() / prophet_df['y'].rolling(7).mean()
    
    # Holiday effects (more sophisticated)
    holiday_dates = set(pd.to_datetime(list(ir_holidays.keys())).date)
    
    # Ensure we can extract dates from the index
    if hasattr(prophet_df.index, 'date'):
        date_series = pd.Series(prophet_df.index.date, index=prophet_df.index)
    else:
        date_series = pd.Series([d.date() for d in prophet_df.index], index=prophet_df.index)
    
    prophet_df['is_holiday'] = date_series.map(lambda x: 1 if x in holiday_dates else 0)
    prophet_df['days_since_holiday'] = 0
    prophet_df['days_until_holiday'] = 0
    
    # Calculate days to/from holidays
    holiday_dates_dt = pd.to_datetime(list(holiday_dates))
    for idx in prophet_df.index:
        past_holidays = holiday_dates_dt[holiday_dates_dt <= idx]
        future_holidays = holiday_dates_dt[holiday_dates_dt > idx]
        
        if len(past_holidays) > 0:
            prophet_df.loc[idx, 'days_since_holiday'] = (idx - past_holidays.max()).days
        if len(future_holidays) > 0:
            prophet_df.loc[idx, 'days_until_holiday'] = (future_holidays.min() - idx).days
    
    # Weekend/weekday patterns
    prophet_df['is_weekend'] = (prophet_df['dayofweek'] >= 5).astype(int)
    prophet_df['is_monday'] = (prophet_df['dayofweek'] == 0).astype(int)
    prophet_df['is_friday'] = (prophet_df['dayofweek'] == 4).astype(int)
    
    # Seasonal patterns
    prophet_df['is_summer'] = prophet_df['month'].isin([6, 7, 8]).astype(int)
    prophet_df['is_winter'] = prophet_df['month'].isin([12, 1, 2]).astype(int)
    
    # Interaction features
    prophet_df['weekend_evening'] = prophet_df['is_weekend'] * (prophet_df['hour'] >= 18).astype(int)
    prophet_df['monday_morning'] = prophet_df['is_monday'] * (prophet_df['hour'] <= 12).astype(int)
    
    return prophet_df.dropna().reset_index()

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
            temp_df['system_trend_7d'] = temp_df['y'].rolling(7*3).mean()  # 7 days * 3 times per day
            temp_df['system_trend_30d'] = temp_df['y'].rolling(30*3).mean()
            temp_df['system_capacity_utilization'] = temp_df['y'] / temp_df['y'].rolling(90*3).max()
            
            all_hospital_features[metric] = temp_df.reset_index()
    
    return all_hospital_features

@st.cache_data
def train_enhanced_models(prophet_df, all_hospital_features, metric_name):
    """Train both Prophet and XGBoost with enhanced features"""
    
    # Enhanced feature list
    base_features = ['year', 'month', 'day', 'dayofweek', 'hour', 'week_of_year']
    cyclical_features = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos']
    lag_features = [f'y_lag{i}' for i in [1, 2, 3, 7, 14]]
    rolling_features = [f'y_rolling_mean_{w}' for w in [3, 7, 14, 30]] + \
                      [f'y_rolling_std_{w}' for w in [3, 7, 14, 30]] + \
                      [f'y_rolling_max_{w}' for w in [3, 7, 14, 30]] + \
                      [f'y_rolling_min_{w}' for w in [3, 7, 14, 30]]
    pct_features = ['y_pct_change_1d', 'y_pct_change_7d', 'y_volatility_7d']
    holiday_features = ['is_holiday', 'days_since_holiday', 'days_until_holiday']
    pattern_features = ['is_weekend', 'is_monday', 'is_friday', 'is_summer', 'is_winter']
    interaction_features = ['weekend_evening', 'monday_morning']
    
    all_features = (base_features + cyclical_features + lag_features + rolling_features + 
                   pct_features + holiday_features + pattern_features + interaction_features)
    
    # Add system-wide features if available
    if metric_name in all_hospital_features:
        system_features = ['system_trend_7d', 'system_trend_30d', 'system_capacity_utilization']
        # Merge system features
        system_df = all_hospital_features[metric_name]
        prophet_df = prophet_df.merge(system_df[['ds'] + system_features], on='ds', how='left')
        prophet_df[system_features] = prophet_df[system_features].bfill().ffill()
        all_features.extend(system_features)
    
    # Filter features that exist in the dataframe
    available_features = [f for f in all_features if f in prophet_df.columns]
    
    # Train Prophet
    prophet_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.05
    )
    
    for feature in available_features:
        prophet_model.add_regressor(feature)
    
    prophet_model.fit(prophet_df[['ds', 'y'] + available_features])
    
    # Train XGBoost with hyperparameter tuning
    X = prophet_df[available_features]
    y = prophet_df['y']
    
    # Scale features for better performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=20
    )
    
    # Split for validation
    train_size = int(0.8 * len(X_scaled))
    X_train, X_val = X_scaled[:train_size], X_scaled[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    return prophet_model, xgb_model, scaler, available_features

# ------------- ENHANCED EVALUATION ---------------

def evaluate_models(prophet_model, xgb_model, scaler, features, prophet_df, test_size=14):
    """Enhanced model evaluation with multiple metrics"""
    test_mask = prophet_df.index >= (len(prophet_df) - test_size)
    test_data = prophet_df[test_mask].copy()
    
    # Prophet predictions
    prophet_preds = prophet_model.predict(test_data[['ds'] + features])
    prophet_forecast = prophet_preds['yhat'].values
    
    # XGBoost predictions
    X_test_scaled = scaler.transform(test_data[features])
    xgb_forecast = xgb_model.predict(X_test_scaled)
    
    # Hybrid predictions (weighted average based on historical performance)
    hybrid_forecast = 0.6 * prophet_forecast + 0.4 * xgb_forecast
    
    y_true = test_data['y'].values
    
    # Calculate multiple metrics
    metrics = {}
    for name, preds in [('Prophet', prophet_forecast), ('XGBoost', xgb_forecast), ('Hybrid', hybrid_forecast)]:
        metrics[name] = {
            'MAE': mean_absolute_error(y_true, preds),
            'RMSE': np.sqrt(mean_squared_error(y_true, preds)),
            'MAPE': np.mean(np.abs((y_true - preds) / y_true)) * 100
        }
    
    return metrics

# ------------- MAIN APP (UPDATED) -------------------

st.set_page_config(layout="wide", page_title="Enhanced ED Forecasting")
st.title("🏥 Enhanced Emergency Department Forecasting")

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
    
    if st.sidebar.button("Run Enhanced Forecast"):
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
        
        st.header(f"Enhanced Forecast Results for {selected_hospital}")
        
        for metric_name, time_suffix in target_metrics_info:
            if metric_name not in df_filtered.columns:
                continue
            
            st.subheader(f"📊 {metric_name}")
            
            # Create enhanced features
            prophet_df = create_advanced_features(df_filtered, metric_name, ir_holidays)
            
            if len(prophet_df) < 50:  # Need sufficient data
                st.warning(f"Insufficient data for {metric_name} (need at least 50 records)")
                continue
            
            # Train enhanced models
            prophet_model, xgb_model, scaler, features = train_enhanced_models(
                prophet_df, all_hospital_features, metric_name
            )
            
            # Evaluate models
            metrics = evaluate_models(prophet_model, xgb_model, scaler, features, prophet_df)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Prophet MAE", f"{metrics['Prophet']['MAE']:.2f}")
                st.metric("Prophet RMSE", f"{metrics['Prophet']['RMSE']:.2f}")
            with col2:
                st.metric("XGBoost MAE", f"{metrics['XGBoost']['MAE']:.2f}")
                st.metric("XGBoost RMSE", f"{metrics['XGBoost']['RMSE']:.2f}")
            with col3:
                st.metric("Hybrid MAE", f"{metrics['Hybrid']['MAE']:.2f}")
                st.metric("Hybrid RMSE", f"{metrics['Hybrid']['RMSE']:.2f}")
            
            # Feature importance for XGBoost
            if hasattr(xgb_model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': xgb_model.feature_importances_
                }).sort_values('Importance', ascending=False).head(10)
                
                st.subheader("Top 10 Most Important Features")
                st.bar_chart(importance_df.set_index('Feature')['Importance'])

else:
    st.info("Please upload an Excel file to begin forecasting.")
