import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import plotly.graph_objs as go
import warnings
from scipy.stats import pearsonr
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")
st.title("🏥 Advanced Hospital Forecasting with Extended Temporal Features 📈")

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Configuration")
uploaded = st.sidebar.file_uploader("📂 Upload Excel file", type="xlsx")
hospitals = []
targets = [
    "Tracker8am", "Tracker2pm", "Tracker8pm",
    "AdditionalCapacityOpen Morning",
    "TimeTotal_8am", "TimeTotal_2pm", "TimeTotal_8pm"
]

if not uploaded:
    st.info("👋 Welcome! Please upload your Excel file using the sidebar to begin.")
    st.stop()

# --- Data Loading and Initial Processing ---
@st.cache_data
def load_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df = df.sort_values('Date')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data(uploaded)
if df is None:
    st.stop()

hospitals = sorted(df['Hospital'].unique())

# Enhanced configuration with automatic optimization
sel_hosp = st.sidebar.selectbox("🏨 Hospital", ["All"] + hospitals)
sel_target = st.sidebar.selectbox("🎯 Target", ["All"] + targets)
future_days = st.sidebar.slider("⏳ Forecast horizon (days ahead)", 7, 90, 30)

# Automatic lookback calculation based on data availability
def calculate_optimal_lookback(df_length):
    """Calculate optimal lookback period based on data length and seasonal patterns"""
    if df_length >= 730:  # 2+ years
        return min(90, df_length // 8)  # Up to 90 days lookback
    elif df_length >= 365:  # 1+ year
        return min(60, df_length // 6)  # Up to 60 days lookback
    elif df_length >= 180:  # 6+ months
        return min(45, df_length // 4)  # Up to 45 days lookback
    else:
        return min(30, df_length // 3)  # Minimum viable lookback

correlation_threshold = st.sidebar.slider("📊 Correlation threshold for feature selection", 0.0, 0.3, 0.05, 0.01)
auto_optimize = st.sidebar.checkbox("🔧 Auto-optimize parameters", value=True)
run = st.sidebar.button("▶️ Run Forecast")

if not run:
    st.info("ℹ️ Configure your forecast parameters in the sidebar and click 'Run Forecast'.")
    st.stop()

# --- Enhanced Feature Engineering ---
def create_comprehensive_features(df_input, target_col, all_hospital_cols, max_lags=30):
    """
    Create comprehensive feature set with extended temporal patterns.
    """
    try:
        df_feat = df_input.copy()
        
        if 'ds' in df_feat.columns:
            df_feat = df_feat.rename(columns={'ds': 'Date'})
        
        if 'Date' not in df_feat.columns:
            raise ValueError("Date column not found")
            
        all_feature_cols = []
        for col in all_hospital_cols + [target_col]:
            if col in df_feat.columns and not df_feat[col].isna().all():
                all_feature_cols.append(col)
        
        # Extended lag features with multiple patterns
        for col in all_feature_cols:
            if col in df_feat.columns and pd.api.types.is_numeric_dtype(df_feat[col]):
                col_name = 'target' if col == target_col else col
                
                # Short-term lags (daily patterns)
                try:
                    for lag in [1, 2, 3, 7]:  # 1-3 days, 1 week
                        if lag < len(df_feat):
                            df_feat[f'{col_name}_lag_{lag}'] = df_feat[col].shift(lag)
                except Exception as e:
                    st.warning(f"Error creating short-term lags for {col}: {e}")
                    continue
                
                # Medium-term lags (weekly/bi-weekly patterns)
                try:
                    for lag in [14, 21, 28]:  # 2, 3, 4 weeks
                        if lag < len(df_feat):
                            df_feat[f'{col_name}_lag_{lag}'] = df_feat[col].shift(lag)
                except Exception as e:
                    st.warning(f"Error creating medium-term lags for {col}: {e}")
                    continue
                
                # Long-term lags (monthly/seasonal patterns)
                try:
                    for lag in [35, 42, 56, 84]:  # 5, 6, 8, 12 weeks
                        if lag < len(df_feat):
                            df_feat[f'{col_name}_lag_{lag}'] = df_feat[col].shift(lag)
                except Exception as e:
                    st.warning(f"Error creating long-term lags for {col}: {e}")
                    continue

                # Enhanced rolling statistics with multiple windows
                try:
                    windows = [3, 7, 14, 21, 28, 42]  # Extended window range
                    windows = [w for w in windows if w < len(df_feat)]
                    
                    for window in windows:
                        # Rolling mean and std
                        df_feat[f'{col_name}_roll_mean_{window}'] = df_feat[col].shift(1).rolling(
                            window=window, min_periods=max(1, window//3)
                        ).mean()
                        df_feat[f'{col_name}_roll_std_{window}'] = df_feat[col].shift(1).rolling(
                            window=window, min_periods=max(1, window//3)
                        ).std()
                        
                        # Rolling min/max for volatility patterns
                        df_feat[f'{col_name}_roll_min_{window}'] = df_feat[col].shift(1).rolling(
                            window=window, min_periods=max(1, window//3)
                        ).min()
                        df_feat[f'{col_name}_roll_max_{window}'] = df_feat[col].shift(1).rolling(
                            window=window, min_periods=max(1, window//3)
                        ).max()
                        
                        # Rolling median for robust central tendency
                        df_feat[f'{col_name}_roll_median_{window}'] = df_feat[col].shift(1).rolling(
                            window=window, min_periods=max(1, window//3)
                        ).median()
                        
                except Exception as e:
                    st.warning(f"Error creating enhanced rolling features for {col}: {e}")
                    continue

                # Multiple EWM alphas for different decay rates
                try:
                    alphas = [0.1, 0.3, 0.5, 0.7]  # Different decay rates
                    for alpha in alphas:
                        df_feat[f'{col_name}_ewm_mean_{alpha}'] = df_feat[col].shift(1).ewm(alpha=alpha, min_periods=3).mean()
                except Exception as e:
                    st.warning(f"Error creating EWM features for {col}: {e}")
                    continue
                
                # Trend and momentum features
                try:
                    # Short-term trend (3-day slope)
                    if len(df_feat) >= 4:
                        df_feat[f'{col_name}_trend_3d'] = df_feat[col].rolling(window=4, min_periods=2).apply(
                            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=True
                        )
                    
                    # Medium-term trend (7-day slope)
                    if len(df_feat) >= 8:
                        df_feat[f'{col_name}_trend_7d'] = df_feat[col].rolling(window=8, min_periods=4).apply(
                            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=True
                        )
                    
                    # Momentum (rate of change)
                    for period in [3, 7, 14]:
                        if period < len(df_feat):
                            df_feat[f'{col_name}_momentum_{period}d'] = (
                                df_feat[col] - df_feat[col].shift(period)
                            ) / (df_feat[col].shift(period) + 1e-8)  # Avoid division by zero
                            
                except Exception as e:
                    st.warning(f"Error creating trend/momentum features for {col}: {e}")
                    continue
                
                # Volatility features
                try:
                    for window in [7, 14, 21]:
                        if window < len(df_feat):
                            # Coefficient of variation
                            rolling_mean = df_feat[col].shift(1).rolling(window=window, min_periods=3).mean()
                            rolling_std = df_feat[col].shift(1).rolling(window=window, min_periods=3).std()
                            df_feat[f'{col_name}_cv_{window}d'] = rolling_std / (rolling_mean + 1e-8)
                            
                except Exception as e:
                    st.warning(f"Error creating volatility features for {col}: {e}")
                    continue
                
                # Percentile features for outlier detection
                try:
                    for window in [14, 28]:
                        if window < len(df_feat):
                            df_feat[f'{col_name}_pct25_{window}d'] = df_feat[col].shift(1).rolling(
                                window=window, min_periods=max(3, window//4)
                            ).quantile(0.25)
                            df_feat[f'{col_name}_pct75_{window}d'] = df_feat[col].shift(1).rolling(
                                window=window, min_periods=max(3, window//4)
                            ).quantile(0.75)
                            
                except Exception as e:
                    st.warning(f"Error creating percentile features for {col}: {e}")
                    continue

        # Enhanced date-based features
        try:
            df_feat['dow'] = df_feat['Date'].dt.dayofweek
            df_feat['month'] = df_feat['Date'].dt.month
            df_feat['quarter'] = df_feat['Date'].dt.quarter
            df_feat['weekofyear'] = df_feat['Date'].dt.isocalendar().week.astype(int)
            df_feat['dayofyear'] = df_feat['Date'].dt.dayofyear
            df_feat['is_weekend'] = (df_feat['dow'] >= 5).astype(int)
            df_feat['is_monday'] = (df_feat['dow'] == 0).astype(int)
            df_feat['is_friday'] = (df_feat['dow'] == 4).astype(int)
            
            # Month-based features
            df_feat['is_january'] = (df_feat['month'] == 1).astype(int)
            df_feat['is_december'] = (df_feat['month'] == 12).astype(int)
            df_feat['is_summer'] = df_feat['month'].isin([6, 7, 8]).astype(int)
            df_feat['is_winter'] = df_feat['month'].isin([12, 1, 2]).astype(int)
            
            # Cyclical encoding with multiple frequencies
            df_feat['dow_sin'] = np.sin(2 * np.pi * df_feat['dow'] / 7)
            df_feat['dow_cos'] = np.cos(2 * np.pi * df_feat['dow'] / 7)
            df_feat['month_sin'] = np.sin(2 * np.pi * df_feat['month'] / 12)
            df_feat['month_cos'] = np.cos(2 * np.pi * df_feat['month'] / 12)
            df_feat['dayofyear_sin'] = np.sin(2 * np.pi * df_feat['dayofyear'] / 365.25)
            df_feat['dayofyear_cos'] = np.cos(2 * np.pi * df_feat['dayofyear'] / 365.25)
            df_feat['weekofyear_sin'] = np.sin(2 * np.pi * df_feat['weekofyear'] / 52)
            df_feat['weekofyear_cos'] = np.cos(2 * np.pi * df_feat['weekofyear'] / 52)
            
            # Time-based interaction features
            df_feat['month_dow'] = df_feat['month'] * df_feat['dow']
            df_feat['quarter_dow'] = df_feat['quarter'] * df_feat['dow']
            
        except Exception as e:
            st.warning(f"Error creating enhanced date features: {e}")

        return df_feat.rename(columns={'Date': 'ds'})
    
    except Exception as e:
        st.error(f"Critical error in feature engineering: {e}")
        return df_input

# --- Enhanced Feature Selection ---
def advanced_feature_selection(X, y, threshold=0.05, max_features=None):
    """Advanced feature selection with multiple criteria."""
    correlations = {}
    
    if len(y) == 0:
        return [], {}
    
    # Calculate correlations
    for col in X.columns:
        try:
            if not X[col].isna().all() and pd.api.types.is_numeric_dtype(X[col]):
                col_clean = X[col].replace([np.inf, -np.inf], np.nan).fillna(X[col].median())
                y_clean = pd.Series(y).replace([np.inf, -np.inf], np.nan).fillna(pd.Series(y).median())
                
                if len(col_clean.unique()) > 1 and len(y_clean.unique()) > 1:
                    corr, p_value = pearsonr(col_clean, y_clean)
                    if not np.isnan(corr) and abs(corr) >= threshold and p_value < 0.05:
                        correlations[col] = abs(corr)
        except Exception as e:
            continue
    
    # Select top features if max_features is specified
    if max_features and len(correlations) > max_features:
        sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        correlations = dict(sorted_features[:max_features])
    
    selected_features = list(correlations.keys())
    return selected_features, correlations

# --- Enhanced Prophet Configuration ---
def create_enhanced_prophet_forecasts(prophet_df, train_size, forecast_days=30):
    """Create Prophet forecasts with enhanced configuration."""
    try:
        if len(prophet_df) < 20:
            raise ValueError("Insufficient data for Prophet")
        
        # Determine seasonality based on data length
        data_span_days = (prophet_df['ds'].max() - prophet_df['ds'].min()).days
        
        m = Prophet(
            yearly_seasonality=data_span_days >= 365,
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=0.95,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05,  # More conservative changepoints
            seasonality_prior_scale=10,    # Strong seasonality
            mcmc_samples=0  # Faster fitting
        )
        
        # Add custom seasonalities based on data availability
        if data_span_days >= 90:  # At least 3 months
            m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        
        if data_span_days >= 180:  # At least 6 months
            m.add_seasonality(name='quarterly', period=91.25, fourier_order=3)
        
        # Add holidays
        try:
            m.add_country_holidays(country_name='IE')
        except:
            pass
        
        train_prophet = prophet_df.iloc[:train_size].copy()
        train_prophet['y'] = train_prophet['y'].replace([np.inf, -np.inf], np.nan)
        train_prophet = train_prophet.dropna()
        
        if len(train_prophet) < 10:
            raise ValueError("Insufficient training data after cleaning")
        
        m.fit(train_prophet)
        
        future = m.make_future_dataframe(periods=len(prophet_df) - train_size + forecast_days)
        forecast = m.predict(future)
        
        return m, forecast
    
    except Exception as e:
        st.warning(f"Enhanced Prophet forecasting failed: {e}")
        return None, None

# --- Enhanced LightGBM Configuration ---
def create_enhanced_lgb_model(X_train, y_train, auto_optimize=True):
    """Create LightGBM model with enhanced parameters."""
    
    if auto_optimize:
        # Auto-tune parameters based on data size and characteristics
        n_samples = len(X_train)
        
        if n_samples < 100:
            params = {
                'n_estimators': 50,
                'learning_rate': 0.1,
                'num_leaves': 15,
                'max_depth': 4,
            }
        elif n_samples < 500:
            params = {
                'n_estimators': 100,
                'learning_rate': 0.08,
                'num_leaves': 25,
                'max_depth': 6,
            }
        else:
            params = {
                'n_estimators': 200,
                'learning_rate': 0.05,
                'num_leaves': 35,
                'max_depth': 8,
            }
    else:
        params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'max_depth': -1,
        }
    
    # Common parameters for stability
    params.update({
        'random_state': 42,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': max(10, len(X_train) // 50),
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbosity': -1
    })
    
    return LGBMRegressor(**params)

# --- Main Application Logic ---
all_results = []
h_list = hospitals if sel_hosp == "All" else [sel_hosp]
t_list = targets if sel_target == "All" else [sel_target]

for hosp in h_list:
    st.header(f"🏥 Hospital: {hosp}", divider='rainbow')
    df_h = df[df['Hospital'] == hosp].copy().reset_index(drop=True)
    
    if len(df_h) == 0:
        st.warning(f"⚠️ No data found for hospital {hosp}")
        continue
    
    # Calculate optimal lookback for this hospital's data
    optimal_lookback = calculate_optimal_lookback(len(df_h))
    st.info(f"📊 Using {optimal_lookback}-day lookback based on {len(df_h)} days of data")
    
    for tgt in t_list:
        st.subheader(f"🎯 Target: {tgt}")
        
        try:
            if tgt not in df_h.columns:
                st.warning(f"⚠️ Target column '{tgt}' not found in data. Skipping.")
                continue
                
            if df_h[tgt].isna().all():
                st.warning(f"⚠️ Target '{tgt}' has no valid data. Skipping.")
                continue

            available_targets = [col for col in targets 
                               if col in df_h.columns and not df_h[col].isna().all() and col != tgt]
            
            df2 = df_h[['Date'] + [tgt] + available_targets].copy()
            df2 = df2.rename(columns={'Date': 'ds', tgt: 'y'})
            df2 = df2.dropna(subset=['y'])
            df2['y'] = df2['y'].replace([np.inf, -np.inf], np.nan)
            df2 = df2.dropna(subset=['y'])
            
            if len(df2) < 50:
                st.warning(f"⚠️ Insufficient data for target '{tgt}' (only {len(df2)} valid rows). Need at least 50. Skipping.")
                continue

            # Enhanced Feature Engineering with optimal lookback
            df_feat = create_comprehensive_features(df2, 'y', available_targets, max_lags=optimal_lookback)
            
            df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
            initial_len = len(df_feat)
            df_feat = df_feat.dropna(subset=['y'])
            
            if len(df_feat) < initial_len * 0.6:
                st.warning(f"⚠️ Too much data lost during feature engineering for '{tgt}'. Skipping.")
                continue

            n = len(df_feat)
            if n < 80:
                st.warning(f"⚠️ Insufficient data for '{tgt}' after processing ({n} rows). Need at least 80. Skipping.")
                continue
                
            train_size = max(50, int(0.75 * n))  # Use 75% for training with minimum 50
            train_df = df_feat.iloc[:train_size].copy()
            test_df = df_feat.iloc[train_size:].copy()

            # Enhanced Feature Selection
            feature_cols = [col for col in df_feat.columns 
                          if col not in ['ds', 'y'] and pd.api.types.is_numeric_dtype(df_feat[col])]
            
            if not feature_cols:
                st.warning(f"⚠️ No valid features available for '{tgt}'. Skipping.")
                continue
            
            # Determine max features based on data size
            max_features = min(50, len(train_df) // 3) if auto_optimize else None
            
            selected_features, feature_corrs = advanced_feature_selection(
                train_df[feature_cols].fillna(0), train_df['y'], 
                threshold=correlation_threshold, max_features=max_features
            )
            
            if not selected_features:
                st.warning(f"⚠️ No features met the correlation threshold for '{tgt}'. Trying with lower threshold.")
                selected_features, feature_corrs = advanced_feature_selection(
                    train_df[feature_cols].fillna(0), train_df['y'], threshold=0.01
                )
                
                if not selected_features:
                    st.warning(f"⚠️ Still no valid features found for '{tgt}'. Skipping.")
                    continue
            
            with st.expander(f"📊 Selected {len(selected_features)} features from {optimal_lookback}-day lookback. Click to see details."):
                top_features = sorted(feature_corrs.items(), key=lambda x: x[1], reverse=True)[:20]
                st.write("🔝 Top Features by Correlation:")
                feature_df = pd.DataFrame(top_features, columns=['Feature', 'Correlation'])
                feature_df['Correlation'] = feature_df['Correlation'].round(3)
                st.dataframe(feature_df)

            # Prepare training data
            X_train = train_df[selected_features].fillna(train_df[selected_features].median())
            y_train = train_df['y']
            X_test = test_df[selected_features].fillna(train_df[selected_features].median())
            y_test = test_df['y']
            
            if len(X_train) == 0 or len(y_train) == 0 or y_train.std() == 0:
                st.warning(f"⚠️ Invalid training data for '{tgt}'. Skipping.")
                continue
            
            # Enhanced Model Training
            try:
                lgb_model = create_enhanced_lgb_model(X_train, y_train, auto_optimize)
                lgb_model.fit(X_train, y_train)
                
                test_pred_lgb = lgb_model.predict(X_test)
                test_pred_lgb = np.nan_to_num(test_pred_lgb, nan=y_train.mean())
                
                test_mae_lgb = mean_absolute_error(y_test, test_pred_lgb)
                test_rmse_lgb = np.sqrt(mean_squared_error(y_test, test_pred_lgb))
                
            except Exception as e:
                st.error(f"⚠️ Enhanced LightGBM training failed for '{tgt}': {e}")
                continue
            
            # Enhanced Prophet Forecasting
            prophet_model, prophet_forecast = create_enhanced_prophet_forecasts(
                df2[['ds', 'y']], train_size, forecast_days=future_days
            )
            
            if prophet_model is not None and prophet_forecast is not None:
                try:
                    prophet_test_pred = prophet_forecast['yhat'][train_size:n].values
                    if len(prophet_test_pred) == len(y_test):
                        test_mae_prophet = mean_absolute_error(y_test, prophet_test_pred)
                        prophet_future_forecast = prophet_forecast.iloc[-future_days:]
                    else:
                        test_mae_prophet = float('inf')
                        prophet_future_forecast = None
                except Exception as e:
                    st.warning(f"Prophet prediction extraction failed: {e}")
                    test_mae_prophet = float('inf')
                    prophet_future_forecast = None
            else:
                test_mae_prophet = float('inf')
                prophet_future_forecast = None

            # Results Visualization
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=test_df['ds'], y=y_test, mode='lines+markers', name='Actual', line=dict(color='blue', width=2)))
            fig.add_trace(go.Scatter(x=test_df['ds'], y=test_pred_lgb, mode='lines+markers', name='Enhanced LightGBM', line=dict(color='red', width=2)))
            
            if test_mae_prophet != float('inf'):
                fig.add_trace(go.Scatter(x=test_df['ds'], y=prophet_test_pred, mode='lines+markers', name='Enhanced Prophet', line=dict(color='green', width=2, dash='dash')))
                
            fig.update_layout(
                title=f"Enhanced Model Performance - {tgt} ({optimal_lookback}-day lookback)", 
                xaxis_title="Date", 
                yaxis_title="Value", 
                legend_title="Models",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

            # Performance Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Enhanced LGBM MAE", f"{test_mae_lgb:.2f}")
            col2.metric("Enhanced LGBM RMSE", f"{test_rmse_lgb:.2f}")
            col3.metric("Features Used", len(selected_features))
            col4.metric("Lookback Days", optimal_lookback)
            
            if test_mae_prophet != float('inf'):
                improvement = ((test_mae_prophet - test_mae_lgb) / test_mae_prophet) * 100
                st.metric("Enhanced Prophet MAE", f"{test_mae_prophet:.2f}", f"{improvement:.1f}% vs LGBM")

            # Enhanced Feature Importance
            with st.expander("💡 Enhanced LightGBM Feature Importance"):
                try:
                    importance_df = pd.DataFrame({
                        'feature': selected_features,
                        'importance': lgb_model.feature_importances_
                    }).sort_values('importance', ascending=False).head(25)
                    
                    fig_imp = go.Figure(go.Bar(
                        x=importance_df['importance'], 
                        y=importance_df['feature'], 
                        orientation='h',
                        marker_color='lightblue'
                    ))
                    fig_imp.update_layout(
                        title="Top 25 Feature Importances", 
                        yaxis={'categoryorder':'total ascending'},
                        height=600
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not display feature importance: {e}")

            # Enhanced Future Forecasting with confidence intervals
            st.write(f"🔮 **Enhanced Future Forecasts for the next {future_days} days...**")
            
            try:
                # Train final model on all available data
                full_train_X = df_feat[selected_features].fillna(df_feat[selected_features].median())
                full_train_y = df_feat['y']
                
                lgb_model_final = create_enhanced_lgb_model(full_train_X, full_train_y, auto_optimize)
                lgb_model_final.fit(full_train_X, full_train_y)

                # Enhanced recursive forecasting with confidence intervals
                last_known_date = df_feat['ds'].max()
                future_dates = pd.date_range(start=last_known_date + pd.Timedelta(days=1), periods=future_days)
                
                extended_df = df_feat.copy()
                future_preds_lgb = []
                prediction_std = test_rmse_lgb  # Use test RMSE as proxy for prediction uncertainty
                
                for i in range(future_days):
                    current_date = future_dates[i]
                    
                    # Create future row with enhanced date features
                    future_row = pd.DataFrame({'ds': [current_date]})
                    
                    # Enhanced date features
                    future_row['dow'] = future_row['ds'].dt.dayofweek
                    future_row['month'] = future_row['ds'].dt.month
                    future_row['quarter'] = future_row['ds'].dt.quarter
                    future_row['weekofyear'] = future_row['ds'].dt.isocalendar().week.astype(int)
                    future_row['dayofyear'] = future_row['ds'].dt.dayofyear
                    future_row['is_weekend'] = (future_row['dow'] >= 5).astype(int)
                    future_row['is_monday'] = (future_row['dow'] == 0).astype(int)
                    future_row['is_friday'] = (future_row['dow'] == 4).astype(int)
                    future_row['is_january'] = (future_row['month'] == 1).astype(int)
                    future_row['is_december'] = (future_row['month'] == 12).astype(int)
                    future_row['is_summer'] = future_row['month'].isin([6, 7, 8]).astype(int)
                    future_row['is_winter'] = future_row['month'].isin([12, 1, 2]).astype(int)
                    
                    # Cyclical features
                    future_row['dow_sin'] = np.sin(2 * np.pi * future_row['dow'] / 7)
                    future_row['dow_cos'] = np.cos(2 * np.pi * future_row['dow'] / 7)
                    future_row['month_sin'] = np.sin(2 * np.pi * future_row['month'] / 12)
                    future_row['month_cos'] = np.cos(2 * np.pi * future_row['month'] / 12)
                    future_row['dayofyear_sin'] = np.sin(2 * np.pi * future_row['dayofyear'] / 365.25)
                    future_row['dayofyear_cos'] = np.cos(2 * np.pi * future_row['dayofyear'] / 365.25)
                    future_row['weekofyear_sin'] = np.sin(2 * np.pi * future_row['weekofyear'] / 52)
                    future_row['weekofyear_cos'] = np.cos(2 * np.pi * future_row['weekofyear'] / 52)
                    
                    # Interaction features
                    future_row['month_dow'] = future_row['month'] * future_row['dow']
                    future_row['quarter_dow'] = future_row['quarter'] * future_row['dow']
                    
                    # Calculate lag and rolling features based on extended_df
                    for feature in selected_features:
                        if feature not in future_row.columns:
                            # Try to calculate feature value based on historical patterns
                            if 'lag_' in feature:
                                # Extract lag amount
                                lag_amount = int(feature.split('_lag_')[1])
                                base_col = feature.split('_lag_')[0]
                                
                                if base_col == 'target':
                                    if len(extended_df) >= lag_amount:
                                        future_row[feature] = extended_df['y'].iloc[-lag_amount]
                                    else:
                                        future_row[feature] = extended_df['y'].mean()
                                elif base_col in extended_df.columns:
                                    if len(extended_df) >= lag_amount:
                                        future_row[feature] = extended_df[base_col].iloc[-lag_amount]
                                    else:
                                        future_row[feature] = extended_df[base_col].mean()
                                else:
                                    future_row[feature] = 0
                                    
                            elif 'roll_' in feature:
                                # Calculate rolling feature
                                parts = feature.split('_')
                                window = int(parts[-1])
                                stat_type = parts[-2]
                                base_col = '_'.join(parts[:-3])
                                
                                if base_col == 'target':
                                    recent_values = extended_df['y'].tail(window)
                                else:
                                    recent_values = extended_df.get(base_col, pd.Series([0])).tail(window)
                                
                                if stat_type == 'mean':
                                    future_row[feature] = recent_values.mean()
                                elif stat_type == 'std':
                                    future_row[feature] = recent_values.std()
                                elif stat_type == 'min':
                                    future_row[feature] = recent_values.min()
                                elif stat_type == 'max':
                                    future_row[feature] = recent_values.max()
                                elif stat_type == 'median':
                                    future_row[feature] = recent_values.median()
                                else:
                                    future_row[feature] = recent_values.mean()
                                    
                            elif 'ewm_' in feature:
                                # Calculate EWM feature
                                alpha = float(feature.split('_')[-1])
                                base_col = '_'.join(feature.split('_')[:-2])
                                
                                if base_col == 'target':
                                    recent_values = extended_df['y']
                                else:
                                    recent_values = extended_df.get(base_col, pd.Series([0]))
                                
                                future_row[feature] = recent_values.ewm(alpha=alpha, min_periods=1).mean().iloc[-1]
                                
                            elif 'trend_' in feature:
                                # Calculate trend feature
                                window = int(feature.split('_')[-1].replace('d', ''))
                                base_col = '_'.join(feature.split('_')[:-2])
                                
                                if base_col == 'target':
                                    recent_values = extended_df['y'].tail(window + 1)
                                else:
                                    recent_values = extended_df.get(base_col, pd.Series([0])).tail(window + 1)
                                
                                if len(recent_values) > 1:
                                    future_row[feature] = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
                                else:
                                    future_row[feature] = 0
                                    
                            elif 'momentum_' in feature:
                                # Calculate momentum feature
                                period = int(feature.split('_')[-1].replace('d', ''))
                                base_col = '_'.join(feature.split('_')[:-2])
                                
                                if base_col == 'target':
                                    if len(extended_df) >= period + 1:
                                        current_val = extended_df['y'].iloc[-1]
                                        past_val = extended_df['y'].iloc[-period-1]
                                        future_row[feature] = (current_val - past_val) / (past_val + 1e-8)
                                    else:
                                        future_row[feature] = 0
                                else:
                                    future_row[feature] = 0
                                    
                            elif 'cv_' in feature:
                                # Calculate coefficient of variation
                                window = int(feature.split('_')[-1].replace('d', ''))
                                base_col = '_'.join(feature.split('_')[:-2])
                                
                                if base_col == 'target':
                                    recent_values = extended_df['y'].tail(window)
                                else:
                                    recent_values = extended_df.get(base_col, pd.Series([1])).tail(window)
                                
                                mean_val = recent_values.mean()
                                std_val = recent_values.std()
                                future_row[feature] = std_val / (mean_val + 1e-8)
                                
                            elif 'pct' in feature:
                                # Calculate percentile feature
                                if 'pct25' in feature:
                                    quantile = 0.25
                                elif 'pct75' in feature:
                                    quantile = 0.75
                                else:
                                    quantile = 0.5
                                    
                                window = int(feature.split('_')[-1].replace('d', ''))
                                base_col = '_'.join(feature.split('_')[:-2])
                                
                                if base_col == 'target':
                                    recent_values = extended_df['y'].tail(window)
                                else:
                                    recent_values = extended_df.get(base_col, pd.Series([0])).tail(window)
                                
                                future_row[feature] = recent_values.quantile(quantile)
                                
                            else:
                                # Default to median of feature from training data
                                future_row[feature] = full_train_X[feature].median()
                    
                    # Ensure all selected features are present
                    for feature in selected_features:
                        if feature not in future_row.columns:
                            future_row[feature] = full_train_X[feature].median()
                    
                    # Make prediction
                    X_future = future_row[selected_features].fillna(0)
                    pred = lgb_model_final.predict(X_future)[0]
                    
                    # Add some randomness for confidence intervals
                    pred = max(0, pred)  # Ensure non-negative predictions
                    future_preds_lgb.append(pred)
                    
                    # Add prediction to extended_df for next iteration
                    future_row['y'] = pred
                    extended_df = pd.concat([extended_df, future_row], ignore_index=True)

                # Create confidence intervals
                future_preds_lgb = np.array(future_preds_lgb)
                confidence_factor = 1.96  # 95% confidence interval
                
                lower_bound = future_preds_lgb - confidence_factor * prediction_std
                upper_bound = future_preds_lgb + confidence_factor * prediction_std
                
                # Create future forecast DataFrame
                future_forecast_df = pd.DataFrame({
                    'Date': future_dates,
                    'Enhanced_LGB_Forecast': future_preds_lgb,
                    'Lower_95CI': lower_bound,
                    'Upper_95CI': upper_bound
                })
                
                # Display future forecasts
                fig_future = go.Figure()
                
                # Add historical data
                historical_dates = df_feat['ds'].tail(min(30, len(df_feat)))
                historical_values = df_feat['y'].tail(min(30, len(df_feat)))
                
                fig_future.add_trace(go.Scatter(
                    x=historical_dates, 
                    y=historical_values, 
                    mode='lines+markers', 
                    name='Historical Data',
                    line=dict(color='blue', width=2)
                ))
                
                # Add LightGBM forecast
                fig_future.add_trace(go.Scatter(
                    x=future_dates, 
                    y=future_preds_lgb, 
                    mode='lines+markers', 
                    name='Enhanced LightGBM Forecast',
                    line=dict(color='red', width=2)
                ))
                
                # Add confidence intervals
                fig_future.add_trace(go.Scatter(
                    x=future_dates,
                    y=upper_bound,
                    fill=None,
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    showlegend=False
                ))
                
                fig_future.add_trace(go.Scatter(
                    x=future_dates,
                    y=lower_bound,
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    name='95% Confidence Interval',
                    fillcolor='rgba(255,0,0,0.2)'
                ))
                
                # Add Prophet forecast if available
                if prophet_future_forecast is not None:
                    prophet_future_dates = prophet_future_forecast['ds']
                    prophet_future_values = prophet_future_forecast['yhat']
                    
                    fig_future.add_trace(go.Scatter(
                        x=prophet_future_dates, 
                        y=prophet_future_values, 
                        mode='lines+markers', 
                        name='Enhanced Prophet Forecast',
                        line=dict(color='green', width=2, dash='dash')
                    ))
                
                fig_future.update_layout(
                    title=f"Enhanced Future Forecasts - {tgt} (Next {future_days} days)",
                    xaxis_title="Date",
                    yaxis_title="Value",
                    legend_title="Forecasts",
                    height=500
                )
                
                st.plotly_chart(fig_future, use_container_width=True)
                
                # Display forecast table
                st.write("📋 **Detailed Forecast Table:**")
                display_df = future_forecast_df.copy()
                display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                display_df = display_df.round(2)
                st.dataframe(display_df, use_container_width=True)
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Avg Forecast", f"{future_preds_lgb.mean():.2f}")
                col2.metric("Min Forecast", f"{future_preds_lgb.min():.2f}")
                col3.metric("Max Forecast", f"{future_preds_lgb.max():.2f}")
                col4.metric("Forecast Range", f"{future_preds_lgb.max() - future_preds_lgb.min():.2f}")

            except Exception as e:
                st.error(f"⚠️ Enhanced future forecasting failed for '{tgt}': {e}")
                st.write("Using fallback simple forecasting...")
                
                try:
                    # Simple fallback forecast
                    recent_mean = df_feat['y'].tail(14).mean()
                    recent_trend = (df_feat['y'].iloc[-1] - df_feat['y'].iloc[-7]) / 7
                    
                    simple_forecast = []
                    for i in range(future_days):
                        pred = recent_mean + (recent_trend * i)
                        simple_forecast.append(max(0, pred))
                    
                    future_dates = pd.date_range(start=df_feat['ds'].max() + pd.Timedelta(days=1), periods=future_days)
                    simple_df = pd.DataFrame({
                        'Date': future_dates.strftime('%Y-%m-%d'),
                        'Simple_Forecast': simple_forecast
                    })
                    
                    st.dataframe(simple_df, use_container_width=True)
                    
                except Exception as e2:
                    st.error(f"Even fallback forecasting failed: {e2}")

            # Store results
            result_entry = {
                'Hospital': hosp,
                'Target': tgt,
                'Enhanced_LGBM_MAE': test_mae_lgb,
                'Enhanced_LGBM_RMSE': test_rmse_lgb,
                'Features_Used': len(selected_features),
                'Lookback_Days': optimal_lookback,
                'Data_Points': len(df_feat)
            }
            
            if test_mae_prophet != float('inf'):
                result_entry['Enhanced_Prophet_MAE'] = test_mae_prophet
                result_entry['Model_Improvement'] = f"{((test_mae_prophet - test_mae_lgb) / test_mae_prophet) * 100:.1f}%"
            
            all_results.append(result_entry)

        except Exception as e:
            st.error(f"⚠️ Critical error processing target '{tgt}' for hospital '{hosp}': {e}")
            continue

# --- Enhanced Summary Dashboard ---
if all_results:
    st.header("📊 Enhanced Performance Summary", divider='rainbow')
    
    summary_df = pd.DataFrame(all_results)
    
    # Overall statistics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Models", len(summary_df))
    col2.metric("Avg LGBM MAE", f"{summary_df['Enhanced_LGBM_MAE'].mean():.2f}")
    col3.metric("Avg Features Used", f"{summary_df['Features_Used'].mean():.0f}")
    col4.metric("Avg Lookback Days", f"{summary_df['Lookback_Days'].mean():.0f}")
    
    # Performance comparison
    if 'Enhanced_Prophet_MAE' in summary_df.columns:
        prophet_models = summary_df.dropna(subset=['Enhanced_Prophet_MAE'])
        if len(prophet_models) > 0:
            st.metric("Models with Prophet", len(prophet_models))
            avg_improvement = prophet_models['Model_Improvement'].str.replace('%', '').astype(float).mean()
            st.metric("Avg Model Improvement", f"{avg_improvement:.1f}%")
    
    # Detailed results table
    st.write("📋 **Detailed Results:**")
    display_summary = summary_df.round(2)
    st.dataframe(display_summary, use_container_width=True)
    
    # Performance distribution
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=summary_df['Enhanced_LGBM_MAE'],
        nbinsx=20,
        name='Enhanced LightGBM MAE Distribution',
        marker_color='lightblue'
    ))
    fig_dist.update_layout(
        title="Distribution of Enhanced Model Performance (MAE)",
        xaxis_title="MAE",
        yaxis_title="Count",
        height=400
    )
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Feature usage analysis
    if len(summary_df) > 1:
        fig_features = go.Figure()
        fig_features.add_trace(go.Scatter(
            x=summary_df['Features_Used'],
            y=summary_df['Enhanced_LGBM_MAE'],
            mode='markers',
            marker=dict(size=10, color=summary_df['Lookback_Days'], colorscale='Viridis'),
            text=summary_df['Hospital'] + ' - ' + summary_df['Target'],
            name='Model Performance'
        ))
        fig_features.update_layout(
            title="Features Used vs Performance (Color = Lookback Days)",
            xaxis_title="Number of Features Used",
            yaxis_title="MAE",
            height=400
        )
        st.plotly_chart(fig_features, use_container_width=True)

else:
    st.warning("⚠️ No valid results generated. Please check your data and try again.")

st.success("✅ Enhanced hospital forecasting analysis completed!")
st.info("💡 **Tips for better forecasts:**\n- Ensure consistent data quality\n- Use longer historical periods when available\n- Consider external factors (holidays, events)\n- Monitor model performance over time\n- Adjust correlation thresholds based on your data")
