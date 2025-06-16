import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import holidays
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SelectKBest, f_regression
from scipy.stats import pearsonr
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

@jit(nopython=True)
def fast_rolling_stats(values, window):
    """Fast rolling statistics using Numba"""
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

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

@jit(nopython=True)
def fast_trend_calculation(values, window=7):
    """Calculate trend slope over a window"""
    n = len(values)
    if n < window:
        window_data = values
        x_data = np.arange(len(window_data))
    else:
        window_data = values[n-window:n]
        x_data = np.arange(window)

    if len(window_data) < 2:
        return 0.0

    # Simple linear regression slope
    n_points = len(window_data)
    sum_x = np.sum(x_data)
    sum_y = np.sum(window_data)
    sum_xy = np.sum(x_data * window_data)
    sum_x2 = np.sum(x_data * x_data)

    denominator = n_points * sum_x2 - sum_x * sum_x
    if denominator == 0:
        return 0.0

    slope = (n_points * sum_xy - sum_x * sum_y) / denominator
    return slope

# ------------- CORRELATION ANALYSIS AND FEATURE SELECTION ---------------

def analyze_feature_correlations(feature_df, target_col='y', correlation_threshold=0.1):
    """Analyze correlations and select best features"""

    # Calculate all features first
    all_features = []

    # Basic temporal features
    temporal_features = [
        'year', 'month', 'day', 'dayofweek', 'hour', 'week_of_year',
        'quarter', 'day_of_year', 'hour_sin', 'hour_cos', 'dow_sin',
        'dow_cos', 'month_sin', 'month_cos', 'quarter_sin', 'quarter_cos'
    ]
    all_features.extend(temporal_features)

    # Pattern features
    pattern_features = [
        'is_weekend', 'is_monday', 'is_friday', 'is_spring', 'is_summer',
        'is_autumn', 'is_winter', 'weekend_evening', 'friday_evening'
    ]
    all_features.extend(pattern_features)

    # All possible lag features (1-30 days)
    lag_features = [f'y_lag{i}' for i in range(1, 31)]
    all_features.extend(lag_features)

    # All possible rolling features
    rolling_windows = [3, 7, 14, 21, 30, 60, 90]
    rolling_stats = ['mean', 'std', 'max', 'min', 'median']
    for window in rolling_windows:
        for stat in rolling_stats:
            all_features.append(f'y_rolling_{stat}_{window}')

    # EMA features
    ema_alphas = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    for alpha in ema_alphas:
        all_features.append(f'y_ema_{alpha}')

    # Percentage change features
    pct_periods = [1, 2, 3, 7, 14, 21, 30]
    for period in pct_periods:
        all_features.append(f'y_pct_change_{period}d')

    # Volatility features
    vol_windows = [7, 14, 30]
    for window in vol_windows:
        all_features.append(f'y_volatility_{window}')

    # Trend features
    trend_windows = [3, 7, 14, 21]
    for window in trend_windows:
        all_features.append(f'y_trend_{window}')

    # Other features
    other_features = ['is_holiday', 'trend']
    all_features.extend(other_features)

    # Check which features exist in the dataframe
    available_features = [f for f in all_features if f in feature_df.columns]

    # Calculate correlations
    correlations = {}
    target_values = feature_df[target_col].dropna()

    for feature in available_features:
        if feature in feature_df.columns:
            feature_values = feature_df[feature]

            # Align the data (remove NaN pairs)
            aligned_data = pd.DataFrame({
                'target': target_values,
                'feature': feature_values
            }).dropna()

            if len(aligned_data) > 10:  # Need sufficient data points
                try:
                    corr, p_value = pearsonr(aligned_data['target'], aligned_data['feature'])
                    correlations[feature] = {
                        'correlation': abs(corr),
                        'p_value': p_value,
                        'raw_correlation': corr
                    }
                except:
                    correlations[feature] = {
                        'correlation': 0.0,
                        'p_value': 1.0,
                        'raw_correlation': 0.0
                    }

    # Sort by correlation strength
    sorted_features = sorted(correlations.items(),
                           key=lambda x: x[1]['correlation'],
                           reverse=True)

    # Select features above threshold
    selected_features = []
    correlation_info = []

    for feature, corr_info in sorted_features:
        if corr_info['correlation'] >= correlation_threshold and corr_info['p_value'] < 0.05:
            selected_features.append(feature)
            correlation_info.append({
                'feature': feature,
                'correlation': corr_info['raw_correlation'],
                'abs_correlation': corr_info['correlation'],
                'p_value': corr_info['p_value']
            })

    # Ensure we have at least some essential features
    essential_features = ['y_lag1', 'y_lag7', 'dayofweek', 'hour', 'month']
    for feature in essential_features:
        if feature in available_features and feature not in selected_features:
            selected_features.append(feature)
            if feature in correlations:
                correlation_info.append({
                    'feature': feature,
                    'correlation': correlations[feature]['raw_correlation'],
                    'abs_correlation': correlations[feature]['correlation'],
                    'p_value': correlations[feature]['p_value']
                })

    return selected_features[:50], pd.DataFrame(correlation_info)  # Limit to top 50 features

@st.cache_data
def get_ireland_holidays(years):
    """Cached holiday calculation"""
    ir_holidays = holidays.Ireland(years=years)
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

    # Cyclical encoding
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
def create_comprehensive_features(df, metric, ir_holidays):
    """Create comprehensive feature set for correlation analysis"""
    temp_df = df.copy()
    temp_df['ds'] = pd.to_datetime(temp_df['Date'])
    feature_df = temp_df[['ds', metric]].rename(columns={metric: 'y'}).dropna().set_index('ds')

    # Create temporal features
    temporal_features = create_temporal_features(feature_df.index)
    feature_df = pd.concat([feature_df, temporal_features], axis=1)

    # Create ALL possible lag features
    for lag in range(1, 31):  # 1 to 30 days
        feature_df[f'y_lag{lag}'] = feature_df['y'].shift(lag)

    # Create ALL possible rolling features
    rolling_windows = [3, 7, 14, 21, 30, 60, 90]
    for window in rolling_windows:
        feature_df[f'y_rolling_mean_{window}'] = feature_df['y'].rolling(window, min_periods=1).mean()
        feature_df[f'y_rolling_std_{window}'] = feature_df['y'].rolling(window, min_periods=1).std()
        feature_df[f'y_rolling_max_{window}'] = feature_df['y'].rolling(window, min_periods=1).max()
        feature_df[f'y_rolling_min_{window}'] = feature_df['y'].rolling(window, min_periods=1).min()
        feature_df[f'y_rolling_median_{window}'] = feature_df['y'].rolling(window, min_periods=1).median()

    # Create ALL EMA features
    ema_alphas = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    for alpha in ema_alphas:
        feature_df[f'y_ema_{alpha}'] = feature_df['y'].ewm(alpha=alpha).mean()

    # Create ALL percentage change features
    pct_periods = [1, 2, 3, 7, 14, 21, 30]
    for period in pct_periods:
        feature_df[f'y_pct_change_{period}d'] = feature_df['y'].pct_change(period)

    # Create volatility features
    vol_windows = [7, 14, 30]
    for window in vol_windows:
        rolling_std = feature_df['y'].rolling(window, min_periods=1).std()
        rolling_mean = feature_df['y'].rolling(window, min_periods=1).mean()
        feature_df[f'y_volatility_{window}'] = rolling_std / (rolling_mean + 1e-8)

    # Create trend features (slope over different windows)
    trend_windows = [3, 7, 14, 21]
    for window in trend_windows:
        feature_df[f'y_trend_{window}'] = feature_df['y'].rolling(window, min_periods=2).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0
        )

    # Holiday effects
    holiday_dates = set(pd.to_datetime(list(ir_holidays.keys())).date)

    if hasattr(feature_df.index, 'date'):
        date_series = pd.Series(feature_df.index.date, index=feature_df.index)
    else:
        date_series = pd.Series([d.date() for d in feature_df.index], index=feature_df.index)

    feature_df['is_holiday'] = date_series.map(lambda x: 1 if x in holiday_dates else 0)

    # Global trend
    feature_df['trend'] = np.arange(len(feature_df))

    # Clean up NaN values
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)

    # Fill NaN values strategically
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col == 'y':
            continue
        if 'pct_change' in col or 'trend' in col:
            feature_df[col] = feature_df[col].fillna(0)
        else:
            feature_df[col] = feature_df[col].fillna(feature_df[col].median())

    return feature_df.reset_index()

@st.cache_data
def train_enhanced_xgboost(feature_df, metric_name, correlation_threshold=0.1):
    """Enhanced XGBoost training with correlation-based feature selection"""

    # Perform correlation analysis and feature selection
    selected_features, correlation_info = analyze_feature_correlations(
        feature_df, 'y', correlation_threshold
    )

    if len(feature_df) < 30:
        raise ValueError(f"Insufficient data: {len(feature_df)} rows")

    # Prepare data with selected features
    model_data = feature_df[['ds', 'y'] + selected_features].copy()
    model_data = model_data.dropna()

    if len(model_data) < 20:
        raise ValueError("Too much data lost after cleaning")

    X = model_data[selected_features]
    y = model_data['y']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Enhanced XGBoost parameters
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        tree_method='hist',
        early_stopping_rounds=20
    )

    # Train/validation split
    train_size = int(0.85 * len(X_scaled))
    X_train, X_val = X_scaled[:train_size], X_scaled[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    # Train with early stopping
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    return xgb_model, scaler, selected_features, model_data, correlation_info

# ------------- TREND-AWARE ITERATIVE FORECASTING ---------------

def trend_aware_iterative_forecast(xgb_model, scaler, features, model_data, forecast_periods=14, ir_holidays=None):
    """Enhanced iterative forecasting that maintains trends"""

    # Pre-allocate arrays
    forecasts = np.zeros(forecast_periods)
    forecast_dates = []

    # Convert to numpy for faster operations
    historical_values = model_data['y'].values
    last_date = pd.to_datetime(model_data['ds'].iloc[-1])

    # Calculate recent trend to guide forecasting
    recent_window = min(14, len(historical_values))
    recent_trend = fast_trend_calculation(historical_values, recent_window)

    # Calculate recent volatility for realistic bounds
    recent_volatility = np.std(historical_values[-recent_window:]) if recent_window > 1 else 0

    # Working array for iterative updates
    working_values = historical_values.copy() # This is the history of 'y'

    # Store original 'ds' and 'y' for historical reference, and for 'trend' feature.
    master_history_df = model_data[['ds', 'y']].copy() # ds is datetime, y is numeric

    # Calculate base patterns from historical data
    base_patterns = {}
    if len(historical_values) >= 7:
        # Weekly patterns
        for dow in range(7):
            dow_values = []
            for i in range(len(model_data)):
                if pd.to_datetime(model_data['ds'].iloc[i]).dayofweek == dow:
                    dow_values.append(historical_values[i])
            if dow_values:
                base_patterns[f'dow_{dow}'] = np.mean(dow_values)

    for i in range(forecast_periods):
        # Calculate next date
        next_date = last_date + pd.Timedelta(days=1)

        # --- Start of New Feature Calculation Logic ---
        next_features_dict = {}

        # Prepare a temporary DataFrame for the next_date to use create_temporal_features
        # create_temporal_features expects a DatetimeIndex
        temp_dates_for_feature_calc = pd.DatetimeIndex([next_date])
        temporal_df_for_step = create_temporal_features(temp_dates_for_feature_calc)

        for feature_name in features: # 'features' is selected_features
            # Temporal features (from create_temporal_features function)
            # This covers: year, month, day, dayofweek, hour, week_of_year, quarter, day_of_year,
            # hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos, quarter_sin, quarter_cos,
            # is_weekend, is_monday, is_friday, is_spring, is_summer, is_autumn, is_winter,
            # weekend_evening, friday_evening
            if feature_name in temporal_df_for_step.columns:
                next_features_dict[feature_name] = temporal_df_for_step[feature_name].iloc[0]

            # Lag features ('y_lagL')
            elif feature_name.startswith('y_lag'):
                lag = int(feature_name.replace('y_lag', ''))
                if lag <= len(working_values):
                    next_features_dict[feature_name] = working_values[-lag]
                else:
                    next_features_dict[feature_name] = np.nan

            # Rolling statistics ('y_rolling_STAT_W')
            elif feature_name.startswith('y_rolling_'):
                parts = feature_name.split('_')
                # Ensure the feature name has the expected structure: y_rolling_STAT_WINDOW
                if len(parts) == 4:
                    stat_type = parts[2]
                    window_str = parts[3]
                    if window_str.isdigit():
                        window_val = int(window_str)
                        # Original logic for using stat_type and window_val:
                        if len(working_values) > 0: # fast_rolling_stats handles window > len(working_values)
                            mean_val, std_val, max_val, min_val, median_val = fast_rolling_stats(working_values, window_val)
                            if stat_type == 'mean': next_features_dict[feature_name] = mean_val
                            elif stat_type == 'std': next_features_dict[feature_name] = std_val
                            elif stat_type == 'max': next_features_dict[feature_name] = max_val
                            elif stat_type == 'min': next_features_dict[feature_name] = min_val
                            elif stat_type == 'median': next_features_dict[feature_name] = median_val
                            else:
                                # This case should ideally not be reached if feature names are correct
                                next_features_dict[feature_name] = np.nan
                        else: # Not enough data for any rolling stat
                            next_features_dict[feature_name] = np.nan
                    else:
                        # parts[3] is not a digit (e.g., 'min' as in the error)
                        # This indicates a malformed feature name or incorrect identification
                        # st.warning(f"Skipping feature {feature_name}: window part '{window_str}' is not an integer.") # Optional: Add warning
                        next_features_dict[feature_name] = np.nan
                else:
                    # Feature name does not have 4 parts as expected for y_rolling_STAT_WINDOW
                    # st.warning(f"Skipping feature {feature_name}: unexpected structure after splitting by '_'.") # Optional: Add warning
                    next_features_dict[feature_name] = np.nan

            # EMA features ('y_ema_A')
            elif feature_name.startswith('y_ema_'):
                alpha_val = float(feature_name.replace('y_ema_', ''))
                if len(working_values) > 0: # fast_ema handles empty
                    next_features_dict[feature_name] = fast_ema(working_values, alpha_val)
                else:
                    next_features_dict[feature_name] = np.nan

            # Percentage change features ('y_pct_change_Pd')
            elif feature_name.startswith('y_pct_change_'):
                period_val = int(feature_name.replace('y_pct_change_', '').replace('d', ''))
                # fast_pct_change needs n > periods.
                if len(working_values) > period_val :
                    next_features_dict[feature_name] = fast_pct_change(working_values, period_val)
                else:
                     # create_comprehensive_features fills pct_change NaN with 0
                     next_features_dict[feature_name] = 0.0

            # Volatility features ('y_volatility_W')
            elif feature_name.startswith('y_volatility_'):
                window_val = int(feature_name.replace('y_volatility_', ''))
                # fast_volatility handles len < window and len <=1 returning 0.0
                # It also returns 0.0 if mean is 0. This matches create_comprehensive_features behavior (div by mean+1e-8)
                next_features_dict[feature_name] = fast_volatility(working_values, window_val)

            # Trend features ('y_trend_W' for slope, or 'trend' for global)
            elif feature_name.startswith('y_trend_'): # Slope trend
                window_val = int(feature_name.replace('y_trend_', ''))
                # fast_trend_calculation handles len < window and len < 2 returning 0.0
                # create_comprehensive_features for y_trend_W also results in 0 for insufficient points or after fillna.
                next_features_dict[feature_name] = fast_trend_calculation(working_values, window_val)

            elif feature_name == 'trend': # Global trend component
                # This should be the count of historical + already forecasted points up to the PREVIOUS step.
                # The current step's trend value is based on its position.
                next_features_dict[feature_name] = len(master_history_df) # master_history_df has not yet been updated for current 'next_date'

            # Holiday features ('is_holiday')
            elif feature_name == 'is_holiday':
                if ir_holidays is not None:
                    # holidays library integration: date object in HolidayBase object
                    next_features_dict[feature_name] = 1 if next_date in ir_holidays else 0
                else:
                    next_features_dict[feature_name] = 0 # Fallback if ir_holidays somehow not provided

            else:
                # Unhandled selected features become NaN, then imputed.
                # This path should ideally not be taken if all feature types from create_comprehensive_features are covered.
                next_features_dict[feature_name] = np.nan


        feature_vector_list = [next_features_dict.get(f, np.nan) for f in features]

        # Imputation:
        # Original code's fallback for next_features was 0.
        # create_comprehensive_features fills NaNs for pct_change and trend with 0.
        # For other features (lags, rolling, ema), it uses median during feature_df creation.
        # Using 0.0 for NaNs here for consistency with the immediate previous state of the loop and simplicity.
        # A more robust strategy could involve passing training set means/medians for imputation if required.
        imputed_value = 0.0
        feature_vector_filled = [imputed_value if np.isnan(x) else x for x in feature_vector_list]
        feature_vector = np.array(feature_vector_filled).reshape(1, -1)
        # --- End of New Feature Calculation Logic ---

        # Scale and predict
        feature_vector_scaled = scaler.transform(feature_vector)
        base_prediction = xgb_model.predict(feature_vector_scaled)[0]

        # Apply trend adjustment to maintain continuity
        if i == 0:
            # For first prediction, apply recent trend more strongly
            trend_adjustment = recent_trend * 0.5
        else:
            # For subsequent predictions, use a mix of model trend and recent trend
            model_trend = forecasts[i-1] - working_values[-1] if i == 1 else forecasts[i-1] - forecasts[i-2]
            trend_adjustment = 0.7 * model_trend + 0.3 * recent_trend

        # Combine base prediction with trend adjustment
        adjusted_prediction = base_prediction + trend_adjustment

        # Apply pattern-based adjustments using historical data
        dow_pattern_key = f'dow_{next_features["dayofweek"]}'
        if dow_pattern_key in base_patterns:
            pattern_adjustment = 0.1 * (base_patterns[dow_pattern_key] - np.mean(historical_values))
            adjusted_prediction += pattern_adjustment

        # Apply realistic bounds based on historical volatility
        recent_mean = np.mean(working_values[-7:]) if len(working_values) >= 7 else np.mean(working_values)
        lower_bound = recent_mean - 3 * recent_volatility
        upper_bound = recent_mean + 3 * recent_volatility

        # Ensure non-negative and within reasonable bounds
        final_prediction = max(0, min(upper_bound, max(lower_bound, adjusted_prediction)))

        # Update working data
        working_values = np.append(working_values, final_prediction) # 'y' history grows

        # Update master_history_df with the new prediction
        new_row_for_master = pd.DataFrame({'ds': [next_date], 'y': [final_prediction]})
        master_history_df = pd.concat([master_history_df, new_row_for_master], ignore_index=True)

        # Store results
        forecasts[i] = final_prediction
        forecast_dates.append(next_date)
        last_date = next_date

    return forecasts.tolist(), forecast_dates

# ------------- EVALUATION AND VISUALIZATION ---------------

def enhanced_evaluate_model(xgb_model, scaler, features, model_data):
    """Enhanced model evaluation with feature importance"""
    test_size = min(14, len(model_data) // 4)
    test_size = max(3, test_size)

    test_data = model_data.tail(test_size).copy()

    X_test = test_data[features].fillna(0)
    X_test_scaled = scaler.transform(X_test)
    y_true = test_data['y'].values
    y_pred = xgb_model.predict(X_test_scaled)

    # Calculate metrics
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)

    metrics = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
    return metrics, y_true, y_pred, test_data['ds'].values, feature_importance

def create_enhanced_plot(historical_dates, historical_values, test_pred_dates, test_pred_values,
                        forecast_dates, forecast_values, metric_name):
    """Enhanced plotting with trend analysis"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'{metric_name} - Forecast Results', 'Recent Trend Analysis'),
        vertical_spacing=0.12,
        row_heights=[0.7, 0.3]
    )

    # Main forecast plot
    fig.add_trace(go.Scatter(
        x=historical_dates, y=historical_values,
        mode='lines', name='Historical Data',
        line=dict(color='blue', width=2)
    ), row=1, col=1)

    # Test predictions
    if len(test_pred_dates) > 0:
        fig.add_trace(go.Scatter(
            x=test_pred_dates, y=test_pred_values,
            mode='lines+markers', name='Test Predictions',
            line=dict(color='orange', width=2, dash='dash'),
            marker=dict(size=6)
        ), row=1, col=1)

    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=forecast_values,
        mode='lines+markers', name='Forecast',
        line=dict(color='red', width=3),
        marker=dict(size=8, symbol='diamond')
    ), row=1, col=1)

    # Recent trend analysis (last 30 days)
    recent_days = min(30, len(historical_values))
    recent_dates = historical_dates[-recent_days:]
    recent_values = historical_values[-recent_days:]

    fig.add_trace(go.Scatter(
        x=recent_dates, y=recent_values,
        mode='lines+markers', name='Recent Trend',
        line=dict(color='green', width=2),
        marker=dict(size=4)
    ), row=2, col=1)

    # Add trend line
    if len(recent_values) >= 2:
        x_numeric = np.arange(len(recent_values))
        trend_coef = np.polyfit(x_numeric, recent_values, 1)
        trend_line = np.poly1d(trend_coef)(x_numeric)

        fig.add_trace(go.Scatter(
            x=recent_dates, y=trend_line,
            mode='lines', name='Trend Line',
            line=dict(color='purple', width=2, dash='dot')
        ), row=2, col=1)

    # Layout
    fig.update_layout(
        height=700,
        title_text=f"Enhanced {metric_name} Forecast Analysis",
        showlegend=True,
        hovermode='x unified'
    )

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text=metric_name, row=1, col=1)
    fig.update_yaxes(title_text="Recent Values", row=2, col=1)

    return fig

# ------------- STREAMLIT APPLICATION ---------------

def main():
    st.set_page_config(page_title="Enhanced Time Series Forecaster", layout="wide")

    st.title("🚀 Enhanced Time Series Forecaster")
    st.markdown("Advanced XGBoost forecasting with correlation analysis and feature selection")

    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            st.success(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")

            # Display data info
            with st.expander("Data Preview"):
                st.dataframe(df.head())
                st.write("**Columns:**", list(df.columns))

            # Column selection
            col1, col2 = st.columns(2)

            with col1:
                date_col = st.selectbox("Select Date Column", df.columns)

            with col2:
                metric_col = st.selectbox("Select Metric Column",
                                        [col for col in df.columns if col != date_col])

            # Parameters
            with st.sidebar:
                st.header("Forecasting Parameters")
                forecast_days = st.slider("Forecast Days", 1, 30, 14)
                correlation_threshold = st.slider("Correlation Threshold", 0.05, 0.3, 0.1, 0.01)

                st.header("Model Parameters")
                show_correlations = st.checkbox("Show Feature Correlations", True)
                show_importance = st.checkbox("Show Feature Importance", True)

            if st.button("🔮 Generate Forecast", type="primary"):
                with st.spinner("Processing... This may take a moment"):

                    # Prepare data
                    df_clean = df[[date_col, metric_col]].copy()
                    df_clean.columns = ['Date', metric_col]
                    df_clean['Date'] = pd.to_datetime(df_clean['Date'])
                    df_clean = df_clean.dropna().sort_values('Date')

                    if len(df_clean) < 30:
                        st.error("Need at least 30 data points for forecasting")
                        return

                    # Get holidays
                    years = list(range(df_clean['Date'].dt.year.min(),
                                     df_clean['Date'].dt.year.max() + 2))
                    ir_holidays = get_ireland_holidays(years)

                    # Create features
                    feature_df = create_comprehensive_features(df_clean, metric_col, ir_holidays)

                    # Train model
                    xgb_model, scaler, selected_features, model_data, correlation_info = train_enhanced_xgboost(
                        feature_df, metric_col, correlation_threshold
                    )

                    # Evaluate model
                    metrics, y_true, y_pred, test_dates, feature_importance = enhanced_evaluate_model(
                        xgb_model, scaler, selected_features, model_data
                    )

                    # Generate forecast
                    forecast_values, forecast_dates = trend_aware_iterative_forecast(
                        xgb_model, scaler, selected_features, model_data, forecast_days, ir_holidays
                    )

                    # Display results
                    st.success("✅ Forecast completed!")

                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("MAE", f"{metrics['MAE']:.3f}")
                    with col2:
                        st.metric("RMSE", f"{metrics['RMSE']:.3f}")
                    with col3:
                        st.metric("MAPE", f"{metrics['MAPE']:.1f}%")

                    # Plot
                    historical_dates = pd.to_datetime(model_data['ds']).tolist()
                    historical_values = model_data['y'].tolist()

                    fig = create_enhanced_plot(
                        historical_dates, historical_values,
                        pd.to_datetime(test_dates).tolist(), y_pred.tolist(),
                        forecast_dates, forecast_values,
                        metric_col
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Additional analysis
                    if show_correlations and len(correlation_info) > 0:
                        with st.expander("🔗 Feature Correlations"):
                            st.dataframe(correlation_info.head(20))

                    if show_importance and len(feature_importance) > 0:
                        with st.expander("⭐ Feature Importance"):
                            st.dataframe(feature_importance.head(15))

                    # Forecast table
                    with st.expander("📊 Forecast Details"):
                        forecast_df = pd.DataFrame({
                            'Date': forecast_dates,
                            'Forecast': forecast_values
                        })
                        st.dataframe(forecast_df)

                        # Download button
                        csv = forecast_df.to_csv(index=False)
                        st.download_button(
                            "Download Forecast CSV",
                            csv,
                            "forecast.csv",
                            "text/csv"
                        )

                    st.info(f"✨ Used {len(selected_features)} selected features out of {len(feature_df.columns)-2} total features")

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.exception(e)

    else:
        st.info("👆 Upload a CSV file to get started")

        # Example data format
        with st.expander("📋 Expected Data Format"):
            st.write("""
            Your CSV should have:
            - A date column (any standard date format)
            - A numeric metric column to forecast

            Example:
            ```
            Date, Sales
            2023-01-01, 100
            2023-01-02, 120
            2023-01-03, 95
            ...
            ```
            """)

if __name__ == "__main__":
    main()
