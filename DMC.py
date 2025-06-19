import streamlit as st
import pandas as pd
import numpy as np
# Import all required ML libraries
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.ensemble import GradientBoostingRegressor # A good scikit-learn equivalent
from prophet import Prophet # Import Prophet
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error # Import MAE for calculation
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday
from pandas.tseries.offsets import DateOffset
from dateutil.rrule import MO
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# --- Define Irish Bank Holidays ---
class IrishBankHolidays(AbstractHolidayCalendar):
    """
    Defines the rules for Irish Bank Holidays.
    Observance logic (nearest_workday, first Monday) is included.
    """
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=nearest_workday),
        Holiday("St. Brigid's Day", month=2, day=1, observance=nearest_workday),
        Holiday("St. Patrick's Day", month=3, day=17, observance=nearest_workday),
        Holiday("May Day", month=5, day=1, offset=DateOffset(weekday=MO(1))),
        Holiday("June Bank Holiday", month=6, day=1, offset=DateOffset(weekday=MO(1))),
        Holiday("August Bank Holiday", month=8, day=1, offset=DateOffset(weekday=MO(1))),
        Holiday("October Bank Holiday", month=10, day=1, offset=DateOffset(weekday=MO(-1))),
        Holiday("Christmas Day", month=12, day=25),
        Holiday("St. Stephen's Day", month=12, day=26),
    ]

@st.cache_data
def prepare_data(df):
    """
    Prepares and feature engineers the data, reshaping it into the desired wide format
    with 'ED Beds', 'Trolleys', and 'Capacity' columns.
    This function is cached to speed up re-runs if the input data doesn't change.
    """
    # Rename columns for clarity and consistency
    df = df.rename(columns={
        'Tracker8am': 'ED_8am',
        'Tracker2pm': 'ED_2pm',
        'Tracker8pm': 'ED_8pm',
        'TimeTotal_8am': 'Trolley_8am',
        'TimeTotal_2pm': 'Trolley_2pm',
        'TimeTotal_8pm': 'Trolley_8pm',
        'AdditionalCapacityOpen Morning': 'Additional_Capacity'
    })

    # Fill Additional_Capacity across the day for each hospital and date
    # This assumes Additional_Capacity is constant for a given Hospital and Date.
    df['Additional_Capacity'] = df.groupby(['Hospital', 'Date'])['Additional_Capacity'].transform('first')
    df['Additional_Capacity'] = df['Additional_Capacity'].fillna(0)

    # Define common id_vars for melting. DayGAR is removed as it's not used as a feature
    # and its future values are not easily predictable for forecasting.
    common_id_vars = ['Hospital Group Name', 'Hospital', 'Date', 'Additional_Capacity']

    # Melt ED counts into a long format
    df_ed = pd.melt(
        df,
        id_vars=common_id_vars,
        value_vars=['ED_8am', 'ED_2pm', 'ED_8pm'],
        var_name='Metric_Time_ED', # Temporary column to extract time
        value_name='ED Beds'      # The target column for ED data
    )

    # Melt Trolley counts into a long format
    df_trolley = pd.melt(
        df,
        id_vars=common_id_vars,
        value_vars=['Trolley_8am', 'Trolley_2pm', 'Trolley_8pm'],
        var_name='Metric_Time_Trolley', # Temporary column to extract time
        value_name='Trolleys'          # The target column for Trolley data
    )

    # Extract time label (e.g., '8am', '2pm', '8pm') from the temporary metric_time columns
    df_ed['TimeLabel'] = df_ed['Metric_Time_ED'].str.extract(r'([\d]+[ap]m)')
    df_trolley['TimeLabel'] = df_trolley['Metric_Time_Trolley'].str.extract(r'([\d]+[ap]m)')

    # Map time labels to standardized 24-hour format (e.g., '08:00', '14:00', '20:00')
    time_map = {'8am': '08:00', '2pm': '14:00', '8pm': '20:00'}
    df_ed['Time'] = df_ed['TimeLabel'].map(time_map)
    df_trolley['Time'] = df_trolley['TimeLabel'].map(time_map)

    # Drop temporary columns before merging to avoid redundancy
    df_ed = df_ed.drop(columns=['Metric_Time_ED', 'TimeLabel'])
    df_trolley = df_trolley.drop(columns=['Metric_Time_Trolley', 'TimeLabel'])

    # Merge the two melted dataframes. This brings 'ED Beds' and 'Trolleys' side-by-side
    # for each unique combination of identifier columns and 'Time'.
    df_merged = pd.merge(
        df_ed,
        df_trolley,
        on=['Hospital Group Name', 'Hospital', 'Date', 'Additional_Capacity', 'Time'], # DayGAR removed
        how='inner' # Use inner join to ensure only complete records (both ED and Trolley for a given time) are kept
    )

    # Convert 'ED Beds' and 'Trolleys' to numeric, coercing errors to NaN
    df_merged['ED Beds'] = pd.to_numeric(df_merged['ED Beds'], errors='coerce')
    df_merged['Trolleys'] = pd.to_numeric(df_merged['Trolleys'], errors='coerce')
    # Drop rows where either 'ED Beds' or 'Trolleys' values are missing
    df_merged = df_merged.dropna(subset=['ED Beds', 'Trolleys'])

    # Rename 'Additional_Capacity' to 'Capacity' to match the desired output
    df_merged = df_merged.rename(columns={'Additional_Capacity': 'Capacity'})

    # Create a unified Datetime column for time-series analysis
    df_merged['Datetime'] = pd.to_datetime(df_merged['Date'].astype(str) + ' ' + df_merged['Time'])
    # Sort the data by Hospital and Datetime, which is crucial for time-series operations like lags
    df_merged = df_merged.sort_values(by=['Hospital', 'Datetime']).reset_index(drop=True)

    # --- Feature Engineering ---
    # Extract various time-based features
    df_merged['Hour'] = df_merged['Datetime'].dt.hour
    df_merged['DayOfWeek'] = df_merged['Datetime'].dt.dayofweek
    df_merged['DayOfMonth'] = df_merged['Datetime'].dt.day
    df_merged['Month'] = df_merged['Datetime'].dt.month
    df_merged['Quarter'] = df_merged['Datetime'].dt.quarter
    df_merged['WeekOfYear'] = df_merged['Datetime'].dt.isocalendar().week
    df_merged['IsWeekend'] = df_merged['DayOfWeek'].isin([5, 6]).astype(int) # Binary indicator for weekend
    df_merged['IsMonday'] = (df_merged['DayOfWeek'] == 0).astype(int) # Binary indicator for Monday
    df_merged['IsFriday'] = (df_merged['DayOfWeek'] == 4).astype(int) # Binary indicator for Friday

    # Apply cyclical encoding to capture periodicity in time features
    df_merged['Hour_sin'] = np.sin(2 * np.pi * df_merged['Hour'] / 24)
    df_merged['Hour_cos'] = np.cos(2 * np.pi * df_merged['Hour'] / 24)
    df_merged['Day_sin'] = np.sin(2 * np.pi * df_merged['DayOfWeek'] / 7)
    df_merged['Day_cos'] = np.cos(2 * np.pi * df_merged['DayOfWeek'] / 7)
    df_merged['Month_sin'] = np.sin(2 * np.pi * df_merged['Month'] / 12)
    df_merged['Month_cos'] = np.cos(2 * np.pi * df_merged['Month'] / 12)

    # Add Irish bank holidays as a feature
    calendar = IrishBankHolidays()
    try:
        # Generate holidays over the entire date range of the dataset
        holidays = calendar.holidays(start=df_merged['Datetime'].min(), end=df_merged['Datetime'].max() + timedelta(days=30))
        df_merged['IsHoliday'] = df_merged['Datetime'].dt.normalize().isin(holidays).astype(int)
    except Exception: # Catch any errors during holiday generation
        df_merged['IsHoliday'] = 0 # Default to 0 if holidays cannot be determined

    # Add seasonal indicators
    df_merged['IsSummer'] = df_merged['Month'].isin([6, 7, 8]).astype(int)
    df_merged['IsWinter'] = df_merged['Month'].isin([12, 1, 2]).astype(int)

    # Add peak/low hour indicators
    df_merged['IsPeakHour'] = df_merged['Hour'].isin([20]).astype(int) # 8 PM is often a peak time
    df_merged['IsLowHour'] = df_merged['Hour'].isin([8]).astype(int)  # 8 AM is often a lower time

    # Encode Hospital names into numerical codes for model consumption
    df_merged['Hospital_Code'] = df_merged['Hospital'].astype('category').cat.codes

    return df_merged

@st.cache_data
def add_lag_features_smart(df, target_column):
    """
    Adds lag features intelligently to the DataFrame based on available data for a specific target column.
    Lag features are time-shifted values of the target, useful for capturing temporal dependencies.
    This function now applies lags and rolling features per hospital group.
    This function is cached to speed up re-runs if the input data doesn't change.
    """
    df_copy = df.copy() # Work on a copy to avoid modifying the original DataFrame during groupby operations
    lag_features = []

    # Ensure the DataFrame is sorted by Hospital and Datetime before group-wise operations
    df_copy = df_copy.sort_values(by=['Hospital', 'Datetime'])

    # Apply lags and rolling means grouped by hospital
    # This is crucial to ensure lags are only calculated within each hospital's time series
    for hospital_name, hospital_group in df_copy.groupby('Hospital'):
        # Determine maximum safe lag based on the size of the filtered data for *this* hospital group
        max_safe_lag = min(7, len(hospital_group) // 4)
        if max_safe_lag < 1:
            # st.warning(f"Very limited data ({len(hospital_group)} records) for {hospital_name}. Skipping lag features for {target_column}.")
            continue

        # Add lag features for the current target_column (e.g., 'ED Beds', 'Trolleys', 'Capacity')
        for i in range(1, max_safe_lag + 1):
            lag_col = f'Lag_{target_column}_{i}' # Lag column names are specific to the target (e.g., 'Lag_ED Beds_1')
            df_copy.loc[hospital_group.index, lag_col] = hospital_group[target_column].shift(i)
            if lag_col not in lag_features: # Add to list only once to avoid duplicates
                lag_features.append(lag_col)

        # Add rolling mean features if enough data exists for the current hospital group
        if len(hospital_group) >= 6:
            roll_mean_3_col = f'Rolling_Mean_3_{target_column}'
            df_copy.loc[hospital_group.index, roll_mean_3_col] = hospital_group[target_column].rolling(window=min(3, len(hospital_group)//2), min_periods=1).mean()
            if roll_mean_3_col not in lag_features:
                lag_features.append(roll_mean_3_col)

        if len(hospital_group) >= 14:
            roll_mean_7_col = f'Rolling_Mean_7_{target_column}'
            df_copy.loc[hospital_group.index, roll_mean_7_col] = hospital_group[target_column].rolling(window=min(7, len(hospital_group)//2), min_periods=1).mean()
            if roll_mean_7_col not in lag_features:
                lag_features.append(roll_mean_7_col)

    # Fill NaN values created by shifting/rolling operations.
    # It's better to fill NaNs after all group-wise operations are complete,
    # or handle them during model training (e.g., CatBoost can handle NaNs).
    # Here, we'll forward-fill, then back-fill, then default to 0 for robustness.
    for feature in lag_features:
        df_copy[feature] = df_copy[feature].fillna(method='ffill').fillna(method='bfill').fillna(0)

    return df_copy, lag_features


def create_future_dates(last_date, hospital, hospital_code, additional_capacity, days=7):
    """
    Creates a DataFrame of future dates and times with corresponding feature values
    for forecasting.
    """
    future_dates = []
    times = ['08:00', '14:00', '20:00'] # Standard observation times

    for day in range(1, days + 1):
        future_date = last_date + timedelta(days=day)
        for time_str in times:
            future_datetime = pd.to_datetime(f"{future_date.date()} {time_str}")

            # Recalculate all relevant features for each future timestamp
            hour = future_datetime.hour
            day_of_week = future_datetime.dayofweek
            day_of_month = future_datetime.day
            month = future_datetime.month
            quarter = (month - 1) // 3 + 1
            week_of_year = future_datetime.isocalendar().week

            is_weekend = int(day_of_week in [5, 6])
            is_monday = int(day_of_week == 0)
            is_friday = int(day_of_week == 4)

            # Cyclical encoding
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            day_sin = np.sin(2 * np.pi * day_of_week / 7)
            day_cos = np.cos(2 * np.pi * day_of_week / 7)
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)

            # Holiday check
            is_holiday = 0
            try:
                calendar = IrishBankHolidays()
                # Check for holidays only within the specific future datetime
                holidays = calendar.holidays(start=future_datetime.normalize(), end=future_datetime.normalize())
                is_holiday = int(future_datetime.normalize() in holidays)
            except Exception: # Catch any errors during holiday calculation
                pass

            # Seasonal and peak hour indicators
            is_summer = int(month in [6, 7, 8])
            is_winter = int(month in [12, 1, 2])
            is_peak_hour = int(hour == 20)
            is_low_hour = int(hour == 8)

            # Append all calculated features for the current future timestamp
            future_dates.append({
                'Date': future_datetime.date(), # Date component
                'Time': time_str,               # Time component (e.g., '08:00')
                'Datetime': future_datetime,    # Full datetime object
                'Hospital': hospital,
                'Hour': hour,
                'DayOfWeek': day_of_week,
                'DayOfMonth': day_of_month,
                'Month': month,
                'Quarter': quarter,
                'WeekOfYear': week_of_year,
                'IsWeekend': is_weekend,
                'IsMonday': is_monday,
                'IsFriday': is_friday,
                'Hour_sin': hour_sin,
                'Hour_cos': hour_cos,
                'Day_sin': day_sin,
                'Day_cos': day_cos,
                'Month_sin': month_sin,
                'Month_cos': month_cos,
                'IsHoliday': is_holiday,
                'IsSummer': is_summer,
                'IsWinter': is_winter,
                'IsPeakHour': is_peak_hour,
                'IsLowHour': is_low_hour,
                'Hospital_Code': hospital_code,
                'Capacity': additional_capacity # Use the 'Capacity' value (from 'Additional_Capacity')
            })

    return pd.DataFrame(future_dates)

def forecast_with_lags(model, historical_data, future_df, features, target_column):
    """
    Generates forecasts iteratively, updating lag and rolling features
    with each new prediction. This is for tree-based models (CatBoost, LightGBM, XGBoost, GB).
    historical_data should already be filtered for the specific hospital and target column.
    Ensures predictions are non-negative and rounded to zero decimal places.
    Also calculates approximate prediction intervals based on historical residuals.
    """
    if historical_data.empty:
        st.error("Historical data for lag forecasting is empty. Cannot generate forecasts.")
        return pd.DataFrame({
            'Predicted': [0] * len(future_df),
            'Predicted_Low': [0] * len(future_df),
            'Predicted_High': [0] * len(future_df)
        }, index=future_df.index)

    predictions = []
    pred_lows = []
    pred_highs = []

    # Get the last few actual values from historical data for initial lag features
    last_values = historical_data[target_column].tail(7).values
    current_lags = list(reversed(last_values.tolist()))
    current_lags = current_lags + [0] * (7 - len(current_lags))

    # Initialize rolling statistics with historical data's tail
    historical_mean_3 = historical_data[target_column].tail(3).mean() if len(historical_data) >= 3 else historical_data[target_column].mean()
    historical_mean_7 = historical_data[target_column].tail(7).mean() if len(historical_data) >= 7 else historical_data[target_column].mean()

    # Calculate approximate prediction interval based on historical residuals
    # First, get predictions on historical data
    try:
        historical_preds = model.predict(historical_data[features])
        # Ensure non-negative and rounded for residual calculation
        historical_preds = np.maximum(0, historical_preds).round(0)
        residuals = historical_data[target_column].values - historical_preds
        # Use 1.96 for approx 95% confidence interval, clamped at 0 for lower bound
        residual_std = np.std(residuals) if len(residuals) > 1 else 0
        interval_width = 1.96 * residual_std # Standard for 95% CI assuming normal distribution
    except Exception as e:
        st.warning(f"Could not calculate historical residuals for interval estimation: {e}. Intervals will be zero.")
        interval_width = 0

    for idx, row in future_df.iterrows():
        try:
            feature_values = []

            for feature in features:
                if feature.startswith(f'Lag_{target_column}_'):
                    lag_num = int(feature.replace(f'Lag_{target_column}_', '')) - 1
                    if lag_num < len(current_lags):
                        feature_values.append(current_lags[lag_num])
                    else:
                        feature_values.append(0)
                elif feature == f'Rolling_Mean_3_{target_column}':
                    feature_values.append(historical_mean_3)
                elif feature == f'Rolling_Mean_7_{target_column}':
                    feature_values.append(historical_mean_7)
                else:
                    if feature in row:
                        feature_values.append(row[feature])
                    else:
                        st.warning(f"Feature '{feature}' not found in future_df row, defaulting to 0. Check feature consistency.")
                        feature_values.append(0)

            if len(feature_values) != len(features):
                st.error(f"Feature vector length mismatch: Expected {len(features)}, got {len(feature_values)}. Skipping prediction.")
                pred = historical_data[target_column].mean()
                pred_low = 0
                pred_high = 0
            else:
                feature_vector = np.array(feature_values).reshape(1, -1)
                pred = model.predict(feature_vector)[0]
                pred = max(0, pred)
                pred = round(pred)

                # Calculate prediction intervals
                pred_low = max(0, round(pred - interval_width))
                pred_high = round(pred + interval_width)

            predictions.append(pred)
            pred_lows.append(pred_low)
            pred_highs.append(pred_high)

            current_lags = [pred] + current_lags[:6]

            temp_rolling_data = list(reversed(predictions)) + list(reversed(historical_data[target_column].values))
            if len(temp_rolling_data) >= 3:
                historical_mean_3 = np.mean(temp_rolling_data[:3])
            if len(temp_rolling_data) >= 7:
                historical_mean_7 = np.mean(temp_rolling_data[:7])

        except Exception as e:
            st.error(f"Error in prediction step {idx} for {target_column}: {e}")
            predictions.append(historical_data[target_column].mean())
            pred_lows.append(0)
            pred_highs.append(0)

    return pd.DataFrame({
        'Predicted': predictions,
        'Predicted_Low': pred_lows,
        'Predicted_High': pred_highs
    }, index=future_df.index)


def predict_prophet(historical_data, future_df_features, target_column):
    """
    Forecasts using Facebook Prophet.
    Returns point forecasts, lower, and upper bounds.
    """
    df_prophet = historical_data[['Datetime', target_column]].rename(columns={'Datetime': 'ds', target_column: 'y'})

    # Initialize Prophet with sensible defaults
    m = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode='additive',
        interval_width=0.95 # Confidence interval width for yhat_lower/yhat_upper
    )

    # Add Irish holidays to Prophet model
    calendar = IrishBankHolidays()
    holidays_df = calendar.holidays(start=df_prophet['ds'].min(), end=future_df_features['Datetime'].max() + timedelta(days=30))
    holidays_df = pd.DataFrame({'ds': holidays_df, 'holiday': 'Irish Holiday'})
    m.add_country_holidays(country_name='IE') 

    m.fit(df_prophet)

    future = future_df_features[['Datetime']].rename(columns={'Datetime': 'ds'})
    forecast = m.predict(future)

    # Extract relevant columns and rename for consistency
    forecast_results = pd.DataFrame({
        'Predicted': np.maximum(0, forecast['yhat']).round(0),
        'Predicted_Low': np.maximum(0, forecast['yhat_lower']).round(0),
        'Predicted_High': np.maximum(0, forecast['yhat_upper']).round(0),
        'Datetime': forecast['ds']
    }, index=future_df_features.index)

    return forecast_results

def predict_hybrid(historical_data, future_df_features, features, target_column, residual_model_name='LightGBM'):
    """
    Hybrid forecasting: Prophet for trend/seasonality, and a specified ML model for residuals.
    Returns point forecasts, lower, and upper bounds (from Prophet).
    """
    if historical_data.empty:
        st.error("Historical data for hybrid forecasting is empty. Cannot generate forecasts.")
        return pd.DataFrame({
            'Predicted': [0] * len(future_df_features),
            'Predicted_Low': [0] * len(future_df_features),
            'Predicted_High': [0] * len(future_df_features)
        }, index=future_df_features.index)

    # --- Step 1: Prophet base forecast ---
    df_prophet_train = historical_data[['Datetime', target_column]].rename(columns={'Datetime': 'ds', target_column: 'y'})
    m_prophet = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode='additive',
        interval_width=0.95
    )
    m_prophet.add_country_holidays(country_name='IE') # Add Irish holidays
    m_prophet.fit(df_prophet_train)

    # Get historical Prophet predictions to calculate residuals
    historical_prophet_future = df_prophet_train[['ds']]
    historical_prophet_forecast = m_prophet.predict(historical_prophet_future)
    historical_data['prophet_yhat'] = historical_prophet_forecast['yhat'].values
    historical_data['prophet_yhat_lower'] = historical_prophet_forecast['yhat_lower'].values
    historical_data['prophet_yhat_upper'] = historical_prophet_forecast['yhat_upper'].values

    # Calculate residuals for ML model training
    historical_data['residuals'] = historical_data[target_column] - historical_data['prophet_yhat']

    # --- Step 2: ML model for residuals ---
    # Generate lags and rolling features specifically for residuals
    residual_data_for_lags, residual_lag_features = add_lag_features_smart(historical_data[['Hospital', 'Datetime', 'residuals']].copy(), 'residuals')

    # Merge lagged residuals back to the main historical_data
    historical_data = pd.merge(historical_data, residual_data_for_lags.drop(columns=['residuals']), on=['Hospital', 'Datetime'], how='left')
    historical_data = historical_data.dropna(subset=['residuals']) # Drop rows where residuals or Prophet yhat were NaN

    # Features for the ML residual model:
    ml_residual_features = [f for f in features if f != target_column and f in historical_data.columns] # Original features
    if 'prophet_yhat' not in ml_residual_features:
        ml_residual_features.append('prophet_yhat') # Add Prophet's point forecast as a feature
    ml_residual_features.extend([f for f in residual_lag_features if f in historical_data.columns]) # Add residual lags

    # Ensure all features exist and are not NaN
    X_ml_res = historical_data.dropna(subset=ml_residual_features)[ml_residual_features]
    y_ml_res = historical_data.dropna(subset=ml_residual_features)['residuals']

    if X_ml_res.empty or len(X_ml_res) < 5:
        st.warning(f"Insufficient data to train {residual_model_name} model for hybrid residuals. Falling back to Prophet only.")
        future_prophet_forecast = m_prophet.predict(future_df_features[['Datetime']].rename(columns={'Datetime': 'ds'}))
        return pd.DataFrame({
            'Predicted': np.maximum(0, future_prophet_forecast['yhat']).round(0),
            'Predicted_Low': np.maximum(0, future_prophet_forecast['yhat_lower']).round(0),
            'Predicted_High': np.maximum(0, future_prophet_forecast['yhat_upper']).round(0),
            'Datetime': future_df_features['Datetime']
        }, index=future_df_features.index)

    # Initialize and train ML model for residuals
    # Using a slightly different learning rate for residual models
    if residual_model_name == 'LightGBM':
        ml_residual_model = lgb.LGBMRegressor(
            n_estimators=min(500, len(X_ml_res) * 2), learning_rate=0.03, num_leaves=20,
            max_depth=-1, min_child_samples=10, subsample=0.7, colsample_bytree=0.7,
            random_state=42, n_jobs=-1, objective='regression_l1' # MAE objective
        )
    elif residual_model_name == 'CatBoost':
        ml_residual_model = cb.CatBoostRegressor(
            iterations=min(500, len(X_ml_res) * 2), learning_rate=0.03, depth=5,
            subsample=0.7, colsample_bylevel=0.7, l2_leaf_reg=2,
            verbose=False, random_state=42, allow_writing_files=False,
            bagging_temperature=1, od_type='Iter', od_wait=20,
            loss_function='MAE' # Changed to MAE as requested
        )
    else:
        st.error(f"Unsupported residual model: {residual_model_name}. Defaulting to LightGBM for residuals.")
        return predict_hybrid(historical_data, future_df_features, features, target_column, residual_model_name='LightGBM') # Fallback


    ml_residual_model.fit(X_ml_res, y_ml_res)

    # --- Step 3: Forecast future with hybrid model ---
    # Get Prophet's future forecast
    future_prophet_forecast = m_prophet.predict(future_df_features[['Datetime']].rename(columns={'Datetime': 'ds'}))
    future_df_features['prophet_yhat'] = future_prophet_forecast['yhat'].values # Add Prophet's yhat to future_df
    future_df_features['prophet_yhat_lower'] = future_prophet_forecast['yhat_lower'].values
    future_df_features['prophet_yhat_upper'] = future_prophet_forecast['yhat_upper'].values

    # Iteratively forecast residuals with ML model for future dates
    predicted_residuals = []
    
    # Initialize residual lags/rolling means for future forecasting based on historical residuals
    last_residual_values_hist = historical_data['residuals'].tail(7).values
    current_residual_lags = list(reversed(last_residual_values_hist.tolist())) + [0] * (7 - len(last_residual_values_hist))

    historical_residual_mean_3 = historical_data['residuals'].tail(3).mean() if len(historical_data) >= 3 else historical_data['residuals'].mean()
    historical_residual_mean_7 = historical_data['residuals'].tail(7).mean() if len(historical_data) >= 7 else historical_data['residuals'].mean()


    for idx, row in future_df_features.iterrows():
        try:
            residual_feature_values = []
            for feature in ml_residual_features:
                if feature.startswith('Lag_residuals_'):
                    lag_num = int(feature.replace('Lag_residuals_', '')) - 1
                    if lag_num < len(current_residual_lags):
                        residual_feature_values.append(current_residual_lags[lag_num])
                    else:
                        residual_feature_values.append(0)
                elif feature == 'Rolling_Mean_3_residuals':
                    residual_feature_values.append(historical_residual_mean_3)
                elif feature == 'Rolling_Mean_7_residuals':
                    residual_feature_values.append(historical_residual_mean_7)
                elif feature in row: # For other features like Prophet's yhat, Hour, DayOfWeek etc.
                    residual_feature_values.append(row[feature])
                else:
                    residual_feature_values.append(0) # Default for missing features

            if len(residual_feature_values) != len(ml_residual_features):
                st.warning(f"Feature vector length mismatch for residual prediction: Expected {len(ml_residual_features)}, got {len(residual_feature_values)}. Skipping.")
                predicted_residuals.append(0)
                continue

            pred_res = ml_residual_model.predict(np.array(residual_feature_values).reshape(1, -1))[0]
            predicted_residuals.append(pred_res)

            # Update residual lags for next iteration
            current_residual_lags = [pred_res] + current_residual_lags[:6]

            # Update rolling residual means
            temp_res_rolling_data = list(reversed(predicted_residuals)) + list(reversed(historical_data['residuals'].values))
            if len(temp_res_rolling_data) >= 3:
                historical_residual_mean_3 = np.mean(temp_res_rolling_data[:3])
            if len(temp_res_rolling_data) >= 7:
                historical_residual_mean_7 = np.mean(temp_res_rolling_data[:7])

        except Exception as e:
            st.warning(f"Error predicting residual for future step {idx}: {e}. Falling back to 0 for residual.")
            predicted_residuals.append(0)

    # Final hybrid prediction = Prophet base + ML residual prediction
    hybrid_predictions = future_prophet_forecast['yhat'].values + np.array(predicted_residuals)
    hybrid_predictions = np.maximum(0, hybrid_predictions).round(0)

    return pd.DataFrame({
        'Predicted': hybrid_predictions,
        # Use Prophet's original intervals for low/high, as residual model intervals are complex
        'Predicted_Low': np.maximum(0, future_prophet_forecast['yhat_lower']).round(0),
        'Predicted_High': np.maximum(0, future_prophet_forecast['yhat_upper']).round(0),
        'Datetime': future_df_features['Datetime'] # Carry Datetime for merge
    }, index=future_df_features.index)


def plot_forecasts(historical_data, forecast_data, metric_name, hospital_name, show_intervals=False):
    """Creates an interactive Plotly chart visualizing historical data and forecasts."""
    fig = go.Figure()

    # Add historical data trace
    fig.add_trace(go.Scatter(
        x=historical_data['Datetime'],
        y=historical_data[metric_name], # Use the actual metric name (e.g., 'ED Beds')
        mode='lines+markers',
        name='Historical',
        line=dict(color='blue'),
        marker=dict(size=4)
    ))

    # Add forecast data trace (point forecast)
    fig.add_trace(go.Scatter(
        x=forecast_data['Datetime'],
        y=forecast_data['Predicted'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='red', dash='dash'), # Dashed line for forecast
        marker=dict(size=6, symbol='diamond')
    ))

    # Add forecast intervals if requested and available
    # Check if 'Predicted_Low' and 'Predicted_High' exist and are not all zeros
    if show_intervals and 'Predicted_Low' in forecast_data.columns and 'Predicted_High' in forecast_data.columns:
        # Check if there's actual variation in intervals to plot them
        if not (forecast_data['Predicted_Low'].eq(forecast_data['Predicted']).all() and forecast_data['Predicted_High'].eq(forecast_data['Predicted']).all()):
            fig.add_trace(go.Scatter(
                x=pd.concat([forecast_data['Datetime'], forecast_data['Datetime'].iloc[::-1]]), # go forward then backward
                y=pd.concat([forecast_data['Predicted_High'], forecast_data['Predicted_Low'].iloc[::-1]]), # upper, then lower reversed
                fill='toself',
                fillcolor='rgba(255,0,0,0.1)', # Light red fill
                line=dict(color='rgba(255,255,255,0)'), # Transparent line
                name='Forecast Interval',
                hoverinfo='skip' # Don't show hover for fill
            ))

    # Add a vertical line to visually separate historical data from the forecast
    last_historical_date = historical_data['Datetime'].max()
    fig.add_shape(
        type="line",
        x0=last_historical_date,
        x1=last_historical_date,
        y0=0,
        y1=1,
        yref="paper", # Relative to plot height
        line=dict(color="gray", width=2, dash="dot"),
    )

    # Add annotation for the forecast start point
    fig.add_annotation(
        x=last_historical_date,
        y=0.95,
        yref="paper",
        text="Forecast Start",
        showarrow=False,
        bgcolor="white",
        bordercolor="gray",
        borderwidth=1
    )

    # Update plot layout with titles and labels
    fig.update_layout(
        title=f'{metric_name} Forecast - {hospital_name}',
        xaxis_title='Date',
        yaxis_title=f'{metric_name} Count',
        hovermode='x unified', # Show all traces' values on hover
        showlegend=True,
        height=500
    )

    return fig

def add_forecasting_insights():
    """Displays insights and tips about the forecasting process."""
    with st.expander("💡 Forecasting Insights & Tips", expanded=False):
        st.subheader("Data Requirements")
        st.markdown("""
        For accurate forecasting, you need:
        * **Minimum 30 records** per hospital-metric combination
        * **Consistent time intervals** (8am, 2pm, 8pm readings are ideal)
        * **Recent data** (within the last 6 months ideally for best relevance)
        * **Complete records** (avoid too many missing values, as they can hinder model performance)
        """)

        st.subheader("Understanding Your Results")
        st.markdown("""
        * **RMSE (Root Mean Square Error)**: Lower values indicate better model accuracy. This metric represents the average magnitude of the errors in your predictions.
        * **Historical vs. Forecast**: The generated chart clearly visualizes your past data patterns and the predicted future values, allowing for easy comparison.
        * **Validation**: The model's performance (RMSE) is calculated on a subset of your historical data, showing how well it generalizes to unseen but similar data.
        * **Prediction Intervals (Forecast Low/High)**: For Prophet and Hybrid models, these directly come from the model's uncertainty estimates. For other models, they are **approximate intervals** derived from the historical prediction residuals, providing a general sense of forecast variability.
        """)

@st.cache_resource # Cache the trained model for faster subsequent runs
def get_ml_model(model_name: str, X_train: pd.DataFrame, y_train: pd.Series, enable_tuning: bool, tuning_iterations: int):
    """
    Initializes and returns the selected machine learning model,
    with optional hyperparameter tuning.
    """
    param_grid = {} # Define hyperparameter search space

    if model_name == "CatBoost":
        model_class = cb.CatBoostRegressor
        base_params = {
            'verbose': False, 'random_state': 42, 'allow_writing_files': False,
            'bagging_temperature': 1, 'od_type': 'Iter', 'od_wait': 50, 'loss_function': 'RMSE'
        }
        if enable_tuning:
            param_grid = {
                'iterations': [100, 250, 500], # Reduced for faster tuning
                'learning_rate': [0.03, 0.05, 0.08],
                'depth': [4, 6, 8],
                'l2_leaf_reg': [1, 3, 5]
            }
        else:
            base_params['iterations'] = min(1000, len(X_train) * 3) # Default iterations
            base_params['learning_rate'] = 0.08
            base_params['depth'] = 6
            base_params['l2_leaf_reg'] = 3

    elif model_name == "LightGBM":
        model_class = lgb.LGBMRegressor
        base_params = {
            'random_state': 42, 'n_jobs': -1, 'objective': 'regression_l1'
        }
        if enable_tuning:
            param_grid = {
                'n_estimators': [100, 250, 500], # Reduced for faster tuning
                'learning_rate': [0.03, 0.05, 0.08],
                'num_leaves': [20, 31, 50],
                'max_depth': [-1, 10, 20],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            }
        else:
            base_params['n_estimators'] = min(1000, len(X_train) * 3)
            base_params['learning_rate'] = 0.05
            base_params['num_leaves'] = 31

    elif model_name == "XGBoost":
        model_class = xgb.XGBRegressor
        base_params = {
            'random_state': 42, 'n_jobs': -1, 'objective': 'reg:squarederror'
        }
        if enable_tuning:
            param_grid = {
                'n_estimators': [100, 250, 500], # Reduced for faster tuning
                'learning_rate': [0.03, 0.05, 0.08],
                'max_depth': [4, 6, 8],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            }
        else:
            base_params['n_estimators'] = min(1000, len(X_train) * 3)
            base_params['learning_rate'] = 0.05
            base_params['max_depth'] = 6

    elif model_name == "GradientBoosting (Scikit-learn)":
        model_class = GradientBoostingRegressor
        base_params = {
            'random_state': 42, 'loss': 'huber'
        }
        if enable_tuning:
            param_grid = {
                'n_estimators': [50, 100, 200], # Reduced for faster tuning
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [3, 5, 7],
                'subsample': [0.7, 0.8, 0.9]
            }
        else:
            base_params['n_estimators'] = min(500, len(X_train) * 2)
            base_params['learning_rate'] = 0.1
            base_params['max_depth'] = 5

    else:
        st.error(f"Invalid model '{model_name}' selected for tuning. Defaulting to CatBoost.")
        return get_ml_model("CatBoost", X_train, y_train, enable_tuning, tuning_iterations)

    if enable_tuning and len(param_grid) > 0 and len(X_train) >= 20: # Only tune if enough data and params defined
        st.info(f"🚀 Starting Hyperparameter Tuning for {model_name} with {tuning_iterations} iterations...")
        # Use TimeSeriesSplit for cross-validation during tuning
        tscv_tuning = TimeSeriesSplit(n_splits=min(5, max(2, len(X_train) // 10))) # At least 2 splits for tuning
        
        # Use n_iter for how many parameter settings are sampled.
        random_search = RandomizedSearchCV(
            estimator=model_class(**base_params),
            param_distributions=param_grid,
            n_iter=tuning_iterations,
            scoring='neg_root_mean_squared_error', # Optimize for RMSE
            cv=tscv_tuning,
            verbose=0, # Suppress verbose output from GridSearchCV
            random_state=42,
            n_jobs=-1 # Use all available cores for tuning
        )
        random_search.fit(X_train, y_train)
        st.success(f"✅ Tuning complete. Best parameters for {model_name}: {random_search.best_params_}")
        st.info(f"Best CV RMSE during tuning: {-random_search.best_score_:.2f}")
        return random_search.best_estimator_
    else:
        if enable_tuning and len(X_train) < 20:
             st.warning(f"Insufficient data ({len(X_train)} records) for hyperparameter tuning. Training {model_name} with default parameters.")
        # Return model with default parameters if tuning is not enabled or not enough data
        return model_class(**base_params).fit(X_train, y_train) # Fit default model if no tuning

# --- Streamlit UI ---
st.title("Emergency Department Forecasting (Ireland)")
st.markdown("Upload your ED Excel file, select hospital(s), and generate 7-day forecasts.")

# Sidebar control for number of forecast days
forecast_days = st.sidebar.slider("Forecast Days", 1, 14, 7)

# Model selection dropdown in sidebar
st.sidebar.header("Model Settings")
model_option = st.sidebar.selectbox(
    "Select ML Model:",
    options=["CatBoost", "LightGBM", "XGBoost", "GradientBoosting (Scikit-learn)", "Prophet", "Prophet-LightGBM Hybrid", "Prophet-CatBoost Hybrid"]
)

# Hyperparameter Tuning options
st.sidebar.subheader("Hyperparameter Tuning (Tree-based models)")
enable_tuning = st.sidebar.checkbox("Enable Tuning", value=False,
    help="Applies RandomizedSearchCV for optimal hyperparameters. Can increase processing time significantly.")
tuning_iterations = st.sidebar.slider("Tuning Iterations (if enabled)", 5, 50, 10,
    help="Number of parameter settings that are sampled. More iterations can lead to better models but take longer.")

if model_option in ["Prophet", "Prophet-LightGBM Hybrid", "Prophet-CatBoost Hybrid"] and enable_tuning:
    st.sidebar.warning("Hyperparameter tuning is not available for Prophet or Hybrid models via this interface.")
    enable_tuning = False # Disable tuning for non-tree models

# File uploader widget
uploaded_file = st.file_uploader("Upload your ED Excel File", type=["xlsx"])

if uploaded_file:
    try:
        # Load and prepare data using the updated function
        df = pd.read_excel(uploaded_file)
        df_processed = prepare_data(df) # Renamed to df_processed as it's no longer just 'long'

        # Show data loading success message and summary
        st.success(f"✅ Data loaded and processed successfully! {len(df_processed)} records found.")

        # Display a sample of the processed data to confirm the new structure
        st.subheader("📋 Sample of Processed Data")
        st.dataframe(df_processed[['Date', 'Time', 'ED Beds', 'Trolleys', 'Capacity']].head())

        # Get unique hospitals for selection in the sidebar
        hospitals = sorted(df_processed['Hospital'].unique())

        # Show data summary by Hospital and the new metrics
        data_summary = df_processed.groupby(['Hospital']).agg(
            ed_records=('ED Beds', 'count'),
            trolley_records=('Trolleys', 'count'),
            capacity_records=('Capacity', 'count') # Added capacity records to summary
        ).reset_index()
        st.subheader("📊 Data Summary by Hospital")
        st.dataframe(data_summary, use_container_width=True)

        # Hospital selection dropdown
        st.sidebar.header("Forecast Settings")
        hospital_option = st.sidebar.selectbox(
            "Select Hospital:",
            options=["All Hospitals"] + hospitals
        )

        # Run forecast button
        run_forecast = st.sidebar.button("🚀 Run Forecast", type="primary")

        if run_forecast:
            st.header("📊 Forecast Results")
            # Use a spinner to indicate ongoing computation
            with st.spinner("Generating forecasts... This may take a few moments depending on data size, model complexity, and tuning settings."):

                # Determine which hospitals to process based on user selection
                if hospital_option == "All Hospitals":
                    selected_hospitals = hospitals
                else:
                    selected_hospitals = [hospital_option]

                # Define base features that are common to all target columns
                base_features = [
                    'Hour', 'DayOfWeek', 'DayOfMonth', 'Month', 'Quarter', 'WeekOfYear',
                    'IsWeekend', 'IsMonday', 'IsFriday',
                    'Hour_sin', 'Hour_cos', 'Day_sin', 'Day_cos', 'Month_sin', 'Month_cos',
                    'IsHoliday', 'IsSummer', 'IsWinter', 'IsPeakHour', 'IsLowHour',
                    'Hospital_Code'
                ]

                # Iterate through each selected hospital and metric ('ED Beds', 'Trolleys', 'Capacity')
                for hospital in selected_hospitals:
                    st.subheader(f"🏥 {hospital}")

                    # Filter data for the current hospital
                    hospital_data = df_processed[df_processed['Hospital'] == hospital].copy()

                    # Extract hospital code and current additional capacity (now 'Capacity') for future dates
                    hospital_code = hospital_data['Hospital_Code'].iloc[0] if not hospital_data.empty else 0
                    # Use the last known capacity value for forecasting future capacity or as a feature for other forecasts
                    current_hospital_capacity_val = hospital_data['Capacity'].fillna(0).iloc[-1] if not hospital_data.empty else 0


                    # Get the last date from the historical data for creating future dates
                    last_date = hospital_data['Datetime'].max().date() if not hospital_data.empty else datetime.now().date()

                    # Process 'ED Beds', 'Trolleys', and 'Capacity' forecasts separately
                    for target_col_name in ['ED Beds', 'Trolleys', 'Capacity']: # Added 'Capacity' as a target
                        # Check if we have sufficient data for the current target column
                        if hospital_data[target_col_name].count() < 10:
                            st.warning(f"⚠️ Insufficient data for '{target_col_name}' at {hospital} ({hospital_data[target_col_name].count()} records). Need at least 10 records for meaningful forecasting.")
                            continue

                        st.info(f"Processing '{target_col_name}' for {hospital} using {model_option}")

                        # Create future dates DataFrame (common for all models)
                        future_df_base = create_future_dates(
                            pd.to_datetime(last_date),
                            hospital,
                            hospital_code,
                            current_hospital_capacity_val,
                            days=forecast_days
                        )

                        avg_rmse = np.nan # Initialize RMSE for all models

                        # --- Model-specific forecasting logic ---
                        if model_option == "Prophet":
                            # Prophet handles its own feature engineering and iterative prediction
                            if target_col_name == 'Capacity':
                                st.warning("Prophet model may not be ideal for Capacity forecasting if it's static or fixed. It's designed for time-varying data.")
                            forecast_results = predict_prophet(hospital_data, future_df_base, target_col_name)
                            # Calculate RMSE for Prophet on historical data for display
                            df_prophet_eval = hospital_data[['Datetime', target_col_name]].rename(columns={'Datetime': 'ds', target_col_name: 'y'})
                            m_eval = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True, seasonality_mode='additive', interval_width=0.95)
                            m_eval.add_country_holidays(country_name='IE')
                            m_eval.fit(df_prophet_eval)
                            historical_prophet_preds = m_eval.predict(df_prophet_eval[['ds']])['yhat'].values
                            historical_prophet_preds = np.maximum(0, historical_prophet_preds).round(0)
                            avg_rmse = np.sqrt(mean_squared_error(hospital_data[target_col_name].values, historical_prophet_preds))
                            st.info(f"Prophet's training RMSE for {target_col_name}: {avg_rmse:.2f} (on historical data)")
                            
                        elif model_option in ["Prophet-LightGBM Hybrid", "Prophet-CatBoost Hybrid"]:
                            # Hybrid model
                            if target_col_name == 'Capacity':
                                st.warning("Hybrid model may not be ideal for Capacity forecasting if it's static or fixed. It's designed for time-varying data.")
                            
                            residual_model = 'LightGBM' if model_option == "Prophet-LightGBM Hybrid" else 'CatBoost'
                            
                            forecast_results = predict_hybrid(hospital_data, future_df_base, base_features, target_col_name, residual_model_name=residual_model)
                            
                            # Calculate RMSE for Hybrid on historical data (Prophet component's RMSE)
                            df_prophet_eval = hospital_data[['Datetime', target_col_name]].rename(columns={'Datetime': 'ds', target_col_name: 'y'})
                            m_eval = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True, seasonality_mode='additive', interval_width=0.95)
                            m_eval.add_country_holidays(country_name='IE')
                            m_eval.fit(df_prophet_eval)
                            historical_prophet_preds = m_eval.predict(df_prophet_eval[['ds']])['yhat'].values
                            historical_prophet_preds = np.maximum(0, historical_prophet_preds).round(0)
                            avg_rmse = np.sqrt(mean_squared_error(hospital_data[target_col_name].values, historical_prophet_preds))
                            st.info(f"Hybrid model's base Prophet training RMSE for {target_col_name}: {avg_rmse:.2f} (on historical data)")

                        else: # Tree-based models (CatBoost, LightGBM, XGBoost, GradientBoosting)
                            # Add lag and rolling features specifically for the current target column
                            data_with_lags, lag_features = add_lag_features_smart(hospital_data.copy(), target_col_name)

                            # Determine features for this specific model
                            model_features = base_features[:]
                            if target_col_name != 'Capacity' and 'Capacity' in data_with_lags.columns:
                                model_features.append('Capacity')

                            all_features_for_model = model_features + lag_features
                            available_features = [f for f in all_features_for_model if f in data_with_lags.columns and f != target_col_name]
                            training_data = data_with_lags.dropna(subset=[target_col_name] + available_features)

                            if len(training_data) < 5:
                                st.warning(f"⚠️ After preprocessing, insufficient data for '{target_col_name}' at {hospital} ({len(training_data)} records). Need at least 5 records to train a model.")
                                continue

                            X = training_data[available_features]
                            y = training_data[target_col_name]

                            # Initialize and/or tune the selected model
                            model = get_ml_model(model_option, X, y, enable_tuning, tuning_iterations)

                            # --- Time Series Cross-Validation for tree-based models ---
                            # If tuning was enabled, get_ml_model would have already reported the best CV RMSE.
                            # Otherwise, calculate RMSE for the default model.
                            if not enable_tuning and len(X) >= 20:
                                tscv = TimeSeriesSplit(n_splits=min(5, max(1, len(X) // 10)))
                                fold_rmses = []
                                for fold_idx, (train_index, test_index) in enumerate(tscv.split(X)):
                                    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
                                    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

                                    if len(X_train_fold) > 0 and len(X_test_fold) > 0:
                                        fold_model = get_ml_model(model_option, X_train_fold, y_train_fold, False, 0) # No tuning in CV folds
                                        y_pred_fold = fold_model.predict(X_test_fold)
                                        y_pred_fold = np.maximum(0, y_pred_fold).round(0)
                                        fold_rmses.append(np.sqrt(mean_squared_error(y_test_fold, y_pred_fold)))
                                    else:
                                        st.warning(f"Skipping fold {fold_idx+1} due to insufficient data for '{target_col_name}' at {hospital}.")

                                if fold_rmses:
                                    avg_rmse = np.mean(fold_rmses)
                                    st.info(f"Cross-Validation RMSE for {target_col_name}: {avg_rmse:.2f} (Avg. over {len(fold_rmses)} folds)")
                                else:
                                    st.warning(f"Could not perform cross-validation for {target_col_name} due to insufficient data or valid folds.")
                            elif not enable_tuning and len(X) > 0: # Fallback to training RMSE if not enough data for CV
                                y_pred_train = model.predict(X)
                                y_pred_train = np.maximum(0, y_pred_train).round(0)
                                avg_rmse = np.sqrt(mean_squared_error(y, y_pred_train))
                                st.info(f"Training RMSE for {target_col_name}: {avg_rmse:.2f} (Trained on all available data)")
                            
                            # Generate predictions for future dates using the trained model
                            forecast_results = forecast_with_lags(model, training_data, future_df_base, available_features, target_col_name)


                        # --- Common display for all models ---
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(f"{target_col_name} RMSE", f"{avg_rmse:.2f}" if avg_rmse is not np.nan else "N/A")
                        with col2:
                            # For Prophet/Hybrid, training records reflect all data passed to Prophet
                            train_records_display = f"{len(X)}" if model_option not in ["Prophet", "Prophet-LightGBM Hybrid", "Prophet-CatBoost Hybrid"] else f"{len(hospital_data)}"
                            st.metric(f"Training Records", train_records_display)
                        with col3:
                            last_val_display = f"{hospital_data[target_col_name].iloc[-1]:.0f}" if not hospital_data.empty else "N/A"
                            st.metric(f"Last {target_col_name} Value", last_val_display)

                        # Create and display the forecast plot
                        # Pass show_intervals=True for all models now that intervals are generated for tree-based too
                        show_intervals = True 
                        fig = plot_forecasts(
                            hospital_data.tail(21), # Show last 21 historical points (approx. 1 week at 3 readings/day)
                            forecast_results, # Use the common forecast_results DataFrame
                            target_col_name,
                            hospital,
                            show_intervals=show_intervals
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Allow users to view detailed forecast data in an expandable section
                        with st.expander(f"📋 {target_col_name} Forecast Details"):
                            display_cols = ['Date', 'Time', 'Predicted']
                            if show_intervals:
                                display_cols.extend(['Predicted_Low', 'Predicted_High'])
                            
                            forecast_display = future_df_base.copy() # Start with base features
                            forecast_display['Predicted'] = forecast_results['Predicted']
                            if show_intervals:
                                forecast_display['Predicted_Low'] = forecast_results['Predicted_Low']
                                forecast_display['Predicted_High'] = forecast_results['Predicted_High']

                            # Ensure all displayed numbers are integers
                            for col in ['Predicted', 'Predicted_Low', 'Predicted_High']:
                                if col in forecast_display.columns:
                                    forecast_display[col] = forecast_display[col].astype(int)

                            st.dataframe(forecast_display[display_cols], use_container_width=True)

                        # Provide a download button for the forecast data
                        csv_data_cols = ['Datetime', 'Hospital', 'Predicted']
                        if show_intervals:
                            csv_data_cols.extend(['Predicted_Low', 'Predicted_High'])
                        
                        csv_data_to_download = future_df_base.copy()
                        csv_data_to_download['Predicted'] = forecast_results['Predicted']
                        if show_intervals:
                            csv_data_to_download['Predicted_Low'] = forecast_results['Predicted_Low']
                            csv_data_to_download['Predicted_High'] = forecast_results['Predicted_High']

                        csv_data_to_download['Metric'] = target_col_name
                        # Ensure all numbers are integers for CSV
                        for col in ['Predicted', 'Predicted_Low', 'Predicted_High']:
                            if col in csv_data_to_download.columns:
                                csv_data_to_download[col] = csv_data_to_download[col].astype(int)

                        st.download_button(
                            f"📥 Download {target_col_name} Forecast CSV",
                            csv_data_to_download[csv_data_cols + ['Metric']].to_csv(index=False),
                            file_name=f"{hospital}_{target_col_name.replace(' ', '_')}_forecast.csv",
                            mime="text/csv",
                            key=f"{hospital}_{target_col_name}_download"
                        )

                        st.divider()

                add_forecasting_insights()

    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please check that your Excel file contains the required columns and data format as described below.")

else:
    st.info("👆 Please upload an Excel file to begin forecasting.")
    with st.expander("📋 Expected File Format"):
        st.markdown("""
        **Required columns:**
        - `Hospital Group Name`: The group the hospital belongs to.
        - `Hospital`: Unique identifier for the hospital.
        - `Date`: The date of the observation (e.g., '30/05/2025').
        - `Tracker8am`, `Tracker2pm`, `Tracker8pm`: ED bed counts at 8 AM, 2 PM, 8 PM respectively.
        - `TimeTotal_8am`, `TimeTotal_2pm`, `TimeTotal_8pm`: Trolley counts at 8 AM, 2 PM, 8 PM respectively.
        - `AdditionalCapacityOpen Morning`: Any additional capacity opened in the morning for that day.

        **Data should contain:**
        - Historical ED and trolley count data.
        - Data for multiple hospitals (optional, but the app supports it).
        - At least **10-15 records** per hospital-metric combination for basic forecasting.
        - At least **30+ records** per hospital-metric combination for more reliable forecasting.
        """)
