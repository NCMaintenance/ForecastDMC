import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday
from pandas.tseries.offsets import DateOffset
from dateutil.rrule import MO
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.model_selection import RandomizedSearchCV
import optuna
from scipy import stats

# --- Define Irish Bank Holidays ---
class IrishBankHolidays(AbstractHolidayCalendar):
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

def prepare_data(df):
    """
    Prepares and feature engineers the data, reshaping it into the desired wide format
    with 'ED Beds', 'Trolleys', and 'Capacity' columns.
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
    df['Additional_Capacity'] = df.groupby(['Hospital', 'Date'])['Additional_Capacity'].transform('first')
    df['Additional_Capacity'] = df['Additional_Capacity'].fillna(0)

    # Define common id_vars for melting, these will become the identifier columns
    common_id_vars = ['Hospital Group Name', 'Hospital', 'Date', 'DayGAR', 'Additional_Capacity']

    # Melt ED counts into a long format
    df_ed = pd.melt(
        df,
        id_vars=common_id_vars,
        value_vars=['ED_8am', 'ED_2pm', 'ED_8pm'],
        var_name='Metric_Time_ED',
        value_name='ED Beds'
    )

    # Melt Trolley counts into a long format
    df_trolley = pd.melt(
        df,
        id_vars=common_id_vars,
        value_vars=['Trolley_8am', 'Trolley_2pm', 'Trolley_8pm'],
        var_name='Metric_Time_Trolley',
        value_name='Trolleys'
    )

    # Extract time label (e.g., '8am', '2pm', '8pm') from the temporary metric_time columns
    df_ed['TimeLabel'] = df_ed['Metric_Time_ED'].str.extract(r'([\d]+[ap]m)')
    df_trolley['TimeLabel'] = df_trolley['Metric_Time_Trolley'].str.extract(r'([\d]+[ap]m)')

    # Map time labels to standardized 24-hour format
    time_map = {'8am': '08:00', '2pm': '14:00', '8pm': '20:00'}
    df_ed['Time'] = df_ed['TimeLabel'].map(time_map)
    df_trolley['Time'] = df_trolley['TimeLabel'].map(time_map)

    # Drop temporary columns before merging
    df_ed = df_ed.drop(columns=['Metric_Time_ED', 'TimeLabel'])
    df_trolley = df_trolley.drop(columns=['Metric_Time_Trolley', 'TimeLabel'])

    # Merge the two melted dataframes
    df_merged = pd.merge(
        df_ed,
        df_trolley,
        on=['Hospital Group Name', 'Hospital', 'Date', 'DayGAR', 'Additional_Capacity', 'Time'],
        how='inner'
    )

    # Convert to numeric and drop missing values
    df_merged['ED Beds'] = pd.to_numeric(df_merged['ED Beds'], errors='coerce')
    df_merged['Trolleys'] = pd.to_numeric(df_merged['Trolleys'], errors='coerce')
    df_merged = df_merged.dropna(subset=['ED Beds', 'Trolleys'])

    # Rename 'Additional_Capacity' to 'Capacity'
    df_merged = df_merged.rename(columns={'Additional_Capacity': 'Capacity'})

    # Create unified Datetime column
    df_merged['Datetime'] = pd.to_datetime(df_merged['Date'].astype(str) + ' ' + df_merged['Time'])
    df_merged = df_merged.sort_values(by=['Hospital', 'Datetime']).reset_index(drop=True)

    return df_merged

def add_advanced_features(df):
    """
    Adds advanced feature engineering including interaction terms, holiday effects,
    and sophisticated time-based features.
    """
    df = df.copy()
    
    # Basic time features
    df['Hour'] = df['Datetime'].dt.hour
    df['DayOfWeek'] = df['Datetime'].dt.dayofweek
    df['DayOfMonth'] = df['Datetime'].dt.day
    df['Month'] = df['Datetime'].dt.month
    df['Quarter'] = df['Datetime'].dt.quarter
    df['WeekOfYear'] = df['Datetime'].dt.isocalendar().week
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    df['IsMonday'] = (df['DayOfWeek'] == 0).astype(int)
    df['IsFriday'] = (df['DayOfWeek'] == 4).astype(int)
    
    # Cyclical encoding
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['Day_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['Day_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    
    # Advanced time-based features
    df['DayOfYear'] = df['Datetime'].dt.dayofyear
    df['WeekOfMonth'] = df['Datetime'].dt.day.apply(lambda x: (x-1) // 7 + 1)
    df['IsMonthStart'] = (df['DayOfMonth'] <= 3).astype(int)
    df['IsMonthEnd'] = (df['DayOfMonth'] >= 28).astype(int)
    df['IsQuarterStart'] = ((df['Month'] % 3 == 1) & (df['DayOfMonth'] <= 7)).astype(int)
    df['IsQuarterEnd'] = ((df['Month'] % 3 == 0) & (df['DayOfMonth'] >= 25)).astype(int)
    
    # Holiday features with effects
    calendar = IrishBankHolidays()
    try:
        holidays = calendar.holidays(start=df['Datetime'].min(), end=df['Datetime'].max() + timedelta(days=30))
        df['IsHoliday'] = df['Datetime'].dt.normalize().isin(holidays).astype(int)
        
        # Holiday proximity effects
        df['DaysToHoliday'] = 0
        df['DaysFromHoliday'] = 0
        
        for holiday in holidays:
            days_diff = (df['Datetime'].dt.date - holiday.date()).dt.days
            df.loc[days_diff > 0, 'DaysFromHoliday'] = np.minimum(
                df.loc[days_diff > 0, 'DaysFromHoliday'], 
                days_diff[days_diff > 0]
            )
            df.loc[days_diff < 0, 'DaysToHoliday'] = np.minimum(
                df.loc[days_diff < 0, 'DaysToHoliday'], 
                -days_diff[days_diff < 0]
            )
        
        # Holiday week effects
        df['IsHolidayWeek'] = 0
        for holiday in holidays:
            holiday_week_start = holiday - timedelta(days=holiday.weekday())
            holiday_week_end = holiday_week_start + timedelta(days=6)
            df.loc[(df['Datetime'].dt.date >= holiday_week_start.date()) & 
                   (df['Datetime'].dt.date <= holiday_week_end.date()), 'IsHolidayWeek'] = 1
            
    except Exception:
        df['IsHoliday'] = 0
        df['DaysToHoliday'] = 0
        df['DaysFromHoliday'] = 0
        df['IsHolidayWeek'] = 0
    
    # Seasonal features
    df['IsSummer'] = df['Month'].isin([6, 7, 8]).astype(int)
    df['IsWinter'] = df['Month'].isin([12, 1, 2]).astype(int)
    df['IsSpring'] = df['Month'].isin([3, 4, 5]).astype(int)
    df['IsAutumn'] = df['Month'].isin([9, 10, 11]).astype(int)
    
    # Peak/low hour indicators
    df['IsPeakHour'] = df['Hour'].isin([20]).astype(int)
    df['IsLowHour'] = df['Hour'].isin([8]).astype(int)
    df['IsMidDay'] = df['Hour'].isin([14]).astype(int)
    
    # Workday patterns
    df['IsWorkingHours'] = ((df['Hour'] >= 8) & (df['Hour'] <= 18) & (~df['IsWeekend'])).astype(int)
    df['IsNightShift'] = ((df['Hour'] >= 20) | (df['Hour'] <= 6)).astype(int)
    
    # Interaction terms
    df['Weekend_x_Hour'] = df['IsWeekend'] * df['Hour']
    df['Holiday_x_Hour'] = df['IsHoliday'] * df['Hour']
    df['Weekend_x_PeakHour'] = df['IsWeekend'] * df['IsPeakHour']
    df['Monday_x_PeakHour'] = df['IsMonday'] * df['IsPeakHour']
    df['Friday_x_PeakHour'] = df['IsFriday'] * df['IsPeakHour']
    df['Winter_x_Weekend'] = df['IsWinter'] * df['IsWeekend']
    df['Summer_x_Weekend'] = df['IsSummer'] * df['IsWeekend']
    df['Holiday_x_Weekend'] = df['IsHoliday'] * df['IsWeekend']
    
    # Capacity interaction terms
    df['Capacity_x_Weekend'] = df['Capacity'] * df['IsWeekend']
    df['Capacity_x_PeakHour'] = df['Capacity'] * df['IsPeakHour']
    df['Capacity_x_Holiday'] = df['Capacity'] * df['IsHoliday']
    
    # Encode Hospital names
    df['Hospital_Code'] = df['Hospital'].astype('category').cat.codes
    
    # Hospital size indicators (based on data patterns)
    hospital_avg_capacity = df.groupby('Hospital')['Capacity'].mean()
    df['Hospital_Size'] = df['Hospital'].map(hospital_avg_capacity)
    df['IsLargeHospital'] = (df['Hospital_Size'] > df['Hospital_Size'].median()).astype(int)
    
    return df

def add_lag_features_advanced(df, target_column, min_data_threshold=20):
    """
    Adds advanced lag features including multiple time horizons and statistical measures.
    """
    df = df.copy()
    
    max_safe_lag = min(14, len(df) // 3)  # Increased lag window
    
    if max_safe_lag < 1:
        st.warning(f"Very limited data ({len(df)} records). Using minimal features for {target_column}.")
        return df, []
    
    lag_features = []
    
    # Standard lag features
    for i in range(1, min(8, max_safe_lag + 1)):
        lag_col = f'Lag_{target_column}_{i}'
        df[lag_col] = df[target_column].shift(i)
        lag_features.append(lag_col)
    
    # Same day previous readings (if available)
    if len(df) >= 3:
        df[f'SameDay_Prev_{target_column}'] = df[target_column].shift(1)
        lag_features.append(f'SameDay_Prev_{target_column}')
    
    # Weekly lags (same time, same day of week)
    if len(df) >= 21:  # 7 days * 3 readings per day
        df[f'WeeklyLag_{target_column}'] = df[target_column].shift(21)
        lag_features.append(f'WeeklyLag_{target_column}')
    
    # Rolling statistics with multiple windows
    if len(df) >= 6:
        # Short-term rolling statistics
        df[f'Rolling_Mean_3_{target_column}'] = df[target_column].rolling(window=3, min_periods=1).mean()
        df[f'Rolling_Std_3_{target_column}'] = df[target_column].rolling(window=3, min_periods=1).std()
        df[f'Rolling_Min_3_{target_column}'] = df[target_column].rolling(window=3, min_periods=1).min()
        df[f'Rolling_Max_3_{target_column}'] = df[target_column].rolling(window=3, min_periods=1).max()
        lag_features.extend([f'Rolling_Mean_3_{target_column}', f'Rolling_Std_3_{target_column}',
                           f'Rolling_Min_3_{target_column}', f'Rolling_Max_3_{target_column}'])
        
        if len(df) >= 14:
            # Medium-term rolling statistics
            df[f'Rolling_Mean_7_{target_column}'] = df[target_column].rolling(window=7, min_periods=1).mean()
            df[f'Rolling_Std_7_{target_column}'] = df[target_column].rolling(window=7, min_periods=1).std()
            df[f'Rolling_Median_7_{target_column}'] = df[target_column].rolling(window=7, min_periods=1).median()
            lag_features.extend([f'Rolling_Mean_7_{target_column}', f'Rolling_Std_7_{target_column}',
                               f'Rolling_Median_7_{target_column}'])
            
            if len(df) >= 42:
                # Long-term rolling statistics
                df[f'Rolling_Mean_14_{target_column}'] = df[target_column].rolling(window=14, min_periods=1).mean()
                df[f'Rolling_Trend_14_{target_column}'] = df[target_column].rolling(window=14, min_periods=1).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=False)
                lag_features.extend([f'Rolling_Mean_14_{target_column}', f'Rolling_Trend_14_{target_column}'])
    
    # Exponential weighted moving averages
    if len(df) >= 6:
        df[f'EWM_Mean_{target_column}'] = df[target_column].ewm(span=5, adjust=False).mean()
        df[f'EWM_Std_{target_column}'] = df[target_column].ewm(span=5, adjust=False).std()
        lag_features.extend([f'EWM_Mean_{target_column}', f'EWM_Std_{target_column}'])
    
    # Ratio features
    if len(df) >= 3:
        df[f'Ratio_Current_vs_Lag1_{target_column}'] = df[target_column] / (df[f'Lag_{target_column}_1'] + 1)
        df[f'Diff_Current_vs_Lag1_{target_column}'] = df[target_column] - df[f'Lag_{target_column}_1']
        lag_features.extend([f'Ratio_Current_vs_Lag1_{target_column}', f'Diff_Current_vs_Lag1_{target_column}'])
    
    # Fill NaN values
    for feature in lag_features:
        if feature in df.columns:
            df[feature] = df[feature].fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return df, lag_features

def optimize_lightgbm_with_optuna(X, y, n_trials=50, cv_folds=3):
    """
    Uses Optuna to optimize LightGBM hyperparameters for MAE.
    """
    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 10, 200),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'verbose': -1
        }
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        mae_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            mae_scores.append(mae)
        
        return np.mean(mae_scores)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    return study.best_params

def create_future_dates(last_date, hospital, hospital_code, additional_capacity, days=7):
    """
    Creates a DataFrame of future dates with advanced features.
    """
    future_dates = []
    times = ['08:00', '14:00', '20:00']
    
    for day in range(1, days + 1):
        future_date = last_date + timedelta(days=day)
        for time_str in times:
            future_datetime = pd.to_datetime(f"{future_date.date()} {time_str}")
            
            # Calculate all features that were used in training
            hour = future_datetime.hour
            day_of_week = future_datetime.dayofweek
            day_of_month = future_datetime.day
            month = future_datetime.month
            quarter = (month - 1) // 3 + 1
            week_of_year = future_datetime.isocalendar().week
            day_of_year = future_datetime.dayofyear
            week_of_month = (day_of_month - 1) // 7 + 1
            
            # Basic indicators
            is_weekend = int(day_of_week in [5, 6])
            is_monday = int(day_of_week == 0)
            is_friday = int(day_of_week == 4)
            is_month_start = int(day_of_month <= 3)
            is_month_end = int(day_of_month >= 28)
            is_quarter_start = int((month % 3 == 1) and (day_of_month <= 7))
            is_quarter_end = int((month % 3 == 0) and (day_of_month >= 25))
            
            # Cyclical encoding
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            day_sin = np.sin(2 * np.pi * day_of_week / 7)
            day_cos = np.cos(2 * np.pi * day_of_week / 7)
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)
            
            # Holiday features
            is_holiday = 0
            days_to_holiday = 0
            days_from_holiday = 0
            is_holiday_week = 0
            try:
                calendar = IrishBankHolidays()
                holidays = calendar.holidays(start=future_datetime, end=future_datetime + timedelta(days=365))
                is_holiday = int(future_datetime.normalize() in holidays)
                
                # Calculate holiday proximity
                for holiday in holidays:
                    days_diff = (future_datetime.date() - holiday.date()).days
                    if days_diff > 0:
                        days_from_holiday = min(days_from_holiday, days_diff) if days_from_holiday > 0 else days_diff
                    elif days_diff < 0:
                        days_to_holiday = min(days_to_holiday, -days_diff) if days_to_holiday > 0 else -days_diff
                    
                    # Holiday week
                    holiday_week_start = holiday - timedelta(days=holiday.weekday())
                    holiday_week_end = holiday_week_start + timedelta(days=6)
                    if holiday_week_start.date() <= future_datetime.date() <= holiday_week_end.date():
                        is_holiday_week = 1
                        
            except Exception:
                pass
            
            # Seasonal features
            is_summer = int(month in [6, 7, 8])
            is_winter = int(month in [12, 1, 2])
            is_spring = int(month in [3, 4, 5])
            is_autumn = int(month in [9, 10, 11])
            
            # Hour-based features
            is_peak_hour = int(hour == 20)
            is_low_hour = int(hour == 8)
            is_mid_day = int(hour == 14)
            is_working_hours = int((hour >= 8) and (hour <= 18) and (not is_weekend))
            is_night_shift = int((hour >= 20) or (hour <= 6))
            
            # Interaction terms
            weekend_x_hour = is_weekend * hour
            holiday_x_hour = is_holiday * hour
            weekend_x_peak_hour = is_weekend * is_peak_hour
            monday_x_peak_hour = is_monday * is_peak_hour
            friday_x_peak_hour = is_friday * is_peak_hour
            winter_x_weekend = is_winter * is_weekend
            summer_x_weekend = is_summer * is_weekend
            holiday_x_weekend = is_holiday * is_weekend
            
            # Capacity interactions
            capacity_x_weekend = additional_capacity * is_weekend
            capacity_x_peak_hour = additional_capacity * is_peak_hour
            capacity_x_holiday = additional_capacity * is_holiday
            
            future_dates.append({
                'Date': future_datetime.date(),
                'Time': time_str,
                'Datetime': future_datetime,
                'Hospital': hospital,
                'Hour': hour,
                'DayOfWeek': day_of_week,
                'DayOfMonth': day_of_month,
                'Month': month,
                'Quarter': quarter,
                'WeekOfYear': week_of_year,
                'DayOfYear': day_of_year,
                'WeekOfMonth': week_of_month,
                'IsWeekend': is_weekend,
                'IsMonday': is_monday,
                'IsFriday': is_friday,
                'IsMonthStart': is_month_start,
                'IsMonthEnd': is_month_end,
                'IsQuarterStart': is_quarter_start,
                'IsQuarterEnd': is_quarter_end,
                'Hour_sin': hour_sin,
                'Hour_cos': hour_cos,
                'Day_sin': day_sin,
                'Day_cos': day_cos,
                'Month_sin': month_sin,
                'Month_cos': month_cos,
                'IsHoliday': is_holiday,
                'DaysToHoliday': days_to_holiday,
                'DaysFromHoliday': days_from_holiday,
                'IsHolidayWeek': is_holiday_week,
                'IsSummer': is_summer,
                'IsWinter': is_winter,
                'IsSpring': is_spring,
                'IsAutumn': is_autumn,
                'IsPeakHour': is_peak_hour,
                'IsLowHour': is_low_hour,
                'IsMidDay': is_mid_day,
                'IsWorkingHours': is_working_hours,
                'IsNightShift': is_night_shift,
                'Weekend_x_Hour': weekend_x_hour,
                'Holiday_x_Hour': holiday_x_hour,
                'Weekend_x_PeakHour': weekend_x_peak_hour,
                'Monday_x_PeakHour': monday_x_peak_hour,
                'Friday_x_PeakHour': friday_x_peak_hour,
                'Winter_x_Weekend': winter_x_weekend,
                'Summer_x_Weekend': summer_x_weekend,
                'Holiday_x_Weekend': holiday_x_weekend,
                'Capacity_x_Weekend': capacity_x_weekend,
                'Capacity_x_PeakHour': capacity_x_peak_hour,
                'Capacity_x_Holiday': capacity_x_holiday,
                'Hospital_Code': hospital_code,
                'Capacity': additional_capacity,
                'Hospital_Size': additional_capacity,  # Simplified for future dates
                'IsLargeHospital': int(additional_capacity > 0)
            })
    
    return pd.DataFrame(future_dates)

def forecast_with_lags_advanced(model, historical_data, future_df, features, target_column):
    """
    Advanced forecasting with lag features and statistical measures.
    """
    last_values = historical_data[target_column].tail(14).values
    predictions = []
    current_lags = list(reversed(last_values))
    
    # Initialize rolling statistics
    historical_mean_3 = historical_data[target_column].tail(3).mean()
    historical_std_3 = historical_data[target_column].tail(3).std()
    historical_min_3 = historical_data[target_column].tail(3).min()
    historical_max_3 = historical_data[target_column].tail(3).max()
    
    historical_mean_7 = historical_data[target_column].tail(7).mean()
    historical_std_7 = historical_data[target_column].tail(7).std()
    historical_median_7 = historical_data[target_column].tail(7).median()
    
    historical_mean_14 = historical_data[target_column].tail(14).mean()
    
    # EWM initialization
    ewm_mean = historical_data[target_column].tail(5).mean()
    ewm_std = historical_data[target_column].tail(5).std()
    
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
                        
                elif feature == f'SameDay_Prev_{target_column}':
                    feature_values.append(current_lags[0] if current_lags else 0)
                    
                elif feature == f'WeeklyLag_{target_column}':
                    if len(current_lags) >= 21:
                        feature_values.append(current_lags[20])
                    else:
                        feature_values.append(historical_data[target_column].mean())
                        
                elif feature == f'Rolling_Mean_3_{target_column}':
                    feature_values.append(historical_mean_3)
                elif feature == f'Rolling_Std_3_{target_column}':
                    feature_values.append(historical_std_3)
                elif feature == f'Rolling_Min_3_{target_column}':
                    feature_values.append(historical_min_3)
                elif feature == f'Rolling_Max_3_{target_column}':
                    feature_values.append(historical_max_3)
                elif feature == f'Rolling_Mean_7_{target_column}':
                    feature_values.append(historical_mean_7)
                elif feature == f'Rolling_Std_7_{target_column}':
                    feature_values.append(historical_std_7)
                elif feature == f'Rolling_Median_7_{target_column}':
                    feature_values.append(historical_median_7)
                elif feature == f'Rolling_Mean_14_{target_column}':
                    feature_values.append(historical_mean_14)
                elif feature == f'Rolling_Trend_14_{target_column}':
                    # Simple trend calculation
                    if len(current_lags) >= 14:
                        trend = np.polyfit(range(14), current_lags[:14], 1)[0]
                    else:
                        trend = 0
                    feature_values.append(trend)
                elif feature == f'EWM_Mean_{target_column}':
                    feature_values.append(ewm_mean)
                elif feature == f'EWM_Std_{target_column}':
                    feature_values.append(ewm_std)
                elif feature == f'Ratio_Current_vs_Lag1_{target_column}':
                    if current_lags:
                        ratio = (current_lags[0] + 1) / (current_lags[0] + 1)  # Will be updated with prediction
                        feature_values.append(ratio)
                    else:
                        feature_values.append(1.0)
                elif feature == f'Diff_Current_vs_Lag1_{target_column}':
                    feature_values.append(0)  # Will be updated with prediction
                else:
                    # Regular feature from the row
                    if feature in row:
                        feature_values.append(row[feature])
                    else:
                        feature_values.append(0)
            
            # Make prediction
            X_pred = np.array(feature_values).reshape(1, -1)
            pred = model.predict(X_pred)[0]
            pred = max(0, pred)  # Ensure non-negative
            predictions.append(pred)
            
            # Update lags for next prediction
            current_lags.insert(0, pred)
            if len(current_lags) > 14:
                current_lags = current_lags[:14]
            
            # Update rolling statistics
            if len(predictions) >= 3:
                historical_mean_3 = np.mean(predictions[-3:])
                historical_std_3 = np.std(predictions[-3:])
                historical_min_3 = np.min(predictions[-3:])
                historical_max_3 = np.max(predictions[-3:])
            
            if len(predictions) >= 7:
                historical_mean_7 = np.mean(predictions[-7:])
                historical_std_7 = np.std(predictions[-7:])
                historical_median_7 = np.median(predictions[-7:])
            
            if len(predictions) >= 14:
                historical_mean_14 = np.mean(predictions[-14:])
            
            # Update EWM (simple approximation)
            alpha = 2 / (5 + 1)  # span=5
            ewm_mean = alpha * pred + (1 - alpha) * ewm_mean
            ewm_std = alpha * abs(pred - ewm_mean) + (1 - alpha) * ewm_std
            
        except Exception as e:
            st.warning(f"Error in prediction step {idx}: {e}")
            predictions.append(historical_data[target_column].mean())
    
    return predictions

def train_and_evaluate_model(df, target_column, hospital_filter=None, optimize_hyperparams=True):
    """
    Trains and evaluates the LightGBM model with advanced features.
    """
    # Filter data if hospital specified
    if hospital_filter:
        df = df[df['Hospital'] == hospital_filter].copy()
    
    if len(df) < 10:
        st.error(f"Insufficient data for {hospital_filter or 'all hospitals'}: {len(df)} records")
        return None, None, None, None, None
    
    # Add advanced features
    df = add_advanced_features(df)
    
    # Add lag features
    df, lag_features = add_lag_features_advanced(df, target_column)
    
    # Prepare features
    feature_columns = [
        'Hour', 'DayOfWeek', 'DayOfMonth', 'Month', 'Quarter', 'WeekOfYear',
        'IsWeekend', 'IsMonday', 'IsFriday', 'IsMonthStart', 'IsMonthEnd',
        'IsQuarterStart', 'IsQuarterEnd', 'Hour_sin', 'Hour_cos', 'Day_sin',
        'Day_cos', 'Month_sin', 'Month_cos', 'DayOfYear', 'WeekOfMonth',
        'IsHoliday', 'DaysToHoliday', 'DaysFromHoliday', 'IsHolidayWeek',
        'IsSummer', 'IsWinter', 'IsSpring', 'IsAutumn', 'IsPeakHour',
        'IsLowHour', 'IsMidDay', 'IsWorkingHours', 'IsNightShift',
        'Weekend_x_Hour', 'Holiday_x_Hour', 'Weekend_x_PeakHour',
        'Monday_x_PeakHour', 'Friday_x_PeakHour', 'Winter_x_Weekend',
        'Summer_x_Weekend', 'Holiday_x_Weekend', 'Capacity_x_Weekend',
        'Capacity_x_PeakHour', 'Capacity_x_Holiday', 'Hospital_Code',
        'Capacity', 'Hospital_Size', 'IsLargeHospital'
    ] + lag_features
    
    # Filter features that exist in the data
    available_features = [f for f in feature_columns if f in df.columns]
    
    if not available_features:
        st.error("No valid features found in the data")
        return None, None, None, None, None
    
    X = df[available_features]
    y = df[target_column]
    
    # Remove rows with missing target values
    valid_rows = ~y.isna()
    X = X[valid_rows]
    y = y[valid_rows]
    
    if len(X) < 10:
        st.error(f"Insufficient valid data after cleaning: {len(X)} records")
        return None, None, None, None, None
    
    # Time series split for evaluation
    tscv = TimeSeriesSplit(n_splits=min(3, len(X) // 5))
    
    # Optimize hyperparameters if requested
    if optimize_hyperparams and len(X) > 20:
        with st.spinner("Optimizing hyperparameters..."):
            try:
                best_params = optimize_lightgbm_with_optuna(X, y, n_trials=30, cv_folds=min(3, len(X) // 10))
                st.success("Hyperparameter optimization completed!")
            except Exception as e:
                st.warning(f"Hyperparameter optimization failed: {e}. Using default parameters.")
                best_params = {
                    'objective': 'regression',
                    'metric': 'mae',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.1,
                    'n_estimators': 200,
                    'verbose': -1
                }
    else:
        best_params = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'verbose': -1
        }
    
    # Train final model
    model = lgb.LGBMRegressor(**best_params)
    
    # Cross-validation evaluation
    mae_scores = []
    mse_scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        
        mae_scores.append(mae)
        mse_scores.append(mse)
    
    # Train on full dataset
    model.fit(X, y)
    
    return model, available_features, np.mean(mae_scores), np.mean(mse_scores), df

def create_forecast_plots(df, predictions_df, target_column, hospital_name):
    """
    Creates comprehensive forecast visualization plots.
    """
    # Historical + Forecast plot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f'Historical vs Forecast - {target_column}',
            f'Daily Patterns - {target_column}',
            f'Weekly Patterns - {target_column}',
            f'Forecast Confidence - {target_column}'
        ],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Historical data
    fig.add_trace(
        go.Scatter(
            x=df['Datetime'],
            y=df[target_column],
            mode='lines+markers',
            name='Historical',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ),
        row=1, col=1
    )
    
    # Forecast data
    fig.add_trace(
        go.Scatter(
            x=predictions_df['Datetime'],
            y=predictions_df[f'Predicted_{target_column}'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=4)
        ),
        row=1, col=1
    )
    
    # Daily patterns
    hourly_avg = df.groupby('Hour')[target_column].mean().reset_index()
    fig.add_trace(
        go.Scatter(
            x=hourly_avg['Hour'],
            y=hourly_avg[target_column],
            mode='lines+markers',
            name='Avg by Hour',
            line=dict(color='green')
        ),
        row=1, col=2
    )
    
    # Weekly patterns
    daily_avg = df.groupby('DayOfWeek')[target_column].mean().reset_index()
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_avg['DayName'] = daily_avg['DayOfWeek'].map(lambda x: day_names[x])
    
    fig.add_trace(
        go.Bar(
            x=daily_avg['DayName'],
            y=daily_avg[target_column],
            name='Avg by Day',
            marker_color='orange'
        ),
        row=2, col=1
    )
    
    # Forecast confidence (simple visualization)
    fig.add_trace(
        go.Scatter(
            x=predictions_df['Datetime'],
            y=predictions_df[f'Predicted_{target_column}'],
            mode='lines',
            name='Forecast',
            line=dict(color='red'),
            fill=None
        ),
        row=2, col=2
    )
    
    # Add confidence bands (simplified)
    forecast_mean = predictions_df[f'Predicted_{target_column}'].mean()
    forecast_std = predictions_df[f'Predicted_{target_column}'].std()
    
    fig.add_trace(
        go.Scatter(
            x=predictions_df['Datetime'],
            y=predictions_df[f'Predicted_{target_column}'] + forecast_std,
            mode='lines',
            name='Upper Bound',
            line=dict(color='rgba(255,0,0,0.3)'),
            fill=None
        ),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=predictions_df['Datetime'],
            y=predictions_df[f'Predicted_{target_column}'] - forecast_std,
            mode='lines',
            name='Lower Bound',
            line=dict(color='rgba(255,0,0,0.3)'),
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.2)'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title=f'Forecast Analysis for {hospital_name} - {target_column}',
        height=800,
        showlegend=True
    )
    
    return fig

def main():
    """
    Main Streamlit application for hospital capacity forecasting.
    """
    st.set_page_config(page_title="Hospital Capacity Forecasting", layout="wide")
    
    st.title("🏥 Advanced Hospital Capacity Forecasting")
    st.markdown("Upload your hospital data to generate accurate capacity and trolley forecasts using LightGBM.")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload Hospital Data (CSV/Excel)",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your hospital capacity data file"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            if uploaded_file.name.endswith('.csv'):
                raw_df = pd.read_csv(uploaded_file)
            else:
                raw_df = pd.read_excel(uploaded_file)
            
            st.success(f"Data loaded successfully! Shape: {raw_df.shape}")
            
            # Show raw data preview
            with st.expander("Raw Data Preview"):
                st.dataframe(raw_df.head(10))
                st.write("Columns:", list(raw_df.columns))
            
            # Prepare data
            with st.spinner("Preparing and feature engineering data..."):
                df = prepare_data(raw_df)
            
            st.success(f"Data prepared! Final shape: {df.shape}")
            
            # Show prepared data
            with st.expander("Prepared Data Preview"):
                st.dataframe(df.head(10))
                st.write("Available columns:", list(df.columns))
            
            # Configuration options
            st.sidebar.subheader("Forecast Settings")
            
            # Hospital selection
            hospitals = ['All Hospitals'] + sorted(df['Hospital'].unique().tolist())
            selected_hospital = st.sidebar.selectbox("Select Hospital", hospitals)
            
            # Target variable selection
            target_options = ['ED Beds', 'Trolleys']
            target_column = st.sidebar.selectbox("Target Variable", target_options)
            
            # Forecast days
            forecast_days = st.sidebar.slider("Forecast Days", 1, 14, 7)
            
            # Optimization option
            optimize_hyperparams = st.sidebar.checkbox("Optimize Hyperparameters", value=True)
            
            # Advanced options
            with st.sidebar.expander("Advanced Options"):
                show_shap = st.checkbox("Show SHAP Analysis", value=False)
                show_detailed_metrics = st.checkbox("Show Detailed Metrics", value=True)
            
            if st.sidebar.button("Generate Forecast", type="primary"):
                hospital_filter = None if selected_hospital == 'All Hospitals' else selected_hospital
                
                with st.spinner("Training model and generating forecasts..."):
                    # Train model
                    model, features, mae, mse, processed_df = train_and_evaluate_model(
                        df, target_column, hospital_filter, optimize_hyperparams
                    )
                    
                    if model is None:
                        st.error("Failed to train model. Please check your data.")
                        return
                    
                    # Display model performance
                    st.header("📊 Model Performance")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Mean Absolute Error", f"{mae:.2f}")
                    with col2:
                        st.metric("Root Mean Squared Error", f"{np.sqrt(mse):.2f}")
                    with col3:
                        st.metric("Features Used", len(features))
                    
                    # Feature importance
                    if hasattr(model, 'feature_importances_'):
                        st.subheader("🔍 Feature Importance")
                        importance_df = pd.DataFrame({
                            'Feature': features,
                            'Importance': model.feature_importances_
                        }).sort_values('Importance', ascending=False).head(15)
                        
                        fig_importance = px.bar(
                            importance_df,
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title='Top 15 Most Important Features'
                        )
                        st.plotly_chart(fig_importance, use_container_width=True)
                    
                    # Generate forecasts
                    st.header("🔮 Forecast Results")
                    
                    if hospital_filter:
                        hospitals_to_forecast = [hospital_filter]
                    else:
                        hospitals_to_forecast = df['Hospital'].unique()[:5]  # Limit to first 5 for performance
                    
                    all_predictions = []
                    
                    for hospital in hospitals_to_forecast:
                        hospital_data = processed_df[processed_df['Hospital'] == hospital].copy()
                        
                        if len(hospital_data) < 5:
                            st.warning(f"Insufficient data for {hospital}: {len(hospital_data)} records")
                            continue
                        
                        # Get hospital-specific information
                        last_date = hospital_data['Datetime'].max().date()
                        hospital_code = hospital_data['Hospital_Code'].iloc[0]
                        additional_capacity = hospital_data['Capacity'].iloc[-1]
                        
                        # Create future dates
                        future_df = create_future_dates(
                            last_date, hospital, hospital_code, 
                            additional_capacity, forecast_days
                        )
                        
                        # Generate predictions
                        predictions = forecast_with_lags_advanced(
                            model, hospital_data, future_df, features, target_column
                        )
                        
                        future_df[f'Predicted_{target_column}'] = predictions
                        future_df['Hospital'] = hospital
                        
                        all_predictions.append(future_df)
                        
                        # Individual hospital plots
                        st.subheader(f"📈 {hospital} - {target_column} Forecast")
                        
                        fig = create_forecast_plots(
                            hospital_data, future_df, target_column, hospital
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Forecast summary table
                        summary_df = future_df[['Date', 'Time', f'Predicted_{target_column}']].copy()
                        summary_df[f'Predicted_{target_column}'] = summary_df[f'Predicted_{target_column}'].round(1)
                        
                        st.subheader(f"📋 {hospital} - Detailed Forecast")
                        st.dataframe(summary_df, use_container_width=True)
                    
                    # Combined analysis if multiple hospitals
                    if len(all_predictions) > 1:
                        st.header("🏥 Combined Hospital Analysis")
                        
                        combined_df = pd.concat(all_predictions, ignore_index=True)
                        
                        # Summary statistics
                        summary_stats = combined_df.groupby('Hospital')[f'Predicted_{target_column}'].agg([
                            'mean', 'std', 'min', 'max'
                        ]).round(2)
                        
                        st.subheader("📊 Hospital Comparison")
                        st.dataframe(summary_stats)
                        
                        # Combined visualization
                        fig_combined = px.line(
                            combined_df,
                            x='Datetime',
                            y=f'Predicted_{target_column}',
                            color='Hospital',
                            title=f'Combined {target_column} Forecast - All Hospitals'
                        )
                        st.plotly_chart(fig_combined, use_container_width=True)
                    
                    # SHAP analysis (if requested and feasible)
                    if show_shap and len(processed_df) < 1000:  # Limit for performance
                        try:
                            st.header("🔍 SHAP Feature Analysis")
                            
                            # Sample data for SHAP to improve performance
                            sample_size = min(100, len(processed_df))
                            sample_data = processed_df.sample(sample_size)
                            X_sample = sample_data[features]
                            
                            explainer = shap.TreeExplainer(model)
                            shap_values = explainer.shap_values(X_sample)
                            
                            # SHAP summary plot
                            fig_shap, ax = plt.subplots(figsize=(10, 6))
                            shap.summary_plot(shap_values, X_sample, show=False)
                            st.pyplot(fig_shap)
                            
                        except Exception as e:
                            st.warning(f"SHAP analysis failed: {e}")
                    
                    # Export functionality
                    st.header("💾 Export Results")
                    
                    if all_predictions:
                        export_df = pd.concat(all_predictions, ignore_index=True)
                        export_df = export_df[['Hospital', 'Date', 'Time', 'Datetime', f'Predicted_{target_column}']]
                        
                        csv_data = export_df.to_csv(index=False)
                        
                        st.download_button(
                            label="Download Forecast Results (CSV)",
                            data=csv_data,
                            file_name=f"hospital_forecast_{target_column.lower().replace(' ', '_')}.csv",
                            mime="text/csv"
                        )
                        
                        st.success("Forecast completed successfully!")
                    
        except Exception as e:
            st.error(f"Error processing data: {e}")
            st.exception(e)
    
    else:
        st.info("Please upload a hospital data file to begin forecasting.")
        
        # Show sample data format
        st.header("📋 Expected Data Format")
        
        sample_data = {
            'Hospital Group Name': ['Group A', 'Group A', 'Group B'],
            'Hospital': ['Hospital 1', 'Hospital 1', 'Hospital 2'],
            'Date': ['2024-01-01', '2024-01-02', '2024-01-01'],
            'DayGAR': ['Monday', 'Tuesday', 'Monday'],
            'Tracker8am': [25, 30, 15],
            'Tracker2pm': [35, 40, 20],
            'Tracker8pm': [45, 50, 25],
            'TimeTotal_8am': [5, 8, 2],
            'TimeTotal_2pm': [12, 15, 7],
            'TimeTotal_8pm': [20, 25, 12],
            'AdditionalCapacityOpen Morning': [10, 10, 5]
        }
        
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df)
        
        st.markdown("""
        **Required Columns:**
        - Hospital Group Name: Name of the hospital group
        - Hospital: Individual hospital name
        - Date: Date in YYYY-MM-DD format
        - DayGAR: Day of the week (optional)
        - Tracker8am, Tracker2pm, Tracker8pm: ED bed counts at different times
        - TimeTotal_8am, TimeTotal_2pm, TimeTotal_8pm: Trolley counts at different times
        - AdditionalCapacityOpen Morning: Additional capacity available
        """)

if __name__ == "__main__":
    main()
