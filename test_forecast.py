# ==========================================
# HOSPITAL ADMISSIONS FORECASTING PIPELINE
# Adjusted for Google Colab
# ==========================================

import subprocess
import sys

# --- Colab Dependency Check ---
def install_packages():
    packages = ['meteostat', 'prophet']
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_packages()

# --- Imports ---
import pandas as pd
import numpy as np
import traceback
import requests
import warnings
import os
from io import StringIO
from datetime import timedelta, datetime, date

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Time Series & Modeling
from prophet import Prophet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA

# Date handling
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday
from pandas.tseries.offsets import DateOffset
from dateutil.rrule import MO

# Weather Data
from meteostat import Point, Hourly

# Colab specific (for file uploads)
try:
    from google.colab import files
except ImportError:
    pass

warnings.filterwarnings("ignore")

# ==========================================
# 1. Configuration & Constants
# ==========================================

# Bank Holiday Calendar Definition
class IrishBankHolidays(AbstractHolidayCalendar):
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=nearest_workday),
        Holiday("St. Brigid's Day", month=2, day=1, observance=nearest_workday),
        Holiday("St. Patrick's Day", month=3, day=17, observance=nearest_workday),
        Holiday("May Day", month=5, day=1, offset=DateOffset(weekday=MO(1))),
        Holiday("June Bank Holiday", month=6, day=1, offset=DateOffset(weekday=MO(1))),
        Holiday("October Bank Holiday", month=10, day=1, offset=DateOffset(weekday=MO(-1))),
        Holiday("Christmas Day", month=12, day=25),
        Holiday("St. Stephen's Day", month=12, day=26),
    ]

# ==========================================
# 2. Data Acquisition Functions
# ==========================================

def download_and_process_virus_data():
    """
    Downloads and processes the HPSC respiratory virus data (Flu, COVID, RSV).
    """
    DATASET_CSV_URL = "https://respiratoryvirus.hpsc.ie/datasets/dbd10a497fd84e44954b858249919997_0.csv"
    print(f"\n--- Downloading virus data from HPSC ---")

    try:
        response = requests.get(DATASET_CSV_URL, timeout=20)
        response.raise_for_status()
        print("  ✅ Successfully downloaded the virus data.")

        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data)

        # Select relevant columns
        df = df[['floor', 'weekly_cases', 'disease']].copy()

        # Convert floor to datetime
        df['Date'] = pd.to_datetime(df['floor']).dt.tz_localize(None).dt.normalize()

        # Pivot table
        df_pivot = df.pivot_table(index='Date', columns='disease', values='weekly_cases', aggfunc='sum').fillna(0)

        # Rename columns to be code-friendly
        df_pivot.columns = [f"{col.replace(' ', '_')}_weekly_cases" for col in df_pivot.columns]

        df_pivot = df_pivot.sort_index().reset_index()
        print("  ✅ Virus data processed successfully.")
        return df_pivot

    except requests.exceptions.RequestException as e:
        print(f"  ❌ [FAILURE] Could not download the virus data file. Error: {e}")
    except Exception as e:
        print(f"  ❌ [FAILURE] An unexpected error occurred while processing virus data: {e}")

    return pd.DataFrame()

def get_weather_data_from_meteostat(start_date, end_date, location_name):
    """
    Fetches raw hourly weather data for a specific location using Meteostat.
    """
    print(f"  Fetching raw hourly weather data from Meteostat for {location_name} from {start_date.date()} to {end_date.date()}...")

    cork_location = Point(51.8979, -8.4706, 25)
    kerry_location = Point(52.2704, -9.7026, 34)

    locations_map = {"Cork": cork_location, "Kerry": kerry_location}

    if location_name not in locations_map:
        print(f"  ❌ Error: Location '{location_name}' not defined for Meteostat.")
        return None

    point = locations_map[location_name]

    try:
        data = Hourly(point, start_date, end_date).fetch()
        if data.empty:
            print(f"  ⚠️ No raw hourly weather data fetched for {location_name}.")
            return None

        data = data.reset_index().rename(columns={'time': 'Datetime'})
        data['Datetime'] = pd.to_datetime(data['Datetime'])

        raw_weather_cols_to_select = ['Datetime', 'temp', 'dwpt', 'rhum', 'prcp', 'snow', 'wdir', 'wspd', 'wpgt', 'pres', 'coco']
        data = data[[col for col in raw_weather_cols_to_select if col in data.columns]]

        print(f"  ✅ Raw hourly Meteostat data fetched for {location_name}: {len(data)} records.")
        return data

    except Exception as e:
        print(f"  ❌ Error fetching raw Meteostat data for {location_name}: {e}")
        return None

def load_weather_data(hospital_name, start_date, end_date):
    print(f"  Loading raw hourly weather data for {hospital_name} region...")
    hospital_weather_mapping = {
        'Cork University Hospital': 'Cork',
        'Mercy University Hospital': 'Cork',
        'UH Kerry': 'Kerry'
    }
    weather_region = hospital_weather_mapping.get(hospital_name, 'Cork')
    weather_data = get_weather_data_from_meteostat(start_date, end_date, weather_region)

    if weather_data is not None:
        print(f"  ✅ {weather_region.capitalize()} raw hourly weather data loaded.")
    return weather_data, weather_region

# ==========================================
# 3. Data Preparation & Engineering
# ==========================================

def merge_weather_data(df_merged, weather_data, hospital_name):
    print(f"  Merging raw hourly weather data for {hospital_name}...")
    df_merged['Datetime'] = pd.to_datetime(df_merged['Datetime'])
    expected_raw_weather_cols = ['temp', 'dwpt', 'rhum', 'prcp', 'snow', 'wdir', 'wspd', 'wpgt', 'pres', 'coco']

    if weather_data is None:
        print("  ⚠️ Weather data not available. Initializing weather columns to 0.")
        for col in expected_raw_weather_cols:
            df_merged[col] = 0.0
        return df_merged

    weather_data['Datetime'] = pd.to_datetime(weather_data['Datetime'])
    merged_hospital = pd.merge(df_merged, weather_data, on='Datetime', how='left')

    fill_with_zero = ['prcp', 'snow']
    forward_fill_cols = ['temp', 'dwpt', 'rhum', 'wdir', 'wspd', 'wpgt', 'pres', 'coco']

    for col in expected_raw_weather_cols:
        if col not in merged_hospital.columns:
            merged_hospital[col] = 0.0
        else:
            if col in fill_with_zero:
                merged_hospital[col] = merged_hospital[col].fillna(0)
            elif col in forward_fill_cols:
                merged_hospital[col] = merged_hospital[col].ffill().bfill()

    merged_hospital[expected_raw_weather_cols] = merged_hospital[expected_raw_weather_cols].fillna(0)
    print(f"  ✅ Raw hourly weather data merged for {hospital_name}.")
    return merged_hospital

def merge_virus_data(df, df_virus):
    print("  Merging virus data with hospital data...")
    if df_virus.empty:
        print("  ⚠️ Virus data is empty. Skipping merge.")
        return df, []

    virus_features = [col for col in df_virus.columns if col != 'Date']
    df['DateOnly'] = df['Datetime'].dt.normalize()

    # Use merge_asof for correct time-series joining
    df_merged = pd.merge_asof(
        df.sort_values('DateOnly'),
        df_virus.sort_values('Date'),
        left_on='DateOnly',
        right_on='Date',
        direction='backward'
    )

    df_merged[virus_features] = df_merged[virus_features].fillna(0)
    print("  ✅ Virus data merged successfully.")
    return df_merged, virus_features

def prepare_data_vectorized(df_raw, hospital_name, weather_thresholds):
    print("  Preparing data: Renaming columns and unpivoting...")
    df = df_raw.copy()

    column_mapping = {
        'Tracker8am': 'ED_8am', 'Tracker2pm': 'ED_2pm', 'Tracker8pm': 'ED_8pm',
        'TimeTotal_8am': 'Trolley_8am', 'TimeTotal_2pm': 'Trolley_2pm', 'TimeTotal_8pm': 'Trolley_8pm',
        'AdditionalCapacityOpen Morning': 'Additional_Capacity'
    }
    df = df.rename(columns=column_mapping)
    df['Additional_Capacity'] = df.groupby(['Hospital', 'Date'])['Additional_Capacity'].transform('first').fillna(0)

    times = ['8am', '2pm', '8pm']
    time_map = {'8am': '08:00', '2pm': '14:00', '8pm': '20:00'}

    rows = []
    for _, row in df.iterrows():
        for time in times:
            rows.append({
                'Hospital Group Name': row['Hospital Group Name'],
                'Hospital': row['Hospital'],
                'Date': row['Date'],
                'Capacity': row['Additional_Capacity'],
                'Time': time_map[time],
                'ED Beds': pd.to_numeric(row.get(f'ED_{time}'), errors='coerce'),
                'Trolleys': pd.to_numeric(row.get(f'Trolley_{time}'), errors='coerce')
            })

    df_merged = pd.DataFrame(rows)
    df_merged['Datetime'] = pd.to_datetime(df_merged['Date'].astype(str) + ' ' + df_merged['Time'])
    df_merged = df_merged.sort_values(['Hospital', 'Datetime']).reset_index(drop=True)

    df_hospital = df_merged[df_merged['Hospital'] == hospital_name].copy()

    min_date, max_date = df_hospital['Datetime'].min(), df_hospital['Datetime'].max()
    raw_hourly_weather_data, _ = load_weather_data(hospital_name, min_date, max_date)

    df_hospital = merge_weather_data(df_hospital, raw_hourly_weather_data, hospital_name)
    df_hospital['DateOnly'] = df_hospital['Datetime'].dt.normalize()

    # Calculate daily sums for ARIMAX layer
    daily_sums = df_hospital.groupby(['Hospital', 'DateOnly']).agg(
        Daily_ED_Beds=('ED Beds', 'sum'),
        Daily_Trolleys=('Trolleys', 'sum')
    ).reset_index()

    df_hospital = pd.merge(df_hospital, daily_sums, on=['Hospital', 'DateOnly'], how='left')

    # Aggregate weather to daily for ARIMAX layer
    if raw_hourly_weather_data is not None:
        raw_hourly_weather_data['DateOnly'] = raw_hourly_weather_data['Datetime'].dt.normalize()
        daily_agg_params = {f'Daily_{col}': (col, agg_func)
                            for col, agg_func in [('temp', 'mean'), ('prcp', 'sum'), ('snow', 'sum'),
                                                  ('wspd', 'mean'), ('pres', 'mean'), ('rhum', 'mean'), ('wpgt', 'max')]
                            if col in raw_hourly_weather_data.columns}

        if daily_agg_params:
            daily_weather_agg = raw_hourly_weather_data.groupby('DateOnly').agg(**daily_agg_params).reset_index()
            df_hospital = pd.merge(df_hospital, daily_weather_agg, on='DateOnly', how='left')

            for col_name in daily_agg_params.keys():
                if col_name in df_hospital.columns:
                    df_hospital[col_name] = df_hospital[col_name].ffill().bfill().fillna(0)

    print("  Preparing data: Adding time-based features...")
    dt = df_hospital['Datetime'].dt
    df_hospital['Hour'] = dt.hour
    df_hospital['DayOfWeek'] = dt.dayofweek
    df_hospital['Month'] = dt.month
    df_hospital['Quarter'] = dt.quarter
    df_hospital['WeekOfYear'] = dt.isocalendar().week.astype(int)
    df_hospital['DayOfMonth'] = dt.day
    df_hospital['DayOfYear'] = dt.dayofyear
    df_hospital['WeekOfMonth'] = (dt.day - 1) // 7 + 1
    df_hospital['IsWeekend'] = dt.dayofweek.isin([5, 6]).astype(int)

    # Specific Day Flags
    df_hospital['IsMonday'] = (dt.dayofweek == 0).astype(int)
    df_hospital['IsTuesday'] = (dt.dayofweek == 1).astype(int)
    df_hospital['IsWednesday'] = (dt.dayofweek == 2).astype(int)
    df_hospital['IsThursday'] = (dt.dayofweek == 3).astype(int)
    df_hospital['IsFriday'] = (dt.dayofweek == 4).astype(int)
    df_hospital['IsSaturday'] = (dt.dayofweek == 5).astype(int)
    df_hospital['IsSunday'] = (dt.dayofweek == 6).astype(int)

    df_hospital['IsSummer'] = dt.month.isin([6, 7, 8]).astype(int)
    df_hospital['IsWinter'] = dt.month.isin([12, 1, 2]).astype(int)
    df_hospital['IsPeakHour'] = (dt.hour == 20).astype(int)
    df_hospital['IsLowHour'] = (dt.hour == 8).astype(int)
    df_hospital['Hour_sin'] = np.sin(2 * np.pi * dt.hour / 24)
    df_hospital['Hour_cos'] = np.cos(2 * np.pi * dt.hour / 24)

    try:
        calendar = IrishBankHolidays()
        holidays = set(calendar.holidays(start=df_hospital['Datetime'].min(), end=df_hospital['Datetime'].max()))
        df_hospital['IsHoliday'] = df_hospital['Datetime'].dt.normalize().isin(holidays).astype(int)
    except Exception:
        df_hospital['IsHoliday'] = 0

    df_hospital['Hospital_Code'] = df_hospital['Hospital'].astype('category').cat.codes

    # --- Advanced Weather Feature Engineering ---
    print("  Adding enhanced weather features (rolling windows)...")

    # Initialize all potential columns
    all_potential_engineered_weather_cols = [
        'prcp_avg_2hr', 'snow_sum_12hr', 'wspd_max_3hr', 'temp_min_2hr', 'rhum_avg_3hr', 'gust_factor',
        'is_rainy', 'is_snowy', 'is_hot', 'is_very_hot', 'is_cold', 'is_freezing', 'is_windy', 'is_very_windy',
        'is_humid', 'is_foggy', 'is_hot_lag1', 'is_hot_lag2', 'is_hot_lag3', 'is_cold_lag1', 'is_cold_lag2',
        'is_cold_lag3', 'is_rainy_lag1', 'is_rainy_lag2', 'is_rainy_lag3', 'hot_duration', 'cold_duration',
        'rainy_and_cold', 'hot_and_weekend', 'windy_and_rainy', 'hot_and_humid', 'pressure_change', 'pressure_drop',
        'winter_rain', 'summer_heat', 'morning_rain', 'afternoon_rain', 'evening_rain', 'morning_heat',
        'afternoon_heat', 'evening_heat', 'morning_cold', 'afternoon_cold', 'evening_cold', 'morning_ice',
        'afternoon_ice', 'evening_ice']

    for col in all_potential_engineered_weather_cols:
        df_hospital[col] = 0.0

    required_raw_weather_cols_for_eng = ['temp', 'prcp', 'snow', 'wspd', 'pres', 'rhum', 'wpgt', 'coco']

    if all(col in df_hospital.columns for col in required_raw_weather_cols_for_eng):
        df_hospital = df_hospital.sort_values(by=['Hospital', 'Datetime'])
        grouped = df_hospital.groupby('Hospital')

        # Calculate rolling stats
        df_hospital['prcp_avg_2hr'] = grouped['prcp'].transform(lambda x: x.rolling(window=2, min_periods=1).mean().shift(1)).fillna(0)
        df_hospital['snow_sum_12hr'] = grouped['snow'].transform(lambda x: x.rolling(window=12, min_periods=1).sum().shift(1)).fillna(0)
        df_hospital['wspd_max_3hr'] = grouped['wspd'].transform(lambda x: x.rolling(window=3, min_periods=1).max().shift(1)).fillna(0)
        df_hospital['temp_min_2hr'] = grouped['temp'].transform(lambda x: x.rolling(window=2, min_periods=1).min().shift(1)).fillna(0)
        df_hospital['rhum_avg_3hr'] = grouped['rhum'].transform(lambda x: x.rolling(window=3, min_periods=1).mean().shift(1)).fillna(0)

        # Derived Binary Flags based on Thresholds
        df_hospital['gust_factor'] = (df_hospital['wpgt'] / df_hospital['wspd']).replace([np.inf, -np.inf], 1).fillna(1)
        df_hospital['is_rainy'] = (df_hospital['prcp_avg_2hr'] > weather_thresholds['rain_threshold']).astype(int)
        df_hospital['is_snowy'] = (df_hospital['snow_sum_12hr'] > weather_thresholds['snow_threshold']).astype(int)
        df_hospital['is_hot'] = (df_hospital['temp'] >= weather_thresholds['hot_temp_threshold']).astype(int)
        df_hospital['is_cold'] = (df_hospital['temp'] < weather_thresholds['cold_temp_threshold']).astype(int)
        df_hospital['is_freezing'] = (df_hospital['temp_min_2hr'] <= weather_thresholds['freezing_temp_threshold']).astype(int)
        df_hospital['is_windy'] = (df_hospital['wspd_max_3hr'] > weather_thresholds['windy_speed_threshold']).astype(int)
        df_hospital['is_humid'] = (df_hospital['rhum_avg_3hr'] > weather_thresholds['humid_rh_threshold']).astype(int)
        df_hospital['is_foggy'] = df_hospital['coco'].isin([4, 5]).astype(int)

        # Weather Lags
        for lag in [1, 2, 3]:
            df_hospital[f'is_hot_lag{lag}'] = grouped['is_hot'].shift(lag).fillna(0).astype(int)
            df_hospital[f'is_cold_lag{lag}'] = grouped['is_cold'].shift(lag).fillna(0).astype(int)
            df_hospital[f'is_rainy_lag{lag}'] = grouped['is_rainy'].shift(lag).fillna(0).astype(int)

        df_hospital['hot_duration'] = grouped['is_hot'].transform(lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)).fillna(0)

        # Interaction Features
        df_hospital['rainy_and_cold'] = (df_hospital['is_rainy'] & df_hospital['is_cold']).astype(int)
        df_hospital['hot_and_weekend'] = (df_hospital['is_hot'] & df_hospital['IsWeekend']).astype(int)
        df_hospital['hot_and_humid'] = (df_hospital['is_hot'] & df_hospital['is_humid']).astype(int)

        df_hospital['pressure_change'] = grouped['pres'].diff().fillna(0)
        df_hospital['pressure_drop'] = (df_hospital['pressure_change'] < weather_thresholds['pressure_drop_threshold']).astype(int)
    else:
        print("  ⚠️ Not all raw weather columns available for enhanced feature engineering.")

    # Fill NaNs in numeric columns
    numeric_cols = df_hospital.select_dtypes(include=[np.number]).columns
    df_hospital[numeric_cols] = df_hospital[numeric_cols].fillna(0)

    print("  Data preparation complete.")
    return df_hospital

def add_lag_features_fast(df, target_column):
    print(f"  Adding lag features for {target_column}...")
    df = df.sort_values(['Hospital', 'Datetime'])
    grouped = df.groupby('Hospital', sort=False)[target_column]

    lag_features = []

    # Simple Lags
    for lag in [1, 2, 3, 6, 9, 12, 24, 48, 72, 168]:
        lag_col = f'Lag_{target_column}_{lag}'
        df[lag_col] = grouped.shift(lag)
        lag_features.append(lag_col)

    # Rolling Means
    for window in [3, 6, 12, 24, 48]:
        roll_col = f'Rolling_Mean_{window}_{target_column}'
        df[roll_col] = grouped.rolling(window=window, min_periods=1).mean().shift(1).reset_index(level=0, drop=True)
        lag_features.append(roll_col)

    df[lag_features] = df[lag_features].bfill().fillna(0)
    return df, lag_features

# ==========================================
# 4. Modeling Layers
# ==========================================

def compare_daily_models(df_prepared, hospital, daily_metric, virus_features, forecast_days=7):
    """
    Trains an ARIMAX model for daily totals.
    """
    print(f"\n--- Training Daily ARIMAX Model for {hospital} - {daily_metric} ---")
    df = df_prepared[df_prepared['Hospital'] == hospital].copy()

    daily_agg_dict = {
        'y': (daily_metric, 'first'),
        'IsHoliday': ('IsHoliday', 'first'),
        'IsMonday': ('IsMonday', 'first'), 'IsTuesday': ('IsTuesday', 'first'),
        'IsWednesday': ('IsWednesday', 'first'), 'IsThursday': ('IsThursday', 'first'),
        'IsFriday': ('IsFriday', 'first'), 'IsSaturday': ('IsSaturday', 'first'),
        'IsSunday': ('IsSunday', 'first'),
        'IsSummer': ('IsSummer', 'first'), 'IsWinter': ('IsWinter', 'first'),
        'Capacity': ('Capacity', 'first')
    }

    weather_cols_for_daily_agg = [
        'temp', 'prcp', 'snow', 'wspd', 'pres', 'rhum', 'is_rainy', 'is_snowy', 'is_hot', 'is_cold', 'is_freezing',
        'is_windy', 'is_humid', 'is_foggy', 'hot_duration', 'rainy_and_cold', 'hot_and_weekend', 'hot_and_humid',
        'pressure_drop', 'winter_rain', 'summer_heat', 'Daily_temp', 'Daily_prcp', 'Daily_snow', 'Daily_wspd',
        'Daily_pres', 'Daily_rhum', 'Daily_wpgt']

    for col in weather_cols_for_daily_agg:
        if col in df.columns:
            daily_agg_dict[col] = (col, 'mean')

    for col in virus_features:
        if col in df.columns:
            daily_agg_dict[col] = (col, 'first')

    df_daily = df.groupby('DateOnly').agg(**daily_agg_dict).reset_index().rename(columns={'DateOnly': 'ds'}).sort_values('ds').reset_index(drop=True)

    if df_daily['y'].notna().sum() < 30:
        print(f"  Not enough daily data for {hospital} {daily_metric}. Skipping daily ARIMAX.")
        return None

    potential_daily_regressors = [
        'IsHoliday', 'IsMonday', 'IsTuesday', 'IsWednesday', 'IsThursday', 'IsFriday', 'IsSaturday',
        'IsSummer', 'IsWinter', 'Capacity'
    ]
    potential_daily_regressors.extend([col for col in weather_cols_for_daily_agg if col in df_daily.columns])
    potential_daily_regressors.extend(virus_features)
    potential_daily_regressors = list(set(potential_daily_regressors))

    exog_cols_initial = [r for r in potential_daily_regressors if r in df_daily.columns]

    # Check for constant columns to avoid LinAlgError
    exog_full_numeric_for_const_check = df_daily[exog_cols_initial].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0)
    constant_cols_full = [col for col in exog_full_numeric_for_const_check.columns if exog_full_numeric_for_const_check[col].nunique() <= 1]
    final_exog_cols = [col for col in exog_cols_initial if col not in constant_cols_full]

    if constant_cols_full:
        print(f"  Dropping constant columns from daily ARIMAX exog: {constant_cols_full}")

    # Prepare Future Dataframe
    last_daily_date = df_daily['ds'].max()
    future_daily_rows = []

    monthly_exog_stats_daily = {
        month_num: {col: month_data[col].mean() for col in exog_cols_initial if col in month_data.columns and pd.api.types.is_numeric_dtype(month_data[col])}
        for month_num in range(1, 13) if not (month_data := df_daily[df_daily['ds'].dt.month == month_num]).empty
    }

    for d in range(1, forecast_days + 1):
        future_date = last_daily_date + timedelta(days=d)
        row = {
            'ds': future_date,
            'IsHoliday': int(future_date.normalize() in IrishBankHolidays().holidays(start=future_date, end=future_date)),
            'IsMonday': int(future_date.dayofweek == 0), 'IsTuesday': int(future_date.dayofweek == 1),
            'IsWednesday': int(future_date.dayofweek == 2), 'IsThursday': int(future_date.dayofweek == 3),
            'IsFriday': int(future_date.dayofweek == 4), 'IsSaturday': int(future_date.dayofweek == 5),
            'IsSummer': int(future_date.month in [6, 7, 8]), 'IsWinter': int(future_date.month in [12, 1, 2]),
            'Capacity': df_daily['Capacity'].iloc[-1]
        }

        current_month_stats_daily = monthly_exog_stats_daily.get(future_date.month, {})

        for col in exog_cols_initial:
            if col in current_month_stats_daily:
                row[col] = current_month_stats_daily[col]
            elif col in virus_features:
                row[col] = df_daily[col].iloc[-1]
            elif col not in row:
                row[col] = df_daily[col].mean() if col in df_daily.columns and pd.api.types.is_numeric_dtype(df_daily[col]) else 0.0

        future_daily_rows.append(row)

    future_daily_df_full = pd.DataFrame(future_daily_rows)
    future_exog_data = future_daily_df_full[final_exog_cols].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)

    try:
        y_full_numeric = pd.to_numeric(df_daily['y'], errors='coerce').fillna(df_daily['y'].median()).astype(float)
        exog_full_numeric = df_daily[final_exog_cols].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)

        y_full_numeric.index = exog_full_numeric.index = pd.to_datetime(df_daily['ds'])
        if not y_full_numeric.index.freq:
             y_full_numeric = y_full_numeric.asfreq('D').fillna(method='ffill')
             exog_full_numeric = exog_full_numeric.asfreq('D').fillna(method='ffill')

        model_arimax_full = ARIMA(y_full_numeric, exog=exog_full_numeric, order=(7, 1, 1)).fit()
        best_daily_forecast_values = model_arimax_full.forecast(steps=forecast_days, exog=future_exog_data)

        daily_forecast_df = pd.DataFrame({'DateOnly': pd.to_datetime(future_daily_df_full['ds']), daily_metric: best_daily_forecast_values.values})
        print(f"  ✅ Daily forecast generated using ARIMAX.")
        return daily_forecast_df

    except Exception as e:
        print(f"  ❌ ARIMAX model failed: {e}")
        traceback.print_exc()
        return None

def get_hourly_features(metric, daily_metric_name, other_metric, other_daily_metric_name, df_columns, virus_features, for_arimax=False):
    base_hourly_features = [
        'DayOfWeek', 'Month', 'Quarter', 'WeekOfYear', 'DayOfMonth', 'DayOfYear', 'WeekOfMonth',
        'IsWeekend', 'IsMonday', 'IsTuesday', 'IsWednesday', 'IsThursday', 'IsFriday', 'IsSaturday', 'IsSunday',
        'IsSummer', 'IsWinter', 'IsHoliday', 'Capacity',
        daily_metric_name, other_metric, other_daily_metric_name,
        'temp', 'prcp', 'snow', 'wspd', 'pres', 'rhum', 'dwpt', 'wpgt', 'gust_factor',
        'prcp_avg_2hr', 'snow_sum_12hr', 'wspd_max_3hr', 'temp_min_2hr', 'rhum_avg_3hr',
        'is_rainy', 'is_snowy', 'is_hot', 'is_very_hot', 'is_cold', 'is_freezing', 'is_windy',
        'is_very_windy', 'is_humid', 'is_foggy', 'hot_duration', 'rainy_and_cold', 'hot_and_weekend',
        'hot_and_humid', 'pressure_change', 'pressure_drop', 'winter_rain', 'summer_heat']

    base_hourly_features.extend(virus_features)

    if not for_arimax:
        base_hourly_features.extend(['Hospital_Code', 'Hour_sin', 'Hour_cos', 'IsPeakHour', 'IsLowHour'])

    lag_features_current_metric = [col for col in df_columns if col.startswith(f'Lag_{metric}') or col.startswith(f'Rolling_Mean_{metric}')]
    lag_features_other_metric = [col for col in df_columns if col.startswith(f'Lag_{other_metric}') or col.startswith(f'Rolling_Mean_{other_metric}')]

    all_potential_features = base_hourly_features + lag_features_current_metric + lag_features_other_metric
    available_features = [f for f in all_potential_features if f in df_columns]

    return available_features

def train_and_predict_hourly_gbm_combined(df, hospital, metric, daily_metric_name, daily_forecast_df,
                                          other_metric, other_daily_metric_name, other_daily_forecast_df,
                                          virus_features, forecast_days=7):
    """
    Trains separate Sklearn GradientBoostingRegressor models for each hourly slot.
    """
    print(f"\n--- Training Hourly Sklearn GBM Models for {hospital} - {metric} ---")
    available_hourly_features = get_hourly_features(metric, daily_metric_name, other_metric, other_daily_metric_name, df.columns, virus_features, for_arimax=False)

    df_filtered_for_gbm = df.dropna(subset=[metric]).copy()
    time_slots = {'08:00': 8, '14:00': 14, '20:00': 20}
    last_date = df['ds'].max()

    future_rows = []

    monthly_exog_stats_hourly = {
        month_num: {col: month_data[col].mean() for col in available_hourly_features if col in month_data.columns and pd.api.types.is_numeric_dtype(month_data[col])}
        for month_num in range(1, 13) if not (month_data := df_filtered_for_gbm[df_filtered_for_gbm['Month'] == month_num]).empty
    }

    for d in range(1, forecast_days + 1):
        future_date_only = last_date.date() + timedelta(days=d)

        # Get pre-calculated daily sums from ARIMAX
        daily_sum_for_future = daily_forecast_df.loc[daily_forecast_df['DateOnly'].dt.date == future_date_only, daily_metric_name].values[0] if not daily_forecast_df[daily_forecast_df['DateOnly'].dt.date == future_date_only].empty else 0
        other_daily_sum_for_future = other_daily_forecast_df.loc[other_daily_forecast_df['DateOnly'].dt.date == future_date_only, other_daily_metric_name].values[0] if not other_daily_forecast_df[other_daily_forecast_df['DateOnly'].dt.date == future_date_only].empty else 0

        for t_str, hour_int in time_slots.items():
            dt = pd.to_datetime(f"{future_date_only} {t_str}")

            row = {'ds': dt, 'Hour': hour_int, 'DayOfWeek': dt.dayofweek, 'Month': dt.month, 'Quarter': dt.quarter,
                   'WeekOfYear': dt.isocalendar().week, 'DayOfMonth': dt.day, 'DayOfYear': dt.dayofyear,
                   'IsWeekend': int(dt.dayofweek in [5,6]), 'IsMonday': int(dt.dayofweek == 0), 'IsTuesday': int(dt.dayofweek == 1),
                   'IsWednesday': int(dt.dayofweek == 2), 'IsThursday': int(dt.dayofweek == 3), 'IsFriday': int(dt.dayofweek == 4),
                   'IsSaturday': int(dt.dayofweek == 5), 'IsSunday': int(dt.dayofweek == 6), 'IsSummer': int(dt.month in [6,7,8]),
                   'IsWinter': int(dt.month in [12,1,2]), 'IsPeakHour': int(dt.hour == 20), 'IsLowHour': int(dt.hour == 8),
                   'Hour_sin': np.sin(2*np.pi*dt.hour/24), 'Hour_cos': np.cos(2*np.pi*dt.hour/24),
                   'IsHoliday': int(dt.normalize() in IrishBankHolidays().holidays(start=dt, end=dt)),
                   'Hospital_Code': df['Hospital_Code'].iloc[-1], 'Capacity': df['Capacity'].iloc[-1],
                   daily_metric_name: daily_sum_for_future, other_daily_metric_name: other_daily_sum_for_future}

            current_month_stats = monthly_exog_stats_hourly.get(dt.month, {})

            for col in available_hourly_features:
                if col not in row:
                    if col in current_month_stats: row[col] = current_month_stats[col]
                    elif col in virus_features: row[col] = df[col].iloc[-1]
                    elif col.startswith('Lag_') or col.startswith('Rolling_Mean_'):
                        row[col] = df_filtered_for_gbm.loc[df_filtered_for_gbm['Hour'] == hour_int, col].iloc[-1] if not df_filtered_for_gbm[df_filtered_for_gbm['Hour'] == hour_int].empty else 0
                    else: row[col] = 0

            future_rows.append(row)

    future_hourly_df = pd.DataFrame(future_rows)
    future_hourly_df['prediction'] = np.nan

    historical_hourly_gbm_preds = df.copy()
    historical_hourly_gbm_preds['gbm_pred'] = np.nan

    for time_str, hour_int in time_slots.items():
        print(f"  Training GradientBoostingRegressor for {time_str}...")
        df_slot = df_filtered_for_gbm[df_filtered_for_gbm['Hour'] == hour_int].copy()

        if df_slot.empty or df_slot[metric].notna().sum() < 30:
            print(f"  ⚠️ Not enough data for {time_str} slot for GBM. Skipping.")
            continue

        X_slot = df_slot[available_hourly_features].apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
        y_slot = df_slot[metric]

        gbm_model = GradientBoostingRegressor(loss='huber', n_estimators=1200, learning_rate=0.05, max_depth=5, subsample=0.8, random_state=42).fit(X_slot, y_slot)

        future_exog_slot = future_hourly_df.loc[future_hourly_df['Hour'] == hour_int, available_hourly_features].apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)

        if not future_exog_slot.empty:
            future_preds = gbm_model.predict(future_exog_slot)
            future_hourly_df.loc[future_hourly_df['Hour'] == hour_int, 'prediction'] = future_preds

        historical_preds = gbm_model.predict(X_slot)
        historical_hourly_gbm_preds.loc[historical_hourly_gbm_preds['Hour'] == hour_int, 'gbm_pred'] = historical_preds

    historical_hourly_gbm_preds['gbm_pred'] = historical_hourly_gbm_preds['gbm_pred'].fillna(historical_hourly_gbm_preds[metric].mean())
    return historical_hourly_gbm_preds, future_hourly_df

# ==========================================
# 5. Main Pipeline
# ==========================================

def prophet_stacking_pipeline(file_name, hospital, metric, forecast_days=7, weather_thresholds=None):
    """
    Main pipeline to forecast a metric using a sophisticated ensemble approach with weather and virus data.
    """
    print(f"--- Starting Full Forecasting Pipeline for {hospital} - {metric} ---")

    if weather_thresholds is None:
        weather_thresholds = {'rain_threshold': 0.1, 'snow_threshold': 0, 'hot_temp_threshold': 18, 'cold_temp_threshold': 10,
                              'freezing_temp_threshold': 0, 'windy_speed_threshold': 20, 'pressure_drop_threshold': -10, 'humid_rh_threshold': 85}

    try:
        # Step 1: Download Virus Data
        df_virus = download_and_process_virus_data()

        # Step 2: Prepare Hospital and Weather Data
        if not os.path.exists(file_name):
            print(f"❌ File '{file_name}' not found.")
            print("Please upload your Excel file now.")
            uploaded = files.upload()
            # Verify file name matches
            if file_name not in uploaded:
                # If they uploaded something else, take the first file
                file_name = list(uploaded.keys())[0]
                print(f"Using uploaded file: {file_name}")

        df_raw = pd.read_excel(file_name)
        df = prepare_data_vectorized(df_raw, hospital, weather_thresholds)

        # Step 3: Merge Virus Data
        df, virus_features = merge_virus_data(df, df_virus)

        # Step 4: Add Lag Features
        current_metric = metric
        other_metric = "Trolleys" if metric == "ED Beds" else "ED Beds"

        df_with_ed_lags, _ = add_lag_features_fast(df.copy(), "ED Beds")
        df_final, _ = add_lag_features_fast(df_with_ed_lags.copy(), "Trolleys")
        df = df_final

        # Step 5: Forecast Daily Metrics with ARIMAX
        current_daily_metric_name = f"Daily_{current_metric.replace(' ', '_')}"
        other_daily_metric_name = f"Daily_{other_metric.replace(' ', '_')}"

        print(f"\nStep 1/3: Forecasting daily metrics for {hospital} using ARIMAX...")
        current_daily_forecast_df = compare_daily_models(df, hospital, current_daily_metric_name, virus_features, forecast_days)
        other_daily_forecast_df = compare_daily_models(df, hospital, other_daily_metric_name, virus_features, forecast_days)

        if current_daily_forecast_df is None or other_daily_forecast_df is None:
            print(f"❌ Skipping main pipeline due to failed daily ARIMAX training for {hospital}.")
            return

        # Step 6: Train Hourly GBM models to create features for Prophet
        df['ds'] = df['Datetime']
        print(f"\nStep 2/3: Training hourly GBM models for {hospital} - {current_metric}...")
        historical_gbm_preds_df, future_gbm_preds_df = train_and_predict_hourly_gbm_combined(
            df.copy(), hospital, current_metric, current_daily_metric_name, current_daily_forecast_df,
            other_metric, other_daily_metric_name, other_daily_forecast_df, virus_features, forecast_days)

        # Step 7: Final Prophet Forecast
        print(f"\nStep 3/3: Generating final Prophet forecast for {current_metric}...")
        m1 = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True,
                     changepoint_prior_scale=0.05, holidays_prior_scale=10.0, seasonality_prior_scale=10.0)

        m1.add_regressor('gbm_pred')
        m1.add_regressor('Hour')

        # Add virus data as regressors to final Prophet model
        for feature in virus_features:
            if feature in historical_gbm_preds_df.columns:
                m1.add_regressor(feature)

        prophet_fit_df = historical_gbm_preds_df[['ds', current_metric, 'gbm_pred', 'Hour'] + virus_features].rename(columns={current_metric: 'y'}).dropna(subset=['y'])
        m1.fit(prophet_fit_df)

        future_df_for_prophet = future_gbm_preds_df.rename(columns={'prediction': 'gbm_pred'})

        # Combine historical and future data for prediction
        combined_df_for_prophet = pd.concat([
            historical_gbm_preds_df[['ds', 'gbm_pred', 'Hour'] + virus_features],
            future_df_for_prophet[['ds', 'gbm_pred', 'Hour'] + virus_features]
        ], ignore_index=True)

        forecast = m1.predict(combined_df_for_prophet)

        print("\n--- ✅ FINAL FORECAST COMPLETE ---")
        cols_to_show = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
        print(forecast[cols_to_show].tail(forecast_days * 3))

        # Plotting results
        plt.figure(figsize=(12, 6))
        plt.plot(prophet_fit_df['ds'], prophet_fit_df['y'], label='Actual', color='black', alpha=0.5)
        plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='blue')
        plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='blue', alpha=0.2)
        plt.title(f'{hospital} - {metric} Forecast')
        plt.legend()
        plt.show()

    except Exception as e:
        print(f"❌ Error in main pipeline: {e}")
        traceback.print_exc()

# ==========================================
# 6. Execution Block
# ==========================================

# Configuration
target_file_name = 'Minus 7 - 2.xlsx'
target_hospital = "Cork University Hospital"
target_metric = "ED Beds"

custom_weather_thresholds = {
    'rain_threshold': 0.5, 'snow_threshold': 0.2, 'hot_temp_threshold': 18,
    'very_hot_temp_threshold': 24, 'cold_temp_threshold': 5, 'freezing_temp_threshold': -2,
    'windy_speed_threshold': 25, 'very_windy_speed_threshold': 35,
    'pressure_drop_threshold': -10, 'humid_rh_threshold': 85
}

# Run Pipeline
if __name__ == "__main__":
    prophet_stacking_pipeline(target_file_name, target_hospital, target_metric,
                              forecast_days=7, weather_thresholds=custom_weather_thresholds)
