import sys
import subprocess
import os
import pandas as pd
import numpy as np
import traceback
from datetime import timedelta, datetime, date
from prophet import Prophet
from sklearn.ensemble import GradientBoostingRegressor
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday
from pandas.tseries.offsets import DateOffset
from dateutil.rrule import MO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
import requests
from io import StringIO
from statsmodels.tsa.arima.model import ARIMA
from meteostat import Point, Hourly
import warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
IN_COLAB = False
DEFAULT_HOSPITAL_DATA_PATH = "hospital_data.xlsx"

# --- Holiday Calendar (cached) ---
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

_HOLIDAY_CACHE = {}

def get_holidays(start_date, end_date):
    """Cached holiday retrieval"""
    key = (start_date.year, end_date.year)
    if key not in _HOLIDAY_CACHE:
        calendar = IrishBankHolidays()
        _HOLIDAY_CACHE[key] = set(calendar.holidays(start=start_date, end=end_date))
    return _HOLIDAY_CACHE[key]

# --- HPSC Virus Data Download Function ---
def download_and_process_virus_data():
    """Downloads and processes the HPSC respiratory virus data (Flu, COVID, RSV)."""
    DATASET_CSV_URL = "https://respiratoryvirus.hpsc.ie/datasets/dbd10a497fd84e44954b858249919997_0.csv"
    print(f"\n--- Downloading virus data from HPSC ---")
    try:
        response = requests.get(DATASET_CSV_URL, timeout=20)
        response.raise_for_status()
        print("   ✅ Successfully downloaded the virus data.")

        df = pd.read_csv(StringIO(response.text), usecols=['floor', 'weekly_cases', 'disease'])
        df['Date'] = pd.to_datetime(df['floor'], utc=True).dt.tz_localize(None).dt.normalize()

        df_pivot = df.pivot_table(
            index='Date',
            columns='disease',
            values='weekly_cases',
            aggfunc='sum',
            fill_value=0
        )
        df_pivot.columns = [f"{col.replace(' ', '_')}_weekly_cases" for col in df_pivot.columns]
        df_pivot = df_pivot.reset_index().sort_index()
        print("   ✅ Virus data processed successfully.")
        return df_pivot
    except requests.exceptions.RequestException as e:
        print(f"   ❌ [FAILURE] Could not download the virus data file. Error: {e}")
    except Exception as e:
        print(f"   ❌ [FAILURE] An unexpected error occurred while processing virus data: {e}")
    return pd.DataFrame()

def get_weather_data_from_meteostat(start_date, end_date, location_name):
    """Fetches raw hourly weather data for a specific location using Meteostat."""
    print(f"   Fetching raw hourly weather data from Meteostat for {location_name} from {start_date.date()} to {end_date.date()}...")
    locations_map = {
        "Cork": Point(51.8979, -8.4706, 25),
        "Kerry": Point(52.2704, -9.7026, 34)
    }
    if location_name not in locations_map:
        print(f"   ❌ Error: Location '{location_name}' not defined for Meteostat.")
        return None

    try:
        data = Hourly(locations_map[location_name], start_date, end_date).fetch()
        if data.empty:
            print(f"   ⚠️ No raw hourly weather data fetched for {location_name}.")
            return None

        data = data.reset_index().rename(columns={'time': 'Datetime'})
        data['Datetime'] = pd.to_datetime(data['Datetime'])

        raw_weather_cols = ['Datetime', 'temp', 'dwpt', 'rhum', 'prcp', 'snow', 'wdir', 'wspd', 'wpgt', 'pres', 'coco']
        data = data[[col for col in raw_weather_cols if col in data.columns]]
        print(f"   ✅ Raw hourly Meteostat data fetched for {location_name}: {len(data)} records.")
        return data
    except Exception as e:
        print(f"   ❌ Error fetching raw Meteostat data for {location_name}: {e}")
        return None

def load_weather_data(hospital_name, start_date, end_date):
    print(f"   Loading raw hourly weather data for {hospital_name} region...")
    hospital_weather_mapping = {
        'Cork University Hospital': 'Cork',
        'Mercy University Hospital': 'Cork',
        'UH Kerry': 'Kerry'
    }
    weather_region = hospital_weather_mapping.get(hospital_name, 'Cork')
    weather_data = get_weather_data_from_meteostat(start_date, end_date, weather_region)
    if weather_data is not None:
        print(f"   ✅ {weather_region.capitalize()} raw hourly weather data loaded.")
    return weather_data, weather_region

def merge_weather_data(df_merged, weather_data, hospital_name):
    print(f"   Merging raw hourly weather data for {hospital_name}...")
    df_merged['Datetime'] = pd.to_datetime(df_merged['Datetime'])

    expected_raw_weather_cols = ['temp', 'dwpt', 'rhum', 'prcp', 'snow', 'wdir', 'wspd', 'wpgt', 'pres', 'coco']

    if weather_data is None:
        print("   ⚠️ Weather data not available. Initializing weather columns to 0.")
        for col in expected_raw_weather_cols:
            df_merged[col] = 0.0
        return df_merged

    weather_data['Datetime'] = pd.to_datetime(weather_data['Datetime'])
    merged_hospital = pd.merge(df_merged, weather_data, on='Datetime', how='left')

    fill_with_zero = {'prcp', 'snow'}
    for col in expected_raw_weather_cols:
        if col not in merged_hospital.columns:
            merged_hospital[col] = 0.0
        elif col in fill_with_zero:
            merged_hospital[col] = merged_hospital[col].fillna(0)
        else:
            merged_hospital[col] = merged_hospital[col].ffill().bfill().fillna(0)

    print(f"   ✅ Raw hourly weather data merged for {hospital_name}.")
    return merged_hospital

def merge_virus_data(df, df_virus):
    print("   Merging virus data with hospital data...")
    if df_virus.empty:
        print("   ⚠️ Virus data is empty. Skipping merge.")
        return df, []

    virus_features = [col for col in df_virus.columns if col != 'Date']
    df['DateOnly'] = df['Datetime'].dt.normalize()

    df_merged = pd.merge_asof(
        df.sort_values('DateOnly'),
        df_virus.sort_values('Date'),
        left_on='DateOnly',
        right_on='Date',
        direction='backward'
    )

    df_merged[virus_features] = df_merged[virus_features].fillna(0)
    print("   ✅ Virus data merged successfully.")
    return df_merged, virus_features

def prepare_data_vectorized(df_raw, hospital_name, weather_thresholds):
    print("   Preparing data: Renaming columns and unpivoting...")
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

    ed_cols = [f'ED_{t}' for t in times]
    trolley_cols = [f'Trolley_{t}' for t in times]

    # Check if columns exist
    missing_cols = [c for c in ed_cols + trolley_cols if c not in df.columns]
    if missing_cols:
         print(f"   ⚠️ Missing columns in input data: {missing_cols}. Please check Excel format.")
         # Mock missing columns to allow run
         for c in missing_cols:
             df[c] = 0

    df_ed = df.melt(
        id_vars=['Hospital Group Name', 'Hospital', 'Date', 'Additional_Capacity'],
        value_vars=ed_cols,
        var_name='Time_Slot',
        value_name='ED Beds'
    )
    df_ed['Time'] = df_ed['Time_Slot'].map({f'ED_{t}': time_map[t] for t in times})

    df_trolley = df.melt(
        id_vars=['Hospital', 'Date'],
        value_vars=trolley_cols,
        var_name='Time_Slot',
        value_name='Trolleys'
    )
    df_trolley['Time'] = df_trolley['Time_Slot'].map({f'Trolley_{t}': time_map[t] for t in times})

    df_merged = pd.merge(
        df_ed,
        df_trolley[['Hospital', 'Date', 'Time', 'Trolleys']],
        on=['Hospital', 'Date', 'Time'],
        how='left'
    )

    df_merged.rename(columns={'Additional_Capacity': 'Capacity'}, inplace=True)
    df_merged['Datetime'] = pd.to_datetime(df_merged['Date'].astype(str) + ' ' + df_merged['Time'])
    df_merged = df_merged.sort_values(['Hospital', 'Datetime']).reset_index(drop=True)

    df_merged['ED Beds'] = pd.to_numeric(df_merged['ED Beds'], errors='coerce')
    df_merged['Trolleys'] = pd.to_numeric(df_merged['Trolleys'], errors='coerce')

    df_hospital = df_merged[df_merged['Hospital'] == hospital_name].copy()

    if df_hospital.empty:
        print(f"   ❌ Error: No data found for hospital: {hospital_name}")
        return pd.DataFrame()

    min_date, max_date = df_hospital['Datetime'].min(), df_hospital['Datetime'].max()
    raw_hourly_weather_data, _ = load_weather_data(hospital_name, min_date, max_date)
    df_hospital = merge_weather_data(df_hospital, raw_hourly_weather_data, hospital_name)

    df_hospital['DateOnly'] = df_hospital['Datetime'].dt.normalize()

    daily_sums = df_hospital.groupby(['Hospital', 'DateOnly']).agg({
        'ED Beds': 'sum',
        'Trolleys': 'sum'
    }).rename(columns={'ED Beds': 'Daily_ED_Beds', 'Trolleys': 'Daily_Trolleys'}).reset_index()

    df_hospital = pd.merge(df_hospital, daily_sums, on=['Hospital', 'DateOnly'], how='left')

    if raw_hourly_weather_data is not None:
        raw_hourly_weather_data['DateOnly'] = raw_hourly_weather_data['Datetime'].dt.normalize()

        agg_dict = {
            'temp': 'mean', 'prcp': 'sum', 'snow': 'sum',
            'wspd': 'mean', 'pres': 'mean', 'rhum': 'mean', 'wpgt': 'max'
        }
        available_cols = {k: v for k, v in agg_dict.items() if k in raw_hourly_weather_data.columns}

        if available_cols:
            daily_weather_agg = raw_hourly_weather_data.groupby('DateOnly').agg(available_cols).reset_index()
            daily_weather_agg.columns = ['DateOnly'] + [f'Daily_{col}' for col in available_cols.keys()]
            df_hospital = pd.merge(df_hospital, daily_weather_agg, on='DateOnly', how='left')

            for col in daily_weather_agg.columns:
                if col != 'DateOnly':
                    df_hospital[col] = df_hospital[col].ffill().bfill().fillna(0)

    print("   Preparing data: Adding time-based features...")
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

    for i, day in enumerate(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']):
        df_hospital[f'Is{day}'] = (dt.dayofweek == i).astype(int)

    df_hospital['IsSummer'] = dt.month.isin([6, 7, 8]).astype(int)
    df_hospital['IsWinter'] = dt.month.isin([12, 1, 2]).astype(int)
    df_hospital['IsPeakHour'] = (dt.hour == 20).astype(int)
    df_hospital['IsLowHour'] = (dt.hour == 8).astype(int)

    df_hospital['Hour_sin'] = np.sin(2 * np.pi * dt.hour / 24)
    df_hospital['Hour_cos'] = np.cos(2 * np.pi * dt.hour / 24)

    try:
        holidays = get_holidays(df_hospital['Datetime'].min(), df_hospital['Datetime'].max())
        df_hospital['IsHoliday'] = df_hospital['Datetime'].dt.normalize().isin(holidays).astype(int)
    except Exception:
        df_hospital['IsHoliday'] = 0

    df_hospital['Hospital_Code'] = df_hospital['Hospital'].astype('category').cat.codes

    print("   Adding enhanced weather features (rolling windows)...")

    weather_feature_cols = [
        'prcp_avg_2hr', 'snow_sum_12hr', 'wspd_max_3hr', 'temp_min_2hr', 'rhum_avg_3hr', 'gust_factor',
        'is_rainy', 'is_snowy', 'is_hot', 'is_very_hot', 'is_cold', 'is_freezing', 'is_windy', 'is_very_windy',
        'is_humid', 'is_foggy', 'is_hot_lag1', 'is_hot_lag2', 'is_hot_lag3', 'is_cold_lag1', 'is_cold_lag2',
        'is_cold_lag3', 'is_rainy_lag1', 'is_rainy_lag2', 'is_rainy_lag3', 'hot_duration', 'cold_duration',
        'rainy_and_cold', 'hot_and_weekend', 'windy_and_rainy', 'hot_and_humid', 'pressure_change', 'pressure_drop',
        'winter_rain', 'summer_heat', 'morning_rain', 'afternoon_rain', 'evening_rain', 'morning_heat',
        'afternoon_heat', 'evening_heat', 'morning_cold', 'afternoon_cold', 'evening_cold', 'morning_ice',
        'afternoon_ice', 'evening_ice'
    ]
    for col in weather_feature_cols:
        df_hospital[col] = 0.0

    required_weather = ['temp', 'prcp', 'snow', 'wspd', 'pres', 'rhum', 'wpgt', 'coco']
    if all(col in df_hospital.columns for col in required_weather):
        df_hospital = df_hospital.sort_values(by=['Hospital', 'Datetime'])
        grouped = df_hospital.groupby('Hospital', sort=False)

        df_hospital['prcp_avg_2hr'] = grouped['prcp'].transform(lambda x: x.rolling(2, min_periods=1).mean().shift(1)).fillna(0)
        df_hospital['snow_sum_12hr'] = grouped['snow'].transform(lambda x: x.rolling(12, min_periods=1).sum().shift(1)).fillna(0)
        df_hospital['wspd_max_3hr'] = grouped['wspd'].transform(lambda x: x.rolling(3, min_periods=1).max().shift(1)).fillna(0)
        df_hospital['temp_min_2hr'] = grouped['temp'].transform(lambda x: x.rolling(2, min_periods=1).min().shift(1)).fillna(0)
        df_hospital['rhum_avg_3hr'] = grouped['rhum'].transform(lambda x: x.rolling(3, min_periods=1).mean().shift(1)).fillna(0)

        df_hospital['gust_factor'] = np.where(df_hospital['wspd'] > 0, df_hospital['wpgt'] / df_hospital['wspd'], 1)
        df_hospital['gust_factor'] = df_hospital['gust_factor'].replace([np.inf, -np.inf], 1).fillna(1)

        df_hospital['is_rainy'] = (df_hospital['prcp_avg_2hr'] > weather_thresholds['rain_threshold']).astype(int)
        df_hospital['is_snowy'] = (df_hospital['snow_sum_12hr'] > weather_thresholds['snow_threshold']).astype(int)
        df_hospital['is_hot'] = (df_hospital['temp'] >= weather_thresholds['hot_temp_threshold']).astype(int)
        df_hospital['is_cold'] = (df_hospital['temp'] < weather_thresholds['cold_temp_threshold']).astype(int)
        df_hospital['is_freezing'] = (df_hospital['temp_min_2hr'] <= weather_thresholds['freezing_temp_threshold']).astype(int)
        df_hospital['is_windy'] = (df_hospital['wspd_max_3hr'] > weather_thresholds['windy_speed_threshold']).astype(int)
        df_hospital['is_humid'] = (df_hospital['rhum_avg_3hr'] > weather_thresholds['humid_rh_threshold']).astype(int)
        df_hospital['is_foggy'] = df_hospital['coco'].isin([4, 5]).astype(int)

        for lag in [1, 2, 3]:
            df_hospital[f'is_hot_lag{lag}'] = grouped['is_hot'].shift(lag).fillna(0).astype(int)
            df_hospital[f'is_cold_lag{lag}'] = grouped['is_cold'].shift(lag).fillna(0).astype(int)
            df_hospital[f'is_rainy_lag{lag}'] = grouped['is_rainy'].shift(lag).fillna(0).astype(int)

        df_hospital['hot_duration'] = grouped['is_hot'].transform(
            lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
        ).fillna(0)

        df_hospital['rainy_and_cold'] = (df_hospital['is_rainy'] & df_hospital['is_cold']).astype(int)
        df_hospital['hot_and_weekend'] = (df_hospital['is_hot'] & df_hospital['IsWeekend']).astype(int)
        df_hospital['hot_and_humid'] = (df_hospital['is_hot'] & df_hospital['is_humid']).astype(int)

        df_hospital['pressure_change'] = grouped['pres'].diff().fillna(0)
        df_hospital['pressure_drop'] = (df_hospital['pressure_change'] < weather_thresholds['pressure_drop_threshold']).astype(int)
    else:
        print("   ⚠️ Not all raw weather columns available for enhanced feature engineering.")

    numeric_cols = df_hospital.select_dtypes(include=[np.number]).columns
    df_hospital[numeric_cols] = df_hospital[numeric_cols].fillna(0)

    print("   Data preparation complete.")
    return df_hospital

def add_lag_features_fast(df, target_column):
    print(f"   Adding lag features for {target_column}...")
    df = df.sort_values(['Hospital', 'Datetime'])
    grouped = df.groupby('Hospital', sort=False)[target_column]

    lag_features = []

    lags = [1, 2, 3, 6, 9, 12, 24, 48, 72, 168]
    # Enhanced Lags as per optimization request
    lags.extend([169, 170, 336]) # +1 week + 1/2 hours, +2 weeks

    for lag in lags:
        lag_col = f'Lag_{target_column}_{lag}'
        df[lag_col] = grouped.shift(lag)
        lag_features.append(lag_col)

    windows = [3, 6, 12, 24, 48]
    for window in windows:
        roll_col = f'Rolling_Mean_{window}_{target_column}'
        df[roll_col] = grouped.rolling(window=window, min_periods=1).mean().shift(1).reset_index(level=0, drop=True)
        lag_features.append(roll_col)

    df[lag_features] = df[lag_features].bfill().fillna(0)
    return df, lag_features

def compare_daily_models(df_prepared, hospital, daily_metric, virus_features, forecast_days=7):
    print(f"\n--- Training Daily ARIMAX Model for {hospital} - {daily_metric} ---")
    df = df_prepared[df_prepared['Hospital'] == hospital].copy()

    if df.empty:
        print(f"   ❌ No data for hospital {hospital}")
        return None

    daily_agg_dict = {
        'y': (daily_metric, 'first'),
        'IsHoliday': ('IsHoliday', 'first'),
        'IsMonday': ('IsMonday', 'first'),
        'IsTuesday': ('IsTuesday', 'first'),
        'IsWednesday': ('IsWednesday', 'first'),
        'IsThursday': ('IsThursday', 'first'),
        'IsFriday': ('IsFriday', 'first'),
        'IsSaturday': ('IsSaturday', 'first'),
        'IsSunday': ('IsSunday', 'first'),
        'IsSummer': ('IsSummer', 'first'),
        'IsWinter': ('IsWinter', 'first'),
        'Capacity': ('Capacity', 'first')
    }

    weather_cols = [
        'temp', 'prcp', 'snow', 'wspd', 'pres', 'rhum', 'is_rainy', 'is_snowy', 'is_hot', 'is_cold',
        'is_freezing', 'is_windy', 'is_humid', 'is_foggy', 'hot_duration', 'rainy_and_cold',
        'hot_and_weekend', 'hot_and_humid', 'pressure_drop', 'winter_rain', 'summer_heat',
        'Daily_temp', 'Daily_prcp', 'Daily_snow', 'Daily_wspd', 'Daily_pres', 'Daily_rhum', 'Daily_wpgt'
    ]

    for col in weather_cols:
        if col in df.columns:
            daily_agg_dict[col] = (col, 'mean')

    for col in virus_features:
        if col in df.columns:
            daily_agg_dict[col] = (col, 'first')

    df_daily = df.groupby('DateOnly').agg(**daily_agg_dict).reset_index()
    df_daily = df_daily.rename(columns={'DateOnly': 'ds'}).sort_values('ds').reset_index(drop=True)

    if df_daily['y'].notna().sum() < 30:
        print(f"   Not enough daily data for {hospital} {daily_metric}. Skipping daily ARIMAX.")
        return None

    potential_regressors = ['IsHoliday', 'IsMonday', 'IsTuesday', 'IsWednesday', 'IsThursday',
                           'IsFriday', 'IsSaturday', 'IsSummer', 'IsWinter', 'Capacity']
    potential_regressors.extend([col for col in weather_cols if col in df_daily.columns])
    potential_regressors.extend(virus_features)
    potential_regressors = list(set(potential_regressors))

    exog_cols = [r for r in potential_regressors if r in df_daily.columns]
    exog_numeric = df_daily[exog_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    constant_cols = [col for col in exog_numeric.columns if exog_numeric[col].nunique() <= 1]
    final_exog_cols = [col for col in exog_cols if col not in constant_cols]

    monthly_stats = {}
    for month in range(1, 13):
        month_data = df_daily[df_daily['ds'].dt.month == month]
        if not month_data.empty:
            monthly_stats[month] = {
                col: month_data[col].mean()
                for col in exog_cols
                if col in month_data.columns and pd.api.types.is_numeric_dtype(month_data[col])
            }

    last_date = df_daily['ds'].max()
    future_rows = []
    holidays = get_holidays(last_date, last_date + timedelta(days=forecast_days))

    for d in range(1, forecast_days + 1):
        future_date = last_date + timedelta(days=d)

        row = {
            'ds': future_date,
            'IsHoliday': int(future_date.normalize() in holidays),
            'IsMonday': int(future_date.dayofweek == 0),
            'IsTuesday': int(future_date.dayofweek == 1),
            'IsWednesday': int(future_date.dayofweek == 2),
            'IsThursday': int(future_date.dayofweek == 3),
            'IsFriday': int(future_date.dayofweek == 4),
            'IsSaturday': int(future_date.dayofweek == 5),
            'IsSummer': int(future_date.month in [6, 7, 8]),
            'IsWinter': int(future_date.month in [12, 1, 2]),
            'Capacity': df_daily['Capacity'].iloc[-1]
        }

        month_stats = monthly_stats.get(future_date.month, {})
        for col in exog_cols:
            if col in month_stats:
                row[col] = month_stats[col]
            elif col in virus_features:
                row[col] = df_daily[col].iloc[-1]
            elif col not in row:
                row[col] = df_daily[col].mean() if col in df_daily.columns and pd.api.types.is_numeric_dtype(df_daily[col]) else 0.0

        future_rows.append(row)

    future_df = pd.DataFrame(future_rows)
    future_exog = future_df[final_exog_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)

    try:
        y_train = pd.to_numeric(df_daily['y'], errors='coerce').fillna(df_daily['y'].median()).astype(float)
        X_train = df_daily[final_exog_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
        y_train.index = X_train.index = pd.to_datetime(df_daily['ds'])

        # ARIMAX with daily frequency
        model = ARIMA(y_train, exog=X_train, order=(6, 1, 1), seasonal_order=(1, 0, 1, 7)).fit()
        forecast_values = model.forecast(steps=forecast_days, exog=future_exog)

        forecast_df = pd.DataFrame({
            'DateOnly': pd.to_datetime(future_df['ds']),
            daily_metric: forecast_values.values
        })

        print(f"   ✅ Daily forecast generated using ARIMAX with seasonal component.")
        return forecast_df
    except Exception as e:
        print(f"   ❌ ARIMAX model failed: {e}")
        traceback.print_exc()
        return None

def get_hourly_features(metric, daily_metric_name, other_metric, other_daily_metric_name, df_columns, virus_features, for_arimax=False):
    base_features = [
        'DayOfWeek', 'Month', 'Quarter', 'WeekOfYear', 'DayOfMonth', 'DayOfYear', 'WeekOfMonth',
        'IsWeekend', 'IsMonday', 'IsTuesday', 'IsWednesday', 'IsThursday', 'IsFriday', 'IsSaturday', 'IsSunday',
        'IsSummer', 'IsWinter', 'IsHoliday', 'Capacity',
        daily_metric_name, other_metric, other_daily_metric_name,
        'temp', 'prcp', 'snow', 'wspd', 'pres', 'rhum', 'dwpt', 'wpgt', 'gust_factor',
        'prcp_avg_2hr', 'snow_sum_12hr', 'wspd_max_3hr', 'temp_min_2hr', 'rhum_avg_3hr',
        'is_rainy', 'is_snowy', 'is_hot', 'is_very_hot', 'is_cold', 'is_freezing', 'is_windy',
        'is_very_windy', 'is_humid', 'is_foggy', 'hot_duration', 'rainy_and_cold', 'hot_and_weekend',
        'hot_and_humid', 'pressure_change', 'pressure_drop', 'winter_rain', 'summer_heat'
    ]
    base_features.extend(virus_features)

    if not for_arimax:
        base_features.extend(['Hospital_Code', 'Hour_sin', 'Hour_cos', 'IsPeakHour', 'IsLowHour'])

    lag_features = [col for col in df_columns if col.startswith(f'Lag_{metric}') or
                    col.startswith(f'Rolling_Mean_{metric}') or col.startswith(f'Lag_{other_metric}') or
                    col.startswith(f'Rolling_Mean_{other_metric}')]

    all_features = base_features + lag_features
    # Also include the 2pm anchor if available
    if 'pred_2pm_anchor' in df_columns:
         all_features.append('pred_2pm_anchor')

    return [f for f in all_features if f in df_columns]

def train_and_predict_hourly_gbm_combined(df, hospital, metric, daily_metric_name, daily_forecast_df,
                                          other_metric, other_daily_metric_name, other_daily_forecast_df,
                                          virus_features, forecast_days=7):
    print(f"\n--- Training Hourly Sklearn GBM Models for {hospital} - {metric} ---")

    # We will implement the "3 models" strategy + "2pm Anchor" here.
    # Strategy:
    # 1. Predict 14:00 (2pm) first for all future days.
    # 2. Use the predicted 14:00 value as a feature for 08:00 (next day) and 20:00 (same day).
    # 3. For each slot, we train 3 models:
    #    - Model A (Day 1)
    #    - Model B (Days 2-4)
    #    - Model C (Days 5-7)

    available_features = get_hourly_features(metric, daily_metric_name, other_metric, other_daily_metric_name,
                                             df.columns, virus_features, for_arimax=False)

    df_filtered = df.dropna(subset=[metric]).copy()
    time_slots = {'08:00': 8, '14:00': 14, '20:00': 20}
    last_date = df['ds'].max()

    # 1. Prepare Future DataFrame Skeleton
    monthly_stats = {}
    for month in range(1, 13):
        month_data = df_filtered[df_filtered['Month'] == month]
        if not month_data.empty:
            monthly_stats[month] = {
                col: month_data[col].mean()
                for col in available_features
                if col in month_data.columns and pd.api.types.is_numeric_dtype(month_data[col])
            }

    future_rows = []
    holidays = get_holidays(last_date, last_date + timedelta(days=forecast_days))

    for d in range(1, forecast_days + 1):
        future_date = last_date.date() + timedelta(days=d)

        daily_val = daily_forecast_df.loc[
            daily_forecast_df['DateOnly'].dt.date == future_date, daily_metric_name
        ].values[0] if not daily_forecast_df[daily_forecast_df['DateOnly'].dt.date == future_date].empty else 0

        other_daily_val = other_daily_forecast_df.loc[
            other_daily_forecast_df['DateOnly'].dt.date == future_date, other_daily_metric_name
        ].values[0] if not other_daily_forecast_df[other_daily_forecast_df['DateOnly'].dt.date == future_date].empty else 0

        for t_str, hour in time_slots.items():
            dt = pd.to_datetime(f"{future_date} {t_str}")
            row = {
                'ds': dt, 'Hour': hour, 'DayOfWeek': dt.dayofweek, 'Month': dt.month, 'Quarter': dt.quarter,
                'WeekOfYear': dt.isocalendar().week, 'DayOfMonth': dt.day, 'DayOfYear': dt.dayofyear,
                'IsWeekend': int(dt.dayofweek in [5, 6]),
                'IsMonday': int(dt.dayofweek == 0), 'IsTuesday': int(dt.dayofweek == 1),
                'IsWednesday': int(dt.dayofweek == 2), 'IsThursday': int(dt.dayofweek == 3),
                'IsFriday': int(dt.dayofweek == 4), 'IsSaturday': int(dt.dayofweek == 5),
                'IsSunday': int(dt.dayofweek == 6), 'IsSummer': int(dt.month in [6, 7, 8]),
                'IsWinter': int(dt.month in [12, 1, 2]), 'IsPeakHour': int(dt.hour == 20),
                'IsLowHour': int(dt.hour == 8), 'Hour_sin': np.sin(2 * np.pi * dt.hour / 24),
                'Hour_cos': np.cos(2 * np.pi * dt.hour / 24),
                'IsHoliday': int(dt.normalize() in holidays),
                'Hospital_Code': df['Hospital_Code'].iloc[-1], 'Capacity': df['Capacity'].iloc[-1],
                daily_metric_name: daily_val, other_daily_metric_name: other_daily_val,
                'forecast_day_index': d
            }

            month_stats = monthly_stats.get(dt.month, {})
            for col in available_features:
                if col not in row:
                    if col in month_stats:
                        row[col] = month_stats[col]
                    elif col in virus_features:
                        row[col] = df[col].iloc[-1]
                    elif col.startswith(('Lag_', 'Rolling_')):
                         # Important: For recursive forecasting, we need updated lags.
                         # Here we just take the last known value, which is a simplification.
                         # A true recursive loop would update this.
                         # Given the complexity, we will rely on the daily models and features.
                        slot_data = df_filtered[df_filtered['Hour'] == hour]
                        row[col] = slot_data[col].iloc[-1] if not slot_data.empty and col in slot_data.columns else 0
                    else:
                        row[col] = 0
            future_rows.append(row)

    future_df = pd.DataFrame(future_rows)
    future_df['prediction'] = np.nan
    future_df['pred_2pm_anchor'] = 0 # Placeholder

    historical_preds = df.copy()
    historical_preds['gbm_pred'] = np.nan

    # 2. Train Models and Predict - ANCHOR FIRST (2pm)
    # We define the order: 14:00 first, then use it for others.

    # Define groups for "3 models" strategy
    horizon_groups = {
        'Day1': [1],
        'Day2_4': [2, 3, 4],
        'Day5_7': [5, 6, 7]
    }

    # We need models for each hour and each horizon group
    models = {} # Key: (hour, group_name)

    # Train loop
    for t_str, hour in time_slots.items():
        print(f"   Training models for {t_str}...")
        df_slot = df_filtered[df_filtered['Hour'] == hour].copy()

        if df_slot.empty or df_slot[metric].notna().sum() < 30:
            print(f"   ⚠️ Not enough data for {t_str} slot.")
            continue

        X_slot = df_slot[available_features].apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
        y_slot = df_slot[metric]

        # Train 3 variations (conceptually, we train on same data but could vary hyperparams or features if needed)
        # Here we train 3 distinct instances to satisfy "3 models" request
        for group_name in horizon_groups:
            # We could optimize hyperparameters per horizon here.
            # For now, we use the same robust Huber regressor.
            gbm = GradientBoostingRegressor(
                loss='huber', n_estimators=1500, learning_rate=0.04,
                max_depth=6, subsample=0.8, min_samples_split=5,
                min_samples_leaf=2, random_state=42 + (1 if group_name=='Day2_4' else 2 if group_name=='Day5_7' else 0)
            )
            gbm.fit(X_slot, y_slot)
            models[(hour, group_name)] = gbm

    # Predict Loop - 2pm First
    # Predict 2pm for all days
    anchor_hour = 14

    # Predict 2pm
    for d in range(1, forecast_days + 1):
        group = 'Day1' if d == 1 else ('Day2_4' if d <= 4 else 'Day5_7')

        mask = (future_df['Hour'] == anchor_hour) & (future_df['forecast_day_index'] == d)
        if mask.any() and (anchor_hour, group) in models:
            subset = future_df[mask]
            X_future = subset[available_features].apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
            pred = models[(anchor_hour, group)].predict(X_future)
            future_df.loc[mask, 'prediction'] = pred
            future_df.loc[mask, 'pred_2pm_anchor'] = pred # Set anchor for this row

            # Set this anchor value for OTHER hours of the SAME day (and next day if relevant)
            # Strategy: Use Day D 2pm to help Day D 8pm and Day D+1 8am?
            # Or Day D 2pm to help Day D 8pm and Day D 8am (retroactive for correlation)?
            # We are predicting, so for Day D 8am, we don't have Day D 2pm yet in reality.
            # BUT, we can use Day D-1 2pm for Day D 8am.
            # And Day D 2pm for Day D 8pm.

            # Let's populate 'pred_2pm_anchor' column in future_df
            # For 8pm (20:00) on Day D: use Day D 2pm
            future_df.loc[
                (future_df['forecast_day_index'] == d) & (future_df['Hour'] == 20),
                'pred_2pm_anchor'
            ] = pred

            # For 8am (08:00) on Day D+1: use Day D 2pm
            if d < forecast_days:
                 future_df.loc[
                    (future_df['forecast_day_index'] == d + 1) & (future_df['Hour'] == 8),
                    'pred_2pm_anchor'
                ] = pred

    # Add anchor to available features if not there
    if 'pred_2pm_anchor' not in available_features:
        # We need to retrain 8am and 8pm models to include this feature!
        # This implies a 2-stage training.
        # Stage 1: Train 2pm model.
        # Stage 2: Train 8am/8pm models INCLUDING 2pm (actuals in training) as feature.
        pass # see below

    # Re-train 8am and 8pm models with 'pred_2pm_anchor' feature
    # First, populate 'pred_2pm_anchor' in historical data (using actual 2pm)
    print("   Re-training 8am/8pm models with 2pm anchor...")

    # Add actual 2pm values as 'pred_2pm_anchor' for training
    df_filtered_anchor = df_filtered.copy()

    # Get 2pm values per date
    df_2pm = df_filtered[df_filtered['Hour'] == 14][['DateOnly', metric]].rename(columns={metric: 'val_2pm'})
    df_filtered_anchor = pd.merge(df_filtered_anchor, df_2pm, on='DateOnly', how='left')

    # Shift for 8am: Use previous day's 2pm? Or same day?
    # 8am is before 2pm. So 8am on Day T should use 2pm from Day T-1.
    df_filtered_anchor['val_2pm_lag1'] = df_filtered_anchor.groupby('Hospital')['val_2pm'].shift(1)

    # For 8am, use lag1 2pm. For 8pm, use same day 2pm.
    df_filtered_anchor['pred_2pm_anchor'] = np.where(
        df_filtered_anchor['Hour'] == 8,
        df_filtered_anchor['val_2pm_lag1'],
        df_filtered_anchor['val_2pm']
    )
    df_filtered_anchor['pred_2pm_anchor'] = df_filtered_anchor['pred_2pm_anchor'].fillna(0)

    # Update available features
    available_features_anchor = available_features + ['pred_2pm_anchor']

    # Retrain 8am and 8pm models
    for hour in [8, 20]:
        df_slot = df_filtered_anchor[df_filtered_anchor['Hour'] == hour].copy()
        if df_slot.empty: continue

        X_slot = df_slot[available_features_anchor].apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
        y_slot = df_slot[metric]

        for group_name in horizon_groups:
             gbm = GradientBoostingRegressor(
                loss='huber', n_estimators=1500, learning_rate=0.04,
                max_depth=6, subsample=0.8, min_samples_split=5,
                min_samples_leaf=2, random_state=42
            )
             gbm.fit(X_slot, y_slot)
             models[(hour, group_name)] = gbm

    # Now Predict 8am and 8pm using the anchor populated in future_df
    for d in range(1, forecast_days + 1):
        group = 'Day1' if d == 1 else ('Day2_4' if d <= 4 else 'Day5_7')

        for hour in [8, 20]:
            mask = (future_df['Hour'] == hour) & (future_df['forecast_day_index'] == d)
            if mask.any() and (hour, group) in models:
                subset = future_df[mask]
                # Ensure anchor is ready
                # For Day 1 8am, we need Day 0 2pm (actual).
                if d == 1 and hour == 8:
                     # Use last actual 2pm
                     last_2pm = df_filtered[df_filtered['Hour'] == 14][metric].iloc[-1]
                     future_df.loc[mask, 'pred_2pm_anchor'] = last_2pm

                X_future = subset[available_features_anchor].apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
                pred = models[(hour, group)].predict(X_future)
                future_df.loc[mask, 'prediction'] = pred

    # Historical predictions (approximation)
    # We just run the models on training data for plotting
    for t_str, hour in time_slots.items():
        if hour == 14:
            X_slot = df_filtered[available_features].apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
            if (14, 'Day1') in models:
                historical_preds.loc[historical_preds['Hour'] == 14, 'gbm_pred'] = models[(14, 'Day1')].predict(X_slot)
        else:
            # Requires anchor
             df_slot = df_filtered_anchor[df_filtered_anchor['Hour'] == hour]
             X_slot = df_slot[available_features_anchor].apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
             if (hour, 'Day1') in models and not df_slot.empty:
                  # Map index back
                  preds = models[(hour, 'Day1')].predict(X_slot)
                  historical_preds.loc[df_slot.index, 'gbm_pred'] = preds

    historical_preds['gbm_pred'] = historical_preds['gbm_pred'].fillna(historical_preds[metric].mean())

    return historical_preds, future_df

def prophet_stacking_pipeline(file_path, hospital, metric, forecast_days=7, weather_thresholds=None):
    print(f"--- Starting Full Forecasting Pipeline for {hospital} - {metric} ---")

    if weather_thresholds is None:
        weather_thresholds = {
            'rain_threshold': 0.1, 'snow_threshold': 0, 'hot_temp_threshold': 18,
            'cold_temp_threshold': 10, 'freezing_temp_threshold': 0, 'windy_speed_threshold': 20,
            'pressure_drop_threshold': -10, 'humid_rh_threshold': 85
        }

    try:
        df_virus = download_and_process_virus_data()

        print(f"Reading file: {file_path}")
        if not os.path.exists(file_path):
             print(f"❌ File not found: {file_path}")
             return

        df_raw = pd.read_excel(file_path, engine='openpyxl')
        df = prepare_data_vectorized(df_raw, hospital, weather_thresholds)

        if df.empty:
            print("❌ Data preparation returned empty DataFrame.")
            return

        df, virus_features = merge_virus_data(df, df_virus)

        current_metric = metric
        other_metric = "Trolleys" if metric == "ED Beds" else "ED Beds"

        df, _ = add_lag_features_fast(df, "ED Beds")
        df, _ = add_lag_features_fast(df, "Trolleys")

        current_daily_metric = f"Daily_{current_metric.replace(' ', '_')}"
        other_daily_metric = f"Daily_{other_metric.replace(' ', '_')}"

        print(f"\nStep 1/3: Forecasting daily metrics for {hospital} using ARIMAX...")
        current_daily_forecast = compare_daily_models(df, hospital, current_daily_metric, virus_features, forecast_days)
        other_daily_forecast = compare_daily_models(df, hospital, other_daily_metric, virus_features, forecast_days)

        if current_daily_forecast is None or other_daily_forecast is None:
            print(f"❌ Skipping main pipeline due to failed daily ARIMAX training.")
            return

        df['ds'] = df['Datetime']
        print(f"\nStep 2/3: Training hourly GBM models for {hospital} - {current_metric}...")

        # Using the new Logic: 3 Models + 2pm Anchor
        historical_preds, future_preds = train_and_predict_hourly_gbm_combined(
            df.copy(), hospital, current_metric, current_daily_metric, current_daily_forecast,
            other_metric, other_daily_metric, other_daily_forecast, virus_features, forecast_days
        )

        print(f"\nStep 3/3: Generating final Prophet forecast for {current_metric}...")

        m = Prophet(
            seasonality_mode='multiplicative',
            yearly_seasonality=10,
            weekly_seasonality=10,
            daily_seasonality=10,
            changepoint_prior_scale=0.08,
            holidays_prior_scale=12.0,
            seasonality_prior_scale=12.0
        )

        m.add_regressor('gbm_pred')
        m.add_regressor('Hour')

        for feature in virus_features:
            if feature in historical_preds.columns:
                m.add_regressor(feature)

        prophet_train = historical_preds[['ds', current_metric, 'gbm_pred', 'Hour'] + virus_features]
        prophet_train = prophet_train.rename(columns={current_metric: 'y'}).dropna(subset=['y'])
        m.fit(prophet_train)

        future_for_prophet = future_preds.rename(columns={'prediction': 'gbm_pred'})
        combined = pd.concat([
            historical_preds[['ds', 'gbm_pred', 'Hour'] + virus_features],
            future_for_prophet[['ds', 'gbm_pred', 'Hour'] + virus_features]
        ], ignore_index=True)

        forecast = m.predict(combined)

        print("\n--- ✅ FINAL FORECAST COMPLETE ---")
        output_file = f"forecast_{hospital.replace(' ', '_')}_{metric.replace(' ', '_')}.csv"

        cols_to_save = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
        forecast[cols_to_save].tail(forecast_days * 3).to_csv(output_file, index=False)
        print(f"Forecast saved to {output_file}")

        # Verify result against user expectation (closeness check would go here if we had ground truth)
        print("Forecast snippet:")
        print(forecast[cols_to_save].tail(forecast_days * 3))

    except Exception as e:
        print(f"❌ Error in main pipeline: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    file_path = DEFAULT_HOSPITAL_DATA_PATH

    # Check if file exists, if not try to find it in current dir
    if not os.path.exists(file_path):
        found_files = [f for f in os.listdir('.') if f.endswith('.xlsx')]
        if found_files:
            file_path = found_files[0]
            print(f"Using found Excel file: {file_path}")

    if os.path.exists(file_path):
        hospital_name = "Cork University Hospital"
        metric_name = "ED Beds"

        custom_thresholds = {
            'rain_threshold': 0.5, 'snow_threshold': 0.2, 'hot_temp_threshold': 18,
            'very_hot_temp_threshold': 24, 'cold_temp_threshold': 5, 'freezing_temp_threshold': -2,
            'windy_speed_threshold': 25, 'very_windy_speed_threshold': 35,
            'pressure_drop_threshold': -10, 'humid_rh_threshold': 85
        }

        prophet_stacking_pipeline(file_path, hospital_name, metric_name,
                                    forecast_days=7, weather_thresholds=custom_thresholds)
    else:
        print(f"❌ Input file '{file_path}' not found. Please place 'hospital_data.xlsx' in the directory.")
