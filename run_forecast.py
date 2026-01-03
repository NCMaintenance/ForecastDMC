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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
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
    key = (start_date.year, end_date.year)
    if key not in _HOLIDAY_CACHE:
        calendar = IrishBankHolidays()
        _HOLIDAY_CACHE[key] = set(calendar.holidays(start=start_date, end=end_date))
    return _HOLIDAY_CACHE[key]

# --- Data Download ---
def download_and_process_virus_data():
    DATASET_CSV_URL = "https://respiratoryvirus.hpsc.ie/datasets/dbd10a497fd84e44954b858249919997_0.csv"
    print(f"\n--- Downloading virus data from HPSC ---")
    try:
        response = requests.get(DATASET_CSV_URL, timeout=10)
        response.raise_for_status()
        print("   ✅ Successfully downloaded the virus data.")
        df = pd.read_csv(StringIO(response.text), usecols=['floor', 'weekly_cases', 'disease'])
        df['Date'] = pd.to_datetime(df['floor'], utc=True).dt.tz_localize(None).dt.normalize()
        df_pivot = df.pivot_table(index='Date', columns='disease', values='weekly_cases', aggfunc='sum', fill_value=0)
        df_pivot.columns = [f"{col.replace(' ', '_')}_weekly_cases" for col in df_pivot.columns]
        df_pivot = df_pivot.reset_index().sort_index()
        return df_pivot
    except Exception as e:
        print(f"   ❌ [FAILURE] processing virus data: {e}")
    return pd.DataFrame()

def get_weather_data_from_meteostat(start_date, end_date, location_name):
    print(f"   Fetching raw hourly weather data for {location_name} (from {start_date.date()} to {end_date.date()})...")
    locations_map = { "Cork": Point(51.8979, -8.4706, 25), "Kerry": Point(52.2704, -9.7026, 34) }
    if location_name not in locations_map: return None
    try:
        data = Hourly(locations_map[location_name], start_date, end_date).fetch()
        if data.empty:
            print("   ⚠️ No weather data found.")
            return None
        data = data.reset_index().rename(columns={'time': 'Datetime'})
        data['Datetime'] = pd.to_datetime(data['Datetime'])
        print(f"   ✅ Fetched {len(data)} weather records.")
        return data
    except Exception as e:
        print(f"   ❌ Error fetching raw Meteostat data: {e}")
        return None

def load_weather_data(hospital_name, start_date, end_date):
    mapping = {'Cork University Hospital': 'Cork', 'Mercy University Hospital': 'Cork', 'UH Kerry': 'Kerry'}
    region = mapping.get(hospital_name, 'Cork')
    return get_weather_data_from_meteostat(start_date, end_date, region), region

def merge_weather_data(df_merged, weather_data, hospital_name):
    print(f"   Merging raw hourly weather data...")
    df_merged['Datetime'] = pd.to_datetime(df_merged['Datetime'])
    expected_cols = ['temp', 'dwpt', 'rhum', 'prcp', 'snow', 'wdir', 'wspd', 'wpgt', 'pres', 'coco']

    if weather_data is None:
        for col in expected_cols: df_merged[col] = 0.0
        return df_merged

    weather_data['Datetime'] = pd.to_datetime(weather_data['Datetime'])
    merged = pd.merge(df_merged, weather_data, on='Datetime', how='left')

    fill_zero = {'prcp', 'snow'}
    for col in expected_cols:
        if col not in merged.columns: merged[col] = 0.0
        elif col in fill_zero: merged[col] = merged[col].fillna(0)
        else: merged[col] = merged[col].ffill().bfill().fillna(0)
    return merged

def merge_virus_data(df, df_virus):
    print("   Merging virus data...")
    if df_virus.empty: return df, []
    virus_features = [col for col in df_virus.columns if col != 'Date']
    df['DateOnly'] = df['Datetime'].dt.normalize()
    df_merged = pd.merge_asof(df.sort_values('DateOnly'), df_virus.sort_values('Date'),
                              left_on='DateOnly', right_on='Date', direction='backward')
    df_merged[virus_features] = df_merged[virus_features].fillna(0)
    return df_merged, virus_features

def prepare_data_vectorized(df_raw, hospital_name, weather_thresholds):
    print("   Preparing data...")
    df = df_raw.copy()
    mapping = {
        'Tracker8am': 'ED_8am', 'Tracker2pm': 'ED_2pm', 'Tracker8pm': 'ED_8pm',
        'TimeTotal_8am': 'Trolley_8am', 'TimeTotal_2pm': 'Trolley_2pm', 'TimeTotal_8pm': 'Trolley_8pm',
        'AdditionalCapacityOpen Morning': 'Additional_Capacity'
    }
    df = df.rename(columns=mapping)
    df['Additional_Capacity'] = df.groupby(['Hospital', 'Date'])['Additional_Capacity'].transform('first').fillna(0)

    times = ['8am', '2pm', '8pm']
    time_map = {'8am': '08:00', '2pm': '14:00', '8pm': '20:00'}

    cols = [f'ED_{t}' for t in times] + [f'Trolley_{t}' for t in times]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"   ⚠️ Missing columns: {missing}. Filling with 0.")
        for c in missing: df[c] = 0

    df_ed = df.melt(id_vars=['Hospital Group Name', 'Hospital', 'Date', 'Additional_Capacity'],
                    value_vars=[f'ED_{t}' for t in times], var_name='Time_Slot', value_name='ED Beds')
    df_ed['Time'] = df_ed['Time_Slot'].map({f'ED_{t}': time_map[t] for t in times})

    df_tr = df.melt(id_vars=['Hospital', 'Date'], value_vars=[f'Trolley_{t}' for t in times],
                    var_name='Time_Slot', value_name='Trolleys')
    df_tr['Time'] = df_tr['Time_Slot'].map({f'Trolley_{t}': time_map[t] for t in times})

    df_merged = pd.merge(df_ed, df_tr[['Hospital', 'Date', 'Time', 'Trolleys']], on=['Hospital', 'Date', 'Time'], how='left')
    df_merged.rename(columns={'Additional_Capacity': 'Capacity'}, inplace=True)
    df_merged['Datetime'] = pd.to_datetime(df_merged['Date'].astype(str) + ' ' + df_merged['Time'])
    df_merged = df_merged.sort_values(['Hospital', 'Datetime']).reset_index(drop=True)

    df_merged['ED Beds'] = pd.to_numeric(df_merged['ED Beds'], errors='coerce')
    df_merged['Trolleys'] = pd.to_numeric(df_merged['Trolleys'], errors='coerce')

    # --- AUTO-DETECT HOSPITAL ---
    if hospital_name not in df_merged['Hospital'].unique():
        print(f"   ⚠️ Hospital '{hospital_name}' not found. Available: {df_merged['Hospital'].unique()}")
        hospital_name = df_merged['Hospital'].unique()[0]
        print(f"   ⚠️ Defaulting to '{hospital_name}'")

    df_hosp = df_merged[df_merged['Hospital'] == hospital_name].copy()
    if df_hosp.empty:
        print(f"   ❌ No data for {hospital_name}")
        return pd.DataFrame(), hospital_name

    min_d, max_d = df_hosp['Datetime'].min(), df_hosp['Datetime'].max()
    print(f"   Data range: {min_d} to {max_d}")
    weather, _ = load_weather_data(hospital_name, min_d, max_d)
    df_hosp = merge_weather_data(df_hosp, weather, hospital_name)
    df_hosp['DateOnly'] = df_hosp['Datetime'].dt.normalize()

    daily = df_hosp.groupby(['Hospital', 'DateOnly']).agg({'ED Beds': 'sum', 'Trolleys': 'sum'}).rename(
        columns={'ED Beds': 'Daily_ED_Beds', 'Trolleys': 'Daily_Trolleys'}).reset_index()
    df_hosp = pd.merge(df_hosp, daily, on=['Hospital', 'DateOnly'], how='left')

    if weather is not None:
        weather['DateOnly'] = weather['Datetime'].dt.normalize()
        agg_d = {'temp': 'mean', 'prcp': 'sum', 'snow': 'sum', 'wspd': 'mean', 'pres': 'mean', 'rhum': 'mean', 'wpgt': 'max'}
        avail = {k: v for k, v in agg_d.items() if k in weather.columns}
        if avail:
            daily_w = weather.groupby('DateOnly').agg(avail).reset_index()
            daily_w.columns = ['DateOnly'] + [f'Daily_{c}' for c in avail]
            df_hosp = pd.merge(df_hosp, daily_w, on='DateOnly', how='left')
            for c in daily_w.columns:
                if c != 'DateOnly': df_hosp[c] = df_hosp[c].ffill().bfill().fillna(0)

    dt = df_hosp['Datetime'].dt
    df_hosp['Hour'] = dt.hour
    df_hosp['DayOfWeek'] = dt.dayofweek
    df_hosp['Month'] = dt.month
    df_hosp['Quarter'] = dt.quarter
    df_hosp['WeekOfYear'] = dt.isocalendar().week.astype(int)
    df_hosp['DayOfMonth'] = dt.day
    df_hosp['IsWeekend'] = dt.dayofweek.isin([5, 6]).astype(int)

    for i, d in enumerate(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']):
        df_hosp[f'Is{d}'] = (dt.dayofweek == i).astype(int)

    df_hosp['IsSummer'] = dt.month.isin([6, 7, 8]).astype(int)
    df_hosp['IsWinter'] = dt.month.isin([12, 1, 2]).astype(int)
    df_hosp['IsPeakHour'] = (dt.hour == 20).astype(int)
    df_hosp['IsLowHour'] = (dt.hour == 8).astype(int)
    df_hosp['Hour_sin'] = np.sin(2 * np.pi * dt.hour / 24)
    df_hosp['Hour_cos'] = np.cos(2 * np.pi * dt.hour / 24)

    try:
        hols = get_holidays(df_hosp['Datetime'].min(), df_hosp['Datetime'].max())
        df_hosp['IsHoliday'] = df_hosp['Datetime'].dt.normalize().isin(hols).astype(int)
    except: df_hosp['IsHoliday'] = 0

    df_hosp['Hospital_Code'] = df_hosp['Hospital'].astype('category').cat.codes

    req_w = ['temp', 'prcp', 'snow', 'wspd', 'pres', 'rhum', 'wpgt', 'coco']
    if all(c in df_hosp.columns for c in req_w):
        df_hosp = df_hosp.sort_values(['Hospital', 'Datetime'])
        g = df_hosp.groupby('Hospital', sort=False)

        df_hosp['prcp_avg_2hr'] = g['prcp'].transform(lambda x: x.rolling(2, min_periods=1).mean().shift(1)).fillna(0)
        df_hosp['snow_sum_12hr'] = g['snow'].transform(lambda x: x.rolling(12, min_periods=1).sum().shift(1)).fillna(0)
        df_hosp['wspd_max_3hr'] = g['wspd'].transform(lambda x: x.rolling(3, min_periods=1).max().shift(1)).fillna(0)
        df_hosp['temp_min_2hr'] = g['temp'].transform(lambda x: x.rolling(2, min_periods=1).min().shift(1)).fillna(0)
        df_hosp['rhum_avg_3hr'] = g['rhum'].transform(lambda x: x.rolling(3, min_periods=1).mean().shift(1)).fillna(0)

        df_hosp['gust_factor'] = np.where(df_hosp['wspd']>0, df_hosp['wpgt']/df_hosp['wspd'], 1)
        df_hosp['is_rainy'] = (df_hosp['prcp_avg_2hr'] > weather_thresholds['rain_threshold']).astype(int)
        df_hosp['is_snowy'] = (df_hosp['snow_sum_12hr'] > weather_thresholds['snow_threshold']).astype(int)
        df_hosp['is_hot'] = (df_hosp['temp'] >= weather_thresholds['hot_temp_threshold']).astype(int)
        df_hosp['is_cold'] = (df_hosp['temp'] < weather_thresholds['cold_temp_threshold']).astype(int)
        df_hosp['is_windy'] = (df_hosp['wspd_max_3hr'] > weather_thresholds['windy_speed_threshold']).astype(int)
        df_hosp['is_humid'] = (df_hosp['rhum_avg_3hr'] > weather_thresholds['humid_rh_threshold']).astype(int)

        df_hosp['pressure_change'] = g['pres'].diff().fillna(0)
        df_hosp['pressure_drop'] = (df_hosp['pressure_change'] < weather_thresholds['pressure_drop_threshold']).astype(int)

        df_hosp['rainy_and_cold'] = (df_hosp['is_rainy'] & df_hosp['is_cold']).astype(int)
        df_hosp['hot_and_weekend'] = (df_hosp['is_hot'] & df_hosp['IsWeekend']).astype(int)

    num_cols = df_hosp.select_dtypes(include=[np.number]).columns
    df_hosp[num_cols] = df_hosp[num_cols].fillna(0)
    print("   Data preparation complete.")
    return df_hosp, hospital_name

def add_lag_features_fast(df, target_column):
    print(f"   Adding lag features for {target_column}...")
    df = df.sort_values(['Hospital', 'Datetime'])
    g = df.groupby('Hospital', sort=False)[target_column]
    lags = [1, 2, 3, 6, 9, 12, 21, 24]
    feats = []

    for l in lags:
        col = f'Lag_{target_column}_{l}'
        df[col] = g.shift(l)
        feats.append(col)

    for w in [3, 6, 12, 21]:
        col = f'Rolling_Mean_{w}_{target_column}'
        df[col] = g.rolling(window=w, min_periods=1).mean().shift(1).reset_index(level=0, drop=True)
        feats.append(col)

    df[feats] = df[feats].bfill().fillna(0)
    return df, feats

def compare_daily_models(df_prepared, hospital, daily_metric, virus_features, forecast_days=7):
    print(f"\n--- Training Daily ARIMAX Model for {hospital} - {daily_metric} ---")
    df = df_prepared[df_prepared['Hospital'] == hospital].copy()
    if df.empty: return None

    cols = ['y', 'IsHoliday', 'IsMonday', 'IsTuesday', 'IsWednesday', 'IsThursday', 'IsFriday', 'IsSaturday', 'IsSunday', 'IsSummer', 'IsWinter', 'Capacity']
    agg = {c: ('first' if c != 'y' else daily_metric) for c in cols if c != 'y'}
    agg['y'] = (daily_metric, 'first')

    extra = [c for c in df.columns if c.startswith('Daily_') or c in virus_features]
    for c in extra: agg[c] = ('mean' if 'Daily_' in c else 'first')

    # Use explicit named aggregation to avoid pandas error
    # "Must provide 'func' or tuples of '(column, aggfunc)."
    # We can just iterate the dictionary.
    df_daily = df.groupby('DateOnly').agg(**agg).reset_index().rename(columns={'DateOnly': 'ds'}).sort_values('ds')

    if df_daily['y'].notna().sum() < 30:
        print(f"   ⚠️ Not enough data ({df_daily['y'].notna().sum()}) for ARIMAX.")
        return None

    exog_cols = [c for c in df_daily.columns if c not in ['ds', 'y']]
    exog_num = df_daily[exog_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    exog_cols = [c for c in exog_cols if exog_num[c].nunique() > 1]

    last_date = df_daily['ds'].max()
    hols = get_holidays(last_date, last_date + timedelta(days=forecast_days))
    future_rows = []

    monthly_avgs = df_daily.groupby(df_daily['ds'].dt.month)[exog_cols].mean().to_dict('index')

    for d in range(1, forecast_days + 1):
        fd = last_date + timedelta(days=d)
        row = {'ds': fd, 'IsHoliday': int(fd.normalize() in hols),
               'Capacity': df_daily['Capacity'].iloc[-1]}
        dow = fd.dayofweek
        for i, day in enumerate(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']):
            row[f'Is{day}'] = int(dow == i)
        row['IsSummer'] = int(fd.month in [6, 7, 8])
        row['IsWinter'] = int(fd.month in [12, 1, 2])

        m_avg = monthly_avgs.get(fd.month, {})
        for c in exog_cols:
            if c not in row:
                row[c] = m_avg.get(c, df_daily[c].mean())
        future_rows.append(row)

    future_df = pd.DataFrame(future_rows)
    future_exog = future_df[exog_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)

    try:
        y = pd.to_numeric(df_daily['y'], errors='coerce').fillna(method='ffill')
        X = df_daily[exog_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
        y.index = X.index = pd.to_datetime(df_daily['ds'])

        model = ARIMA(y, exog=X, order=(6, 1, 1), seasonal_order=(1, 0, 1, 7)).fit()
        preds = model.forecast(steps=forecast_days, exog=future_exog)

        return pd.DataFrame({'DateOnly': future_df['ds'], daily_metric: preds.values})
    except Exception as e:
        print(f"   ❌ ARIMAX failed: {e}")
        return None

def get_hourly_features(metric, daily_metric_name, other_metric, other_daily_metric_name, df_columns, virus_features):
    base = ['DayOfWeek', 'Month', 'Quarter', 'WeekOfYear', 'DayOfMonth', 'IsWeekend', 'IsHoliday', 'Capacity',
            daily_metric_name, other_metric, other_daily_metric_name, 'temp', 'prcp', 'snow', 'wspd', 'pres', 'rhum',
            'prcp_avg_2hr', 'snow_sum_12hr', 'wspd_max_3hr', 'temp_min_2hr', 'is_rainy', 'is_snowy', 'is_hot', 'is_cold',
            'pressure_drop', 'rainy_and_cold', 'Hospital_Code', 'Hour_sin', 'Hour_cos', 'IsPeakHour', 'IsLowHour']
    base += virus_features
    lags = [c for c in df_columns if c.startswith(f'Lag_{metric}') or c.startswith(f'Rolling_Mean')]
    lags_other = [c for c in df_columns if c.startswith(f'Lag_{other_metric}')]
    all_f = base + lags + lags_other
    if 'pred_2pm_anchor' in df_columns: all_f.append('pred_2pm_anchor')
    return [f for f in all_f if f in df_columns]

def update_lags(df, target_col, new_val, date, hour, time_slots_per_day=3):
    mask = (df['ds'] == pd.to_datetime(f"{date} {hour}:00"))
    if not mask.any(): return df

    idx = df.index[mask][0]
    df.at[idx, target_col] = new_val

    lags = [1, 2, 3, 6, 9, 12, 21, 24]
    for l in lags:
        target_idx = idx + l
        if target_idx in df.index:
            col_name = f'Lag_{target_col}_{l}'
            if col_name in df.columns:
                df.at[target_idx, col_name] = new_val
    return df

def train_and_predict_hourly_gbm_combined(df, hospital, metric, daily_metric_name, daily_forecast_df,
                                          other_metric, other_daily_metric_name, other_daily_forecast_df,
                                          virus_features, forecast_days=7):
    print(f"\n--- Training Hourly Sklearn GBM Models for {hospital} - {metric} ---")

    df_train = df.dropna(subset=[metric]).copy()
    time_slots = {'08:00': 8, '14:00': 14, '20:00': 20}
    last_date = df['ds'].max()

    # 1. Prepare Future DataFrame Skeleton
    future_rows = []
    hols = get_holidays(last_date, last_date + timedelta(days=forecast_days))

    base_feats = get_hourly_features(metric, daily_metric_name, other_metric, other_daily_metric_name, df.columns, virus_features)

    monthly_stats = {}
    for m in range(1, 13):
        d = df_train[df_train['Month'] == m]
        if not d.empty:
            monthly_stats[m] = {c: d[c].mean() for c in base_feats if pd.api.types.is_numeric_dtype(d[c])}

    for d in range(1, forecast_days + 1):
        f_date = last_date.date() + timedelta(days=d)
        d_val = daily_forecast_df.loc[daily_forecast_df['DateOnly'].dt.date == f_date, daily_metric_name].values[0] if not daily_forecast_df.empty else 0
        od_val = other_daily_forecast_df.loc[other_daily_forecast_df['DateOnly'].dt.date == f_date, other_daily_metric_name].values[0] if not other_daily_forecast_df.empty else 0

        for t_str, h in time_slots.items():
            dt = pd.to_datetime(f"{f_date} {t_str}")
            row = {
                'ds': dt, 'Hour': h, 'DayOfWeek': dt.dayofweek, 'Month': dt.month, 'Quarter': dt.quarter,
                'WeekOfYear': dt.isocalendar().week, 'DayOfMonth': dt.day, 'DayOfYear': dt.dayofyear,
                'IsWeekend': int(dt.dayofweek in [5, 6]),
                'IsMonday': int(dt.dayofweek==0), 'IsTuesday': int(dt.dayofweek==1), 'IsWednesday': int(dt.dayofweek==2),
                'IsThursday': int(dt.dayofweek==3), 'IsFriday': int(dt.dayofweek==4), 'IsSaturday': int(dt.dayofweek==5),
                'IsSunday': int(dt.dayofweek==6), 'IsSummer': int(dt.month in [6, 7, 8]), 'IsWinter': int(dt.month in [12, 1, 2]),
                'IsPeakHour': int(h==20), 'IsLowHour': int(h==8), 'Hour_sin': np.sin(2*np.pi*h/24), 'Hour_cos': np.cos(2*np.pi*h/24),
                'IsHoliday': int(dt.normalize() in hols), 'Hospital_Code': df['Hospital_Code'].iloc[-1], 'Capacity': df['Capacity'].iloc[-1],
                daily_metric_name: d_val, other_daily_metric_name: od_val,
                'forecast_day_index': d, 'pred_2pm_anchor': 0
            }
            m_stats = monthly_stats.get(dt.month, {})
            for c in base_feats:
                if c not in row:
                    if c in m_stats: row[c] = m_stats[c]
                    elif c in virus_features: row[c] = df[c].iloc[-1]
                    else: row[c] = 0
            future_rows.append(row)

    future_df = pd.DataFrame(future_rows)
    future_df['prediction'] = np.nan

    cols = list(df_train.columns)
    for c in future_df.columns:
        if c not in cols: cols.append(c)

    df_combined = pd.concat([df_train, future_df]).reset_index(drop=True)
    df_combined = df_combined.sort_values('ds').reset_index(drop=True)

    models = {}
    horizons = {'Day1': [1], 'Day2_4': [2,3,4], 'Day5_7': [5,6,7]}

    print("   Training 2pm Anchor models...")
    for grp in horizons:
        d = df_train[df_train['Hour'] == 14].copy()
        if d.empty: continue
        X = d[base_feats].apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
        y = d[metric]
        m = GradientBoostingRegressor(loss='huber', n_estimators=800, learning_rate=0.05, max_depth=5, random_state=42)
        m.fit(X, y)
        models[(14, grp)] = m

    print("   Training 8am/8pm Dependent models...")
    df_anchor = df_train.copy()
    vals_2pm = df_train[df_train['Hour'] == 14][['DateOnly', metric]].rename(columns={metric: 'val_2pm'})
    df_anchor = pd.merge(df_anchor, vals_2pm, on='DateOnly', how='left')
    vals_2pm['val_2pm_lag1'] = vals_2pm['val_2pm'].shift(1)
    df_anchor = pd.merge(df_anchor, vals_2pm[['DateOnly', 'val_2pm_lag1']], on='DateOnly', how='left')

    df_anchor['pred_2pm_anchor'] = np.where(df_anchor['Hour'] == 8,
                                            df_anchor['val_2pm_lag1'],
                                            df_anchor['val_2pm'])
    df_anchor['pred_2pm_anchor'] = df_anchor['pred_2pm_anchor'].fillna(0)

    anchor_feats = base_feats + ['pred_2pm_anchor']

    for h in [8, 20]:
        for grp in horizons:
            d = df_anchor[df_anchor['Hour'] == h].copy()
            if d.empty: continue
            X = d[anchor_feats].apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
            y = d[metric]
            m = GradientBoostingRegressor(loss='huber', n_estimators=800, learning_rate=0.05, max_depth=5, random_state=42)
            m.fit(X, y)
            models[(h, grp)] = m

    print("   Generating in-sample predictions...")
    hist_preds = []
    d14 = df_train[df_train['Hour'] == 14].copy()
    if (14, 'Day1') in models:
        X14 = d14[base_feats].apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
        d14['gbm_pred'] = models[(14, 'Day1')].predict(X14)
        hist_preds.append(d14)

    for h in [8, 20]:
        dh = df_anchor[df_anchor['Hour'] == h].copy()
        if (h, 'Day1') in models:
            Xh = dh[anchor_feats].apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
            dh['gbm_pred'] = models[(h, 'Day1')].predict(Xh)
            dh = dh.drop(columns=['val_2pm', 'val_2pm_lag1'], errors='ignore')
            hist_preds.append(dh)

    historical_preds = pd.concat(hist_preds).sort_values('ds')

    print("   Running recursive forecast...")

    future_start_idx = len(df_train)

    for d in range(1, forecast_days + 1):
        grp = 'Day1' if d == 1 else ('Day2_4' if d <= 4 else 'Day5_7')
        f_date = last_date.date() + timedelta(days=d)

        # 1. Predict 14:00
        idx_14 = df_combined[(df_combined['ds'].dt.date == f_date) & (df_combined['Hour'] == 14)].index
        if len(idx_14) > 0:
            idx = idx_14[0]
            row = df_combined.iloc[[idx]]
            X = row[base_feats].apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
            pred = models[(14, grp)].predict(X)[0]
            df_combined = update_lags(df_combined, metric, pred, f_date, 14)
            df_combined.at[idx, 'prediction'] = pred
            val_14 = pred

        # 2. Predict 08:00
        idx_8 = df_combined[(df_combined['ds'].dt.date == f_date) & (df_combined['Hour'] == 8)].index
        if len(idx_8) > 0:
            idx = idx_8[0]
            if d == 1:
                anchor = df_train[df_train['Hour']==14][metric].iloc[-1]
            else:
                prev_date = f_date - timedelta(days=1)
                anchor = df_combined[(df_combined['ds'].dt.date == prev_date) & (df_combined['Hour'] == 14)]['prediction'].values[0]

            df_combined.at[idx, 'pred_2pm_anchor'] = anchor
            row = df_combined.iloc[[idx]]
            X = row[anchor_feats].apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
            pred = models[(8, grp)].predict(X)[0]
            df_combined = update_lags(df_combined, metric, pred, f_date, 8)
            df_combined.at[idx, 'prediction'] = pred

        # 3. Predict 20:00
        idx_20 = df_combined[(df_combined['ds'].dt.date == f_date) & (df_combined['Hour'] == 20)].index
        if len(idx_20) > 0:
            idx = idx_20[0]
            df_combined.at[idx, 'pred_2pm_anchor'] = val_14
            row = df_combined.iloc[[idx]]
            X = row[anchor_feats].apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
            pred = models[(20, grp)].predict(X)[0]
            df_combined = update_lags(df_combined, metric, pred, f_date, 20)
            df_combined.at[idx, 'prediction'] = pred

    future_preds = df_combined[df_combined['ds'] > last_date].copy()

    return historical_preds, future_preds

def prophet_stacking_pipeline(file_path, hospital, metric, forecast_days=7, weather_thresholds=None):
    print(f"--- Starting Full Forecasting Pipeline for {hospital} - {metric} ---")

    if weather_thresholds is None:
        weather_thresholds = {'rain_threshold': 0.1, 'snow_threshold': 0, 'hot_temp_threshold': 18,
                              'cold_temp_threshold': 10, 'freezing_temp_threshold': 0, 'windy_speed_threshold': 20,
                              'pressure_drop_threshold': -10, 'humid_rh_threshold': 85}

    try:
        df_virus = download_and_process_virus_data()

        if not os.path.exists(file_path):
             alt_path = os.path.join("/tmp/file_attachments", os.path.basename(file_path))
             if os.path.exists(alt_path): file_path = alt_path
             else:
                 import glob
                 files = glob.glob("*.xlsx")
                 if files: file_path = files[0]
                 else:
                     print(f"❌ File not found: {file_path}")
                     return

        print(f"Reading file: {file_path}")
        df_raw = pd.read_excel(file_path, engine='openpyxl')

        # AUTO-DETECT HOSPITAL NAME
        print("   Auto-detecting available hospitals...")
        mapping = {'Tracker8am': 'ED_8am', 'Tracker2pm': 'ED_2pm', 'Tracker8pm': 'ED_8pm'}
        temp_df = df_raw.rename(columns=mapping)
        if 'Hospital' in temp_df.columns:
            available_hospitals = temp_df['Hospital'].unique()
            if hospital not in available_hospitals:
                print(f"   ⚠️ Hospital '{hospital}' not in file. Switching to '{available_hospitals[0]}'.")
                hospital = available_hospitals[0]

        df, final_hospital_name = prepare_data_vectorized(df_raw, hospital, weather_thresholds)

        if df.empty: return

        df, virus_features = merge_virus_data(df, df_virus)
        current_metric = metric
        other_metric = "Trolleys" if metric == "ED Beds" else "ED Beds"

        df, _ = add_lag_features_fast(df, "ED Beds")
        df, _ = add_lag_features_fast(df, "Trolleys")

        print(f"\nStep 1/3: Forecasting daily metrics...")
        # Add fallback for ARIMAX if too little data
        if len(df) < 30:
            print("   ⚠️ Data insufficient for daily ARIMAX. Creating dummy daily forecast.")
            # Dummy daily forecast
            current_daily_forecast = pd.DataFrame({
                'DateOnly': pd.date_range(start=df['DateOnly'].max() + timedelta(days=1), periods=forecast_days),
                f"Daily_{current_metric.replace(' ', '_')}": df[f"Daily_{current_metric.replace(' ', '_')}"].mean()
            })
            other_daily_forecast = pd.DataFrame({
                'DateOnly': pd.date_range(start=df['DateOnly'].max() + timedelta(days=1), periods=forecast_days),
                f"Daily_{other_metric.replace(' ', '_')}": df[f"Daily_{other_metric.replace(' ', '_')}"].mean()
            })
        else:
            current_daily_forecast = compare_daily_models(df, final_hospital_name, f"Daily_{current_metric.replace(' ', '_')}", virus_features, forecast_days)
            other_daily_forecast = compare_daily_models(df, final_hospital_name, f"Daily_{other_metric.replace(' ', '_')}", virus_features, forecast_days)

        if current_daily_forecast is None:
            print("   ❌ Failed to generate daily forecast.")
            return

        df['ds'] = df['Datetime']
        print(f"\nStep 2/3: Training hourly GBM models...")

        hist_preds, fut_preds = train_and_predict_hourly_gbm_combined(
            df.copy(), final_hospital_name, current_metric, f"Daily_{current_metric.replace(' ', '_')}", current_daily_forecast,
            other_metric, f"Daily_{other_metric.replace(' ', '_')}", other_daily_forecast, virus_features, forecast_days
        )

        print(f"\nStep 3/3: Generating final Prophet forecast...")
        m = Prophet(seasonality_mode='multiplicative', yearly_seasonality=10, weekly_seasonality=10,
                    daily_seasonality=10, changepoint_prior_scale=0.08, holidays_prior_scale=12.0)

        m.add_regressor('gbm_pred')
        m.add_regressor('Hour')
        for f in virus_features:
            if f in hist_preds.columns: m.add_regressor(f)

        p_train = hist_preds[['ds', current_metric, 'gbm_pred', 'Hour'] + virus_features].rename(columns={current_metric: 'y'}).dropna()
        m.fit(p_train)

        # In-Sample Fit MAE
        in_sample = m.predict(p_train)
        y_true = p_train['y'].values
        y_pred = in_sample['yhat'].values
        mae = mean_absolute_error(y_true, y_pred)
        print(f"\n--- 📊 MODEL PERFORMANCE (In-Sample) ---")
        print(f"MAE: {mae:.2f}")

        fut_for_p = fut_preds.rename(columns={'prediction': 'gbm_pred'})
        combined = pd.concat([hist_preds[['ds', 'gbm_pred', 'Hour'] + virus_features],
                              fut_for_p[['ds', 'gbm_pred', 'Hour'] + virus_features]], ignore_index=True)

        forecast = m.predict(combined)

        print("\n--- ✅ FINAL FORECAST COMPLETE ---")
        output_file = f"forecast_{final_hospital_name.replace(' ', '_')}_{metric.replace(' ', '_')}.csv"
        cols = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
        forecast[cols].tail(forecast_days * 3).to_csv(output_file, index=False)
        print(f"Forecast saved to {output_file}")
        print(forecast[cols].tail(forecast_days * 3))

    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    prophet_stacking_pipeline(DEFAULT_HOSPITAL_DATA_PATH, "Cork University Hospital", "ED Beds")
