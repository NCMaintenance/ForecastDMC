import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.ensemble import GradientBoostingRegressor
from prophet import Prophet
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday
from pandas.tseries.offsets import DateOffset
from dateutil.rrule import MO
import plotly.graph_objects as go
import requests
from io import StringIO
import warnings
import traceback

# --- Library Checks ---
try:
    from meteostat import Point, Hourly
    METEOSTAT_AVAILABLE = True
except ImportError:
    METEOSTAT_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Page Config ---
st.set_page_config(page_title="ED Admissions Forecasting", layout="wide")

# --- Irish Bank Holidays ---
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

# --- External Data Fetching Functions ---

@st.cache_data(ttl=24*3600) # Cache for 24 hours
def download_and_process_virus_data():
    """Downloads and processes HPSC respiratory virus data."""
    DATASET_CSV_URL = "https://respiratoryvirus.hpsc.ie/datasets/dbd10a497fd84e44954b858249919997_0.csv"
    try:
        response = requests.get(DATASET_CSV_URL, timeout=10)
        response.raise_for_status()
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data)
        df = df[['floor', 'weekly_cases', 'disease']].copy()
        df['Date'] = pd.to_datetime(df['floor']).dt.tz_localize(None).dt.normalize()
        df_pivot = df.pivot_table(index='Date', columns='disease', values='weekly_cases', aggfunc='sum').fillna(0)
        df_pivot.columns = [f"{col.replace(' ', '_')}_weekly_cases" for col in df_pivot.columns]
        df_pivot = df_pivot.sort_index().reset_index()
        return df_pivot
    except Exception as e:
        st.warning(f"Could not download virus data: {e}. Proceeding without it.")
        return pd.DataFrame()

@st.cache_data(ttl=24*3600)
def get_weather_data(start_date, end_date, location_name="Cork"):
    """Fetches hourly weather data using Meteostat."""
    if not METEOSTAT_AVAILABLE:
        return None

    # Coordinates
    loc_map = {
        "Cork": Point(51.8979, -8.4706, 25),
        "Kerry": Point(52.2704, -9.7026, 34)
    }
    point = loc_map.get(location_name, loc_map["Cork"])
    
    try:
        data = Hourly(point, start_date, end_date).fetch()
        if data.empty:
            return None
        data = data.reset_index().rename(columns={'time': 'Datetime'})
        data['Datetime'] = pd.to_datetime(data['Datetime'])
        cols_to_keep = ['Datetime', 'temp', 'dwpt', 'rhum', 'prcp', 'snow', 'wspd', 'pres', 'coco']
        return data[[c for c in cols_to_keep if c in data.columns]]
    except Exception:
        return None

# --- Data Preparation ---

@st.cache_data
def prepare_data(df):
    """
    Prepares data, renames to 'ED Admissions', and adds Weather + Virus features.
    """
    # 1. Cleaning and Renaming
    df.columns = [col.strip() for col in df.columns]
    
    rename_mapping = {
        'Tracker8am': 'ED_8am', 'Tracker2pm': 'ED_2pm', 'Tracker8pm': 'ED_8pm',
        'TimeTotal_8am': 'Trolley_8am', 'TimeTotal_2pm': 'Trolley_2pm', 'TimeTotal_8pm': 'Trolley_8pm',
        'AdditionalCapacityOpen Morning': 'Capacity'
    }
    actual_rename = {k: v for k, v in rename_mapping.items() if k in df.columns}
    df = df.rename(columns=actual_rename)

    # 2. Melting
    common_id = ['Hospital Group Name', 'Hospital', 'Date', 'Capacity']
    if 'Capacity' not in df.columns: 
        df['Capacity'] = 0
        common_id = ['Hospital Group Name', 'Hospital', 'Date', 'Capacity']

    # Melt ED Admissions (formerly ED Beds)
    df_ed = pd.melt(df, id_vars=common_id, value_vars=['ED_8am', 'ED_2pm', 'ED_8pm'],
                    var_name='Metric_Time', value_name='ED Admissions')
    
    # Melt Trolleys
    df_trolley = pd.melt(df, id_vars=common_id, value_vars=['Trolley_8am', 'Trolley_2pm', 'Trolley_8pm'],
                         var_name='Metric_Time_T', value_name='Trolleys')

    # Time Mapping
    time_map = {'8am': '08:00', '2pm': '14:00', '8pm': '20:00'}
    
    # Extract time labels
    df_ed['TimeLabel'] = df_ed['Metric_Time'].str.extract(r'(_\d+[ap]m)')[0].str.replace('_', '')
    df_trolley['TimeLabel'] = df_trolley['Metric_Time_T'].str.extract(r'(_\d+[ap]m)')[0].str.replace('_', '')
    
    df_ed['Time'] = df_ed['TimeLabel'].map(time_map)
    df_trolley['Time'] = df_trolley['TimeLabel'].map(time_map)

    # Merge
    df_merged = pd.merge(
        df_ed.drop(columns=['Metric_Time', 'TimeLabel']),
        df_trolley.drop(columns=['Metric_Time_T', 'TimeLabel']),
        on=['Hospital Group Name', 'Hospital', 'Date', 'Capacity', 'Time'],
        how='inner'
    )

    df_merged['ED Admissions'] = pd.to_numeric(df_merged['ED Admissions'], errors='coerce')
    df_merged['Trolleys'] = pd.to_numeric(df_merged['Trolleys'], errors='coerce')
    df_merged = df_merged.dropna(subset=['ED Admissions', 'Trolleys'])
    
    # Datetime creation
    df_merged['Datetime'] = pd.to_datetime(df_merged['Date'].astype(str) + ' ' + df_merged['Time'])
    df_merged = df_merged.sort_values(['Hospital', 'Datetime']).reset_index(drop=True)

    # 3. Basic Feature Engineering
    df_merged['Hour'] = df_merged['Datetime'].dt.hour
    df_merged['DayOfWeek'] = df_merged['Datetime'].dt.dayofweek
    df_merged['Month'] = df_merged['Datetime'].dt.month
    df_merged['WeekOfYear'] = df_merged['Datetime'].dt.isocalendar().week.astype(int)
    df_merged['IsWeekend'] = df_merged['DayOfWeek'].isin([5, 6]).astype(int)
    
    calendar = IrishBankHolidays()
    holidays = calendar.holidays(start=df_merged['Datetime'].min(), end=df_merged['Datetime'].max() + timedelta(days=60))
    df_merged['IsHoliday'] = df_merged['Datetime'].dt.normalize().isin(holidays).astype(int)
    
    df_merged['Hospital_Code'] = df_merged['Hospital'].astype('category').cat.codes
    
    # Day flags for ARIMAX
    df_merged['IsMonday'] = (df_merged['DayOfWeek'] == 0).astype(int)
    df_merged['IsTuesday'] = (df_merged['DayOfWeek'] == 1).astype(int)
    df_merged['IsWednesday'] = (df_merged['DayOfWeek'] == 2).astype(int)
    df_merged['IsThursday'] = (df_merged['DayOfWeek'] == 3).astype(int)
    df_merged['IsFriday'] = (df_merged['DayOfWeek'] == 4).astype(int)
    df_merged['IsSaturday'] = (df_merged['DayOfWeek'] == 5).astype(int)
    df_merged['IsSunday'] = (df_merged['DayOfWeek'] == 6).astype(int)
    df_merged['IsSummer'] = df_merged['Month'].isin([6, 7, 8]).astype(int)
    df_merged['IsWinter'] = df_merged['Month'].isin([12, 1, 2]).astype(int)


    # 4. MERGE VIRUS DATA
    df_virus = download_and_process_virus_data()
    virus_cols = []
    if not df_virus.empty:
        df_merged['DateOnly'] = df_merged['Datetime'].dt.normalize()
        df_merged = pd.merge_asof(
            df_merged.sort_values('DateOnly'),
            df_virus.sort_values('Date'),
            left_on='DateOnly',
            right_on='Date',
            direction='backward'
        )
        virus_cols = [c for c in df_virus.columns if c != 'Date']
        df_merged[virus_cols] = df_merged[virus_cols].fillna(0)
        df_merged = df_merged.drop(columns=['DateOnly', 'Date_y'], errors='ignore')
        if 'Date_x' in df_merged.columns:
            df_merged = df_merged.rename(columns={'Date_x': 'Date'})

    # 5. MERGE WEATHER DATA
    min_date = df_merged['Datetime'].min()
    max_date = df_merged['Datetime'].max()
    # Simple mapping for demo; ideally map every hospital to closest station
    weather_df = get_weather_data(min_date, max_date, "Cork") 
    
    weather_features = []
    if weather_df is not None:
        # Merge on exact hour
        df_merged = pd.merge(df_merged, weather_df, on='Datetime', how='left')
        
        # Fill missing weather (forward fill then 0)
        w_cols = ['temp', 'dwpt', 'rhum', 'prcp', 'snow', 'wspd', 'pres', 'coco']
        df_merged[w_cols] = df_merged[w_cols].ffill().fillna(0)
        
        # Engineering Weather Features (from your first script)
        df_merged['is_rainy'] = (df_merged['prcp'] > 0.1).astype(int)
        df_merged['is_cold'] = (df_merged['temp'] < 5).astype(int)
        df_merged['is_hot'] = (df_merged['temp'] > 18).astype(int)
        df_merged['is_windy'] = (df_merged['wspd'] > 20).astype(int)
        
        # Additional Daily Aggregations needed for Stacked Model
        df_merged['DateOnly'] = df_merged['Datetime'].dt.normalize()
        
        # Daily Aggregations
        daily_weather_agg = df_merged.groupby('DateOnly').agg({
            'temp': 'mean', 'prcp': 'sum', 'snow': 'sum', 'wspd': 'mean', 
            'pres': 'mean', 'rhum': 'mean'
        }).rename(columns=lambda x: f'Daily_{x}').reset_index()
        
        df_merged = pd.merge(df_merged, daily_weather_agg, on='DateOnly', how='left')

        weather_features = w_cols + ['is_rainy', 'is_cold', 'is_hot', 'is_windy'] + [f'Daily_{c}' for c in ['temp', 'prcp', 'snow', 'wspd', 'pres', 'rhum']]
    else:
        # Create dummy columns if weather fails so models don't break
        dummy_weather = ['temp', 'prcp', 'is_rainy', 'is_cold', 'is_hot', 'is_windy', 'Daily_temp', 'Daily_prcp']
        for c in dummy_weather:
            df_merged[c] = 0
            weather_features.append(c)

    return df_merged, virus_cols, weather_features

@st.cache_data
def add_lag_features_smart(df, target_column):
    """Adds lag and rolling features."""
    df_copy = df.copy()
    lag_features = []
    df_copy = df_copy.sort_values(by=['Hospital', 'Datetime'])

    for hospital_name, hospital_group in df_copy.groupby('Hospital'):
        max_safe_lag = min(7, len(hospital_group) // 4)
        if max_safe_lag < 1: continue

        # Lags 1 to 7
        for i in range(1, max_safe_lag + 1):
            lag_col = f'Lag_{target_column}_{i}'
            df_copy.loc[hospital_group.index, lag_col] = hospital_group[target_column].shift(i)
            if lag_col not in lag_features: lag_features.append(lag_col)
        
        # Rolling Means
        if len(hospital_group) >= 6:
            roll_3 = f'Rolling_Mean_3_{target_column}'
            df_copy.loc[hospital_group.index, roll_3] = hospital_group[target_column].rolling(3, min_periods=1).mean().shift(1)
            if roll_3 not in lag_features: lag_features.append(roll_3)
            
        if len(hospital_group) >= 14:
            roll_7 = f'Rolling_Mean_7_{target_column}'
            df_copy.loc[hospital_group.index, roll_7] = hospital_group[target_column].rolling(7, min_periods=1).mean().shift(1)
            if roll_7 not in lag_features: lag_features.append(roll_7)

    df_copy[lag_features] = df_copy[lag_features].fillna(0)
    return df_copy, lag_features

def create_future_dates(last_date, hospital_data, days=7, virus_cols=[], weather_cols=[]):
    """
    Creates future dataframe. 
    Crucially: It estimates future weather/virus data using monthly averages/carry-forward.
    """
    future_dates = []
    times = ['08:00', '14:00', '20:00']
    
    hospital = hospital_data['Hospital'].iloc[0]
    h_code = hospital_data['Hospital_Code'].iloc[0]
    cap_val = hospital_data['Capacity'].iloc[-1]
    
    # Calculate monthly averages for weather/virus from historical data to use as proxies
    monthly_stats = {}
    numeric_cols = weather_cols + virus_cols
    if numeric_cols:
        monthly_stats = hospital_data.groupby('Month')[numeric_cols].mean().to_dict('index')
    
    # Get last known values for carry-forward fallback
    last_known = hospital_data.iloc[-1]

    calendar = IrishBankHolidays()
    future_holidays = calendar.holidays(start=last_date, end=last_date + timedelta(days=days+10))

    for day in range(1, days + 1):
        future_date = last_date + timedelta(days=day)
        for time_str in times:
            dt = pd.to_datetime(f"{future_date.date()} {time_str}")
            
            # Base Time Features
            month = dt.month
            row = {
                'Datetime': dt,
                'Date': dt.date(),
                'Time': time_str,
                'Hospital': hospital,
                'Hospital_Code': h_code,
                'Capacity': cap_val,
                'Hour': dt.hour,
                'DayOfWeek': dt.dayofweek,
                'Month': month,
                'WeekOfYear': dt.isocalendar().week,
                'IsWeekend': int(dt.dayofweek in [5, 6]),
                'IsHoliday': int(dt.normalize() in future_holidays),
                'IsMonday': int(dt.dayofweek == 0),
                'IsTuesday': int(dt.dayofweek == 1),
                'IsWednesday': int(dt.dayofweek == 2),
                'IsThursday': int(dt.dayofweek == 3),
                'IsFriday': int(dt.dayofweek == 4),
                'IsSaturday': int(dt.dayofweek == 5),
                'IsSunday': int(dt.dayofweek == 6),
                'IsSummer': int(month in [6,7,8]),
                'IsWinter': int(month in [12,1,2]),
                'IsPeakHour': int(dt.hour == 20),
                'IsLowHour': int(dt.hour == 8)
            }
            
            # Fill Weather/Virus features
            # Strategy: Try monthly average, fallback to last known
            stats = monthly_stats.get(month, {})
            for col in numeric_cols:
                if col in stats:
                    row[col] = stats[col]
                elif col in last_known:
                    row[col] = last_known[col]
                else:
                    row[col] = 0
            
            future_dates.append(row)
            
    return pd.DataFrame(future_dates)

# --- Stacked Ensemble Helper Functions (from improved script) ---

def compare_daily_models(df, daily_metric, virus_features, forecast_days=7):
    """Trains ARIMAX for daily totals."""
    if not STATSMODELS_AVAILABLE: return None
    
    # Create 'y' column for aggregation - simplifies dictionary syntax
    df['y'] = df[daily_metric]
    
    # Aggregation logic - using standard pandas syntax {col: func}
    daily_agg_dict = {
        'y': 'sum', 
        'IsHoliday': 'first',
        'IsMonday': 'first', 'IsTuesday': 'first',
        'IsWednesday': 'first', 'IsThursday': 'first',
        'IsFriday': 'first', 'IsSaturday': 'first',
        'IsSunday': 'first', 'IsSummer': 'first',
        'IsWinter': 'first', 'Capacity': 'first'
    }
    
    # Add Weather/Virus means
    extra_cols = virus_features + [c for c in df.columns if 'Daily_' in c or c in ['temp', 'prcp']]
    for c in extra_cols:
        if c in df.columns: daily_agg_dict[c] = 'mean'
    
    # Map raw metric name to 'y'
    df['DateOnly'] = df['Datetime'].dt.normalize()
    
    # Perform Aggregation
    df_daily = df.groupby('DateOnly').agg(daily_agg_dict).reset_index().rename(columns={'DateOnly': 'ds'}).sort_values('ds')
    
    if len(df_daily) < 30: return None
    
    # Exogenous variables
    potential_exog = ['IsHoliday', 'IsMonday', 'IsTuesday', 'IsWednesday', 'IsThursday', 'IsFriday', 'IsSaturday', 'IsSummer', 'IsWinter', 'Capacity'] + extra_cols
    valid_exog = [c for c in potential_exog if c in df_daily.columns]
    
    # Train ARIMAX
    try:
        exog_data = df_daily[valid_exog].fillna(0).astype(float)
        model = ARIMA(df_daily['y'].astype(float), exog=exog_data, order=(7,1,1)).fit()
        
        # Forecast
        last_date = df_daily['ds'].max()
        future_exog = []
        for i in range(1, forecast_days + 1):
            fut_date = last_date + timedelta(days=i)
            row = {}
            for c in valid_exog:
                # Simple logic: carry forward or calc date based
                if c == 'IsMonday': row[c] = int(fut_date.dayofweek == 0)
                elif c == 'IsTuesday': row[c] = int(fut_date.dayofweek == 1)
                elif c == 'IsWednesday': row[c] = int(fut_date.dayofweek == 2)
                elif c == 'IsThursday': row[c] = int(fut_date.dayofweek == 3)
                elif c == 'IsFriday': row[c] = int(fut_date.dayofweek == 4)
                elif c == 'IsSaturday': row[c] = int(fut_date.dayofweek == 5)
                elif c == 'IsSunday': row[c] = int(fut_date.dayofweek == 6)
                elif c == 'IsHoliday': row[c] = 0 # Simplified for demo
                else: row[c] = df_daily[c].iloc[-1] # Carry forward others
            future_exog.append(row)
            
        future_exog_df = pd.DataFrame(future_exog)
        forecast_vals = model.forecast(steps=forecast_days, exog=future_exog_df)
        
        res_df = pd.DataFrame({'DateOnly': [last_date + timedelta(days=i) for i in range(1, forecast_days+1)], 
                               'Daily_Forecast': forecast_vals.values})
        return res_df
        
    except Exception as e:
        return None

def train_hourly_gbm(df, metric, daily_forecast, forecast_days, future_df_base):
    """Trains separate GBMs for 8am, 2pm, 8pm."""
    time_slots = {8: '08:00', 14: '14:00', 20: '20:00'}
    
    gbm_preds_hist = []
    gbm_preds_fut = []
    
    # Calculate daily total for historical data to use as feature
    df['DateOnly'] = df['Datetime'].dt.normalize()
    daily_sums = df.groupby('DateOnly')[metric].transform('sum')
    df['Daily_Total_Metric'] = daily_sums
    
    # Add Daily Forecast to future_df
    future_df_base['DateOnly'] = future_df_base['Datetime'].dt.normalize()
    future_df_base = pd.merge(future_df_base, daily_forecast, on='DateOnly', how='left')
    future_df_base['Daily_Total_Metric'] = future_df_base['Daily_Forecast'] # Rename for consistency

    features = ['DayOfWeek', 'Month', 'IsWeekend', 'IsHoliday', 'Capacity', 'Daily_Total_Metric']
    # Add extra numeric columns available in df
    extra_feats = [c for c in df.columns if c in ['temp', 'prcp', 'is_rainy', 'is_cold', 'Daily_temp']]
    features += extra_feats

    # Train model per hour
    for hour, time_str in time_slots.items():
        # Historical
        subset = df[df['Hour'] == hour].copy()
        if subset.empty: continue
        
        # Future
        fut_subset = future_df_base[future_df_base['Hour'] == hour].copy()
        
        # Train GBM
        X = subset[features].fillna(0)
        y = subset[metric]
        
        model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, random_state=42)
        model.fit(X, y)
        
        # Predict
        subset['gbm_pred'] = model.predict(X)
        gbm_preds_hist.append(subset[['Datetime', 'gbm_pred']])
        
        if not fut_subset.empty:
            fut_subset['gbm_pred'] = model.predict(fut_subset[features].fillna(0))
            gbm_preds_fut.append(fut_subset[['Datetime', 'gbm_pred']])
            
    # Combine
    hist_res = pd.concat(gbm_preds_hist).sort_values('Datetime')
    fut_res = pd.concat(gbm_preds_fut).sort_values('Datetime')
    
    return hist_res, fut_res

def validate_and_steer(forecast, actuals, target_col):
    """Steers forecast based on last 4 weeks bias."""
    last_date = actuals['Datetime'].max()
    validation_start = last_date - timedelta(days=28)
    
    # Merge forecast (historical portion) with actuals
    # Note: forecast here contains full history fitted + future
    merged = pd.merge(forecast, actuals[['Datetime', target_col]], left_on='ds', right_on='Datetime', how='inner')
    merged = merged[merged['ds'] > validation_start]
    
    if merged.empty: return 0.0
    
    merged['error'] = merged[target_col] - merged['yhat']
    
    # Weighted bias (recent weeks matter more)
    weeks = [1, 2, 3, 4]
    weighted_bias = 0
    total_weight = 0
    
    for i in weeks:
        w_end = last_date - timedelta(days=(i-1)*7)
        w_start = last_date - timedelta(days=i*7)
        week_data = merged[(merged['ds'] > w_start) & (merged['ds'] <= w_end)]
        
        if not week_data.empty:
            bias = week_data['error'].mean()
            weight = 5 - i
            weighted_bias += (bias * weight)
            total_weight += weight
            
    steering_factor = weighted_bias / total_weight if total_weight > 0 else 0
    return steering_factor

def predict_stacked_ensemble(historical, future, target_col, virus_cols):
    """
    Orchestrates: 
    1. ARIMAX Daily Forecast 
    2. Hourly GBM (using daily as feature)
    3. Prophet Stacking (using GBM as regressor)
    4. Steering
    """
    if not STATSMODELS_AVAILABLE:
        st.error("Statsmodels not available. Falling back to simple Prophet.")
        return predict_prophet(historical, future, target_col)
        
    # 1. Forecast Daily Total
    daily_forecast = compare_daily_models(historical.copy(), target_col, virus_cols)
    
    if daily_forecast is None:
        return predict_prophet(historical, future, target_col)
        
    # 2. Train Hourly GBMs
    hist_gbm, fut_gbm = train_hourly_gbm(historical.copy(), target_col, daily_forecast, 7, future.copy())
    
    # 3. Final Prophet Stack
    # Merge gbm_pred into historical/future
    hist_fit = pd.merge(historical, hist_gbm, on='Datetime', how='left')
    fut_fit = pd.merge(future, fut_gbm, on='Datetime', how='left')
    
    # Fill any missing GBM preds (e.g. if hour wasn't 8/14/20) with mean
    hist_fit['gbm_pred'] = hist_fit['gbm_pred'].fillna(hist_fit[target_col].mean())
    fut_fit['gbm_pred'] = fut_fit['gbm_pred'].fillna(hist_fit[target_col].mean())
    
    m = Prophet(daily_seasonality=True, weekly_seasonality=True)
    m.add_country_holidays(country_name='IE')
    m.add_regressor('gbm_pred')
    
    df_train = hist_fit[['Datetime', target_col, 'gbm_pred']].rename(columns={'Datetime': 'ds', target_col: 'y'})
    m.fit(df_train)
    
    df_fut = fut_fit[['Datetime', 'gbm_pred']].rename(columns={'Datetime': 'ds'})
    forecast = m.predict(df_fut)
    
    # 4. Steering
    # Get historical fit to calc steering
    hist_forecast = m.predict(df_train)
    steering_val = validate_and_steer(hist_forecast, historical, target_col)
    
    # Apply results
    res = future.copy()
    res['Predicted'] = np.maximum(0, forecast['yhat'] + steering_val).round(0)
    res['Predicted_Low'] = np.maximum(0, forecast['yhat_lower'] + steering_val).round(0)
    res['Predicted_High'] = np.maximum(0, forecast['yhat_upper'] + steering_val).round(0)
    
    return res

# --- Forecasting Functions (Standard) ---

def forecast_with_lags(model, historical, future, features, target_col):
    """Iterative forecasting for Tree models."""
    preds, lows, highs = [], [], []
    
    # Prepare historical lag basis
    last_vals = historical[target_col].tail(7).values.tolist()
    current_lags = list(reversed(last_vals)) + [0]*7
    
    hist_mean_3 = historical[target_col].tail(3).mean()
    hist_mean_7 = historical[target_col].tail(7).mean()
    
    # Calculate approximate residuals for intervals
    try:
        hist_preds = model.predict(historical[features])
        residuals = historical[target_col].values - hist_preds
        std = np.std(residuals)
        interval = 1.96 * std
    except:
        interval = 0

    # Ensure future has all features
    for f in features:
        if f not in future.columns and f not in [f'Lag_{target_col}_{i}' for i in range(1,8)]:
             future[f] = 0

    for idx, row in future.iterrows():
        # Build feature vector
        row_feats = []
        for f in features:
            if f.startswith(f'Lag_{target_col}_'):
                lag_n = int(f.split('_')[-1]) - 1
                val = current_lags[lag_n] if lag_n < len(current_lags) else 0
                row_feats.append(val)
            elif f == f'Rolling_Mean_3_{target_col}':
                row_feats.append(hist_mean_3)
            elif f == f'Rolling_Mean_7_{target_col}':
                row_feats.append(hist_mean_7)
            else:
                row_feats.append(row.get(f, 0))
        
        # Predict
        try:
            pred = model.predict(np.array(row_feats).reshape(1, -1))[0]
        except:
            pred = hist_mean_7 # Fallback
            
        pred = max(0, round(pred))
        preds.append(pred)
        lows.append(max(0, round(pred - interval)))
        highs.append(round(pred + interval))
        
        # Update Lags
        current_lags = [pred] + current_lags[:6]
        hist_mean_3 = (hist_mean_3 * 2 + pred) / 3
        hist_mean_7 = (hist_mean_7 * 6 + pred) / 7
        
    future['Predicted'] = preds
    future['Predicted_Low'] = lows
    future['Predicted_High'] = highs
    return future

def predict_prophet(historical, future, target_col, regressors=[]):
    """Prophet forecasting with extra regressors (Virus/Weather)."""
    df_train = historical[['Datetime', target_col] + regressors].rename(columns={'Datetime': 'ds', target_col: 'y'})
    
    m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    m.add_country_holidays(country_name='IE')
    
    # Add external regressors
    for reg in regressors:
        m.add_regressor(reg)
        
    m.fit(df_train)
    
    df_future = future[['Datetime'] + regressors].rename(columns={'Datetime': 'ds'})
    forecast = m.predict(df_future)
    
    res = future.copy()
    res['Predicted'] = np.maximum(0, forecast['yhat']).round(0)
    res['Predicted_Low'] = np.maximum(0, forecast['yhat_lower']).round(0)
    res['Predicted_High'] = np.maximum(0, forecast['yhat_upper']).round(0)
    return res

def predict_hybrid(historical, future, base_features, target_col, residual_model_name, ml_iter=500, regressors=[]):
    """Hybrid Prophet + ML model."""
    # 1. Prophet
    df_train = historical[['Datetime', target_col] + regressors].rename(columns={'Datetime': 'ds', target_col: 'y'})
    m = Prophet(daily_seasonality=True, weekly_seasonality=True)
    m.add_country_holidays(country_name='IE')
    for reg in regressors:
        m.add_regressor(reg)
    m.fit(df_train)
    
    # Get residuals
    prophet_preds = m.predict(df_train[['ds'] + regressors])
    historical['prophet_yhat'] = prophet_preds['yhat'].values
    historical['residuals'] = historical[target_col] - historical['prophet_yhat']
    
    # 2. ML Residuals
    # Add lag features for residuals
    hist_lags, resid_lag_feats = add_lag_features_smart(historical[['Hospital', 'Datetime', 'residuals']].copy(), 'residuals')
    historical = pd.merge(historical, hist_lags.drop(columns=['residuals']), on=['Hospital', 'Datetime'])
    
    ml_feats = [f for f in base_features if f in historical.columns and f != target_col] + resid_lag_feats
    
    train_clean = historical.dropna(subset=['residuals'] + ml_feats)
    X = train_clean[ml_feats]
    y = train_clean['residuals']
    
    if len(X) < 10:
        # Fallback to pure Prophet if not enough data for residual model
        df_fut = future[['Datetime'] + regressors].rename(columns={'Datetime': 'ds'})
        p_fut = m.predict(df_fut)
        future['Predicted'] = np.maximum(0, p_fut['yhat']).round(0)
        future['Predicted_Low'] = np.maximum(0, p_fut['yhat_lower']).round(0)
        future['Predicted_High'] = np.maximum(0, p_fut['yhat_upper']).round(0)
        return future

    if residual_model_name == 'LightGBM':
        model = lgb.LGBMRegressor(n_estimators=ml_iter, random_state=42)
    elif residual_model_name == 'CatBoost':
        model = cb.CatBoostRegressor(iterations=ml_iter, verbose=False, random_state=42)
    else:
        model = GradientBoostingRegressor(n_estimators=ml_iter, random_state=42)
        
    model.fit(X, y)
    
    # 3. Predict Future
    # Prepare future for residual prediction using simplified lag logic (carry forward mean)
    # A full iterative loop is omitted here for brevity/speed in Hybrid, trusting the Stacked model for max accuracy
    df_fut_prophet = future[['Datetime'] + regressors].rename(columns={'Datetime': 'ds'})
    prophet_fut = m.predict(df_fut_prophet)
    
    # For future residual features, we will just use 0 lags/mean rolling as approximation
    # To do this robustly requires the loop, but we have that in "Stacked Ensemble"
    
    # Approximate residual pred (single shot)
    future_for_res = future.copy()
    for f in ml_feats:
        if f not in future_for_res.columns: future_for_res[f] = 0
    
    res_pred = model.predict(future_for_res[ml_feats])
    
    final_pred = prophet_fut['yhat'].values + res_pred
    future['Predicted'] = np.maximum(0, final_pred).round(0)
    future['Predicted_Low'] = np.maximum(0, prophet_fut['yhat_lower']).round(0) # Use Prophet intervals
    future['Predicted_High'] = np.maximum(0, prophet_fut['yhat_upper']).round(0)
    
    return future

# --- Plotting ---

def plot_forecasts(historical, forecast, metric_name, hospital_name):
    fig = go.Figure()
    
    # Historical
    fig.add_trace(go.Scatter(
        x=historical['Datetime'], y=historical[metric_name],
        mode='lines+markers', name='Historical', line=dict(color='blue')
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast['Datetime'], y=forecast['Predicted'],
        mode='lines+markers', name='Forecast', line=dict(color='red', dash='dash')
    ))
    
    # Intervals
    if 'Predicted_Low' in forecast.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast['Datetime'], forecast['Datetime'].iloc[::-1]]),
            y=pd.concat([forecast['Predicted_High'], forecast['Predicted_Low'].iloc[::-1]]),
            fill='toself', fillcolor='rgba(255,0,0,0.1)', line=dict(color='rgba(0,0,0,0)'),
            name='Confidence Interval'
        ))
        
    fig.update_layout(title=f'{metric_name} Forecast - {hospital_name}', hovermode='x unified')
    return fig

# --- UI ---

st.title("🏥 ED Admissions & Trolley Forecasting")
st.markdown("Combines **Meteostat Weather**, **HPSC Virus Data**, and **Machine Learning** for 7-day forecasts.")

with st.sidebar:
    st.header("⚙️ Configuration")
    days_to_forecast = st.slider("Forecast Horizon (Days)", 1, 14, 7)
    model_choice = st.selectbox("Select Model", 
        ["Stacked Ensemble (Best Accuracy)", "Prophet", "LightGBM", "CatBoost", "XGBoost", "Hybrid (Prophet + LightGBM)"])
    
    st.divider()
    st.write("Feature Status:")
    st.caption("✅ HPSC Virus Data (Auto-downloaded)")
    st.caption(f"✅ Meteostat Weather ({'Available' if METEOSTAT_AVAILABLE else 'Not Found'})")
    st.caption(f"✅ Statsmodels ({'Available' if STATSMODELS_AVAILABLE else 'Not Found'})")

uploaded_file = st.file_uploader("Upload ED Excel File", type=['xlsx'])

if uploaded_file:
    with st.spinner("Processing data, downloading weather & virus stats..."):
        df_raw = pd.read_excel(uploaded_file)
        df_proc, virus_features, weather_features = prepare_data(df_raw)
        
    st.success(f"Loaded {len(df_proc)} records.")
    
    # Metric Selection
    metric_map = {"ED Admissions": "ED Admissions", "Trolleys": "Trolleys", "Capacity": "Capacity"}
    
    # Hospital Selection
    hospitals = df_proc['Hospital'].unique()
    selected_hospital = st.selectbox("Select Hospital", hospitals)
    
    if st.button("🚀 Run Forecast"):
        h_data = df_proc[df_proc['Hospital'] == selected_hospital].copy()
        last_date = h_data['Datetime'].max()
        
        # Combine external features
        external_regressors = virus_features + weather_features
        
        # Define Base Features for ML models
        base_features = [
            'Hour', 'DayOfWeek', 'Month', 'IsWeekend', 'IsHoliday', 'Capacity',
            'Hospital_Code'
        ] + external_regressors 
        
        # Create Future Frame
        future_df = create_future_dates(last_date, h_data, days_to_forecast, virus_features, weather_features)
        
        for metric_name, col_name in metric_map.items():
            st.subheader(f"📊 {metric_name}")
            
            # --- Forecasting Logic ---
            with st.spinner(f"Forecasting {metric_name}..."):
                
                if model_choice == "Stacked Ensemble (Best Accuracy)":
                    forecast = predict_stacked_ensemble(h_data, future_df, col_name, virus_features)

                elif model_choice == "Prophet":
                    forecast = predict_prophet(h_data, future_df, col_name, external_regressors)
                    
                elif "Hybrid" in model_choice:
                    forecast = predict_hybrid(h_data, future_df, base_features, col_name, "LightGBM", regressors=external_regressors)
                    
                else: # Tree Models (LightGBM, CatBoost, XGBoost)
                    # Add Lags
                    data_lagged, lag_names = add_lag_features_smart(h_data, col_name)
                    full_features = base_features + lag_names
                    
                    train = data_lagged.dropna(subset=[col_name] + full_features)
                    X = train[full_features]
                    y = train[col_name]
                    
                    if model_choice == "LightGBM":
                        model = lgb.LGBMRegressor(random_state=42)
                    elif model_choice == "CatBoost":
                        model = cb.CatBoostRegressor(verbose=False, random_state=42)
                    else:
                        model = xgb.XGBRegressor(random_state=42)
                        
                    model.fit(X, y)
                    forecast = forecast_with_lags(model, train, future_df, full_features, col_name)
                    forecast['Datetime'] = future_df['Datetime'] # Ensure datetime alignment
            
            # Plot
            fig = plot_forecasts(h_data.tail(14*3), forecast, metric_name, selected_hospital)
            st.plotly_chart(fig, use_container_width=True)
            
            # Data Table
            with st.expander("View Data"):
                st.dataframe(forecast[['Date', 'Time', 'Predicted', 'Predicted_Low', 'Predicted_High']])

else:
    st.info("Please upload your ED Data Excel file to begin.")
