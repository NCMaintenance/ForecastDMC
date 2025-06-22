import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday
from pandas.tseries.offsets import DateOffset
from dateutil.rrule import MO
import subprocess
import sys

# Install openpyxl, as it's needed by pandas to read .xlsx files
subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl"])

# --- Define Irish Bank Holidays (Copied from DMC.py) ---
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

# --- prepare_data function (Copied from DMC.py) ---
def prepare_data(df):
    df = df.rename(columns={
        'Tracker8am': 'ED_8am',
        'Tracker2pm': 'ED_2pm',
        'Tracker8pm': 'ED_8pm',
        'TimeTotal_8am': 'Trolley_8am',
        'TimeTotal_2pm': 'Trolley_2pm',
        'TimeTotal_8pm': 'Trolley_8pm',
        'AdditionalCapacityOpen Morning': 'Additional_Capacity'
    })
    df['Additional_Capacity'] = df.groupby(['Hospital', 'Date'])['Additional_Capacity'].transform('first')
    df['Additional_Capacity'] = df['Additional_Capacity'].fillna(0)
    common_id_vars = ['Hospital Group Name', 'Hospital', 'Date', 'DayGAR', 'Additional_Capacity']
    df_ed = pd.melt(
        df,
        id_vars=common_id_vars,
        value_vars=['ED_8am', 'ED_2pm', 'ED_8pm'],
        var_name='Metric_Time_ED',
        value_name='ED Beds'
    )
    df_trolley = pd.melt(
        df,
        id_vars=common_id_vars,
        value_vars=['Trolley_8am', 'Trolley_2pm', 'Trolley_8pm'],
        var_name='Metric_Time_Trolley',
        value_name='Trolleys'
    )
    df_ed['TimeLabel'] = df_ed['Metric_Time_ED'].str.extract(r'([\d]+[ap]m)')
    df_trolley['TimeLabel'] = df_trolley['Metric_Time_Trolley'].str.extract(r'([\d]+[ap]m)')
    time_map = {'8am': '08:00', '2pm': '14:00', '8pm': '20:00'}
    df_ed['Time'] = df_ed['TimeLabel'].map(time_map)
    df_trolley['Time'] = df_trolley['TimeLabel'].map(time_map)
    df_ed = df_ed.drop(columns=['Metric_Time_ED', 'TimeLabel'])
    df_trolley = df_trolley.drop(columns=['Metric_Time_Trolley', 'TimeLabel'])
    df_merged = pd.merge(
        df_ed,
        df_trolley,
        on=['Hospital Group Name', 'Hospital', 'Date', 'DayGAR', 'Additional_Capacity', 'Time'],
        how='inner'
    )
    df_merged['ED Beds'] = pd.to_numeric(df_merged['ED Beds'], errors='coerce')
    df_merged['Trolleys'] = pd.to_numeric(df_merged['Trolleys'], errors='coerce')
    df_merged = df_merged.dropna(subset=['ED Beds', 'Trolleys'])
    df_merged = df_merged.rename(columns={'Additional_Capacity': 'Capacity'})
    # Ensure 'Date' is datetime before trying to access .astype(str)
    if not pd.api.types.is_datetime64_any_dtype(df_merged['Date']):
        df_merged['Date'] = pd.to_datetime(df_merged['Date'], errors='coerce')

    df_merged['Datetime'] = pd.to_datetime(df_merged['Date'].dt.strftime('%Y-%m-%d') + ' ' + df_merged['Time'])
    df_merged = df_merged.sort_values(by=['Hospital', 'Datetime']).reset_index(drop=True)
    df_merged['Hour'] = df_merged['Datetime'].dt.hour
    df_merged['DayOfWeek'] = df_merged['Datetime'].dt.dayofweek
    df_merged['DayOfMonth'] = df_merged['Datetime'].dt.day
    df_merged['Month'] = df_merged['Datetime'].dt.month
    df_merged['Quarter'] = df_merged['Datetime'].dt.quarter
    df_merged['WeekOfYear'] = df_merged['Datetime'].dt.isocalendar().week.astype(int) # Ensure it's int
    df_merged['IsWeekend'] = df_merged['DayOfWeek'].isin([5, 6]).astype(int)
    df_merged['IsMonday'] = (df_merged['DayOfWeek'] == 0).astype(int)
    df_merged['IsFriday'] = (df_merged['DayOfWeek'] == 4).astype(int)
    df_merged['Hour_sin'] = np.sin(2 * np.pi * df_merged['Hour'] / 24)
    df_merged['Hour_cos'] = np.cos(2 * np.pi * df_merged['Hour'] / 24)
    df_merged['Day_sin'] = np.sin(2 * np.pi * df_merged['DayOfWeek'] / 7)
    df_merged['Day_cos'] = np.cos(2 * np.pi * df_merged['DayOfWeek'] / 7)
    df_merged['Month_sin'] = np.sin(2 * np.pi * df_merged['Month'] / 12)
    df_merged['Month_cos'] = np.cos(2 * np.pi * df_merged['Month'] / 12)
    calendar = IrishBankHolidays()
    try:
        holidays = calendar.holidays(start=df_merged['Datetime'].min(), end=df_merged['Datetime'].max() + timedelta(days=30))
        df_merged['IsHoliday'] = df_merged['Datetime'].dt.normalize().isin(holidays).astype(int)
    except Exception:
        df_merged['IsHoliday'] = 0
    df_merged['IsSummer'] = df_merged['Month'].isin([6, 7, 8]).astype(int)
    df_merged['IsWinter'] = df_merged['Month'].isin([12, 1, 2]).astype(int)
    df_merged['IsPeakHour'] = df_merged['Hour'].isin([20]).astype(int)
    df_merged['IsLowHour'] = df_merged['Hour'].isin([8]).astype(int)
    df_merged['Hospital_Code'] = df_merged['Hospital'].astype('category').cat.codes
    return df_merged

# --- add_lag_features_smart function (Copied from DMC.py, with st.warning removed) ---
def add_lag_features_smart(df, target_column, min_data_threshold=20):
    df = df.copy()
    max_safe_lag = min(7, len(df) // 4)
    if max_safe_lag < 1:
        print(f"Warning: Very limited data ({len(df)} records). Using minimal features for {target_column}.")
        return df, []
    lag_features = []
    for i in range(1, max_safe_lag + 1):
        lag_col = f'Lag_{target_column}_{i}'
        df[lag_col] = df[target_column].shift(i)
        lag_features.append(lag_col)
    if len(df) >= 6:
        df[f'Rolling_Mean_3_{target_column}'] = df[target_column].rolling(window=min(3, len(df)//2), min_periods=1).mean()
        lag_features.append(f'Rolling_Mean_3_{target_column}')
        if len(df) >= 14:
            df[f'Rolling_Mean_7_{target_column}'] = df[target_column].rolling(window=min(7, len(df)//2), min_periods=1).mean()
            lag_features.append(f'Rolling_Mean_7_{target_column}')
    for feature in lag_features:
        df[feature] = df[feature].ffill().bfill().fillna(0)
    return df, lag_features

# Task 2 & 3: Download and load the Excel file
file_url = "https://raw.githubusercontent.com/NCMaintenance/ForecastDMC/feat/dmc-mae-tuning-phase1/may.xlsx"
print(f"Downloading data from {file_url}...")
df_may = pd.read_excel(file_url)
print("Data downloaded and loaded into DataFrame.")
print(f"Initial shape of may.xlsx: {df_may.shape}")
print("First 5 rows of may.xlsx:")
print(df_may.head())

# Task 4 & 5: Preprocess the data
print("Preprocessing data using prepare_data function...")
df_processed = prepare_data(df_may.copy()) # Use a copy to avoid modifying original df_may
print("Data preprocessing complete.")
print(f"Shape after prepare_data: {df_processed.shape}")

# Task 6: Add lag features
print("Adding lag features...")
# We need to apply lag features per hospital, as they are group-specific
# Create a list to store dataframes with lag features for each hospital
all_hospital_data_with_lags = []
hospitals = df_processed['Hospital'].unique()

for hospital in hospitals:
    hospital_data = df_processed[df_processed['Hospital'] == hospital].copy()
    # Add lags for 'ED Beds'
    hospital_data, ed_lags = add_lag_features_smart(hospital_data, 'ED Beds')
    # Add lags for 'Trolleys'
    hospital_data, trolley_lags = add_lag_features_smart(hospital_data, 'Trolleys')
    # Add lags for 'Capacity'
    hospital_data, capacity_lags = add_lag_features_smart(hospital_data, 'Capacity')
    all_hospital_data_with_lags.append(hospital_data)

# Concatenate all hospital data back into a single DataFrame
df_final_processed = pd.concat(all_hospital_data_with_lags, ignore_index=True)
# Sort again as concatenation might mess up the order needed for time series analysis
df_final_processed = df_final_processed.sort_values(by=['Hospital', 'Datetime']).reset_index(drop=True)

print("Lag feature addition complete.")

# Task 7: Print shape and head of the final processed DataFrame
print(f"Shape of final processed DataFrame: {df_final_processed.shape}")
print("First 5 rows of final processed DataFrame:")
print(df_final_processed.head())

# Task 8: Save the processed DataFrame
output_path = 'processed_may_data.csv'
df_final_processed.to_csv(output_path, index=False)
print(f"Processed data saved to {output_path}")

print("Script finished successfully.")
