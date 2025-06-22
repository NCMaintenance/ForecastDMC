import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday
from pandas.tseries.offsets import DateOffset
from dateutil.rrule import MO

print("Script started: Preparing dataset with all Iteration 3 features.")

# --- IrishBankHolidays Class ---
class IrishBankHolidays(AbstractHolidayCalendar):
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=nearest_workday),
        Holiday("St. Brigid's Day", month=2, day=1, observance=nearest_workday),
        Holiday("St. Patrick's Day", month=3, day=17, observance=nearest_workday),
        Holiday("Easter Monday", month=1, day=1, offset=[pd.DateOffset(weekday=MO(1), weeks=1), pd.offsets.Easter()]),
        Holiday("May Day", month=5, day=1, offset=DateOffset(weekday=MO(1))),
        Holiday("June Bank Holiday", month=6, day=1, offset=DateOffset(weekday=MO(1))),
        Holiday("August Bank Holiday", month=8, day=1, offset=DateOffset(weekday=MO(1))),
        Holiday("October Bank Holiday", month=10, day=1, offset=DateOffset(weekday=MO(-1))),
        Holiday("Christmas Day", month=12, day=25),
        Holiday("St. Stephen's Day", month=12, day=26),
    ]

try:
    df = pd.read_csv('processed_may_data.csv') # This is the original from Subtask 1
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    print(f"Base data loaded. Shape: {df.shape}")
except FileNotFoundError:
    print("Error: processed_may_data.csv not found. Exiting.")
    exit()

# Apply Iteration 2 features
print("Applying Iteration 2 features...")
df['Hour_DayOfWeek'] = df['Hour'].astype(str) + "_" + df['DayOfWeek'].astype(str)
df['Hour_DayOfWeek'] = df['Hour_DayOfWeek'].astype('category').cat.codes
df['Hospital_Code_Hour'] = df['Hospital_Code'].astype(str) + "_" + df['Hour'].astype(str)
df['Hospital_Code_Hour'] = df['Hospital_Code_Hour'].astype('category').cat.codes
if 'Capacity' in df.columns and 'Hour' in df.columns:
    df['Capacity_Hour_Interaction'] = df['Capacity'] * df['Hour']
actual_holidays = []
if not df.empty:
    calendar = IrishBankHolidays()
    min_date = df['Datetime'].min(); max_date = df['Datetime'].max()
    # Ensure holidays are generated if calendar is used.
    try:
        actual_holidays = calendar.holidays(start=min_date - timedelta(days=60), end=max_date + timedelta(days=60)).normalize()
        actual_holidays = sorted(list(set(actual_holidays)))
    except Exception as e_holiday:
        print(f"Warning: Holiday generation failed: {e_holiday}. Proximity features might be affected.")
        actual_holidays = [] # Ensure it's an empty list
if not actual_holidays: # Check if list is empty
    df['DaysToNextHoliday'] = 365; df['DaysFromLastHoliday'] = 365
else:
    df['DaysToNextHoliday'] = df['Datetime'].apply(lambda x: min([ (h - x.normalize()).days for h in actual_holidays if h >= x.normalize() ], default=365) )
    df['DaysFromLastHoliday'] = df['Datetime'].apply(lambda x: min([ (x.normalize() - h).days for h in actual_holidays if h <= x.normalize() ], default=365) )
df['DaysToNextHoliday'] = df['DaysToNextHoliday'].fillna(365) # Fill NaNs that can occur if no holiday is found
df['DaysFromLastHoliday'] = df['DaysFromLastHoliday'].fillna(365)
print(f"Shape after Iter2 features: {df.shape}")

# Apply Iteration 3 features (ED Beds specific, calculated per hospital)
print("Applying Iteration 3 (ED Beds specific) features...")
df_list_iter3_features = []
for hospital_code_val in df['Hospital_Code'].unique():
    hospital_data = df[df['Hospital_Code'] == hospital_code_val].copy()
    hospital_data = hospital_data.sort_values(by='Datetime')
    if 'Lag_ED Beds_1' in hospital_data.columns and 'Lag_ED Beds_2' in hospital_data.columns:
        hospital_data['Lag1_Diff_ED_Beds'] = hospital_data['Lag_ED Beds_1'] - hospital_data['Lag_ED Beds_2']
    else: hospital_data['Lag1_Diff_ED_Beds'] = 0
    if 'Lag_ED Beds_1' in hospital_data.columns and 'Lag_ED Beds_7' in hospital_data.columns:
        hospital_data['Lag_1_minus_7_ED_Beds'] = hospital_data['Lag_ED Beds_1'] - hospital_data['Lag_ED Beds_7']
    else: hospital_data['Lag_1_minus_7_ED_Beds'] = 0
    hospital_data['Rolling_Std_7_ED_Beds'] = hospital_data['ED Beds'].rolling(window=7, min_periods=1).std()
    # Fill NaNs for these new features
    for col in ['Lag1_Diff_ED_Beds', 'Lag_1_minus_7_ED_Beds', 'Rolling_Std_7_ED_Beds']:
        hospital_data[col] = hospital_data[col].fillna(method='bfill').fillna(method='ffill').fillna(0)
    df_list_iter3_features.append(hospital_data)

if df_list_iter3_features:
    df_processed_iter3 = pd.concat(df_list_iter3_features).reset_index(drop=True)
    print(f"Shape after Iter3 features: {df_processed_iter3.shape}")
    df_processed_iter3.to_csv('processed_data_iter3_features.csv', index=False)
    print("Saved processed_data_iter3_features.csv")
else:
    print("Error: No data processed after hospital grouping for Iter3 features. Saving original df instead.")
    df.to_csv('processed_data_iter3_features.csv', index=False) # Save something to avoid breaking next step

print("Script finished: Dataset preparation complete.")
