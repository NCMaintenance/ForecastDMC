import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday

# --- Define Irish Bank Holidays ---
class IrishBankHolidays(AbstractHolidayCalendar):
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=nearest_workday),
        Holiday("St. Brigid's Day", month=2, day=1, observance=nearest_workday),
        Holiday("St. Patrick's Day", month=3, day=17, observance=nearest_workday),
        Holiday("Easter Monday", month=4, day=1, offset=pd.DateOffset(weekday=0)),  # Will adjust manually later
        Holiday("May Day", month=5, day=1, offset=pd.DateOffset(weekday=0(1))),
        Holiday("June Bank Holiday", month=6, day=1, offset=pd.DateOffset(weekday=0(1))),
        Holiday("August Bank Holiday", month=8, day=1, offset=pd.DateOffset(weekday=0(1))),
        Holiday("October Bank Holiday", month=10, day=1, offset=pd.DateOffset(weekday=0(-1))),
        Holiday("Christmas Day", month=12, day=25),
        Holiday("St. Stephen's Day", month=12, day=26),
    ]

# --- Streamlit UI ---
st.title("Emergency Department Forecasting (Ireland)")
uploaded_file = st.file_uploader("Upload your ED Excel File", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Rename columns
    df = df.rename(columns={
        'Tracker8am': 'ED_8am',
        'Tracker2pm': 'ED_2pm',
        'Tracker8pm': 'ED_8pm',
        'TimeTotal_8am': 'Trolley_8am',
        'TimeTotal_2pm': 'Trolley_2pm',
        'TimeTotal_8pm': 'Trolley_8pm',
        'AdditionalCapacityOpen Morning': 'Additional_Capacity'
    })

    # Forward-fill Additional Capacity across time points of the same day
    df['Additional_Capacity'] = df.groupby(['Hospital', 'Date'])['Additional_Capacity'].transform('first')

    # Reshape to long format
    df_long = pd.melt(
        df,
        id_vars=['Hospital Group Name', 'Hospital', 'Date', 'DayGAR', 'Additional_Capacity'],
        value_vars=['ED_8am', 'ED_2pm', 'ED_8pm', 'Trolley_8am', 'Trolley_2pm', 'Trolley_8pm'],
        var_name='Metric_Time',
        value_name='Value'
    )

    df_long[['Metric', 'TimeLabel']] = df_long['Metric_Time'].str.extract(r'(\w+)_([\d]+[ap]m)')
    time_map = {'8am': '08:00', '2pm': '14:00', '8pm': '20:00'}
    df_long['TimeStr'] = df_long['TimeLabel'].map(time_map)
    df_long['Datetime'] = pd.to_datetime(df_long['Date'].astype(str) + ' ' + df_long['TimeStr'])

    # Feature engineering
    df_long['Hour'] = df_long['Datetime'].dt.hour
    df_long['DayOfWeek'] = df_long['Datetime'].dt.dayofweek
    df_long['IsWeekend'] = df_long['DayOfWeek'].isin([5, 6]).astype(int)
    df_long['Hour_sin'] = np.sin(2 * np.pi * df_long['Hour'] / 24)
    df_long['Hour_cos'] = np.cos(2 * np.pi * df_long['Hour'] / 24)
    df_long['Day_sin'] = np.sin(2 * np.pi * df_long['DayOfWeek'] / 7)
    df_long['Day_cos'] = np.cos(2 * np.pi * df_long['DayOfWeek'] / 7)

    # Add Irish holidays
    calendar = IrishBankHolidays()
    holidays = calendar.holidays(start=df_long['Datetime'].min(), end=df_long['Datetime'].max())
    df_long['IsHoliday'] = df_long['Datetime'].dt.normalize().isin(holidays).astype(int)

    # Encode hospital as category
    df_long['Hospital_Code'] = df_long['Hospital'].astype('category').cat.codes

    # Filter for ED only
    df_ed = df_long[df_long['Metric'] == 'ED']

    # Create lag features
    df_ed = df_ed.sort_values(by=['Hospital', 'Datetime'])
    df_ed['Lag_1'] = df_ed.groupby('Hospital')['Value'].shift(1)
    df_ed['Lag_2'] = df_ed.groupby('Hospital')['Value'].shift(2)
    df_ed['Lag_3'] = df_ed.groupby('Hospital')['Value'].shift(3)

    # Drop rows with NA lag values
    df_ed = df_ed.dropna()

    # Features and target
    features = [
        'Hour', 'DayOfWeek', 'IsWeekend', 'Hour_sin', 'Hour_cos',
        'Day_sin', 'Day_cos', 'IsHoliday', 'Hospital_Code',
        'Lag_1', 'Lag_2', 'Lag_3', 'Additional_Capacity'
    ]
    X = df_ed[features]
    y = df_ed['Value']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    # Train LightGBM
    model = lgb.LGBMRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Show results
    st.subheader("Model Performance")
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    st.write(f"RMSE: {rmse:.2f}")

    # Show predictions
    df_ed_test = df_ed.iloc[y_test.index]
    df_ed_test = df_ed_test.copy()
    df_ed_test['Predicted'] = y_pred
    st.subheader("Sample Predictions")
    st.dataframe(df_ed_test[['Datetime', 'Hospital', 'Value', 'Predicted']].head(20))
