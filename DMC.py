import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday
from pandas.tseries.offsets import DateOffset
from dateutil.rrule import MO
import shap
import matplotlib.pyplot as plt

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

# --- Streamlit UI ---
st.title("🇮🇪 Emergency Department Forecasting (Ireland)")

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

    # Fill Additional Capacity across the day
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

    # Encode hospital
    df_long['Hospital_Code'] = df_long['Hospital'].astype('category').cat.codes

    # -------------------------
    # ED Forecasting
    # -------------------------
    df_ed = df_long[df_long['Metric'] == 'ED'].copy()
    df_ed = df_ed.sort_values(by=['Hospital', 'Datetime'])
    for i in range(1, 4):
        df_ed[f'Lag_{i}'] = df_ed.groupby('Hospital')['Value'].shift(i)

    df_ed.dropna(inplace=True)
    features = [
        'Hour', 'DayOfWeek', 'IsWeekend', 'Hour_sin', 'Hour_cos',
        'Day_sin', 'Day_cos', 'IsHoliday', 'Hospital_Code',
        'Lag_1', 'Lag_2', 'Lag_3', 'Additional_Capacity'
    ]
    X = df_ed[features]
    y = df_ed['Value']
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    model = lgb.LGBMRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    st.subheader("📈 ED Model Performance")
    st.write(f"RMSE: {rmse:.2f}")

    df_ed_test = df_ed.iloc[y_test.index].copy()
    df_ed_test['Predicted'] = y_pred
    st.dataframe(df_ed_test[['Datetime', 'Hospital', 'Value', 'Predicted']].head(20))

    st.download_button(
        "📥 Download ED Predictions as CSV",
        df_ed_test[['Datetime', 'Hospital', 'Value', 'Predicted']].to_csv(index=False).encode(),
        file_name="ed_predictions.csv",
        mime="text/csv"
    )

    # SHAP Plot
    st.subheader("🧠 SHAP Feature Importance (ED)")
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.bar(shap_values, max_display=10, show=False)
    st.pyplot(fig)

    # -------------------------
    # Trolley Forecasting
    # -------------------------
    st.subheader("🛒 Trolley Model Performance")
    df_trolley = df_long[df_long['Metric'] == 'Trolley'].copy()
    df_trolley = df_trolley.sort_values(by=['Hospital', 'Datetime'])
    for i in range(1, 4):
        df_trolley[f'Lag_{i}'] = df_trolley.groupby('Hospital')['Value'].shift(i)
    df_trolley.dropna(inplace=True)

    X_t = df_trolley[features]
    y_t = df_trolley['Value']
    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_t, y_t, shuffle=False, test_size=0.2)

    model_t = lgb.LGBMRegressor()
    model_t.fit(X_train_t, y_train_t)
    y_pred_t = model_t.predict(X_test_t)
    rmse_t = np.sqrt(mean_squared_error(y_test_t, y_pred_t))
    st.write(f"Trolley RMSE: {rmse_t:.2f}")

    df_t_test = df_trolley.iloc[y_test_t.index].copy()
    df_t_test['Predicted'] = y_pred_t
    st.dataframe(df_t_test[['Datetime', 'Hospital', 'Value', 'Predicted']].head(20))

    st.download_button(
        "📥 Download Trolley Predictions as CSV",
        df_t_test[['Datetime', 'Hospital', 'Value', 'Predicted']].to_csv(index=False).encode(),
        file_name="trolley_predictions.csv",
        mime="text/csv"
    )

    st.subheader("🧠 SHAP Feature Importance (Trolley)")
    explainer_t = shap.Explainer(model_t)
    shap_values_t = explainer_t(X_test_t)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    shap.plots.bar(shap_values_t, max_display=10, show=False)
    st.pyplot(fig2)
