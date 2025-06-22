import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.ensemble import GradientBoostingRegressor
from prophet import Prophet
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday
from pandas.tseries.offsets import DateOffset
from dateutil.rrule import MO
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import sys

# Suppress specific warnings for a cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")
try:
    warnings.filterwarnings("ignore", category=lgb.LightGBMWarning)
except AttributeError:
    print("Warning: lgb.LightGBMWarning not found.")

# --- Global storage for category mappings (from Script 1) ---
category_mappings = {}

# --- Define Irish Bank Holidays ---
class IrishBankHolidays(AbstractHolidayCalendar):
    """
    Defines the rules for Irish Bank Holidays.
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

# --- Feature from Script 1 Start ---
def add_iteration2_features(df, calendar_obj):
    """
    Adds advanced interaction and holiday proximity features from Script 1.
    Uses a global mapping dictionary for robust categorical encoding.
    """
    global category_mappings
    print("Adding Iteration 2 features...")
    df_eng = df.copy()
    df_eng['Hour_DayOfWeek_str'] = df_eng['Hour'].astype(str) + "_" + df_eng['DayOfWeek'].astype(str)
    if 'Hour_DayOfWeek' not in category_mappings:
        unique_cats = sorted(df_eng['Hour_DayOfWeek_str'].unique())
        category_mappings['Hour_DayOfWeek'] = pd.CategoricalDtype(categories=unique_cats, ordered=False)
    df_eng['Hour_DayOfWeek'] = df_eng['Hour_DayOfWeek_str'].astype(category_mappings['Hour_DayOfWeek']).cat.codes
    df_eng['Hospital_Code_Hour_str'] = df_eng['Hospital_Code'].astype(str) + "_" + df_eng['Hour'].astype(str)
    if 'Hospital_Code_Hour' not in category_mappings:
        unique_cats_hc_hr = sorted(df_eng['Hospital_Code_Hour_str'].unique())
        category_mappings['Hospital_Code_Hour'] = pd.CategoricalDtype(categories=unique_cats_hc_hr, ordered=False)
    df_eng['Hospital_Code_Hour'] = df_eng['Hospital_Code_Hour_str'].astype(category_mappings['Hospital_Code_Hour']).cat.codes
    if 'Capacity' in df_eng.columns and 'Hour' in df_eng.columns:
        df_eng['Capacity_Hour_Interaction'] = df_eng['Capacity'] * df_eng['Hour']
    actual_holidays = []
    if not df_eng.empty:
        min_date, max_date = df_eng['Datetime'].min(), df_eng['Datetime'].max()
        try:
            actual_holidays = calendar_obj.holidays(start=min_date - timedelta(days=45), end=max_date + timedelta(days=45)).normalize()
            actual_holidays = sorted(list(set(actual_holidays)))
        except Exception as e:
            print(f"Warning: Holiday generation failed for proximity features: {e}")
            actual_holidays = []
    if not actual_holidays:
        df_eng['DaysToNextHoliday'] = 365
        df_eng['DaysFromLastHoliday'] = 365
    else:
        df_eng['DaysToNextHoliday'] = df_eng['Datetime'].apply(lambda x: min([ (h - x.normalize()).days for h in actual_holidays if h >= x.normalize() ], default=365))
        df_eng['DaysFromLastHoliday'] = df_eng['Datetime'].apply(lambda x: min([ (x.normalize() - h).days for h in actual_holidays if h <= x.normalize() ], default=365))
    df_eng['DaysToNextHoliday'] = df_eng['DaysToNextHoliday'].fillna(365)
    df_eng['DaysFromLastHoliday'] = df_eng['DaysFromLastHoliday'].fillna(365)
    df_eng = df_eng.drop(columns=['Hour_DayOfWeek_str', 'Hospital_Code_Hour_str'], errors='ignore')
    return df_eng
# --- Feature from Script 1 End ---


@st.cache_data
def prepare_data(df):
    """
    Prepares and feature engineers the data, reshaping it into the desired wide format
    and adding both basic and advanced features.
    """
    df = df.rename(columns={
        'Tracker8am': 'ED_8am', 'Tracker2pm': 'ED_2pm', 'Tracker8pm': 'ED_8pm',
        'TimeTotal_8am': 'Trolley_8am', 'TimeTotal_2pm': 'Trolley_2pm', 'TimeTotal_8pm': 'Trolley_8pm',
        'AdditionalCapacityOpen Morning': 'Additional_Capacity'
    })
    df['Additional_Capacity'] = df.groupby(['Hospital', 'Date'])['Additional_Capacity'].transform('first').fillna(0)
    common_id_vars = ['Hospital Group Name', 'Hospital', 'Date', 'Additional_Capacity']
    df_ed = pd.melt(df, id_vars=common_id_vars, value_vars=['ED_8am', 'ED_2pm', 'ED_8pm'], var_name='Metric_Time_ED', value_name='ED Beds')
    df_trolley = pd.melt(df, id_vars=common_id_vars, value_vars=['Trolley_8am', 'Trolley_2pm', 'Trolley_8pm'], var_name='Metric_Time_Trolley', value_name='Trolleys')
    time_map = {'8am': '08:00', '2pm': '14:00', '8pm': '20:00'}
    df_ed['TimeLabel'] = df_ed['Metric_Time_ED'].str.extract(r'([\d]+[ap]m)')
    df_trolley['TimeLabel'] = df_trolley['Metric_Time_Trolley'].str.extract(r'([\d]+[ap]m)')
    df_ed['Time'] = df_ed['TimeLabel'].map(time_map)
    df_trolley['Time'] = df_trolley['TimeLabel'].map(time_map)
    df_merged = pd.merge(df_ed.drop(columns=['Metric_Time_ED', 'TimeLabel']), df_trolley.drop(columns=['Metric_Time_Trolley', 'TimeLabel']), on=['Hospital Group Name', 'Hospital', 'Date', 'Additional_Capacity', 'Time'], how='inner')
    df_merged['ED Beds'] = pd.to_numeric(df_merged['ED Beds'], errors='coerce')
    df_merged['Trolleys'] = pd.to_numeric(df_merged['Trolleys'], errors='coerce')
    df_merged = df_merged.dropna(subset=['ED Beds', 'Trolleys'])
    df_merged = df_merged.rename(columns={'Additional_Capacity': 'Capacity'})
    # Convert Date column to datetime just in case it's not
    df_merged['Date'] = pd.to_datetime(df_merged['Date'], errors='coerce')
    df_merged = df_merged.dropna(subset=['Date'])
    df_merged['Datetime'] = pd.to_datetime(df_merged['Date'].dt.strftime('%Y-%m-%d') + ' ' + df_merged['Time'])
    df_merged = df_merged.sort_values(by=['Hospital', 'Datetime']).reset_index(drop=True)
    df_merged['Hour'] = df_merged['Datetime'].dt.hour
    df_merged['DayOfWeek'] = df_merged['Datetime'].dt.dayofweek
    df_merged['DayOfMonth'] = df_merged['Datetime'].dt.day
    df_merged['Month'] = df_merged['Datetime'].dt.month
    df_merged['Quarter'] = df_merged['Datetime'].dt.quarter
    df_merged['WeekOfYear'] = df_merged['Datetime'].dt.isocalendar().week.astype(int)
    df_merged['IsWeekend'] = df_merged['DayOfWeek'].isin([5, 6]).astype(int)
    df_merged['IsMonday'] = (df_merged['DayOfWeek'] == 0).astype(int)
    df_merged['IsFriday'] = (df_merged['DayOfWeek'] == 4).astype(int)
    df_merged['Hour_sin'] = np.sin(2 * np.pi * df_merged['Hour'] / 24)
    df_merged['Hour_cos'] = np.cos(2 * np.pi * df_merged['Hour'] / 24)
    df_merged['Day_sin'] = np.sin(2 * np.pi * df_merged['DayOfWeek'] / 7)
    df_merged['Day_cos'] = np.cos(2 * np.pi * df_merged['DayOfWeek'] / 7)
    df_merged['Month_sin'] = np.sin(2 * np.pi * df_merged['Month'] / 12)
    df_merged['Month_cos'] = np.cos(2 * np.pi * df_merged['Month'] / 12)
    df_merged['IsSummer'] = df_merged['Month'].isin([6, 7, 8]).astype(int)
    df_merged['IsWinter'] = df_merged['Month'].isin([12, 1, 2]).astype(int)
    df_merged['IsPeakHour'] = df_merged['Hour'].isin([20]).astype(int)
    df_merged['IsLowHour'] = df_merged['Hour'].isin([8]).astype(int)
    df_merged['Hospital_Code'] = df_merged['Hospital'].astype('category').cat.codes
    calendar = IrishBankHolidays()
    try:
        holidays = calendar.holidays(start=df_merged['Datetime'].min(), end=df_merged['Datetime'].max() + timedelta(days=30))
        df_merged['IsHoliday'] = df_merged['Datetime'].dt.normalize().isin(holidays).astype(int)
    except Exception:
        df_merged['IsHoliday'] = 0
    df_featured = add_iteration2_features(df_merged, calendar)
    return df_featured


@st.cache_data
def add_lag_features_smart(df, target_column):
    df_copy = df.copy()
    lag_features = []
    df_copy = df_copy.sort_values(by=['Hospital', 'Datetime'])
    for hospital_name, hospital_group in df_copy.groupby('Hospital'):
        max_safe_lag = min(7, len(hospital_group) // 4)
        if max_safe_lag < 1:
            continue
        for i in range(1, max_safe_lag + 1):
            lag_col = f'Lag_{target_column}_{i}'
            df_copy.loc[hospital_group.index, lag_col] = hospital_group[target_column].shift(i)
            if lag_col not in lag_features:
                lag_features.append(lag_col)
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
    for feature in lag_features:
        df_copy[feature] = df_copy[feature].fillna(method='ffill').fillna(method='bfill').fillna(0)
    return df_copy, lag_features


def create_future_dates(last_date, hospital, hospital_code, additional_capacity, days=7, existing_df_for_holiday_range=None):
    global category_mappings
    future_dates_list = []
    times = ['08:00', '14:00', '20:00']
    calendar_obj = IrishBankHolidays()
    actual_holidays = []
    if existing_df_for_holiday_range is not None and not existing_df_for_holiday_range.empty:
        min_hist_date = existing_df_for_holiday_range['Datetime'].min()
        try:
            actual_holidays = calendar_obj.holidays(start=min_hist_date - timedelta(days=45), end=last_date + timedelta(days=days + 45)).normalize()
            actual_holidays = sorted(list(set(actual_holidays)))
        except Exception as e:
            print(f"Warning: Holiday generation for future dates failed: {e}")
            actual_holidays = []
    for day in range(1, days + 1):
        future_date = last_date + timedelta(days=day)
        for time_str in times:
            future_datetime = pd.to_datetime(f"{future_date.date()} {time_str}")
            h, dow, dom, m, q, woy = future_datetime.hour, future_datetime.dayofweek, future_datetime.day, future_datetime.month, (future_datetime.month-1)//3+1, int(future_datetime.isocalendar().week)
            hour_dayofweek_str_future = f"{h}_{dow}"
            hour_dayofweek_code = pd.Series([hour_dayofweek_str_future]).astype(category_mappings.get('Hour_DayOfWeek', 'category')).cat.codes[0]
            hospital_code_hour_str_future = f"{hospital_code}_{h}"
            hospital_code_hour_code = pd.Series([hospital_code_hour_str_future]).astype(category_mappings.get('Hospital_Code_Hour', 'category')).cat.codes[0]
            cap_hour_inter = additional_capacity * h
            days_to_next = 365
            days_from_last = 365
            if actual_holidays:
                days_to_next = min([(hol_date - future_datetime.normalize()).days for hol_date in actual_holidays if hol_date >= future_datetime.normalize()], default=365)
                days_from_last = min([(future_datetime.normalize() - hol_date).days for hol_date in actual_holidays if hol_date <= future_datetime.normalize()], default=365)
            future_dates_list.append({
                'Date': future_datetime.date(), 'Time': time_str, 'Datetime': future_datetime, 'Hospital': hospital,
                'Hour': h, 'DayOfWeek': dow, 'DayOfMonth': dom, 'Month': m, 'Quarter': q, 'WeekOfYear': woy,
                'IsWeekend': int(dow in [5,6]), 'IsMonday': int(dow==0), 'IsFriday': int(dow==4),
                'Hour_sin': np.sin(2*np.pi*h/24), 'Hour_cos': np.cos(2*np.pi*h/24),
                'Day_sin': np.sin(2*np.pi*dow/7), 'Day_cos': np.cos(2*np.pi*dow/7),
                'Month_sin': np.sin(2*np.pi*m/12), 'Month_cos': np.cos(2*np.pi*m/12),
                'IsHoliday': int(future_datetime.normalize() in actual_holidays if actual_holidays else 0),
                'IsSummer': int(m in [6,7,8]), 'IsWinter': int(m in [12,1,2]),
                'IsPeakHour': int(h==20), 'IsLowHour': int(h==8),
                'Hospital_Code': hospital_code, 'Capacity': additional_capacity,
                'Hour_DayOfWeek': hour_dayofweek_code,
                'Hospital_Code_Hour': hospital_code_hour_code,
                'Capacity_Hour_Interaction': cap_hour_inter,
                'DaysToNextHoliday': days_to_next,
                'DaysFromLastHoliday': days_from_last
            })
    return pd.DataFrame(future_dates_list)


def forecast_with_lags(model, historical_data, future_df, features, target_column):
    if historical_data.empty:
        st.error("Historical data for lag forecasting is empty.")
        return pd.DataFrame({'Predicted': [0]*len(future_df)}, index=future_df.index)
    temp_target_history = historical_data[target_column].tolist()
    predictions = []
    for i in range(len(future_df)):
        current_feature_values_map = {}
        current_future_row = future_df.iloc[i]
        for f_name in features:
            if not (f_name.startswith(f'Lag_{target_column}_') or f_name.startswith(f'Rolling_Mean_')):
                if f_name in current_future_row:
                    current_feature_values_map[f_name] = current_future_row[f_name]
                else:
                    current_feature_values_map[f_name] = 0
            else:
                if f_name.startswith(f'Lag_{target_column}_'):
                    lag_order = int(f_name.split('_')[-1])
                    if lag_order <= len(temp_target_history):
                        current_feature_values_map[f_name] = temp_target_history[-lag_order]
                    else:
                        current_feature_values_map[f_name] = 0
                elif f_name == f'Rolling_Mean_3_{target_column}':
                    current_feature_values_map[f_name] = np.mean(temp_target_history[-3:]) if len(temp_target_history) >= 3 else (np.mean(temp_target_history) if temp_target_history else 0)
                elif f_name == f'Rolling_Mean_7_{target_column}':
                    current_feature_values_map[f_name] = np.mean(temp_target_history[-7:]) if len(temp_target_history) >= 7 else (np.mean(temp_target_history) if temp_target_history else 0)
        ordered_feature_values = [current_feature_values_map.get(f_name, 0) for f_name in features]
        feature_vector = np.array(ordered_feature_values).reshape(1, -1)
        pred = model.predict(feature_vector)[0]
        pred = max(0, round(pred))
        predictions.append(pred)
        temp_target_history.append(pred)
    return pd.DataFrame({'Predicted': predictions}, index=future_df.index)


def plot_forecasts(historical_data, forecast_data, metric_name, hospital_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical_data['Datetime'], y=historical_data[metric_name], mode='lines+markers', name='Historical', line=dict(color='blue'), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=forecast_data['Datetime'], y=forecast_data['Predicted'], mode='lines+markers', name='Forecast', line=dict(color='red', dash='dash'), marker=dict(size=6, symbol='diamond')))
    if not historical_data.empty:
        last_historical_date = historical_data['Datetime'].max()
        fig.add_vline(x=last_historical_date, line_width=2, line_dash="dot", line_color="gray", annotation_text="Forecast Start", annotation_position="top right")
    fig.update_layout(title=f'{metric_name} Forecast for {hospital_name}', xaxis_title='Date', yaxis_title=f'{metric_name} Count', hovermode='x unified', showlegend=True, height=500)
    return fig


def main():
    st.title("Emergency Department Forecasting - Integrated Model")
    
    # --- CORRECTED CODE: Use Streamlit file uploader for Excel files ---
    st.sidebar.header("1. Upload Data")
    uploaded_file = st.sidebar.file_uploader(
        "Choose an Excel file",
        type=['xlsx', 'xls']
    )

    if uploaded_file is None:
        st.info("Please upload an Excel data file to begin.")
        return
    # --- END OF CORRECTION ---

    # Reset mappings for each run in a Streamlit context
    global category_mappings
    category_mappings = {}

    # Read the uploaded excel file
    df_raw = pd.read_excel(uploaded_file)
    
    st.success(f"File '{uploaded_file.name}' uploaded successfully!")

    with st.spinner("Preparing data and engineering features..."):
        df_featured = prepare_data(df_raw)
    
    st.success(f"Data prepared with {len(df_featured.columns)} features.")
    with st.expander("Show Data Head with New Features"):
        st.dataframe(df_featured.head())
    
    st.sidebar.header("2. Configure Forecast")
    hospitals = sorted(df_featured['Hospital'].unique())
    hospital_option = st.sidebar.selectbox("Select Hospital:", options=hospitals)
    
    target_metric = st.sidebar.selectbox("Select Metric to Forecast:", options=['ED Beds', 'Trolleys', 'Capacity'])

    if st.sidebar.button("🚀 Run Forecast", type="primary"):
        st.header(f"Forecast for {hospital_option}")

        hospital_data = df_featured[df_featured['Hospital'] == hospital_option].copy()
        
        with st.spinner("Adding time-series features (lags and rolling means)..."):
            hospital_data_with_lags, lag_features = add_lag_features_smart(hospital_data, target_metric)

        base_features = [
            'Hour', 'DayOfWeek', 'DayOfMonth', 'Month', 'Quarter', 'WeekOfYear', 'IsWeekend',
            'IsMonday', 'IsFriday', 'Hour_sin', 'Hour_cos', 'Day_sin', 'Day_cos', 'Month_sin',
            'Month_cos', 'IsHoliday', 'IsSummer', 'IsWinter', 'IsPeakHour', 'IsLowHour', 'Hospital_Code'
        ]
        iter2_features = ['Hour_DayOfWeek', 'Hospital_Code_Hour', 'Capacity_Hour_Interaction', 'DaysToNextHoliday', 'DaysFromLastHoliday']
        model_features = base_features + iter2_features + lag_features
        if target_metric != 'Capacity' and 'Capacity' in hospital_data_with_lags.columns:
            if 'Capacity' not in model_features:
                model_features.append('Capacity')
        model_features = sorted(list(set(f for f in model_features if f != target_metric and f in hospital_data_with_lags.columns)))
        
        with st.expander("Features Used for Modeling"):
            st.write(model_features)

        split_date = hospital_data_with_lags['Datetime'].max() - timedelta(days=7)
        train_df = hospital_data_with_lags[hospital_data_with_lags['Datetime'] <= split_date].copy()
        
        if train_df.empty or len(train_df) < 20:
            st.warning("Insufficient data to train the model for the selected hospital.")
            return

        X_train = train_df.dropna(subset=[target_metric] + model_features)[model_features]
        y_train = train_df.loc[X_train.index, target_metric]
        
        if X_train.empty:
            st.warning("Training data is empty after removing rows with missing values.")
            return

        with st.spinner("Training the forecasting model..."):
            model = lgb.LGBMRegressor(random_state=42, verbose=-1)
            model.fit(X_train, y_train)
        
        last_train_date = train_df['Datetime'].max()
        hospital_code_val = train_df['Hospital_Code'].iloc[0]
        mean_capacity_val = train_df['Capacity'].mean()
        
        future_df = create_future_dates(
            last_train_date, hospital_option, hospital_code_val, 
            mean_capacity_val, days=7, existing_df_for_holiday_range=df_featured
        )

        with st.spinner("Generating future predictions..."):
            forecast_results_df = forecast_with_lags(model, train_df, future_df, model_features, target_metric)
            forecast_df_final = future_df.join(forecast_results_df)

        fig = plot_forecasts(hospital_data.tail(90), forecast_df_final, target_metric, hospital_option)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
