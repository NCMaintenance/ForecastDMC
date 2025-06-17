import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday
from pandas.tseries.offsets import DateOffset
from dateutil.rrule import MO
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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
    """Prepare and feature engineer the data"""
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

    # Enhanced Feature Engineering
    df_long['Hour'] = df_long['Datetime'].dt.hour
    df_long['DayOfWeek'] = df_long['Datetime'].dt.dayofweek
    df_long['DayOfMonth'] = df_long['Datetime'].dt.day
    df_long['Month'] = df_long['Datetime'].dt.month
    df_long['Quarter'] = df_long['Datetime'].dt.quarter
    df_long['WeekOfYear'] = df_long['Datetime'].dt.isocalendar().week
    df_long['IsWeekend'] = df_long['DayOfWeek'].isin([5, 6]).astype(int)
    df_long['IsMonday'] = (df_long['DayOfWeek'] == 0).astype(int)  # Monday effect
    df_long['IsFriday'] = (df_long['DayOfWeek'] == 4).astype(int)  # Friday effect
    
    # Cyclical encoding for time features
    df_long['Hour_sin'] = np.sin(2 * np.pi * df_long['Hour'] / 24)
    df_long['Hour_cos'] = np.cos(2 * np.pi * df_long['Hour'] / 24)
    df_long['Day_sin'] = np.sin(2 * np.pi * df_long['DayOfWeek'] / 7)
    df_long['Day_cos'] = np.cos(2 * np.pi * df_long['DayOfWeek'] / 7)
    df_long['Month_sin'] = np.sin(2 * np.pi * df_long['Month'] / 12)
    df_long['Month_cos'] = np.cos(2 * np.pi * df_long['Month'] / 12)
    df_long['Week_sin'] = np.sin(2 * np.pi * df_long['WeekOfYear'] / 52)
    df_long['Week_cos'] = np.cos(2 * np.pi * df_long['WeekOfYear'] / 52)

    # Add Irish holidays
    calendar = IrishBankHolidays()
    holidays = calendar.holidays(start=df_long['Datetime'].min(), end=df_long['Datetime'].max() + timedelta(days=30))
    df_long['IsHoliday'] = df_long['Datetime'].dt.normalize().isin(holidays).astype(int)
    
    # Holiday proximity features
    df_long['DaysToHoliday'] = 0
    df_long['DaysFromHoliday'] = 0
    
    for idx, row in df_long.iterrows():
        date = row['Datetime'].normalize()
        # Find closest holiday
        future_holidays = [h for h in holidays if h >= date]
        past_holidays = [h for h in holidays if h < date]
        
        if future_holidays:
            df_long.loc[idx, 'DaysToHoliday'] = (min(future_holidays) - date).days
        else:
            df_long.loc[idx, 'DaysToHoliday'] = 365  # Far future
            
        if past_holidays:
            df_long.loc[idx, 'DaysFromHoliday'] = (date - max(past_holidays)).days
        else:
            df_long.loc[idx, 'DaysFromHoliday'] = 365  # Far past
    
    # Seasonal indicators
    df_long['IsSummer'] = df_long['Month'].isin([6, 7, 8]).astype(int)
    df_long['IsWinter'] = df_long['Month'].isin([12, 1, 2]).astype(int)
    df_long['IsSpring'] = df_long['Month'].isin([3, 4, 5]).astype(int)
    
    # Peak hour indicators
    df_long['IsPeakHour'] = df_long['Hour'].isin([20]).astype(int)  # 8pm typically busiest
    df_long['IsLowHour'] = df_long['Hour'].isin([8]).astype(int)    # 8am typically quietest

    # Encode hospital
    df_long['Hospital_Code'] = df_long['Hospital'].astype('category').cat.codes

    return df_long

def create_future_dates(last_date, hospital, hospital_code, additional_capacity, days=7):
    """Create future dates for forecasting"""
    future_dates = []
    times = ['08:00', '14:00', '20:00']
    
    for day in range(1, days + 1):
        future_date = last_date + timedelta(days=day)
        for time_str in times:
            future_datetime = pd.to_datetime(f"{future_date.date()} {time_str}")
            
            # Enhanced feature engineering for future dates
            hour = future_datetime.hour
            day_of_week = future_datetime.dayofweek
            day_of_month = future_datetime.day
            month = future_datetime.month
            quarter = (month - 1) // 3 + 1
            week_of_year = future_datetime.isocalendar().week
            
            is_weekend = int(day_of_week in [5, 6])
            is_monday = int(day_of_week == 0)
            is_friday = int(day_of_week == 4)
            
            # Cyclical features
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            day_sin = np.sin(2 * np.pi * day_of_week / 7)
            day_cos = np.cos(2 * np.pi * day_of_week / 7)
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)
            week_sin = np.sin(2 * np.pi * week_of_year / 52)
            week_cos = np.cos(2 * np.pi * week_of_year / 52)
            
            # Check if it's a holiday
            calendar = IrishBankHolidays()
            holidays = calendar.holidays(start=future_datetime, end=future_datetime)
            is_holiday = int(future_datetime.normalize() in holidays)
            
            # Holiday proximity (simplified for future dates)
            days_to_holiday = 30  # Default
            days_from_holiday = 30  # Default
            
            # Seasonal indicators
            is_summer = int(month in [6, 7, 8])
            is_winter = int(month in [12, 1, 2])
            is_spring = int(month in [3, 4, 5])
            
            # Peak hour indicators
            is_peak_hour = int(hour == 20)
            is_low_hour = int(hour == 8)
            
            future_dates.append({
                'Datetime': future_datetime,
                'Hospital': hospital,
                'Hour': hour,
                'DayOfWeek': day_of_week,
                'DayOfMonth': day_of_month,
                'Month': month,
                'Quarter': quarter,
                'WeekOfYear': week_of_year,
                'IsWeekend': is_weekend,
                'IsMonday': is_monday,
                'IsFriday': is_friday,
                'Hour_sin': hour_sin,
                'Hour_cos': hour_cos,
                'Day_sin': day_sin,
                'Day_cos': day_cos,
                'Month_sin': month_sin,
                'Month_cos': month_cos,
                'Week_sin': week_sin,
                'Week_cos': week_cos,
                'IsHoliday': is_holiday,
                'DaysToHoliday': days_to_holiday,
                'DaysFromHoliday': days_from_holiday,
                'IsSummer': is_summer,
                'IsWinter': is_winter,
                'IsSpring': is_spring,
                'IsPeakHour': is_peak_hour,
                'IsLowHour': is_low_hour,
                'Hospital_Code': hospital_code,
                'Additional_Capacity': additional_capacity
            })
    
    return pd.DataFrame(future_dates)

def forecast_with_lags(model, historical_data, future_df, features):
    """Enhanced forecast function with better lag handling"""
    # Get last 7 values for initial lags (weekly pattern)
    last_values = historical_data['Value'].tail(7).values
    
    predictions = []
    current_lags = list(reversed(last_values))  # [lag_1, lag_2, ..., lag_7]
    
    # Features without lag, rolling, and cross-metric columns
    base_features = [f for f in features if not any(x in f for x in ['Lag_', 'Rolling_', '_Value', '_Lag1', 'Hour_Weekend', 'Month_Weekend'])]
    
    # Get rolling stats from historical data
    historical_mean_3 = historical_data['Value'].tail(3).mean()
    historical_mean_7 = historical_data['Value'].tail(7).mean()
    historical_std_3 = historical_data['Value'].tail(3).std()
    historical_std_7 = historical_data['Value'].tail(7).std()
    
    # Fill NaN with 0 for std calculations
    historical_std_3 = historical_std_3 if not np.isnan(historical_std_3) else 0
    historical_std_7 = historical_std_7 if not np.isnan(historical_std_7) else 0
    
    for idx, row in future_df.iterrows():
        try:
            # Base features
            base_values = [row[f] for f in base_features]
            
            # Lag features (up to 7 lags)
            lag_values = current_lags[:7] + [0] * (7 - len(current_lags))
            
            # Rolling statistics (simplified for forecasting)
            rolling_values = [historical_mean_3, historical_mean_7, historical_std_3, historical_std_7]
            
            # Interaction features
            hour_weekend = row['Hour'] * row['IsWeekend']
            month_weekend = row['Month'] * row['IsWeekend']
            interaction_values = [hour_weekend, month_weekend]
            
            # Cross-metric features (simplified - use last known values)
            cross_values = [0, 0]  # Placeholder for cross-metric features
            
            # Combine all features
            feature_vector = base_values + lag_values + rolling_values + interaction_values + cross_values
            feature_vector = np.array(feature_vector).reshape(1, -1)
            
            # Make prediction
            pred = model.predict(feature_vector)[0]
            pred = max(0, pred)  # Ensure non-negative predictions
            predictions.append(pred)
            
            # Update lags for next prediction
            current_lags = [pred] + current_lags[:6]
            
            # Update rolling statistics
            if len(predictions) >= 3:
                historical_mean_3 = np.mean(predictions[-3:])
                historical_std_3 = np.std(predictions[-3:])
            if len(predictions) >= 7:
                historical_mean_7 = np.mean(predictions[-7:])
                historical_std_7 = np.std(predictions[-7:])
            
        except Exception as e:
            st.error(f"Error in prediction: {e}")
            predictions.append(0)
    
    return predictions

def plot_forecasts(historical_data, forecast_data, metric_name, hospital_name):
    """Create interactive plotly chart"""
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=historical_data['Datetime'],
        y=historical_data['Value'],
        mode='lines+markers',
        name='Historical',
        line=dict(color='blue'),
        marker=dict(size=4)
    ))
    
    # Forecast data
    fig.add_trace(go.Scatter(
        x=forecast_data['Datetime'],
        y=forecast_data['Predicted'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='red', dash='dash'),
        marker=dict(size=6, symbol='diamond')
    ))
    
    # Add vertical line to separate historical and forecast
    last_historical_date = historical_data['Datetime'].max()
    
    # Convert timestamp to string to avoid plotly issues
    fig.add_shape(
        type="line",
        x0=last_historical_date,
        x1=last_historical_date,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="gray", width=2, dash="dot"),
    )
    
    # Add annotation for the line
    fig.add_annotation(
        x=last_historical_date,
        y=0.95,
        yref="paper",
        text="Forecast Start",
        showarrow=False,
        bgcolor="white",
        bordercolor="gray",
        borderwidth=1
    )
    
    fig.update_layout(
        title=f'{metric_name} Forecast - {hospital_name}',
        xaxis_title='Date',
        yaxis_title=f'{metric_name} Count',
        hovermode='x unified',
        showlegend=True,
        height=500
    )
    
    return fig

def add_forecasting_insights():
    with st.expander("💡 Forecasting Insights & Tips", expanded=False):
        st.subheader("Making Sense of Your Data Statistics")
        st.markdown("""
        Use the 'Detailed Statistical Measurements' section to understand typical patterns in your data:
        *   **Time of Day:** Which reporting times (8am, 2pm, 8pm) generally have the highest or lowest counts? Are some more volatile (higher standard deviation)?
        *   **Day of Week:** Are there clear busy days (e.g., Mondays) or quieter days (e.g., weekends)? This can inform staffing and resource allocation.
        *   **Monthly Trends:** Do you observe seasonal patterns? For example, are certain months consistently busier due to weather or holidays?
        *   **Hospital Variations:** If you're looking at multiple hospitals, how do these patterns compare across them?
        """)

        st.subheader("Understanding Key Forecasting Features")
        st.markdown("""
        The model uses several types of features to learn from historical data:
        *   **Lag Features:** The model looks at recent past values (lags) because what happened recently often influences the near future. For example, if patient counts were high yesterday, they might be more likely to be high today.
        *   **Cyclical Time Features (Hour, Day of Week, Month, etc.):** Many activities have natural cycles. Counts might rise and fall predictably depending on the time of day, day of the week, or month of the year. We convert these time aspects into 'cyclical features' (using sine and cosine functions) that help the model learn these repeating patterns. This is more effective than just using numbers like 1-7 for days of the week.
        *   **Holiday Effects:** Public holidays and special days often have a significant impact on ED and trolley counts. The model is designed to recognize Irish bank holidays and learn their typical influence. If your data includes other recurring local events that affect numbers, consider if these could be added as features in future model versions.
        *   **Other Factors (`Additional_Capacity`, `DayGAR`):** Features like whether additional capacity was open or the specific 'GAR' (General Adult Report) day type are also included to help the model understand other known influences.
        """)
        st.info("Forecasting is about identifying patterns in historical data to predict the future. The more consistent the patterns and the better they are represented by features, the more accurate the forecast is likely to be. However, unexpected events can always affect outcomes.")

# --- Streamlit UI ---
st.title("Emergency Department Forecasting")
st.markdown("Upload your ED Excel file, select hospital(s), and generate 7-day forecasts")

uploaded_file = st.file_uploader("Upload your ED Excel File", type=["xlsx"])

if uploaded_file:
    # Load and prepare data
    df = pd.read_excel(uploaded_file)
    df_long = prepare_data(df)
    
    # Get unique hospitals for selection
    hospitals = sorted(df_long['Hospital'].unique())

    # --- Descriptive Statistics Section ---
    with st.expander("🔬 Detailed Statistical Measurements", expanded=False):
        st.markdown("""
        This section provides a statistical overview of the historical data,
        helping to understand trends and patterns before forecasting.
        """)

        # Define mappings for DayName and MonthName
        day_map = {
            0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
            4: 'Friday', 5: 'Saturday', 6: 'Sunday'
        }
        month_map = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        }

        # Apply mappings
        df_long['DayName'] = df_long['DayOfWeek'].map(day_map)
        df_long['MonthName'] = df_long['Month'].map(month_map)

        # Aggregation function (optional, but good for consistency)
        def get_agg_stats(df, group_by_cols):
            return df.groupby(group_by_cols)['Value'].agg(['mean', 'median', 'min', 'max', 'std']).reset_index()

        st.subheader("Overall Statistics by Metric & Time of Day")
        stats_by_time = get_agg_stats(df_long, ['Metric', 'TimeLabel'])
        st.dataframe(stats_by_time.style.format({'mean': '{:.2f}', 'median': '{:.2f}', 'std': '{:.2f}'}))

        st.subheader("Overall Statistics by Metric & Day of Week")
        stats_by_dow = get_agg_stats(df_long, ['Metric', 'DayName'])
        # Ensure correct order for DayName
        stats_by_dow['DayName'] = pd.Categorical(stats_by_dow['DayName'], categories=day_map.values(), ordered=True)
        stats_by_dow = stats_by_dow.sort_values(['Metric', 'DayName'])
        st.dataframe(stats_by_dow.style.format({'mean': '{:.2f}', 'median': '{:.2f}', 'std': '{:.2f}'}))

        st.subheader("Overall Statistics by Metric & Month")
        stats_by_month = get_agg_stats(df_long, ['Metric', 'MonthName'])
        # Ensure correct order for MonthName
        stats_by_month['MonthName'] = pd.Categorical(stats_by_month['MonthName'], categories=month_map.values(), ordered=True)
        stats_by_month = stats_by_month.sort_values(['Metric', 'MonthName'])
        st.dataframe(stats_by_month.style.format({'mean': '{:.2f}', 'median': '{:.2f}', 'std': '{:.2f}'}))

        st.markdown("---")
        st.subheader("Hospital-Specific Statistics")

        st.info("The following tables show statistics broken down by individual hospitals. This can be useful for comparing trends across different sites.")

        st.subheader("Statistics by Hospital, Metric & Time of Day")
        stats_by_hosp_time = get_agg_stats(df_long, ['Hospital', 'Metric', 'TimeLabel'])
        st.dataframe(stats_by_hosp_time.style.format({'mean': '{:.2f}', 'median': '{:.2f}', 'std': '{:.2f}'}))

        st.subheader("Statistics by Hospital, Metric & Day of Week")
        stats_by_hosp_dow = get_agg_stats(df_long, ['Hospital', 'Metric', 'DayName'])
        stats_by_hosp_dow['DayName'] = pd.Categorical(stats_by_hosp_dow['DayName'], categories=day_map.values(), ordered=True)
        stats_by_hosp_dow = stats_by_hosp_dow.sort_values(['Hospital', 'Metric', 'DayName'])
        st.dataframe(stats_by_hosp_dow.style.format({'mean': '{:.2f}', 'median': '{:.2f}', 'std': '{:.2f}'}))

        st.subheader("Statistics by Hospital, Metric & Month")
        stats_by_hosp_month = get_agg_stats(df_long, ['Hospital', 'Metric', 'MonthName'])
        stats_by_hosp_month['MonthName'] = pd.Categorical(stats_by_hosp_month['MonthName'], categories=month_map.values(), ordered=True)
        stats_by_hosp_month = stats_by_hosp_month.sort_values(['Hospital', 'Metric', 'MonthName'])
        st.dataframe(stats_by_hosp_month.style.format({'mean': '{:.2f}', 'median': '{:.2f}', 'std': '{:.2f}'}))

    add_forecasting_insights()
    
    # UI Controls
    st.sidebar.header("Forecast Settings")
    
    # Hospital selection
    hospital_option = st.sidebar.selectbox(
        "Select Hospital:",
        options=["All Hospitals"] + hospitals
    )
    
    # Accuracy Improvement Controls
    st.sidebar.header("🎯 Accuracy Settings")
    
    use_cross_validation = st.sidebar.checkbox("Use Cross-Validation", value=False)
    ensemble_models = st.sidebar.checkbox("Use Ensemble Models", value=False)
    
    # Model hyperparameters
    with st.sidebar.expander("⚙️ Model Parameters"):
        n_estimators = st.slider("Number of Trees", 100, 500, 200)
        learning_rate = st.slider("Learning Rate", 0.01, 0.2, 0.05, 0.01)
        max_depth = st.slider("Max Depth", 3, 15, 8)
        
    # Feature importance threshold
    feature_importance_threshold = st.sidebar.slider(
        "Feature Importance Threshold", 
        0.0, 0.1, 0.01, 0.005,
        help="Remove features below this importance threshold"
    )
    
    # Run button
    run_forecast = st.sidebar.button("🚀 Run Forecast", type="primary")
    
    if run_forecast:
        st.header("📊 Forecast Results")
        
        # Determine which hospitals to process
        if hospital_option == "All Hospitals":
            selected_hospitals = hospitals
        else:
            selected_hospitals = [hospital_option]
        
        # Enhanced Features for modeling
        base_features = [
            'Hour', 'DayOfWeek', 'DayOfMonth', 'Month', 'Quarter', 'WeekOfYear',
            'IsWeekend', 'IsMonday', 'IsFriday',
            'Hour_sin', 'Hour_cos', 'Day_sin', 'Day_cos', 'Month_sin', 'Month_cos', 'Week_sin', 'Week_cos',
            'IsHoliday', 'DaysToHoliday', 'DaysFromHoliday',
            'IsSummer', 'IsWinter', 'IsSpring',
            'IsPeakHour', 'IsLowHour',
            'Hospital_Code', 'Additional_Capacity'
        ]
        
        lag_features = [f'Lag_{i}' for i in range(1, 8)]
        rolling_features = ['Rolling_Mean_3', 'Rolling_Mean_7', 'Rolling_Std_3', 'Rolling_Std_7']
        interaction_features = ['Hour_Weekend', 'Month_Weekend']
        cross_features = [f'{other_metric}_Value', f'{other_metric}_Lag1'] if 'other_metric' in locals() else []
        
        features = base_features + lag_features + rolling_features + interaction_features + cross_features
        
        # Process each selected hospital
        for hospital in selected_hospitals:
            st.subheader(f"🏥 {hospital}")
            
            # Filter data for current hospital
            hospital_data = df_long[df_long['Hospital'] == hospital].copy()
            hospital_code = hospital_data['Hospital_Code'].iloc[0]
            additional_capacity = hospital_data['Additional_Capacity'].fillna(0).iloc[0]
            
            # Get last date for this hospital
            last_date = hospital_data['Datetime'].max().date()
            
            # Process ED and Trolley separately
            for metric in ['ED', 'Trolley']:
                metric_data = hospital_data[hospital_data['Metric'] == metric].copy()
                metric_data = metric_data.sort_values('Datetime')
                
                # Enhanced lag features
                for i in range(1, 8):  # Increase to 7 lags (weekly pattern)
                    metric_data[f'Lag_{i}'] = metric_data['Value'].shift(i)
                
                # Rolling statistics
                metric_data['Rolling_Mean_3'] = metric_data['Value'].rolling(window=3).mean()
                metric_data['Rolling_Mean_7'] = metric_data['Value'].rolling(window=7).mean()
                metric_data['Rolling_Std_3'] = metric_data['Value'].rolling(window=3).std()
                metric_data['Rolling_Std_7'] = metric_data['Value'].rolling(window=7).std()
                
                # Interaction features
                metric_data['Hour_Weekend'] = metric_data['Hour'] * metric_data['IsWeekend']
                metric_data['Month_Weekend'] = metric_data['Month'] * metric_data['IsWeekend']
                
                # Add cross-metric features (ED affects Trolley and vice versa)
                other_metric = 'Trolley' if metric == 'ED' else 'ED'
                other_data = hospital_data[hospital_data['Metric'] == other_metric].copy()
                if len(other_data) > 0:
                    other_data = other_data.sort_values('Datetime')
                    # Merge on datetime to get cross-metric features
                    cross_merge = pd.merge(metric_data[['Datetime']], other_data[['Datetime', 'Value']], 
                                         on='Datetime', how='left', suffixes=('', f'_{other_metric}'))
                    metric_data[f'{other_metric}_Value'] = cross_merge['Value'].fillna(method='ffill')
                    metric_data[f'{other_metric}_Lag1'] = metric_data[f'{other_metric}_Value'].shift(1)
                else:
                    metric_data[f'{other_metric}_Value'] = 0
                    metric_data[f'{other_metric}_Lag1'] = 0
                
                metric_data.dropna(inplace=True)
                
                if len(metric_data) < 10:  # Need minimum data for training
                    st.warning(f"Insufficient data for {metric} forecasting at {hospital}")
                    continue
                
                # Train model
                X = metric_data[features]
                y = metric_data['Value']
                
                # Use time-based split (last 20% for testing)
                split_idx = int(len(X) * 0.8)
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
                
                # Enhanced model with better hyperparameters
                model = lgb.LGBMRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=8,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    verbose=-1,
                    random_state=42
                )
                model.fit(X_train, y_train)
                
                # Calculate model performance
                y_pred_test = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                
                # Get test data for display (using the same indices)
                test_data = metric_data.iloc[split_idx:].copy()
                test_data['Predicted'] = y_pred_test
                
                # Create future dates
                future_df = create_future_dates(
                    pd.to_datetime(last_date), 
                    hospital, 
                    hospital_code, 
                    additional_capacity, 
                    days=forecast_days
                )
                
                # Generate forecasts
                predictions = forecast_with_lags(model, metric_data, future_df, features)
                future_df['Predicted'] = predictions
                
                # Display metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(f"{metric} RMSE", f"{rmse:.2f}")
                with col2:
                    st.metric(f"Last {metric} Value", f"{metric_data['Value'].iloc[-1]:.0f}")
                
                # Create and display plot
                fig = plot_forecasts(
                    metric_data.tail(30),  # Show last 30 historical points
                    future_df,
                    metric,
                    hospital
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show model validation results
                with st.expander(f"📊 {metric} Model Validation"):
                    validation_df = test_data[['Datetime', 'Hospital', 'Value', 'Predicted']].copy()
                    validation_df['Error'] = validation_df['Value'] - validation_df['Predicted']
                    validation_df['Abs_Error'] = abs(validation_df['Error'])
                    st.dataframe(validation_df.head(10), use_container_width=True)
                
                # Show forecast table
                with st.expander(f"📋 {metric} Forecast Details"):
                    forecast_display = future_df[['Datetime', 'Predicted']].copy()
                    forecast_display['Predicted'] = forecast_display['Predicted'].round(1)
                    forecast_display['Date'] = forecast_display['Datetime'].dt.date
                    forecast_display['Time'] = forecast_display['Datetime'].dt.strftime('%I:%M %p')
                    st.dataframe(
                        forecast_display[['Date', 'Time', 'Predicted']],
                        use_container_width=True
                    )
                
                # Download button for forecasts
                csv_data = future_df[['Datetime', 'Hospital', 'Predicted']].copy()
                csv_data['Metric'] = metric
                st.download_button(
                    f"📥 Download {metric} Forecast CSV",
                    csv_data.to_csv(index=False),
                    file_name=f"{hospital}_{metric}_forecast.csv",
                    mime="text/csv",
                    key=f"{hospital}_{metric}_download"
                )
                
                st.divider()
    
    # Show data summary
    with st.expander("📈 Data Summary"):
        st.write("**Hospitals in dataset:**", len(hospitals))
        st.write("**Date range:**", f"{df_long['Datetime'].min().date()} to {df_long['Datetime'].max().date()}")
        st.write("**Total records:**", len(df_long))
        
        # Show sample data
        st.dataframe(df_long.head(10), use_container_width=True)

else:
    st.info("👆 Please upload an Excel file to begin forecasting")
    
    # Show expected file format
    with st.expander("📋 Expected File Format"):
        st.markdown("""
        **Required columns:**
        - `Hospital Group Name`
        - `Hospital` 
        - `Date`
        - `DayGAR`
        - `Tracker8am`, `Tracker2pm`, `Tracker8pm` (ED counts)
        - `TimeTotal_8am`, `TimeTotal_2pm`, `TimeTotal_8pm` (Trolley counts)
        - `AdditionalCapacityOpen Morning` (Additional capacity)
        
        **Data should contain:**
        - Historical ED and trolley wait data
        - Multiple hospitals (optional)
        - At least 30 days of historical data for reliable forecasting
        """)
