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
import seaborn as sns

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

def prepare_data_enhanced(df):
    """Enhanced data preparation with capacity analysis"""
    # Rename columns
    df = df.rename(columns={
        'Tracker8am': 'ED_8am',
        'Tracker2pm': 'ED_2pm',
        'Tracker8pm': 'ED_8pm',
        'TimeTotal_8am': 'Trolley_8am',
        'TimeTotal_2pm': 'Trolley_2pm',
        'TimeTotal_8pm': 'Trolley_8pm',
        'AdditionalCapacityOpen Morning': 'Additional_Capacity_8am'
    })

    # ENHANCED CAPACITY HANDLING
    # Method 1: Forward fill Additional Capacity to 2pm and 8pm (assumes it stays open)
    df['Additional_Capacity_2pm'] = df['Additional_Capacity_8am']  # Assume stays open
    df['Additional_Capacity_8pm'] = df['Additional_Capacity_8am']  # Assume stays open
    
    # Method 2: Alternative - decay the capacity throughout the day
    # df['Additional_Capacity_2pm'] = df['Additional_Capacity_8am'] * 0.8  # 80% by afternoon
    # df['Additional_Capacity_8pm'] = df['Additional_Capacity_8am'] * 0.6  # 60% by evening

    # Fill missing values
    df['Additional_Capacity_8am'] = df.groupby(['Hospital'])['Additional_Capacity_8am'].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill')).fillna(0)
    df['Additional_Capacity_2pm'] = df['Additional_Capacity_2pm'].fillna(0)
    df['Additional_Capacity_8pm'] = df['Additional_Capacity_8pm'].fillna(0)

    # Reshape to long format with time-specific additional capacity
    df_long = pd.melt(
        df,
        id_vars=['Hospital Group Name', 'Hospital', 'Date', 'DayGAR'],
        value_vars=[
            'ED_8am', 'ED_2pm', 'ED_8pm', 
            'Trolley_8am', 'Trolley_2pm', 'Trolley_8pm',
            'Additional_Capacity_8am', 'Additional_Capacity_2pm', 'Additional_Capacity_8pm'
        ],
        var_name='Metric_Time',
        value_name='Value'
    )

    # Clean and handle missing values
    df_long = df_long.dropna(subset=['Value'])
    df_long['Value'] = pd.to_numeric(df_long['Value'], errors='coerce')
    df_long = df_long.dropna(subset=['Value'])

    # Extract metric and time
    df_long[['Metric', 'TimeLabel']] = df_long['Metric_Time'].str.extract(r'(\w+(?:_\w+)*)_([\d]+[ap]m)')
    
    # Separate Additional_Capacity from other metrics
    capacity_data = df_long[df_long['Metric'] == 'Additional_Capacity'].copy()
    main_data = df_long[df_long['Metric'].isin(['ED', 'Trolley'])].copy()
    
    # Merge capacity data back to main data
    if not capacity_data.empty:
        capacity_pivot = capacity_data.pivot_table(
            index=['Hospital', 'Date', 'TimeLabel'], 
            values='Value', 
            aggfunc='first'
        ).reset_index()
        capacity_pivot = capacity_pivot.rename(columns={'Value': 'Additional_Capacity'})
        
        main_data = main_data.merge(
            capacity_pivot, 
            on=['Hospital', 'Date', 'TimeLabel'], 
            how='left'
        )
    else:
        main_data['Additional_Capacity'] = 0
    
    main_data['Additional_Capacity'] = main_data['Additional_Capacity'].fillna(0)

    # Create datetime and time features
    time_map = {'8am': '08:00', '2pm': '14:00', '8pm': '20:00'}
    main_data['TimeStr'] = main_data['TimeLabel'].map(time_map)
    main_data['Datetime'] = pd.to_datetime(main_data['Date'].astype(str) + ' ' + main_data['TimeStr'])

    # Basic time features
    main_data['Hour'] = main_data['Datetime'].dt.hour
    main_data['DayOfWeek'] = main_data['Datetime'].dt.dayofweek
    main_data['DayOfMonth'] = main_data['Datetime'].dt.day
    main_data['Month'] = main_data['Datetime'].dt.month
    main_data['Quarter'] = main_data['Datetime'].dt.quarter
    main_data['WeekOfYear'] = main_data['Datetime'].dt.isocalendar().week
    main_data['IsWeekend'] = main_data['DayOfWeek'].isin([5, 6]).astype(int)
    main_data['IsMonday'] = (main_data['DayOfWeek'] == 0).astype(int)
    main_data['IsFriday'] = (main_data['DayOfWeek'] == 4).astype(int)
    
    # Cyclical encoding
    main_data['Hour_sin'] = np.sin(2 * np.pi * main_data['Hour'] / 24)
    main_data['Hour_cos'] = np.cos(2 * np.pi * main_data['Hour'] / 24)
    main_data['Day_sin'] = np.sin(2 * np.pi * main_data['DayOfWeek'] / 7)
    main_data['Day_cos'] = np.cos(2 * np.pi * main_data['DayOfWeek'] / 7)
    main_data['Month_sin'] = np.sin(2 * np.pi * main_data['Month'] / 12)
    main_data['Month_cos'] = np.cos(2 * np.pi * main_data['Month'] / 12)

    # Add Irish holidays
    calendar = IrishBankHolidays()
    try:
        holidays = calendar.holidays(start=main_data['Datetime'].min(), end=main_data['Datetime'].max() + timedelta(days=30))
        main_data['IsHoliday'] = main_data['Datetime'].dt.normalize().isin(holidays).astype(int)
    except:
        main_data['IsHoliday'] = 0

    # Seasonal and time indicators
    main_data['IsSummer'] = main_data['Month'].isin([6, 7, 8]).astype(int)
    main_data['IsWinter'] = main_data['Month'].isin([12, 1, 2]).astype(int)
    main_data['IsPeakHour'] = main_data['Hour'].isin([20]).astype(int)
    main_data['IsLowHour'] = main_data['Hour'].isin([8]).astype(int)

    # Hospital encoding
    main_data['Hospital_Code'] = main_data['Hospital'].astype('category').cat.codes

    return main_data

def add_capacity_features(df):
    """Add capacity-related features and cross-metric relationships"""
    df = df.copy()
    
    # Sort by hospital and datetime for proper feature engineering
    df = df.sort_values(['Hospital', 'Datetime']).reset_index(drop=True)
    
    # Create pivot table to get ED and Trolley values side by side
    metrics_pivot = df.pivot_table(
        index=['Hospital', 'Datetime', 'Additional_Capacity'], 
        columns='Metric', 
        values='Value', 
        aggfunc='first'
    ).reset_index()
    
    # Fill missing values
    if 'ED' in metrics_pivot.columns:
        metrics_pivot['ED'] = metrics_pivot['ED'].fillna(0)
    else:
        metrics_pivot['ED'] = 0
        
    if 'Trolley' in metrics_pivot.columns:
        metrics_pivot['Trolley'] = metrics_pivot['Trolley'].fillna(0)
    else:
        metrics_pivot['Trolley'] = 0
    
    # CAPACITY UTILIZATION FEATURES
    metrics_pivot['Total_Patients'] = metrics_pivot['ED'] + metrics_pivot['Trolley']
    
    # Capacity utilization ratios (avoid division by zero)
    metrics_pivot['ED_to_Capacity_Ratio'] = np.where(
        metrics_pivot['Additional_Capacity'] > 0,
        metrics_pivot['ED'] / metrics_pivot['Additional_Capacity'],
        0
    )
    
    metrics_pivot['Trolley_to_Capacity_Ratio'] = np.where(
        metrics_pivot['Additional_Capacity'] > 0,
        metrics_pivot['Trolley'] / metrics_pivot['Additional_Capacity'],
        0
    )
    
    metrics_pivot['Total_to_Capacity_Ratio'] = np.where(
        metrics_pivot['Additional_Capacity'] > 0,
        metrics_pivot['Total_Patients'] / metrics_pivot['Additional_Capacity'],
        0
    )
    
    # ED to Trolley relationship
    metrics_pivot['ED_to_Trolley_Ratio'] = np.where(
        metrics_pivot['Trolley'] > 0,
        metrics_pivot['ED'] / metrics_pivot['Trolley'],
        0
    )
    
    # Capacity pressure indicators
    metrics_pivot['Is_Over_Capacity'] = (metrics_pivot['Total_Patients'] > metrics_pivot['Additional_Capacity']).astype(int)
    metrics_pivot['Capacity_Pressure'] = np.maximum(0, metrics_pivot['Total_Patients'] - metrics_pivot['Additional_Capacity'])
    
    # Merge back to original dataframe
    merge_cols = ['Hospital', 'Datetime', 'Additional_Capacity']
    capacity_features = [
        'Total_Patients', 'ED_to_Capacity_Ratio', 'Trolley_to_Capacity_Ratio',
        'Total_to_Capacity_Ratio', 'ED_to_Trolley_Ratio', 'Is_Over_Capacity', 'Capacity_Pressure'
    ]
    
    # Add cross-metric features for ED predictions
    ed_data = df[df['Metric'] == 'ED'].copy()
    if not ed_data.empty:
        ed_enhanced = ed_data.merge(
            metrics_pivot[merge_cols + capacity_features + ['Trolley']],
            on=merge_cols, how='left'
        )
        ed_enhanced['Concurrent_Trolley'] = ed_enhanced['Trolley']
        ed_enhanced = ed_enhanced.drop('Trolley', axis=1)
    else:
        ed_enhanced = pd.DataFrame()
    
    # Add cross-metric features for Trolley predictions
    trolley_data = df[df['Metric'] == 'Trolley'].copy()
    if not trolley_data.empty:
        trolley_enhanced = trolley_data.merge(
            metrics_pivot[merge_cols + capacity_features + ['ED']],
            on=merge_cols, how='left'
        )
        trolley_enhanced['Concurrent_ED'] = trolley_enhanced['ED']
        trolley_enhanced = trolley_enhanced.drop('ED', axis=1)
    else:
        trolley_enhanced = pd.DataFrame()
    
    # Combine enhanced datasets
    if not ed_enhanced.empty and not trolley_enhanced.empty:
        enhanced_df = pd.concat([ed_enhanced, trolley_enhanced], ignore_index=True)
    elif not ed_enhanced.empty:
        enhanced_df = ed_enhanced
    elif not trolley_enhanced.empty:
        enhanced_df = trolley_enhanced
    else:
        enhanced_df = df.copy()
        # Add empty capacity features
        for feature in capacity_features + ['Concurrent_Trolley', 'Concurrent_ED']:
            enhanced_df[feature] = 0
    
    # Fill any remaining NaN values
    numeric_cols = enhanced_df.select_dtypes(include=[np.number]).columns
    enhanced_df[numeric_cols] = enhanced_df[numeric_cols].fillna(0)
    
    return enhanced_df

def create_capacity_correlation_analysis(df):
    """Create correlation analysis between capacity, ED, and trolley metrics"""
    st.subheader("🔍 Capacity Correlation Analysis")
    
    # Create pivot for analysis
    analysis_data = df.pivot_table(
        index=['Hospital', 'Datetime'], 
        columns='Metric', 
        values='Value', 
        aggfunc='first'
    ).reset_index()
    
    # Add capacity data
    capacity_data = df.groupby(['Hospital', 'Datetime'])['Additional_Capacity'].first().reset_index()
    analysis_data = analysis_data.merge(capacity_data, on=['Hospital', 'Datetime'], how='left')
    
    # Fill NaN values
    analysis_data = analysis_data.fillna(0)
    
    if 'ED' in analysis_data.columns and 'Trolley' in analysis_data.columns:
        # Calculate correlations
        corr_data = analysis_data[['ED', 'Trolley', 'Additional_Capacity']].corr()
        
        # Display correlation matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Correlation Matrix: ED, Trolley & Additional Capacity')
        st.pyplot(fig)
        
        # Key insights
        ed_trolley_corr = corr_data.loc['ED', 'Trolley']
        ed_capacity_corr = corr_data.loc['ED', 'Additional_Capacity']
        trolley_capacity_corr = corr_data.loc['Trolley', 'Additional_Capacity']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("ED ↔ Trolley Correlation", f"{ed_trolley_corr:.3f}")
        with col2:
            st.metric("ED ↔ Capacity Correlation", f"{ed_capacity_corr:.3f}")
        with col3:
            st.metric("Trolley ↔ Capacity Correlation", f"{trolley_capacity_corr:.3f}")
        
        # Time series plot
        if len(analysis_data) > 0:
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=('ED Counts', 'Trolley Counts', 'Additional Capacity'),
                shared_xaxes=True
            )
            
            fig.add_trace(
                go.Scatter(x=analysis_data['Datetime'], y=analysis_data['ED'], 
                          name='ED', line=dict(color='blue')),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=analysis_data['Datetime'], y=analysis_data['Trolley'], 
                          name='Trolley', line=dict(color='red')),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=analysis_data['Datetime'], y=analysis_data['Additional_Capacity'], 
                          name='Additional Capacity', line=dict(color='green')),
                row=3, col=1
            )
            
            fig.update_layout(height=600, title_text="Time Series: ED, Trolley & Capacity")
            st.plotly_chart(fig, use_container_width=True)

def forecast_additional_capacity(historical_data, future_df, features):
    """Forecast additional capacity based on patterns"""
    capacity_data = historical_data.groupby(['Hospital', 'Datetime'])['Additional_Capacity'].first().reset_index()
    capacity_data = capacity_data[capacity_data['Additional_Capacity'] > 0]  # Only days with additional capacity
    
    if len(capacity_data) < 5:
        # If insufficient data, use historical average
        avg_capacity = historical_data['Additional_Capacity'].mean()
        return [avg_capacity] * len(future_df)
    
    # Add time features to capacity data
    capacity_data['DayOfWeek'] = capacity_data['Datetime'].dt.dayofweek
    capacity_data['Month'] = capacity_data['Datetime'].dt.month
    capacity_data['IsWeekend'] = capacity_data['DayOfWeek'].isin([5, 6]).astype(int)
    
    # Simple pattern-based forecasting for capacity
    # Use day-of-week patterns
    dow_patterns = capacity_data.groupby('DayOfWeek')['Additional_Capacity'].mean().to_dict()
    
    predictions = []
    for _, row in future_df.iterrows():
        future_dow = pd.to_datetime(row['Datetime']).dayofweek
        if future_dow in dow_patterns:
            pred_capacity = dow_patterns[future_dow]
        else:
            pred_capacity = capacity_data['Additional_Capacity'].mean()
        predictions.append(pred_capacity)
    
    return predictions

# Enhanced forecasting function
def forecast_with_lags_enhanced(model, historical_data, future_df, features):
    """Enhanced forecast function with capacity features"""
    # Get last values for initial lags
    last_values = historical_data['Value'].tail(7).values
    
    predictions = []
    current_lags = list(reversed(last_values))
    
    # Calculate rolling statistics
    historical_mean_3 = historical_data['Value'].tail(3).mean()
    historical_mean_7 = historical_data['Value'].tail(7).mean()
    
    for idx, row in future_df.iterrows():
        try:
            feature_values = []
            
            for feature in features:
                if feature.startswith('Lag_'):
                    lag_num = int(feature.split('_')[1]) - 1
                    if lag_num < len(current_lags):
                        feature_values.append(current_lags[lag_num])
                    else:
                        feature_values.append(0)
                elif feature == 'Rolling_Mean_3':
                    feature_values.append(historical_mean_3)
                elif feature == 'Rolling_Mean_7':
                    feature_values.append(historical_mean_7)
                else:
                    if feature in row:
                        feature_values.append(row[feature])
                    else:
                        feature_values.append(0)
            
            # Make prediction
            feature_vector = np.array(feature_values).reshape(1, -1)
            pred = model.predict(feature_vector)[0]
            pred = max(0, pred)
            predictions.append(pred)
            
            # Update lags
            current_lags = [pred] + current_lags[:6]
            
            # Update rolling statistics
            if len(predictions) >= 3:
                historical_mean_3 = np.mean(predictions[-3:])
            if len(predictions) >= 7:
                historical_mean_7 = np.mean(predictions[-7:])
                
        except Exception as e:
            st.error(f"Error in prediction step {idx}: {e}")
            predictions.append(historical_data['Value'].mean())
    
    return predictions

def add_lag_features_smart_enhanced(df, min_data_threshold=20):
    """Enhanced lag features with capacity considerations"""
    df = df.copy()
    
    max_safe_lag = min(7, len(df) // 4)
    
    if max_safe_lag < 1:
        st.warning(f"Very limited data ({len(df)} records). Using minimal features.")
        return df, []
    
    lag_features = []
    
    # Standard lag features
    for i in range(1, max_safe_lag + 1):
        lag_col = f'Lag_{i}'
        df[lag_col] = df['Value'].shift(i)
        lag_features.append(lag_col)
    
    # Rolling features
    if len(df) >= 6:
        df['Rolling_Mean_3'] = df['Value'].rolling(window=min(3, len(df)//2), min_periods=1).mean()
        lag_features.append('Rolling_Mean_3')
        
        if len(df) >= 14:
            df['Rolling_Mean_7'] = df['Value'].rolling(window=min(7, len(df)//2), min_periods=1).mean()
            lag_features.append('Rolling_Mean_7')
    
    # Capacity-related rolling features
    if 'Additional_Capacity' in df.columns and len(df) >= 6:
        df['Rolling_Capacity_Mean_3'] = df['Additional_Capacity'].rolling(window=min(3, len(df)//2), min_periods=1).mean()
        lag_features.append('Rolling_Capacity_Mean_3')
    
    # Fill NaN values
    for feature in lag_features:
        df[feature] = df[feature].fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return df, lag_features

def create_future_dates_enhanced(last_date, hospital, hospital_code, additional_capacity, days=7):
    """Enhanced future dates with forecasted additional capacity"""
    future_dates = []
    times = ['08:00', '14:00', '20:00']
    
    for day in range(1, days + 1):
        future_date = last_date + timedelta(days=day)
        for time_str in times:
            future_datetime = pd.to_datetime(f"{future_date.date()} {time_str}")
            
            # Basic features
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
            
            # Holiday check
            is_holiday = 0
            try:
                calendar = IrishBankHolidays()
                holidays = calendar.holidays(start=future_datetime, end=future_datetime)
                is_holiday = int(future_datetime.normalize() in holidays)
            except:
                pass
            
            # Seasonal indicators
            is_summer = int(month in [6, 7, 8])
            is_winter = int(month in [12, 1, 2])
            is_peak_hour = int(hour == 20)
            is_low_hour = int(hour == 8)
            
            # For forecasting, we'll estimate additional capacity based on patterns
            # This will be updated in the main forecasting loop
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
                'IsHoliday': is_holiday,
                'IsSummer': is_summer,
                'IsWinter': is_winter,
                'IsPeakHour': is_peak_hour,
                'IsLowHour': is_low_hour,
                'Hospital_Code': hospital_code,
                'Additional_Capacity': additional_capacity  # Will be updated with forecasted values
            })
    
    return pd.DataFrame(future_dates)

# --- Streamlit UI ---
st.title("🇮🇪 Enhanced Emergency Department Forecasting (Ireland)")
st.markdown("Upload your ED Excel file for comprehensive forecasting with capacity analysis")

# Sidebar controls
st.sidebar.header("Forecast Settings")
forecast_days = st.sidebar.slider("Forecast Days", 1, 14, 7)
capacity_method = st.sidebar.selectbox(
    "Additional Capacity Handling",
    ["Forward Fill (stays open all day)", "Decay (reduces during day)", "Pattern-based forecast"],
    help="How to handle additional capacity for 2pm and 8pm when only 8am data is available"
)

uploaded_file = st.file_uploader("Upload your ED Excel File", type=["xlsx"])

if uploaded_file:
    try:
        # Load data
        df = pd.read_excel(uploaded_file)
        df_long = prepare_data_enhanced(df)
        
        st.success(f"✅ Data loaded successfully! {len(df_long)} records found.")
        
        # Add capacity analysis
        create_capacity_correlation_analysis(df_long)
        
        # Enhanced data with capacity features
        df_enhanced = add_capacity_features(df_long)
        
        # Show enhanced data summary
        st.subheader("📊 Enhanced Data Summary")
        capacity_summary = df_enhanced.groupby(['Hospital', 'Metric']).agg({
            'Value': ['count', 'mean', 'std'],
            'Additional_Capacity': 'mean',
            'Total_to_Capacity_Ratio': 'mean'
        }).round(2)
        st.dataframe(capacity_summary, use_container_width=True)
        
        # Hospital selection
        hospitals = sorted(df_enhanced['Hospital'].unique())
        hospital_option = st.sidebar.selectbox("Select Hospital:", ["All Hospitals"] + hospitals)
        
        # Run forecast
        if st.sidebar.button("🚀 Run Enhanced Forecast", type="primary"):
            st.header("📊 Enhanced Forecast Results")
            
            selected_hospitals = hospitals if hospital_option == "All Hospitals" else [hospital_option]
            
            # Enhanced base features
            base_features = [
                'Hour', 'DayOfWeek', 'DayOfMonth', 'Month', 'Quarter', 'WeekOfYear',
                'IsWeekend', 'IsMonday', 'IsFriday',
                'Hour_sin', 'Hour_cos', 'Day_sin', 'Day_cos', 'Month_sin', 'Month_cos',
                'IsHoliday', 'IsSummer', 'IsWinter', 'IsPeakHour', 'IsLowHour',
                'Hospital_Code', 'Additional_Capacity',
                # New capacity features
                'Total_to_Capacity_Ratio', 'ED_to_Capacity_Ratio', 'Trolley_to_Capacity_Ratio',
                'ED_to_Trolley_Ratio', 'Is_Over_Capacity', 'Capacity_Pressure'
            ]
            
            for hospital in selected_hospitals:
                st.subheader(f"🏥 {hospital}")
                
                hospital_data = df_enhanced[df_enhanced['Hospital'] == hospital].copy()
                hospital_code = hospital_data['Hospital_Code'].iloc[0]
                additional_capacity = hospital_data['Additional_Capacity'].fillna(0).iloc[0]
                last_date = hospital_data['Datetime'].max().date()
                
                for metric in ['ED', 'Trolley']:
                    metric_data = hospital_data[hospital_data['Metric'] == metric].copy()
                    metric_data = metric_data.sort_values('Datetime').reset_index(drop=True)
                    
                    if len(metric_data) < 10:
                        st.warning(f"⚠️ Insufficient data for {metric} at {hospital}")
                        continue
                    
                    # Add cross-metric features
                    if metric == 'ED' and 'Concurrent_Trolley' in metric_data.columns:
                        base_features_enhanced = base_features + ['Concurrent_Trolley']
                    elif metric == 'Trolley' and 'Concurrent_ED' in metric_data.columns:
                        base_features_enhanced = base_features + ['Concurrent_ED']
                    else:
                        base_features_enhanced = base_features
                    
                    # Add lag features
                    metric_data, lag_features = add_lag_features_smart_enhanced(metric_data)
                    
                    # Combine all features
                    all_features = base_features_enhanced + lag_features
                    available_features = [f for f in all_features if f in metric_data.columns]
                    
                    # Prepare training data
                    training_data = metric_data.dropna(subset=available_features + ['Value'])
                    
                    if len(training_data) < 5:
                        st.warning(f"⚠️ After preprocessing, insufficient data for {metric} at {hospital}")
                        continue
                    
                    st.info(f"Processing {metric} for {hospital} ({len(training_data)} records) with {len(available_features)} features")
                    
                    # Train model
                    X = training_data[available_features]
                    y = training_data['Value']
                    
                    # Split data
                    if len(X) < 20:
                        X_train, X_test = X, X.tail(min(3, len(X)))
                        y_train, y_test = y, y.tail(min(3, len(y)))
                    else:
                        split_idx = int(len(X) * 0.8)
                        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
                    
                    # Enhanced model configuration
                    model = lgb.LGBMRegressor(
                        n_estimators=min(500, len(X_train) * 3),
                        learning_rate=0.05,
                        max_depth=min(8, len(available_features) // 2 + 2),
                        num_leaves=min(50, 2 ** min(8, len(available_features) // 2 + 2) - 1),
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_alpha=0.1,
                        reg_lambda=0.1,
                        verbose=-1,
                        random_state=42,
                        force_col_wise=True
                    )
                    
                    try:
                        model.fit(X_train, y_train)
                        
                        # Model performance
                        y_pred_test = model.predict(X_test)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                        
                        # Feature importance
                        feature_importance = pd.DataFrame({
                            'feature': available_features,
                            'importance': model.feature_importances_
                        }).sort_values('importance', ascending=False)
                        
                        # Create future dates
                        future_df = create_future_dates_enhanced(
                            pd.to_datetime(last_date), 
                            hospital, 
                            hospital_code, 
                            additional_capacity, 
                            days=forecast_days
                        )
                        
                        # Forecast additional capacity if using pattern-based method
                        if capacity_method == "Pattern-based forecast":
                            forecasted_capacity = forecast_additional_capacity(training_data, future_df, available_features)
                            future_df['Additional_Capacity'] = forecasted_capacity
                        elif capacity_method == "Decay (reduces during day)":
                            # Apply decay based on time of day
                            decay_factors = {'08:00': 1.0, '14:00': 0.8, '20:00': 0.6}
                            future_df['Additional_Capacity'] = future_df.apply(
                                lambda row: additional_capacity * decay_factors.get(
                                    pd.to_datetime(row['Datetime']).strftime('%H:%M'), 1.0
                                ), axis=1
                            )
                        
                        # Add capacity-related features to future data
                        # Estimate cross-metric values for future (simplified approach)
                        if metric == 'ED':
                            # Estimate future trolley values based on historical relationship
                            avg_trolley = training_data.get('Concurrent_Trolley', pd.Series([0])).mean()
                            future_df['Concurrent_Trolley'] = avg_trolley
                        elif metric == 'Trolley':
                            # Estimate future ED values based on historical relationship
                            avg_ed = training_data.get('Concurrent_ED', pd.Series([0])).mean()
                            future_df['Concurrent_ED'] = avg_ed
                        
                        # Calculate capacity features for future data
                        for _, row in future_df.iterrows():
                            capacity = row['Additional_Capacity']
                            if capacity > 0:
                                # Use average historical values for cross-metric estimation
                                if metric == 'ED':
                                    est_ed = training_data['Value'].mean()
                                    est_trolley = future_df.loc[future_df.index[0], 'Concurrent_Trolley']
                                else:
                                    est_ed = future_df.loc[future_df.index[0], 'Concurrent_ED']
                                    est_trolley = training_data['Value'].mean()
                                
                                total_patients = est_ed + est_trolley
                                
                                future_df.loc[row.name, 'Total_to_Capacity_Ratio'] = total_patients / capacity if capacity > 0 else 0
                                future_df.loc[row.name, 'ED_to_Capacity_Ratio'] = est_ed / capacity if capacity > 0 else 0
                                future_df.loc[row.name, 'Trolley_to_Capacity_Ratio'] = est_trolley / capacity if capacity > 0 else 0
                                future_df.loc[row.name, 'ED_to_Trolley_Ratio'] = est_ed / est_trolley if est_trolley > 0 else 0
                                future_df.loc[row.name, 'Is_Over_Capacity'] = int(total_patients > capacity)
                                future_df.loc[row.name, 'Capacity_Pressure'] = max(0, total_patients - capacity)
                            else:
                                # No additional capacity
                                future_df.loc[row.name, 'Total_to_Capacity_Ratio'] = 0
                                future_df.loc[row.name, 'ED_to_Capacity_Ratio'] = 0
                                future_df.loc[row.name, 'Trolley_to_Capacity_Ratio'] = 0
                                future_df.loc[row.name, 'ED_to_Trolley_Ratio'] = 0
                                future_df.loc[row.name, 'Is_Over_Capacity'] = 0
                                future_df.loc[row.name, 'Capacity_Pressure'] = 0
                        
                        # Fill any missing features with 0
                        for feature in available_features:
                            if feature not in future_df.columns:
                                future_df[feature] = 0
                        
                        # Generate forecasts
                        predictions = forecast_with_lags_enhanced(model, training_data, future_df, available_features)
                        future_df['Predicted'] = predictions
                        
                        # Display enhanced metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric(f"{metric} RMSE", f"{rmse:.2f}")
                        with col2:
                            st.metric("Training Records", f"{len(X_train)}")
                        with col3:
                            st.metric(f"Last {metric} Value", f"{training_data['Value'].iloc[-1]:.0f}")
                        with col4:
                            avg_capacity_ratio = training_data.get('Total_to_Capacity_Ratio', pd.Series([0])).mean()
                            st.metric("Avg Capacity Ratio", f"{avg_capacity_ratio:.2f}")
                        
                        # Enhanced plotting
                        fig = plot_forecasts_enhanced(
                            training_data.tail(21),
                            future_df,
                            metric,
                            hospital
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Feature importance chart
                        with st.expander(f"📊 {metric} Feature Importance"):
                            top_features = feature_importance.head(10)
                            fig_importance = px.bar(
                                top_features, 
                                x='importance', 
                                y='feature',
                                orientation='h',
                                title=f'Top 10 Features for {metric} Prediction'
                            )
                            fig_importance.update_layout(height=400)
                            st.plotly_chart(fig_importance, use_container_width=True)
                        
                        # Enhanced forecast table
                        with st.expander(f"📋 {metric} Enhanced Forecast Details"):
                            forecast_display = future_df[['Datetime', 'Predicted', 'Additional_Capacity', 'Total_to_Capacity_Ratio']].copy()
                            forecast_display['Predicted'] = forecast_display['Predicted'].round(1)
                            forecast_display['Date'] = forecast_display['Datetime'].dt.date
                            forecast_display['Time'] = forecast_display['Datetime'].dt.strftime('%I:%M %p')
                            forecast_display['Capacity_Utilization'] = (forecast_display['Total_to_Capacity_Ratio'] * 100).round(1)
                            
                            display_cols = ['Date', 'Time', 'Predicted', 'Additional_Capacity', 'Capacity_Utilization']
                            st.dataframe(
                                forecast_display[display_cols].rename(columns={
                                    'Predicted': f'{metric} Forecast',
                                    'Additional_Capacity': 'Additional Capacity',
                                    'Capacity_Utilization': 'Capacity Use %'
                                }),
                                use_container_width=True
                            )
                        
                        # Download enhanced CSV
                        csv_data = future_df[['Datetime', 'Hospital', 'Predicted', 'Additional_Capacity', 'Total_to_Capacity_Ratio']].copy()
                        csv_data['Metric'] = metric
                        csv_data['Capacity_Utilization_Percent'] = (csv_data['Total_to_Capacity_Ratio'] * 100).round(1)
                        
                        st.download_button(
                            f"📥 Download {metric} Enhanced Forecast CSV",
                            csv_data.to_csv(index=False),
                            file_name=f"{hospital}_{metric}_enhanced_forecast.csv",
                            mime="text/csv",
                            key=f"{hospital}_{metric}_enhanced_download"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error training model for {metric} at {hospital}: {str(e)}")
                        st.info("This might be due to insufficient data or data quality issues.")
                    
                    st.divider()
        
        # Enhanced insights
        add_enhanced_forecasting_insights()
        
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please check that your Excel file contains the required columns and data format.")

else:
    st.info("👆 Please upload an Excel file to begin enhanced forecasting")
    
    with st.expander("📋 Expected File Format & Enhancements"):
        st.markdown("""
        **Required columns (same as before):**
        - `Hospital Group Name`, `Hospital`, `Date`, `DayGAR`
        - `Tracker8am`, `Tracker2pm`, `Tracker8pm` (ED counts)
        - `TimeTotal_8am`, `TimeTotal_2pm`, `TimeTotal_8pm` (Trolley counts)  
        - `AdditionalCapacityOpen Morning` (Additional capacity)
        
        **🆕 New Enhanced Features:**
        
        **1. Additional Capacity Handling:**
        - Duplicates 8am capacity value to 2pm and 8pm (configurable)
        - Option for decay throughout the day
        - Pattern-based capacity forecasting
        
        **2. Capacity Utilization Features:**
        - ED/Trolley to capacity ratios
        - Over-capacity indicators
        - Capacity pressure metrics
        
        **3. Cross-Metric Relationships:**
        - ED predictions consider concurrent trolley counts
        - Trolley predictions consider concurrent ED counts
        - Total patient load analysis
        
        **4. Enhanced Analysis:**
        - Correlation analysis between ED, Trolley, and Capacity
        - Feature importance rankings
        - Capacity utilization forecasts
        """)

def plot_forecasts_enhanced(historical_data, forecast_data, metric_name, hospital_name):
    """Create enhanced interactive plotly chart with capacity information"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'{metric_name} Forecast', 'Additional Capacity & Utilization'),
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=historical_data['Datetime'],
        y=historical_data['Value'],
        mode='lines+markers',
        name='Historical',
        line=dict(color='blue'),
        marker=dict(size=4)
    ), row=1, col=1)
    
    # Forecast data
    fig.add_trace(go.Scatter(
        x=forecast_data['Datetime'],
        y=forecast_data['Predicted'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='red', dash='dash'),
        marker=dict(size=6, symbol='diamond')
    ), row=1, col=1)
    
    # Additional capacity
    if 'Additional_Capacity' in historical_data.columns:
        fig.add_trace(go.Scatter(
            x=historical_data['Datetime'],
            y=historical_data['Additional_Capacity'],
            mode='lines',
            name='Historical Capacity',
            line=dict(color='green', width=2),
            opacity=0.7
        ), row=2, col=1)
    
    # Forecasted capacity
    fig.add_trace(go.Scatter(
        x=forecast_data['Datetime'],
        y=forecast_data['Additional_Capacity'],
        mode='lines',
        name='Forecasted Capacity',
        line=dict(color='green', dash='dot', width=2),
        opacity=0.7
    ), row=2, col=1)
    
    # Capacity utilization
    if 'Total_to_Capacity_Ratio' in forecast_data.columns:
        fig.add_trace(go.Scatter(
            x=forecast_data['Datetime'],
            y=forecast_data['Total_to_Capacity_Ratio'] * 100,
            mode='lines+markers',
            name='Capacity Utilization %',
            line=dict(color='orange'),
            yaxis='y3'
        ), row=2, col=1)
    
    # Add vertical line
    last_historical_date = historical_data['Datetime'].max()
    fig.add_shape(
        type="line",
        x0=last_historical_date,
        x1=last_historical_date,
        y0=0, y1=1,
        yref="paper",
        line=dict(color="gray", width=2, dash="dot")
    )
    
    fig.update_layout(
        title=f'Enhanced {metric_name} Forecast - {hospital_name}',
        height=700,
        hovermode='x unified',
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text=f'{metric_name} Count', row=1, col=1)
    fig.update_yaxes(title_text='Additional Capacity', row=2, col=1)
    
    return fig

def add_enhanced_forecasting_insights():
    """Add enhanced insights section"""
    with st.expander("💡 Enhanced Forecasting Insights & Tips", expanded=False):
        st.subheader("🆕 New Capacity Features")
        st.markdown("""
        **Additional Capacity Handling:**
        * **Forward Fill**: Assumes additional capacity stays open all day
        * **Decay**: Reduces capacity throughout the day (100% → 80% → 60%)
        * **Pattern-based**: Forecasts capacity based on historical patterns
        
        **Capacity Utilization Metrics:**
        * **Utilization Ratio**: Total patients / Additional capacity
        * **Over-capacity**: When patient load exceeds additional capacity
        * **Capacity Pressure**: How many patients above capacity threshold
        """)
        
        st.subheader("🔍 Cross-Metric Analysis")
        st.markdown("""
        * **ED ↔ Trolley Correlation**: Shows relationship between ED attendance and trolley waits
        * **Concurrent Features**: ED predictions consider trolley levels and vice versa
        * **Total Patient Load**: Combined ED + Trolley for capacity planning
        """)
        
        st.subheader("📊 Enhanced Results")
        st.markdown("""
        * **Feature Importance**: Shows which factors most influence predictions
        * **Capacity Forecasting**: Predicts both patient numbers AND capacity utilization
        * **Time-series Visualization**: Shows capacity alongside patient forecasts
        * **Utilization %**: Helps identify potential overcrowding periods
        """)
        
        st.subheader("💡 Recommendations")
        st.markdown("""
        * **High Correlation (>0.7)**: Strong relationship between metrics - good for cross-prediction
        * **Over-capacity Periods**: Plan additional resources when utilization >100%
        * **Pattern Recognition**: Use day-of-week and seasonal patterns for capacity planning
        * **Early Warning**: Monitor forecasted capacity pressure for proactive management
        """)
