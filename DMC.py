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
    """Prepare and feature engineer the data with improved handling"""
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
    df['Additional_Capacity'] = df['Additional_Capacity'].fillna(0)

    # Reshape to long format
    df_long = pd.melt(
        df,
        id_vars=['Hospital Group Name', 'Hospital', 'Date', 'DayGAR', 'Additional_Capacity'],
        value_vars=['ED_8am', 'ED_2pm', 'ED_8pm', 'Trolley_8am', 'Trolley_2pm', 'Trolley_8pm'],
        var_name='Metric_Time',
        value_name='Value'
    )

    # Clean and handle missing values early
    df_long = df_long.dropna(subset=['Value'])
    df_long['Value'] = pd.to_numeric(df_long['Value'], errors='coerce')
    df_long = df_long.dropna(subset=['Value'])

    df_long[['Metric', 'TimeLabel']] = df_long['Metric_Time'].str.extract(r'(\w+)_([\d]+[ap]m)')
    time_map = {'8am': '08:00', '2pm': '14:00', '8pm': '20:00'}
    df_long['TimeStr'] = df_long['TimeLabel'].map(time_map)
    df_long['Datetime'] = pd.to_datetime(df_long['Date'].astype(str) + ' ' + df_long['TimeStr'])

    # Basic time features
    df_long['Hour'] = df_long['Datetime'].dt.hour
    df_long['DayOfWeek'] = df_long['Datetime'].dt.dayofweek
    df_long['DayOfMonth'] = df_long['Datetime'].dt.day
    df_long['Month'] = df_long['Datetime'].dt.month
    df_long['Quarter'] = df_long['Datetime'].dt.quarter
    df_long['WeekOfYear'] = df_long['Datetime'].dt.isocalendar().week
    df_long['IsWeekend'] = df_long['DayOfWeek'].isin([5, 6]).astype(int)
    df_long['IsMonday'] = (df_long['DayOfWeek'] == 0).astype(int)
    df_long['IsFriday'] = (df_long['DayOfWeek'] == 4).astype(int)
    
    # Cyclical encoding for time features
    df_long['Hour_sin'] = np.sin(2 * np.pi * df_long['Hour'] / 24)
    df_long['Hour_cos'] = np.cos(2 * np.pi * df_long['Hour'] / 24)
    df_long['Day_sin'] = np.sin(2 * np.pi * df_long['DayOfWeek'] / 7)
    df_long['Day_cos'] = np.cos(2 * np.pi * df_long['DayOfWeek'] / 7)
    df_long['Month_sin'] = np.sin(2 * np.pi * df_long['Month'] / 12)
    df_long['Month_cos'] = np.cos(2 * np.pi * df_long['Month'] / 12)

    # Add Irish holidays
    calendar = IrishBankHolidays()
    try:
        holidays = calendar.holidays(start=df_long['Datetime'].min(), end=df_long['Datetime'].max() + timedelta(days=30))
        df_long['IsHoliday'] = df_long['Datetime'].dt.normalize().isin(holidays).astype(int)
    except:
        df_long['IsHoliday'] = 0

    # Seasonal indicators
    df_long['IsSummer'] = df_long['Month'].isin([6, 7, 8]).astype(int)
    df_long['IsWinter'] = df_long['Month'].isin([12, 1, 2]).astype(int)
    
    # Peak hour indicators
    df_long['IsPeakHour'] = df_long['Hour'].isin([20]).astype(int)
    df_long['IsLowHour'] = df_long['Hour'].isin([8]).astype(int)

    # Encode hospital
    df_long['Hospital_Code'] = df_long['Hospital'].astype('category').cat.codes

    return df_long

def add_lag_features_smart(df, min_data_threshold=20):
    """Add lag features intelligently based on available data"""
    df = df.copy()
    
    # Determine maximum safe lag based on data size
    max_safe_lag = min(7, len(df) // 4)  # Use at most 1/4 of data for lags
    
    if max_safe_lag < 1:
        st.warning(f"Very limited data ({len(df)} records). Using minimal features.")
        return df, []
    
    lag_features = []
    
    # Add lag features progressively
    for i in range(1, max_safe_lag + 1):
        lag_col = f'Lag_{i}'
        df[lag_col] = df['Value'].shift(i)
        lag_features.append(lag_col)
    
    # Add rolling features if we have enough data
    if len(df) >= 6:
        df['Rolling_Mean_3'] = df['Value'].rolling(window=min(3, len(df)//2), min_periods=1).mean()
        lag_features.append('Rolling_Mean_3')
        
        if len(df) >= 14:
            df['Rolling_Mean_7'] = df['Value'].rolling(window=min(7, len(df)//2), min_periods=1).mean()
            lag_features.append('Rolling_Mean_7')
    
    # Fill NaN values with forward fill, then backward fill, then 0
    for feature in lag_features:
        df[feature] = df[feature].fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return df, lag_features

def create_future_dates(last_date, hospital, hospital_code, additional_capacity, days=7):
    """Create future dates for forecasting"""
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
                'IsHoliday': is_holiday,
                'IsSummer': is_summer,
                'IsWinter': is_winter,
                'IsPeakHour': is_peak_hour,
                'IsLowHour': is_low_hour,
                'Hospital_Code': hospital_code,
                'Additional_Capacity': additional_capacity
            })
    
    return pd.DataFrame(future_dates)

def forecast_with_lags(model, historical_data, future_df, features):
    """Enhanced forecast function with better lag handling"""
    # Get last values for initial lags
    last_values = historical_data['Value'].tail(7).values
    
    predictions = []
    current_lags = list(reversed(last_values))  # Most recent first
    
    # Calculate rolling statistics from historical data
    historical_mean_3 = historical_data['Value'].tail(3).mean()
    historical_mean_7 = historical_data['Value'].tail(7).mean()
    
    for idx, row in future_df.iterrows():
        try:
            # Start with base features
            feature_values = []
            
            for feature in features:
                if feature.startswith('Lag_'):
                    # Handle lag features
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
                    # Regular features
                    if feature in row:
                        feature_values.append(row[feature])
                    else:
                        feature_values.append(0)
            
            # Make prediction
            feature_vector = np.array(feature_values).reshape(1, -1)
            pred = model.predict(feature_vector)[0]
            pred = max(0, pred)  # Ensure non-negative
            predictions.append(pred)
            
            # Update lags for next prediction
            current_lags = [pred] + current_lags[:6]
            
            # Update rolling statistics
            if len(predictions) >= 3:
                historical_mean_3 = np.mean(predictions[-3:])
            if len(predictions) >= 7:
                historical_mean_7 = np.mean(predictions[-7:])
            
        except Exception as e:
            st.error(f"Error in prediction step {idx}: {e}")
            predictions.append(historical_data['Value'].mean())  # Fallback to mean
    
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
    
    fig.add_shape(
        type="line",
        x0=last_historical_date,
        x1=last_historical_date,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="gray", width=2, dash="dot"),
    )
    
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
        st.subheader("Data Requirements")
        st.markdown("""
        For accurate forecasting, you need:
        * **Minimum 30 records** per hospital-metric combination
        * **Consistent time intervals** (8am, 2pm, 8pm readings)
        * **Recent data** (within last 6 months ideally)
        * **Complete records** (avoid too many missing values)
        """)
        
        st.subheader("Understanding Your Results")
        st.markdown("""
        * **RMSE (Root Mean Square Error)**: Lower is better. This shows average prediction error.
        * **Historical vs Forecast**: The chart shows your data pattern and predicted future values.
        * **Validation**: Shows how well the model predicts on recent historical data.
        """)

# --- Streamlit UI ---
st.title("🇮🇪 Emergency Department Forecasting (Ireland)")
st.markdown("Upload your ED Excel file, select hospital(s), and generate 7-day forecasts")

# Add forecast days control
forecast_days = st.sidebar.slider("Forecast Days", 1, 14, 7)

uploaded_file = st.file_uploader("Upload your ED Excel File", type=["xlsx"])

if uploaded_file:
    try:
        # Load and prepare data
        df = pd.read_excel(uploaded_file)
        df_long = prepare_data(df)
        
        # Show data info
        st.success(f"✅ Data loaded successfully! {len(df_long)} records found.")
        
        # Get unique hospitals for selection
        hospitals = sorted(df_long['Hospital'].unique())
        
        # Show data summary
        data_summary = df_long.groupby(['Hospital', 'Metric']).size().reset_index(name='Records')
        st.subheader("📊 Data Summary by Hospital & Metric")
        st.dataframe(data_summary, use_container_width=True)
        
        # Hospital selection
        st.sidebar.header("Forecast Settings")
        hospital_option = st.sidebar.selectbox(
            "Select Hospital:",
            options=["All Hospitals"] + hospitals
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
            
            # Base features (without lags)
            base_features = [
                'Hour', 'DayOfWeek', 'DayOfMonth', 'Month', 'Quarter', 'WeekOfYear',
                'IsWeekend', 'IsMonday', 'IsFriday',
                'Hour_sin', 'Hour_cos', 'Day_sin', 'Day_cos', 'Month_sin', 'Month_cos',
                'IsHoliday', 'IsSummer', 'IsWinter', 'IsPeakHour', 'IsLowHour',
                'Hospital_Code', 'Additional_Capacity'
            ]
            
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
                    metric_data = metric_data.sort_values('Datetime').reset_index(drop=True)
                    
                    # Check if we have sufficient data
                    if len(metric_data) < 10:
                        st.warning(f"⚠️ Insufficient data for {metric} at {hospital} ({len(metric_data)} records). Need at least 10 records.")
                        continue
                    
                    st.info(f"Processing {metric} for {hospital} ({len(metric_data)} records)")
                    
                    # Add lag features intelligently
                    metric_data, lag_features = add_lag_features_smart(metric_data)
                    
                    # Combine all features
                    all_features = base_features + lag_features
                    
                    # Remove any features that don't exist in the data
                    available_features = [f for f in all_features if f in metric_data.columns]
                    
                    # Final check for training data
                    training_data = metric_data.dropna(subset=available_features + ['Value'])
                    
                    if len(training_data) < 5:
                        st.warning(f"⚠️ After preprocessing, insufficient data for {metric} at {hospital} ({len(training_data)} records). Need at least 5 records.")
                        continue
                    
                    # Train model
                    X = training_data[available_features]
                    y = training_data['Value']
                    
                    # Use simple split for small datasets
                    if len(X) < 20:
                        # Use all data for training with small datasets
                        X_train, X_test = X, X.tail(min(3, len(X)))
                        y_train, y_test = y, y.tail(min(3, len(y)))
                    else:
                        # Use time-based split for larger datasets
                        split_idx = int(len(X) * 0.8)
                        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
                    
                    # Simple model configuration for small datasets
                    model = lgb.LGBMRegressor(
                        n_estimators=min(200, len(X_train) * 2),  # Adapt to data size was 100.......................................................................
                        learning_rate=0.1,
                        max_depth=min(6, len(available_features) // 2 + 1),
                        num_leaves=min(31, 2 ** min(6, len(available_features) // 2 + 1) - 1),
                        subsample=0.8,
                        colsample_bytree=0.8,
                        verbose=-1,
                        random_state=42,
                        force_col_wise=True
                    )
                    
                    try:
                        model.fit(X_train, y_train)
                        
                        # Calculate model performance
                        y_pred_test = model.predict(X_test)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                        
                        # Create future dates
                        future_df = create_future_dates(
                            pd.to_datetime(last_date), 
                            hospital, 
                            hospital_code, 
                            additional_capacity, 
                            days=forecast_days
                        )
                        
                        # Generate forecasts
                        predictions = forecast_with_lags(model, training_data, future_df, available_features)
                        future_df['Predicted'] = predictions
                        
                        # Display metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(f"{metric} RMSE", f"{rmse:.2f}")
                        with col2:
                            st.metric(f"Training Records", f"{len(X_train)}")
                        with col3:
                            st.metric(f"Last {metric} Value", f"{training_data['Value'].iloc[-1]:.0f}")
                        
                        # Create and display plot
                        fig = plot_forecasts(
                            training_data.tail(21),  # Show last 21 points (1 week at 3x/day)
                            future_df,
                            metric,
                            hospital
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
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
                        
                    except Exception as e:
                        st.error(f"❌ Error training model for {metric} at {hospital}: {str(e)}")
                        st.info("This might be due to insufficient data or data quality issues.")
                    
                    st.divider()
        
        # Add insights
        add_forecasting_insights()
        
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please check that your Excel file contains the required columns and data format.")

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
        - At least 10-15 records per hospital-metric combination for basic forecasting
        - At least 30+ records per hospital-metric combination for reliable forecasting
        """)
