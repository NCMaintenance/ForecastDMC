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
    holidays = calendar.holidays(start=df_long['Datetime'].min(), end=df_long['Datetime'].max() + timedelta(days=30))
    df_long['IsHoliday'] = df_long['Datetime'].dt.normalize().isin(holidays).astype(int)

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
            
            # Feature engineering for future dates
            hour = future_datetime.hour
            day_of_week = future_datetime.dayofweek
            is_weekend = int(day_of_week in [5, 6])
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            day_sin = np.sin(2 * np.pi * day_of_week / 7)
            day_cos = np.cos(2 * np.pi * day_of_week / 7)
            
            # Check if it's a holiday
            calendar = IrishBankHolidays()
            holidays = calendar.holidays(start=future_datetime, end=future_datetime)
            is_holiday = int(future_datetime.normalize() in holidays)
            
            future_dates.append({
                'Datetime': future_datetime,
                'Hospital': hospital,
                'Hour': hour,
                'DayOfWeek': day_of_week,
                'IsWeekend': is_weekend,
                'Hour_sin': hour_sin,
                'Hour_cos': hour_cos,
                'Day_sin': day_sin,
                'Day_cos': day_cos,
                'IsHoliday': is_holiday,
                'Hospital_Code': hospital_code,
                'Additional_Capacity': additional_capacity
            })
    
    return pd.DataFrame(future_dates)

def forecast_with_lags(model, historical_data, future_df, features):
    """Forecast future values using lag features"""
    # Get last 3 values for initial lags
    last_values = historical_data['Value'].tail(3).values
    
    predictions = []
    current_lags = list(reversed(last_values))  # [lag_1, lag_2, lag_3]
    
    for idx, row in future_df.iterrows():
        # Create feature vector with current lags
        feature_vector = row[features[:-3]].values.tolist() + current_lags
        feature_vector = np.array(feature_vector).reshape(1, -1)
        
        # Make prediction
        pred = model.predict(feature_vector)[0]
        predictions.append(pred)
        
        # Update lags for next prediction
        current_lags = [pred] + current_lags[:2]
    
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
    fig.add_vline(
        x=last_historical_date,
        line_dash="dot",
        line_color="gray",
        annotation_text="Forecast Start"
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

# --- Streamlit UI ---
st.title("🇮🇪 Emergency Department Forecasting (Ireland)")
st.markdown("Upload your ED Excel file, select hospital(s), and generate 7-day forecasts")

uploaded_file = st.file_uploader("Upload your ED Excel File", type=["xlsx"])

if uploaded_file:
    # Load and prepare data
    df = pd.read_excel(uploaded_file)
    df_long = prepare_data(df)
    
    # Get unique hospitals for selection
    hospitals = sorted(df_long['Hospital'].unique())
    
    # UI Controls
    st.sidebar.header("Forecast Settings")
    
    # Hospital selection
    hospital_option = st.sidebar.selectbox(
        "Select Hospital:",
        options=["All Hospitals"] + hospitals
    )
    
    # Forecast days
    forecast_days = st.sidebar.slider(
        "Forecast Days:",
        min_value=1,
        max_value=14,
        value=7
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
        
        # Features for modeling
        features = [
            'Hour', 'DayOfWeek', 'IsWeekend', 'Hour_sin', 'Hour_cos',
            'Day_sin', 'Day_cos', 'IsHoliday', 'Hospital_Code',
            'Lag_1', 'Lag_2', 'Lag_3', 'Additional_Capacity'
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
                metric_data = metric_data.sort_values('Datetime')
                
                # Create lag features
                for i in range(1, 4):
                    metric_data[f'Lag_{i}'] = metric_data['Value'].shift(i)
                
                metric_data.dropna(inplace=True)
                
                if len(metric_data) < 10:  # Need minimum data for training
                    st.warning(f"Insufficient data for {metric} forecasting at {hospital}")
                    continue
                
                # Train model
                X = metric_data[features]
                y = metric_data['Value']
                
                # Use time-based split (last 20% for testing)
                split_idx = int(len(X) * 0.8)
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                
                model = lgb.LGBMRegressor(verbose=-1, random_state=42)
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
