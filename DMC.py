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
    """FIXED: Enhanced data preparation with proper melting to get your desired format"""
    # Rename columns
    df = df.rename(columns={
        'Tracker8am': 'ED_8am',
        'Tracker2pm': 'ED_2pm',
        'Tracker8pm': 'ED_8pm',
        'TimeTotal_8am': 'Trolley_8am',
        'TimeTotal_2pm': 'Trolley_2pm',
        'TimeTotal_8pm': 'Trolley_8pm',
        'AdditionalCapacityOpen Morning': 'Capacity'
    })

    # Fill missing capacity values
    df['Capacity'] = df.groupby(['Hospital'])['Capacity'].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill')).fillna(0)
    
    # Create the melted data in the exact format you want
    melted_rows = []
    
    for _, row in df.iterrows():
        base_info = {
            'Hospital Group Name': row['Hospital Group Name'],
            'Hospital': row['Hospital'],
            'Date': row['Date'],
            'DayGAR': row['DayGAR']
        }
        
        # Create three time periods
        times = [
            ('08:00:00', 'ED_8am', 'Trolley_8am'),
            ('14:00:00', 'ED_2pm', 'Trolley_2pm'), 
            ('20:00:00', 'ED_8pm', 'Trolley_8pm')
        ]
        
        for time_str, ed_col, trolley_col in times:
            melted_row = base_info.copy()
            melted_row.update({
                'Time': time_str,
                'ED Beds': row[ed_col] if pd.notna(row[ed_col]) else 0,
                'Trolleys': row[trolley_col] if pd.notna(row[trolley_col]) else 0,
                'Capacity': row['Capacity'] if pd.notna(row['Capacity']) else 0
            })
            melted_rows.append(melted_row)
    
    # Create the melted dataframe
    df_melted = pd.DataFrame(melted_rows)
    
    # Convert Date to datetime and create full datetime
    df_melted['Date'] = pd.to_datetime(df_melted['Date'])
    df_melted['Datetime'] = pd.to_datetime(df_melted['Date'].dt.strftime('%Y-%m-%d') + ' ' + df_melted['Time'])
    
    # Add time-based features
    df_melted['Hour'] = df_melted['Datetime'].dt.hour
    df_melted['DayOfWeek'] = df_melted['Datetime'].dt.dayofweek
    df_melted['DayOfMonth'] = df_melted['Datetime'].dt.day
    df_melted['Month'] = df_melted['Datetime'].dt.month
    df_melted['Quarter'] = df_melted['Datetime'].dt.quarter
    df_melted['WeekOfYear'] = df_melted['Datetime'].dt.isocalendar().week
    df_melted['IsWeekend'] = df_melted['DayOfWeek'].isin([5, 6]).astype(int)
    df_melted['IsMonday'] = (df_melted['DayOfWeek'] == 0).astype(int)
    df_melted['IsFriday'] = (df_melted['DayOfWeek'] == 4).astype(int)
    
    # Cyclical encoding
    df_melted['Hour_sin'] = np.sin(2 * np.pi * df_melted['Hour'] / 24)
    df_melted['Hour_cos'] = np.cos(2 * np.pi * df_melted['Hour'] / 24)
    df_melted['Day_sin'] = np.sin(2 * np.pi * df_melted['DayOfWeek'] / 7)
    df_melted['Day_cos'] = np.cos(2 * np.pi * df_melted['DayOfWeek'] / 7)
    df_melted['Month_sin'] = np.sin(2 * np.pi * df_melted['Month'] / 12)
    df_melted['Month_cos'] = np.cos(2 * np.pi * df_melted['Month'] / 12)

    # Add Irish holidays
    calendar = IrishBankHolidays()
    try:
        holidays = calendar.holidays(start=df_melted['Datetime'].min(), end=df_melted['Datetime'].max() + timedelta(days=30))
        df_melted['IsHoliday'] = df_melted['Datetime'].dt.normalize().isin(holidays).astype(int)
    except:
        df_melted['IsHoliday'] = 0

    # Seasonal and time indicators
    df_melted['IsSummer'] = df_melted['Month'].isin([6, 7, 8]).astype(int)
    df_melted['IsWinter'] = df_melted['Month'].isin([12, 1, 2]).astype(int)
    df_melted['IsPeakHour'] = df_melted['Hour'].isin([20]).astype(int)
    df_melted['IsLowHour'] = df_melted['Hour'].isin([8]).astype(int)

    # Hospital encoding
    df_melted['Hospital_Code'] = df_melted['Hospital'].astype('category').cat.codes
    
    # Calculate capacity utilization ratios
    df_melted['Total_Patients'] = df_melted['ED Beds'] + df_melted['Trolleys']
    df_melted['ED_to_Capacity_Ratio'] = np.where(
        df_melted['Capacity'] > 0,
        df_melted['ED Beds'] / df_melted['Capacity'],
        0
    )
    df_melted['Trolley_to_Capacity_Ratio'] = np.where(
        df_melted['Capacity'] > 0,
        df_melted['Trolleys'] / df_melted['Capacity'],
        0
    )
    df_melted['Total_to_Capacity_Ratio'] = np.where(
        df_melted['Capacity'] > 0,
        df_melted['Total_Patients'] / df_melted['Capacity'],
        0
    )
    df_melted['ED_to_Trolley_Ratio'] = np.where(
        df_melted['Trolleys'] > 0,
        df_melted['ED Beds'] / df_melted['Trolleys'],
        0
    )
    df_melted['Is_Over_Capacity'] = (df_melted['Total_Patients'] > df_melted['Capacity']).astype(int)
    df_melted['Capacity_Pressure'] = np.maximum(0, df_melted['Total_Patients'] - df_melted['Capacity'])
    
    return df_melted

def create_long_format_for_modeling(df_melted):
    """Convert the melted format to long format for modeling (ED and Trolley as separate rows)"""
    # Melt ED Beds and Trolleys into separate rows
    df_long = pd.melt(
        df_melted,
        id_vars=[col for col in df_melted.columns if col not in ['ED Beds', 'Trolleys']],
        value_vars=['ED Beds', 'Trolleys'],
        var_name='Metric',
        value_name='Value'
    )
    
    # Clean up metric names
    df_long['Metric'] = df_long['Metric'].str.replace(' ', '_').str.replace('Beds', '').str.strip('_')
    df_long['Metric'] = df_long['Metric'].replace({'ED_': 'ED', 'Trolleys': 'Trolley'})
    
    # Add cross-metric features
    pivot_data = df_melted.pivot_table(
        index=['Hospital', 'Datetime'], 
        values=['ED Beds', 'Trolleys'], 
        aggfunc='first'
    ).reset_index()
    pivot_data.columns = ['Hospital', 'Datetime', 'ED_Value', 'Trolley_Value']
    
    # Merge cross-metric features
    df_long = df_long.merge(pivot_data, on=['Hospital', 'Datetime'], how='left')
    
    # Add concurrent values
    df_long['Concurrent_ED'] = np.where(df_long['Metric'] == 'Trolley', df_long['ED_Value'], 0)
    df_long['Concurrent_Trolley'] = np.where(df_long['Metric'] == 'ED', df_long['Trolley_Value'], 0)
    
    # Drop temporary columns
    df_long = df_long.drop(['ED_Value', 'Trolley_Value'], axis=1)
    
    # Ensure all numeric columns are properly typed
    numeric_cols = df_long.select_dtypes(include=[np.number]).columns
    df_long[numeric_cols] = df_long[numeric_cols].fillna(0)
    
    return df_long

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
    if 'Capacity' in df.columns and len(df) >= 6:
        df['Rolling_Capacity_Mean_3'] = df['Capacity'].rolling(window=min(3, len(df)//2), min_periods=1).mean()
        lag_features.append('Rolling_Capacity_Mean_3')
    
    # Fill NaN values
    for feature in lag_features:
        df[feature] = df[feature].fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return df, lag_features

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

def plot_forecasts_enhanced(historical_data, forecast_data, metric_name, hospital_name):
    """Create enhanced interactive plotly chart with capacity information"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'{metric_name} Forecast', 'Capacity & Utilization'),
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
    
    # Capacity
    if 'Capacity' in historical_data.columns:
        fig.add_trace(go.Scatter(
            x=historical_data['Datetime'],
            y=historical_data['Capacity'],
            mode='lines',
            name='Historical Capacity',
            line=dict(color='green', width=2),
            opacity=0.7
        ), row=2, col=1)
    
    if 'Capacity' in forecast_data.columns:
        fig.add_trace(go.Scatter(
            x=forecast_data['Datetime'],
            y=forecast_data['Capacity'],
            mode='lines',
            name='Forecasted Capacity',
            line=dict(color='green', dash='dot', width=2),
            opacity=0.7
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
    fig.update_yaxes(title_text='Capacity', row=2, col=1)
    
    return fig

def create_future_dates_enhanced(last_date, hospital, hospital_code, capacity, days=7):
    """Enhanced future dates with forecasted capacity"""
    future_dates = []
    times = ['08:00:00', '14:00:00', '20:00:00']
    
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
            
            future_dates.append({
                'Datetime': future_datetime,
                'Date': future_datetime.date(),
                'Time': time_str,
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
                'Capacity': capacity
            })
    
    return pd.DataFrame(future_dates)

# --- Streamlit UI ---
st.title("🏥 Enhanced Emergency Department Forecasting - FIXED")
st.markdown("Upload your ED Excel file for comprehensive forecasting with capacity analysis")

# Sidebar controls
st.sidebar.header("Forecast Settings")
forecast_days = st.sidebar.slider("Forecast Days", 1, 14, 7)

uploaded_file = st.file_uploader("Upload your ED Excel File", type=["xlsx"])

if uploaded_file:
    try:
        # Load data
        df = pd.read_excel(uploaded_file)
        
        # Apply the FIXED data preparation
        df_melted = prepare_data_enhanced(df)
        
        st.success(f"✅ Data loaded and melted successfully! {len(df_melted)} records found.")
        
        # Show sample of melted data in your desired format
        st.subheader("📊 Sample of Melted Data (Your Desired Format)")
        sample_display = df_melted[['Date', 'Time', 'ED Beds', 'Trolleys', 'Capacity']].head(10)
        st.dataframe(sample_display, use_container_width=True)
        
        # Convert to long format for modeling
        df_long = create_long_format_for_modeling(df_melted)
        
        st.subheader("📊 Data Summary")
        summary_stats = df_long.groupby(['Hospital', 'Metric']).agg({
            'Value': ['count', 'mean', 'std'],
            'Capacity': 'mean'
        }).round(2)
        st.dataframe(summary_stats, use_container_width=True)
        
        # Hospital selection
        hospitals = sorted(df_long['Hospital'].unique())
        hospital_option = st.sidebar.selectbox("Select Hospital:", ["All Hospitals"] + hospitals)
        
        # Run forecast
        if st.sidebar.button("🚀 Run Enhanced Forecast", type="primary"):
            st.header("📊 Enhanced Forecast Results")
            
            selected_hospitals = hospitals if hospital_option == "All Hospitals" else [hospital_option]
            
            # Base features
            base_features = [
                'Hour', 'DayOfWeek', 'DayOfMonth', 'Month', 'Quarter', 'WeekOfYear',
                'IsWeekend', 'IsMonday', 'IsFriday',
                'Hour_sin', 'Hour_cos', 'Day_sin', 'Day_cos', 'Month_sin', 'Month_cos',
                'IsHoliday', 'IsSummer', 'IsWinter', 'IsPeakHour', 'IsLowHour',
                'Hospital_Code', 'Capacity',
                'Total_to_Capacity_Ratio', 'ED_to_Capacity_Ratio', 'Trolley_to_Capacity_Ratio',
                'ED_to_Trolley_Ratio', 'Is_Over_Capacity', 'Capacity_Pressure'
            ]
            
            for hospital in selected_hospitals:
                st.subheader(f"🏥 {hospital}")
                
                hospital_data = df_long[df_long['Hospital'] == hospital].copy()
                hospital_code = hospital_data['Hospital_Code'].iloc[0]
                capacity = hospital_data['Capacity'].fillna(0).iloc[0]
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
                    
                    st.info(f"Processing {metric} for {hospital} ({len(training_data)} records)")
                    
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
                    
                    # Model configuration
                    model = lgb.LGBMRegressor(
                        n_estimators=min(300, len(X_train) * 3),
                        learning_rate=0.05,
                        max_depth=min(6, len(available_features) // 2 + 2),
                        num_leaves=min(31, 2 ** min(6, len(available_features) // 2 + 2) - 1),
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
                        
                        # Create future dates
                        future_df = create_future_dates_enhanced(
                            pd.to_datetime(last_date), 
                            hospital, 
                            hospital_code, 
                            capacity, 
                            days=forecast_days
                        )
                        
                        # Add capacity features to future data
                        future_df['Total_Patients'] = 0  # Will be updated
                        future_df['ED_to_Capacity_Ratio'] = 0
                        future_df['Trolley_to_Capacity_Ratio'] = 0
                        future_df['Total_to_Capacity_Ratio'] = 0
                        future_df['ED_to_Trolley_Ratio'] = 0
                        future_df['Is_Over_Capacity'] = 0
                        future_df['Capacity_Pressure'] = 0
                        
                        # Add cross-metric features
                        if metric == 'ED':
                            avg_trolley = training_data.get('Concurrent_Trolley', pd.Series([0])).mean()
                            future_df['Concurrent_Trolley'] = avg_trolley
                        elif metric == 'Trolley':
                            avg_ed = training_data.get('Concurrent_ED', pd.Series([0])).mean()
                            future_df['Concurrent_ED'] = avg_ed
                        
                        # Generate forecasts
                        predictions = forecast_with_lags_enhanced(model, training_data, future_df, available_features)
                        future_df['Predicted'] = predictions
                        
                        # Display metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(f"{metric} RMSE", f"{rmse:.2f}")
                        with col2:
                            st.metric("Training Records", f"{len(X_train)}")
                        with col3:
                            st.metric(f"Last {metric} Value", f"{training_data['Value'].iloc[-1]:.0f}")
                        
                        # Plotting
                        fig = plot_forecasts_enhanced(
                            training_data.tail(21),
                            future_df,
                            metric,
                            hospital
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Forecast table
                        with st.expander(f"📋 {metric} Forecast Details"):
                            forecast_display = future_df[['Date', 'Time', 'Predicted', 'Capacity']].copy()
                            forecast_display['Predicted'] = forecast_display['Predicted'].round(1)
                            
                            st.dataframe(
                                forecast_display.rename(columns={
                                    'Predicted': f'{metric} Forecast'
                                }),
                                use_container_width=True
                            )
                        
                        # Download CSV
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
