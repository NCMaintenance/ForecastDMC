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
    """
    Prepares and feature engineers the data, reshaping it into the desired wide format
    with 'ED Beds', 'Trolleys', and 'Capacity' columns.
    """
    # Rename columns for clarity and consistency
    df = df.rename(columns={
        'Tracker8am': 'ED_8am',
        'Tracker2pm': 'ED_2pm',
        'Tracker8pm': 'ED_8pm',
        'TimeTotal_8am': 'Trolley_8am',
        'TimeTotal_2pm': 'Trolley_2pm',
        'TimeTotal_8pm': 'Trolley_8pm',
        'AdditionalCapacityOpen Morning': 'Additional_Capacity'
    })

    # Fill Additional_Capacity across the day for each hospital and date
    df['Additional_Capacity'] = df.groupby(['Hospital', 'Date'])['Additional_Capacity'].transform('first')
    df['Additional_Capacity'] = df['Additional_Capacity'].fillna(0)

    # Define common id_vars for melting, these will become the identifier columns
    common_id_vars = ['Hospital Group Name', 'Hospital', 'Date', 'DayGAR', 'Additional_Capacity']

    # Melt ED counts into a long format
    df_ed = pd.melt(
        df,
        id_vars=common_id_vars,
        value_vars=['ED_8am', 'ED_2pm', 'ED_8pm'],
        var_name='Metric_Time_ED', # Temporary column to extract time
        value_name='ED Beds'      # The target column for ED data
    )

    # Melt Trolley counts into a long format
    df_trolley = pd.melt(
        df,
        id_vars=common_id_vars,
        value_vars=['Trolley_8am', 'Trolley_2pm', 'Trolley_8pm'],
        var_name='Metric_Time_Trolley', # Temporary column to extract time
        value_name='Trolleys'          # The target column for Trolley data
    )

    # Extract time label (e.g., '8am', '2pm', '8pm') from the temporary metric_time columns
    df_ed['TimeLabel'] = df_ed['Metric_Time_ED'].str.extract(r'([\d]+[ap]m)')
    df_trolley['TimeLabel'] = df_trolley['Metric_Time_Trolley'].str.extract(r'([\d]+[ap]m)')

    # Map time labels to standardized 24-hour format (e.g., '08:00', '14:00', '20:00')
    time_map = {'8am': '08:00', '2pm': '14:00', '8pm': '20:00'}
    df_ed['Time'] = df_ed['TimeLabel'].map(time_map)
    df_trolley['Time'] = df_trolley['TimeLabel'].map(time_map)

    # Drop temporary columns before merging to avoid redundancy
    df_ed = df_ed.drop(columns=['Metric_Time_ED', 'TimeLabel'])
    df_trolley = df_trolley.drop(columns=['Metric_Time_Trolley', 'TimeLabel'])

    # Merge the two melted dataframes. This brings 'ED Beds' and 'Trolleys' side-by-side
    # for each unique combination of identifier columns and 'Time'.
    df_merged = pd.merge(
        df_ed,
        df_trolley,
        on=['Hospital Group Name', 'Hospital', 'Date', 'DayGAR', 'Additional_Capacity', 'Time'],
        how='inner' # Use inner join to ensure only complete records (both ED and Trolley for a given time) are kept
    )

    # Convert 'ED Beds' and 'Trolleys' to numeric, coercing errors to NaN
    df_merged['ED Beds'] = pd.to_numeric(df_merged['ED Beds'], errors='coerce')
    df_merged['Trolleys'] = pd.to_numeric(df_merged['Trolleys'], errors='coerce')
    # Drop rows where either 'ED Beds' or 'Trolleys' values are missing
    df_merged = df_merged.dropna(subset=['ED Beds', 'Trolleys'])

    # Rename 'Additional_Capacity' to 'Capacity' to match the desired output
    df_merged = df_merged.rename(columns={'Additional_Capacity': 'Capacity'})

    # Create a unified Datetime column for time-series analysis
    df_merged['Datetime'] = pd.to_datetime(df_merged['Date'].astype(str) + ' ' + df_merged['Time'])
    # Sort the data by Hospital and Datetime, which is crucial for time-series operations like lags
    df_merged = df_merged.sort_values(by=['Hospital', 'Datetime']).reset_index(drop=True)

    # --- Feature Engineering ---
    # Extract various time-based features
    df_merged['Hour'] = df_merged['Datetime'].dt.hour
    df_merged['DayOfWeek'] = df_merged['Datetime'].dt.dayofweek
    df_merged['DayOfMonth'] = df_merged['Datetime'].dt.day
    df_merged['Month'] = df_merged['Datetime'].dt.month
    df_merged['Quarter'] = df_merged['Datetime'].dt.quarter
    df_merged['WeekOfYear'] = df_merged['Datetime'].dt.isocalendar().week
    df_merged['IsWeekend'] = df_merged['DayOfWeek'].isin([5, 6]).astype(int) # Binary indicator for weekend
    df_merged['IsMonday'] = (df_merged['DayOfWeek'] == 0).astype(int) # Binary indicator for Monday
    df_merged['IsFriday'] = (df_merged['DayOfWeek'] == 4).astype(int) # Binary indicator for Friday

    # Apply cyclical encoding to capture periodicity in time features
    df_merged['Hour_sin'] = np.sin(2 * np.pi * df_merged['Hour'] / 24)
    df_merged['Hour_cos'] = np.cos(2 * np.pi * df_merged['Hour'] / 24)
    df_merged['Day_sin'] = np.sin(2 * np.pi * df_merged['DayOfWeek'] / 7)
    df_merged['Day_cos'] = np.cos(2 * np.pi * df_merged['DayOfWeek'] / 7)
    df_merged['Month_sin'] = np.sin(2 * np.pi * df_merged['Month'] / 12)
    df_merged['Month_cos'] = np.cos(2 * np.pi * df_merged['Month'] / 12)

    # Add Irish bank holidays as a feature
    calendar = IrishBankHolidays()
    try:
        # Generate holidays over the entire date range of the dataset
        holidays = calendar.holidays(start=df_merged['Datetime'].min(), end=df_merged['Datetime'].max() + timedelta(days=30))
        df_merged['IsHoliday'] = df_merged['Datetime'].dt.normalize().isin(holidays).astype(int)
    except Exception: # Catch any errors during holiday generation
        df_merged['IsHoliday'] = 0 # Default to 0 if holidays cannot be determined

    # Add seasonal indicators
    df_merged['IsSummer'] = df_merged['Month'].isin([6, 7, 8]).astype(int)
    df_merged['IsWinter'] = df_merged['Month'].isin([12, 1, 2]).astype(int)

    # Add peak/low hour indicators
    df_merged['IsPeakHour'] = df_merged['Hour'].isin([20]).astype(int) # 8 PM is often a peak time
    df_merged['IsLowHour'] = df_merged['Hour'].isin([8]).astype(int)  # 8 AM is often a lower time

    # Encode Hospital names into numerical codes for model consumption
    df_merged['Hospital_Code'] = df_merged['Hospital'].astype('category').cat.codes

    return df_merged

def add_lag_features_smart(df, target_column, min_data_threshold=20):
    """
    Adds lag features intelligently to the DataFrame based on available data for a specific target column.
    Lag features are time-shifted values of the target, useful for capturing temporal dependencies.
    """
    df = df.copy()

    # Determine maximum safe lag based on the size of the filtered data
    # This prevents creating too many NaNs or using disproportionately large lags on small datasets.
    max_safe_lag = min(7, len(df) // 4)

    if max_safe_lag < 1:
        # If data is too limited, no lag features can be meaningfully created
        st.warning(f"Very limited data ({len(df)} records). Using minimal features for {target_column}.")
        return df, []

    lag_features = []

    # Generate lag features for the target_column (e.g., 'ED Beds', 'Trolleys')
    for i in range(1, max_safe_lag + 1):
        lag_col = f'Lag_{target_column}_{i}' # Lag column names are specific to the target (e.g., 'Lag_ED Beds_1')
        df[lag_col] = df[target_column].shift(i)
        lag_features.append(lag_col)

    # Add rolling mean features if enough data exists
    if len(df) >= 6:
        # Rolling mean over 3 periods
        df[f'Rolling_Mean_3_{target_column}'] = df[target_column].rolling(window=min(3, len(df)//2), min_periods=1).mean()
        lag_features.append(f'Rolling_Mean_3_{target_column}')

        if len(df) >= 14:
            # Rolling mean over 7 periods
            df[f'Rolling_Mean_7_{target_column}'] = df[target_column].rolling(window=min(7, len(df)//2), min_periods=1).mean()
            lag_features.append(f'Rolling_Mean_7_{target_column}')

    # Fill NaN values created by shifting/rolling operations
    # Prioritize forward fill, then backward fill, then a default of 0
    for feature in lag_features:
        df[feature] = df[feature].fillna(method='ffill').fillna(method='bfill').fillna(0)

    return df, lag_features

def create_future_dates(last_date, hospital, hospital_code, additional_capacity, days=7):
    """
    Creates a DataFrame of future dates and times with corresponding feature values
    for forecasting.
    """
    future_dates = []
    times = ['08:00', '14:00', '20:00'] # Standard observation times

    for day in range(1, days + 1):
        future_date = last_date + timedelta(days=day)
        for time_str in times:
            future_datetime = pd.to_datetime(f"{future_date.date()} {time_str}")

            # Recalculate all relevant features for each future timestamp
            hour = future_datetime.hour
            day_of_week = future_datetime.dayofweek
            day_of_month = future_datetime.day
            month = future_datetime.month
            quarter = (month - 1) // 3 + 1
            week_of_year = future_datetime.isocalendar().week

            is_weekend = int(day_of_week in [5, 6])
            is_monday = int(day_of_week == 0)
            is_friday = int(day_of_week == 4)

            # Cyclical encoding
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
            except Exception: # Handle potential errors during holiday calculation
                pass

            # Seasonal and peak hour indicators
            is_summer = int(month in [6, 7, 8])
            is_winter = int(month in [12, 1, 2])
            is_peak_hour = int(hour == 20)
            is_low_hour = int(hour == 8)

            # Append all calculated features for the current future timestamp
            future_dates.append({
                'Date': future_datetime.date(), # Date component
                'Time': time_str,               # Time component (e.g., '08:00')
                'Datetime': future_datetime,    # Full datetime object
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
                'Capacity': additional_capacity # Use the 'Capacity' value (from 'Additional_Capacity')
            })

    return pd.DataFrame(future_dates)

def forecast_with_lags(model, historical_data, future_df, features, target_column):
    """
    Generates forecasts iteratively, updating lag and rolling features
    with each new prediction.
    """
    # Get the last few actual values from historical data for initial lag features
    last_values = historical_data[target_column].tail(7).values # Get last 7 values

    predictions = []
    # Initialize current_lags with historical values (most recent first)
    current_lags = list(reversed(last_values))

    # Initialize rolling statistics with historical data's tail
    historical_mean_3 = historical_data[target_column].tail(3).mean()
    historical_mean_7 = historical_data[target_column].tail(7).mean()

    for idx, row in future_df.iterrows():
        try:
            feature_values = [] # List to build the feature vector for the current prediction

            for feature in features:
                if feature.startswith(f'Lag_{target_column}_'): # Check if the feature is a lag for the current target
                    lag_num = int(feature.replace(f'Lag_{target_column}_', '')) - 1 # Extract lag order (0-indexed)
                    if lag_num < len(current_lags):
                        feature_values.append(current_lags[lag_num]) # Use the dynamically updated lag value
                    else:
                        feature_values.append(0) # Default to 0 if lag is not available
                elif feature == f'Rolling_Mean_3_{target_column}': # Check for target-specific rolling mean 3
                    feature_values.append(historical_mean_3)
                elif feature == f'Rolling_Mean_7_{target_column}': # Check for target-specific rolling mean 7
                    feature_values.append(historical_mean_7)
                else:
                    # For all other (non-lag, non-rolling) features, use their values from the future_df row
                    if feature in row:
                        feature_values.append(row[feature])
                    else:
                        feature_values.append(0) # Default to 0 if feature is missing

            # Make the prediction using the constructed feature vector
            feature_vector = np.array(feature_values).reshape(1, -1)
            pred = model.predict(feature_vector)[0]
            pred = max(0, pred) # Ensure predictions are non-negative
            predictions.append(pred)

            # Update the lag values for the next prediction step
            # The most recent prediction becomes the first lag for the next step
            current_lags = [pred] + current_lags[:6] # Keep a window of 7 lags

            # Update rolling statistics based on the new prediction
            if len(predictions) >= 3:
                historical_mean_3 = np.mean(predictions[-3:])
            if len(predictions) >= 7:
                historical_mean_7 = np.mean(predictions[-7:])

        except Exception as e:
            # Fallback in case of prediction error
            st.error(f"Error in prediction step {idx} for {target_column}: {e}")
            predictions.append(historical_data[target_column].mean()) # Use historical mean as a fallback

    return predictions

def plot_forecasts(historical_data, forecast_data, metric_name, hospital_name):
    """Creates an interactive Plotly chart visualizing historical data and forecasts."""
    fig = go.Figure()

    # Add historical data trace
    fig.add_trace(go.Scatter(
        x=historical_data['Datetime'],
        y=historical_data[metric_name], # Use the actual metric name (e.g., 'ED Beds')
        mode='lines+markers',
        name='Historical',
        line=dict(color='blue'),
        marker=dict(size=4)
    ))

    # Add forecast data trace
    fig.add_trace(go.Scatter(
        x=forecast_data['Datetime'],
        y=forecast_data['Predicted'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='red', dash='dash'), # Dashed line for forecast
        marker=dict(size=6, symbol='diamond')
    ))

    # Add a vertical line to visually separate historical data from the forecast
    last_historical_date = historical_data['Datetime'].max()
    fig.add_shape(
        type="line",
        x0=last_historical_date,
        x1=last_historical_date,
        y0=0,
        y1=1,
        yref="paper", # Relative to plot height
        line=dict(color="gray", width=2, dash="dot"),
    )

    # Add annotation for the forecast start point
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

    # Update plot layout with titles and labels
    fig.update_layout(
        title=f'{metric_name} Forecast - {hospital_name}',
        xaxis_title='Date',
        yaxis_title=f'{metric_name} Count',
        hovermode='x unified', # Show all traces' values on hover
        showlegend=True,
        height=500
    )

    return fig

def add_forecasting_insights():
    """Displays insights and tips about the forecasting process."""
    with st.expander("💡 Forecasting Insights & Tips", expanded=False):
        st.subheader("Data Requirements")
        st.markdown("""
        For accurate forecasting, you need:
        * **Minimum 30 records** per hospital-metric combination
        * **Consistent time intervals** (8am, 2pm, 8pm readings are ideal)
        * **Recent data** (within the last 6 months ideally for best relevance)
        * **Complete records** (avoid too many missing values, as they can hinder model performance)
        """)

        st.subheader("Understanding Your Results")
        st.markdown("""
        * **RMSE (Root Mean Square Error)**: Lower values indicate better model accuracy. This metric represents the average magnitude of the errors in your predictions.
        * **Historical vs. Forecast**: The generated chart clearly visualizes your past data patterns and the predicted future values, allowing for easy comparison.
        * **Validation**: The model's performance (RMSE) is calculated on a subset of your historical data, showing how well it generalizes to unseen but similar data.
        """)

# --- Streamlit UI ---
st.title("Emergency Department Forecasting (Ireland)")
st.markdown("Upload your ED Excel file, select hospital(s), and generate 7-day forecasts.")

# Sidebar control for number of forecast days
forecast_days = st.sidebar.slider("Forecast Days", 1, 14, 7)

# File uploader widget
uploaded_file = st.file_uploader("Upload your ED Excel File", type=["xlsx"])

if uploaded_file:
    try:
        # Load and prepare data using the updated function
        df = pd.read_excel(uploaded_file)
        df_processed = prepare_data(df) # Renamed to df_processed as it's no longer just 'long'

        # Show data loading success message and summary
        st.success(f"✅ Data loaded and processed successfully! {len(df_processed)} records found.")

        # Display a sample of the processed data to confirm the new structure
        st.subheader("📋 Sample of Processed Data")
        st.dataframe(df_processed[['Date', 'Time', 'ED Beds', 'Trolleys', 'Capacity']].head())

        # Get unique hospitals for selection in the sidebar
        hospitals = sorted(df_processed['Hospital'].unique())

        # Show data summary by Hospital and the new metrics
        data_summary = df_processed.groupby(['Hospital']).agg(
            ed_records=('ED Beds', 'count'),
            trolley_records=('Trolleys', 'count')
        ).reset_index()
        st.subheader("📊 Data Summary by Hospital")
        st.dataframe(data_summary, use_container_width=True)

        # Hospital selection dropdown
        st.sidebar.header("Forecast Settings")
        hospital_option = st.sidebar.selectbox(
            "Select Hospital:",
            options=["All Hospitals"] + hospitals
        )

        # Run forecast button
        run_forecast = st.sidebar.button("🚀 Run Forecast", type="primary")

        if run_forecast:
            st.header("📊 Forecast Results")

            # Determine which hospitals to process based on user selection
            if hospital_option == "All Hospitals":
                selected_hospitals = hospitals
            else:
                selected_hospitals = [hospital_option]

            # Define base features that are common to both ED Beds and Trolleys
            base_features = [
                'Hour', 'DayOfWeek', 'DayOfMonth', 'Month', 'Quarter', 'WeekOfYear',
                'IsWeekend', 'IsMonday', 'IsFriday',
                'Hour_sin', 'Hour_cos', 'Day_sin', 'Day_cos', 'Month_sin', 'Month_cos',
                'IsHoliday', 'IsSummer', 'IsWinter', 'IsPeakHour', 'IsLowHour',
                'Hospital_Code', 'Capacity' # 'Capacity' is now the column name
            ]

            # Iterate through each selected hospital and metric ('ED Beds', 'Trolleys')
            for hospital in selected_hospitals:
                st.subheader(f"🏥 {hospital}")

                # Filter data for the current hospital
                hospital_data = df_processed[df_processed['Hospital'] == hospital].copy()

                # Extract hospital code and additional capacity (now 'Capacity') for future dates
                hospital_code = hospital_data['Hospital_Code'].iloc[0] if not hospital_data.empty else 0
                additional_capacity_val = hospital_data['Capacity'].fillna(0).iloc[0] if not hospital_data.empty else 0

                # Get the last date from the historical data for creating future dates
                last_date = hospital_data['Datetime'].max().date() if not hospital_data.empty else datetime.now().date()

                # Process 'ED Beds' and 'Trolleys' forecasts separately
                for target_col_name in ['ED Beds', 'Trolleys']:
                    # Check if we have sufficient data for the current target column
                    # Count non-null values in the target column
                    if hospital_data[target_col_name].count() < 10:
                        st.warning(f"⚠️ Insufficient data for '{target_col_name}' at {hospital} ({hospital_data[target_col_name].count()} records). Need at least 10 records for meaningful forecasting.")
                        continue

                    st.info(f"Processing '{target_col_name}' for {hospital} ({hospital_data[target_col_name].count()} records)")

                    # Add lag and rolling features specifically for the current target column
                    # add_lag_features_smart modifies hospital_data in place and returns new feature names
                    data_with_lags, lag_features = add_lag_features_smart(hospital_data, target_col_name)

                    # Combine base features with newly generated lag features
                    all_features = base_features + lag_features

                    # Filter available features to only include those present in the DataFrame and not the target itself
                    available_features = [f for f in all_features if f in data_with_lags.columns and f != target_col_name]

                    # Prepare training data: drop rows with any missing values in features or target
                    training_data = data_with_lags.dropna(subset=[target_col_name] + available_features)

                    if len(training_data) < 5:
                        st.warning(f"⚠️ After preprocessing, insufficient data for '{target_col_name}' at {hospital} ({len(training_data)} records). Need at least 5 records to train a model.")
                        continue

                    # Define X (features) and y (target) for model training
                    X = training_data[available_features]
                    y = training_data[target_col_name]

                    # Split data into training and testing sets. For smaller datasets, use all data for training.
                    if len(X) < 20:
                        X_train, X_test = X, X.tail(min(3, len(X))) # Train on all, test on a small recent portion
                        y_train, y_test = y, y.tail(min(3, len(y)))
                    else:
                        # For larger datasets, use a time-based split (80% train, 20% test)
                        split_idx = int(len(X) * 0.8)
                        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

                    # Initialize and train the LightGBM Regressor model
                    model = lgb.LGBMRegressor(
                        n_estimators=min(100, len(X_train) * 2), # Adapt n_estimators to data size
                        learning_rate=0.1,
                        max_depth=min(4, len(available_features) // 2 + 1), # Adapt max_depth
                        num_leaves=min(32, 2 ** min(6, len(available_features) // 2 + 1) - 1), # Adapt num_leaves
                        subsample=0.8,
                        colsample_bytree=0.8,
                        verbose=-1, # Suppress verbose output
                        random_state=42,
                        force_col_wise=True # Optimize for column-wise data access
                    )

                    try:
                        model.fit(X_train, y_train)

                        # Evaluate model performance on the test set
                        y_pred_test = model.predict(X_test)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

                        # Create future timestamps for which to generate forecasts
                        future_df = create_future_dates(
                            pd.to_datetime(last_date),
                            hospital,
                            hospital_code,
                            additional_capacity_val, # Pass the correct capacity value
                            days=forecast_days
                        )

                        # Generate predictions for future dates using the trained model
                        predictions = forecast_with_lags(model, training_data, future_df, available_features, target_col_name)
                        future_df['Predicted'] = predictions # Add predictions to the future DataFrame

                        # Display key metrics using Streamlit columns
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(f"{target_col_name} RMSE", f"{rmse:.2f}")
                        with col2:
                            st.metric(f"Training Records", f"{len(X_train)}")
                        with col3:
                            # Display the last actual value of the target column
                            st.metric(f"Last {target_col_name} Value", f"{training_data[target_col_name].iloc[-1]:.0f}")

                        # Create and display the forecast plot
                        fig = plot_forecasts(
                            training_data.tail(21), # Show last 21 historical points (approx. 1 week at 3 readings/day)
                            future_df,
                            target_col_name, # Pass the metric name for plot title/labels
                            hospital
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Allow users to view detailed forecast data in an expandable section
                        with st.expander(f"📋 {target_col_name} Forecast Details"):
                            forecast_display = future_df[['Date', 'Time', 'Predicted']].copy()
                            forecast_display['Predicted'] = forecast_display['Predicted'].round(1) # Round predictions
                            st.dataframe(
                                forecast_display,
                                use_container_width=True
                            )

                        # Provide a download button for the forecast data
                        csv_data = future_df[['Datetime', 'Hospital', 'Predicted']].copy()
                        csv_data['Metric'] = target_col_name # Identify the metric in the CSV
                        st.download_button(
                            f"📥 Download {target_col_name} Forecast CSV",
                            csv_data.to_csv(index=False),
                            file_name=f"{hospital}_{target_col_name.replace(' ', '_')}_forecast.csv", # Sanitize filename
                            mime="text/csv",
                            key=f"{hospital}_{target_col_name}_download"
                        )

                    except Exception as e:
                        st.error(f"❌ Error training model or generating forecast for '{target_col_name}' at {hospital}: {str(e)}")
                        st.info("This might be due to insufficient data for model training or data quality issues preventing proper feature engineering.")

                    st.divider() # Separator for each metric's results

            # Display general forecasting insights after all forecasts are run
            add_forecasting_insights()

    except Exception as e:
        # Catch and display errors during file processing or initial data preparation
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please check that your Excel file contains the required columns and data format as described below.")

else:
    # Instructions for users when no file is uploaded
    st.info("👆 Please upload an Excel file to begin forecasting.")

    # Expandable section detailing the expected file format
    with st.expander("📋 Expected File Format"):
        st.markdown("""
        **Required columns:**
        - `Hospital Group Name`: The group the hospital belongs to.
        - `Hospital`: Unique identifier for the hospital.
        - `Date`: The date of the observation (e.g., '30/05/2025').
        - `DayGAR`: Categorical day information (if available, e.g., 'Day X').
        - `Tracker8am`, `Tracker2pm`, `Tracker8pm`: ED bed counts at 8 AM, 2 PM, 8 PM respectively.
        - `TimeTotal_8am`, `TimeTotal_2pm`, `TimeTotal_8pm`: Trolley counts at 8 AM, 2 PM, 8 PM respectively.
        - `AdditionalCapacityOpen Morning`: Any additional capacity opened in the morning for that day.

        **Data should contain:**
        - Historical ED and trolley count data.
        - Data for multiple hospitals (optional, but the app supports it).
        - At least **10-15 records** per hospital-metric combination for basic forecasting.
        - At least **30+ records** per hospital-metric combination for more reliable forecasting.
        """)

# import streamlit as st
# import pandas as pd
# import numpy as np
# import lightgbm as lgb
# from datetime import datetime, timedelta
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday
# from pandas.tseries.offsets import DateOffset
# from dateutil.rrule import MO
# import shap
# import matplotlib.pyplot as plt
# import plotly.graph_objects as go
# import plotly.express as px
# from plotly.subplots import make_subplots

# # --- Define Irish Bank Holidays ---
# class IrishBankHolidays(AbstractHolidayCalendar):
#     rules = [
#         Holiday("New Year's Day", month=1, day=1, observance=nearest_workday),
#         Holiday("St. Brigid's Day", month=2, day=1, observance=nearest_workday),
#         Holiday("St. Patrick's Day", month=3, day=17, observance=nearest_workday),
#         Holiday("May Day", month=5, day=1, offset=DateOffset(weekday=MO(1))),
#         Holiday("June Bank Holiday", month=6, day=1, offset=DateOffset(weekday=MO(1))),
#         Holiday("August Bank Holiday", month=8, day=1, offset=DateOffset(weekday=MO(1))),
#         Holiday("October Bank Holiday", month=10, day=1, offset=DateOffset(weekday=MO(-1))),
#         Holiday("Christmas Day", month=12, day=25),
#         Holiday("St. Stephen's Day", month=12, day=26),
#     ]

# def prepare_data(df):
#     """Prepare and feature engineer the data with improved handling"""
#     # Rename columns
#     df = df.rename(columns={
#         'Tracker8am': 'ED_8am',
#         'Tracker2pm': 'ED_2pm',
#         'Tracker8pm': 'ED_8pm',
#         'TimeTotal_8am': 'Trolley_8am',
#         'TimeTotal_2pm': 'Trolley_2pm',
#         'TimeTotal_8pm': 'Trolley_8pm',
#         'AdditionalCapacityOpen Morning': 'Additional_Capacity'
#     })

#     # Fill Additional Capacity across the day
#     df['Additional_Capacity'] = df.groupby(['Hospital', 'Date'])['Additional_Capacity'].transform('first')
#     df['Additional_Capacity'] = df['Additional_Capacity'].fillna(0)

#     # Reshape to long format
#     df_long = pd.melt(
#         df,
#         id_vars=['Hospital Group Name', 'Hospital', 'Date', 'DayGAR', 'Additional_Capacity'],
#         value_vars=['ED_8am', 'ED_2pm', 'ED_8pm', 'Trolley_8am', 'Trolley_2pm', 'Trolley_8pm'],
#         var_name='Metric_Time',
#         value_name='Value'
#     )

#     # Clean and handle missing values early
#     df_long = df_long.dropna(subset=['Value'])
#     df_long['Value'] = pd.to_numeric(df_long['Value'], errors='coerce')
#     df_long = df_long.dropna(subset=['Value'])

#     df_long[['Metric', 'TimeLabel']] = df_long['Metric_Time'].str.extract(r'(\w+)_([\d]+[ap]m)')
#     time_map = {'8am': '08:00', '2pm': '14:00', '8pm': '20:00'}
#     df_long['TimeStr'] = df_long['TimeLabel'].map(time_map)
#     df_long['Datetime'] = pd.to_datetime(df_long['Date'].astype(str) + ' ' + df_long['TimeStr'])

#     # Basic time features
#     df_long['Hour'] = df_long['Datetime'].dt.hour
#     df_long['DayOfWeek'] = df_long['Datetime'].dt.dayofweek
#     df_long['DayOfMonth'] = df_long['Datetime'].dt.day
#     df_long['Month'] = df_long['Datetime'].dt.month
#     df_long['Quarter'] = df_long['Datetime'].dt.quarter
#     df_long['WeekOfYear'] = df_long['Datetime'].dt.isocalendar().week
#     df_long['IsWeekend'] = df_long['DayOfWeek'].isin([5, 6]).astype(int)
#     df_long['IsMonday'] = (df_long['DayOfWeek'] == 0).astype(int)
#     df_long['IsFriday'] = (df_long['DayOfWeek'] == 4).astype(int)
    
#     # Cyclical encoding for time features
#     df_long['Hour_sin'] = np.sin(2 * np.pi * df_long['Hour'] / 24)
#     df_long['Hour_cos'] = np.cos(2 * np.pi * df_long['Hour'] / 24)
#     df_long['Day_sin'] = np.sin(2 * np.pi * df_long['DayOfWeek'] / 7)
#     df_long['Day_cos'] = np.cos(2 * np.pi * df_long['DayOfWeek'] / 7)
#     df_long['Month_sin'] = np.sin(2 * np.pi * df_long['Month'] / 12)
#     df_long['Month_cos'] = np.cos(2 * np.pi * df_long['Month'] / 12)

#     # Add Irish holidays
#     calendar = IrishBankHolidays()
#     try:
#         holidays = calendar.holidays(start=df_long['Datetime'].min(), end=df_long['Datetime'].max() + timedelta(days=30))
#         df_long['IsHoliday'] = df_long['Datetime'].dt.normalize().isin(holidays).astype(int)
#     except:
#         df_long['IsHoliday'] = 0

#     # Seasonal indicators
#     df_long['IsSummer'] = df_long['Month'].isin([6, 7, 8]).astype(int)
#     df_long['IsWinter'] = df_long['Month'].isin([12, 1, 2]).astype(int)
    
#     # Peak hour indicators
#     df_long['IsPeakHour'] = df_long['Hour'].isin([20]).astype(int)
#     df_long['IsLowHour'] = df_long['Hour'].isin([8]).astype(int)

#     # Encode hospital
#     df_long['Hospital_Code'] = df_long['Hospital'].astype('category').cat.codes

#     return df_long

# def add_lag_features_smart(df, min_data_threshold=20):
#     """Add lag features intelligently based on available data"""
#     df = df.copy()
    
#     # Determine maximum safe lag based on data size
#     max_safe_lag = min(7, len(df) // 4)  # Use at most 1/4 of data for lags
    
#     if max_safe_lag < 1:
#         st.warning(f"Very limited data ({len(df)} records). Using minimal features.")
#         return df, []
    
#     lag_features = []
    
#     # Add lag features progressively
#     for i in range(1, max_safe_lag + 1):
#         lag_col = f'Lag_{i}'
#         df[lag_col] = df['Value'].shift(i)
#         lag_features.append(lag_col)
    
#     # Add rolling features if we have enough data
#     if len(df) >= 6:
#         df['Rolling_Mean_3'] = df['Value'].rolling(window=min(3, len(df)//2), min_periods=1).mean()
#         lag_features.append('Rolling_Mean_3')
        
#         if len(df) >= 14:
#             df['Rolling_Mean_7'] = df['Value'].rolling(window=min(7, len(df)//2), min_periods=1).mean()
#             lag_features.append('Rolling_Mean_7')
    
#     # Fill NaN values with forward fill, then backward fill, then 0
#     for feature in lag_features:
#         df[feature] = df[feature].fillna(method='ffill').fillna(method='bfill').fillna(0)
    
#     return df, lag_features

# def create_future_dates(last_date, hospital, hospital_code, additional_capacity, days=7):
#     """Create future dates for forecasting"""
#     future_dates = []
#     times = ['08:00', '14:00', '20:00']
    
#     for day in range(1, days + 1):
#         future_date = last_date + timedelta(days=day)
#         for time_str in times:
#             future_datetime = pd.to_datetime(f"{future_date.date()} {time_str}")
            
#             # Basic features
#             hour = future_datetime.hour
#             day_of_week = future_datetime.dayofweek
#             day_of_month = future_datetime.day
#             month = future_datetime.month
#             quarter = (month - 1) // 3 + 1
#             week_of_year = future_datetime.isocalendar().week
            
#             is_weekend = int(day_of_week in [5, 6])
#             is_monday = int(day_of_week == 0)
#             is_friday = int(day_of_week == 4)
            
#             # Cyclical features
#             hour_sin = np.sin(2 * np.pi * hour / 24)
#             hour_cos = np.cos(2 * np.pi * hour / 24)
#             day_sin = np.sin(2 * np.pi * day_of_week / 7)
#             day_cos = np.cos(2 * np.pi * day_of_week / 7)
#             month_sin = np.sin(2 * np.pi * month / 12)
#             month_cos = np.cos(2 * np.pi * month / 12)
            
#             # Holiday check
#             is_holiday = 0
#             try:
#                 calendar = IrishBankHolidays()
#                 holidays = calendar.holidays(start=future_datetime, end=future_datetime)
#                 is_holiday = int(future_datetime.normalize() in holidays)
#             except:
#                 pass
            
#             # Seasonal indicators
#             is_summer = int(month in [6, 7, 8])
#             is_winter = int(month in [12, 1, 2])
            
#             # Peak hour indicators
#             is_peak_hour = int(hour == 20)
#             is_low_hour = int(hour == 8)
            
#             future_dates.append({
#                 'Datetime': future_datetime,
#                 'Hospital': hospital,
#                 'Hour': hour,
#                 'DayOfWeek': day_of_week,
#                 'DayOfMonth': day_of_month,
#                 'Month': month,
#                 'Quarter': quarter,
#                 'WeekOfYear': week_of_year,
#                 'IsWeekend': is_weekend,
#                 'IsMonday': is_monday,
#                 'IsFriday': is_friday,
#                 'Hour_sin': hour_sin,
#                 'Hour_cos': hour_cos,
#                 'Day_sin': day_sin,
#                 'Day_cos': day_cos,
#                 'Month_sin': month_sin,
#                 'Month_cos': month_cos,
#                 'IsHoliday': is_holiday,
#                 'IsSummer': is_summer,
#                 'IsWinter': is_winter,
#                 'IsPeakHour': is_peak_hour,
#                 'IsLowHour': is_low_hour,
#                 'Hospital_Code': hospital_code,
#                 'Additional_Capacity': additional_capacity
#             })
    
#     return pd.DataFrame(future_dates)

# def forecast_with_lags(model, historical_data, future_df, features):
#     """Enhanced forecast function with better lag handling"""
#     # Get last values for initial lags
#     last_values = historical_data['Value'].tail(7).values
    
#     predictions = []
#     current_lags = list(reversed(last_values))  # Most recent first
    
#     # Calculate rolling statistics from historical data
#     historical_mean_3 = historical_data['Value'].tail(3).mean()
#     historical_mean_7 = historical_data['Value'].tail(7).mean()
    
#     for idx, row in future_df.iterrows():
#         try:
#             # Start with base features
#             feature_values = []
            
#             for feature in features:
#                 if feature.startswith('Lag_'):
#                     # Handle lag features
#                     lag_num = int(feature.split('_')[1]) - 1
#                     if lag_num < len(current_lags):
#                         feature_values.append(current_lags[lag_num])
#                     else:
#                         feature_values.append(0)
#                 elif feature == 'Rolling_Mean_3':
#                     feature_values.append(historical_mean_3)
#                 elif feature == 'Rolling_Mean_7':
#                     feature_values.append(historical_mean_7)
#                 else:
#                     # Regular features
#                     if feature in row:
#                         feature_values.append(row[feature])
#                     else:
#                         feature_values.append(0)
            
#             # Make prediction
#             feature_vector = np.array(feature_values).reshape(1, -1)
#             pred = model.predict(feature_vector)[0]
#             pred = max(0, pred)  # Ensure non-negative
#             predictions.append(pred)
            
#             # Update lags for next prediction
#             current_lags = [pred] + current_lags[:6]
            
#             # Update rolling statistics
#             if len(predictions) >= 3:
#                 historical_mean_3 = np.mean(predictions[-3:])
#             if len(predictions) >= 7:
#                 historical_mean_7 = np.mean(predictions[-7:])
            
#         except Exception as e:
#             st.error(f"Error in prediction step {idx}: {e}")
#             predictions.append(historical_data['Value'].mean())  # Fallback to mean
    
#     return predictions

# def plot_forecasts(historical_data, forecast_data, metric_name, hospital_name):
#     """Create interactive plotly chart"""
#     fig = go.Figure()
    
#     # Historical data
#     fig.add_trace(go.Scatter(
#         x=historical_data['Datetime'],
#         y=historical_data['Value'],
#         mode='lines+markers',
#         name='Historical',
#         line=dict(color='blue'),
#         marker=dict(size=4)
#     ))
    
#     # Forecast data
#     fig.add_trace(go.Scatter(
#         x=forecast_data['Datetime'],
#         y=forecast_data['Predicted'],
#         mode='lines+markers',
#         name='Forecast',
#         line=dict(color='red', dash='dash'),
#         marker=dict(size=6, symbol='diamond')
#     ))
    
#     # Add vertical line to separate historical and forecast
#     last_historical_date = historical_data['Datetime'].max()
    
#     fig.add_shape(
#         type="line",
#         x0=last_historical_date,
#         x1=last_historical_date,
#         y0=0,
#         y1=1,
#         yref="paper",
#         line=dict(color="gray", width=2, dash="dot"),
#     )
    
#     fig.add_annotation(
#         x=last_historical_date,
#         y=0.95,
#         yref="paper",
#         text="Forecast Start",
#         showarrow=False,
#         bgcolor="white",
#         bordercolor="gray",
#         borderwidth=1
#     )
    
#     fig.update_layout(
#         title=f'{metric_name} Forecast - {hospital_name}',
#         xaxis_title='Date',
#         yaxis_title=f'{metric_name} Count',
#         hovermode='x unified',
#         showlegend=True,
#         height=500
#     )
    
#     return fig

# def add_forecasting_insights():
#     with st.expander("💡 Forecasting Insights & Tips", expanded=False):
#         st.subheader("Data Requirements")
#         st.markdown("""
#         For accurate forecasting, you need:
#         * **Minimum 30 records** per hospital-metric combination
#         * **Consistent time intervals** (8am, 2pm, 8pm readings)
#         * **Recent data** (within last 6 months ideally)
#         * **Complete records** (avoid too many missing values)
#         """)
        
#         st.subheader("Understanding Your Results")
#         st.markdown("""
#         * **RMSE (Root Mean Square Error)**: Lower is better. This shows average prediction error.
#         * **Historical vs Forecast**: The chart shows your data pattern and predicted future values.
#         * **Validation**: Shows how well the model predicts on recent historical data.
#         """)

# # --- Streamlit UI ---
# st.title("Emergency Department Forecasting (Ireland)")
# st.markdown("Upload your ED Excel file, select hospital(s), and generate 7-day forecasts")

# # Add forecast days control
# forecast_days = st.sidebar.slider("Forecast Days", 1, 14, 7)

# uploaded_file = st.file_uploader("Upload your ED Excel File", type=["xlsx"])

# if uploaded_file:
#     try:
#         # Load and prepare data
#         df = pd.read_excel(uploaded_file)
#         df_long = prepare_data(df)
        
#         # Show data info
#         st.success(f"✅ Data loaded successfully! {len(df_long)} records found.")
        
#         # Get unique hospitals for selection
#         hospitals = sorted(df_long['Hospital'].unique())
        
#         # Show data summary
#         data_summary = df_long.groupby(['Hospital', 'Metric']).size().reset_index(name='Records')
#         st.subheader("📊 Data Summary by Hospital & Metric")
#         st.dataframe(data_summary, use_container_width=True)
        
#         # Hospital selection
#         st.sidebar.header("Forecast Settings")
#         hospital_option = st.sidebar.selectbox(
#             "Select Hospital:",
#             options=["All Hospitals"] + hospitals
#         )
        
#         # Run button
#         run_forecast = st.sidebar.button("🚀 Run Forecast", type="primary")
        
#         if run_forecast:
#             st.header("📊 Forecast Results")
            
#             # Determine which hospitals to process
#             if hospital_option == "All Hospitals":
#                 selected_hospitals = hospitals
#             else:
#                 selected_hospitals = [hospital_option]
            
#             # Base features (without lags)
#             base_features = [
#                 'Hour', 'DayOfWeek', 'DayOfMonth', 'Month', 'Quarter', 'WeekOfYear',
#                 'IsWeekend', 'IsMonday', 'IsFriday',
#                 'Hour_sin', 'Hour_cos', 'Day_sin', 'Day_cos', 'Month_sin', 'Month_cos',
#                 'IsHoliday', 'IsSummer', 'IsWinter', 'IsPeakHour', 'IsLowHour',
#                 'Hospital_Code', 'Additional_Capacity'
#             ]
            
#             # Process each selected hospital
#             for hospital in selected_hospitals:
#                 st.subheader(f"🏥 {hospital}")
                
#                 # Filter data for current hospital
#                 hospital_data = df_long[df_long['Hospital'] == hospital].copy()
#                 hospital_code = hospital_data['Hospital_Code'].iloc[0]
#                 additional_capacity = hospital_data['Additional_Capacity'].fillna(0).iloc[0]
                
#                 # Get last date for this hospital
#                 last_date = hospital_data['Datetime'].max().date()
                
#                 # Process ED and Trolley separately
#                 for metric in ['ED', 'Trolley']:
#                     metric_data = hospital_data[hospital_data['Metric'] == metric].copy()
#                     metric_data = metric_data.sort_values('Datetime').reset_index(drop=True)
                    
#                     # Check if we have sufficient data
#                     if len(metric_data) < 10:
#                         st.warning(f"⚠️ Insufficient data for {metric} at {hospital} ({len(metric_data)} records). Need at least 10 records.")
#                         continue
                    
#                     st.info(f"Processing {metric} for {hospital} ({len(metric_data)} records)")
                    
#                     # Add lag features intelligently
#                     metric_data, lag_features = add_lag_features_smart(metric_data)
                    
#                     # Combine all features
#                     all_features = base_features + lag_features
                    
#                     # Remove any features that don't exist in the data
#                     available_features = [f for f in all_features if f in metric_data.columns]
                    
#                     # Final check for training data
#                     training_data = metric_data.dropna(subset=available_features + ['Value'])
                    
#                     if len(training_data) < 5:
#                         st.warning(f"⚠️ After preprocessing, insufficient data for {metric} at {hospital} ({len(training_data)} records). Need at least 5 records.")
#                         continue
                    
#                     # Train model
#                     X = training_data[available_features]
#                     y = training_data['Value']
                    
#                     # Use simple split for small datasets
#                     if len(X) < 20:
#                         # Use all data for training with small datasets
#                         X_train, X_test = X, X.tail(min(3, len(X)))
#                         y_train, y_test = y, y.tail(min(3, len(y)))
#                     else:
#                         # Use time-based split for larger datasets
#                         split_idx = int(len(X) * 0.8)
#                         X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
#                         y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
                    
#                     # Simple model configuration for small datasets
#                     model = lgb.LGBMRegressor(
#                         n_estimators=min(100, len(X_train) * 2),  # Adapt to data size was 100.......................................................................
#                         learning_rate=0.1,
#                         max_depth=min(6, len(available_features) // 2 + 1),
#                         num_leaves=min(31, 2 ** min(6, len(available_features) // 2 + 1) - 1),
#                         subsample=0.8,
#                         colsample_bytree=0.8,
#                         verbose=-1,
#                         random_state=42,
#                         force_col_wise=True
#                     )
                    
#                     try:
#                         model.fit(X_train, y_train)
                        
#                         # Calculate model performance
#                         y_pred_test = model.predict(X_test)
#                         rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                        
#                         # Create future dates
#                         future_df = create_future_dates(
#                             pd.to_datetime(last_date), 
#                             hospital, 
#                             hospital_code, 
#                             additional_capacity, 
#                             days=forecast_days
#                         )
                        
#                         # Generate forecasts
#                         predictions = forecast_with_lags(model, training_data, future_df, available_features)
#                         future_df['Predicted'] = predictions
                        
#                         # Display metrics
#                         col1, col2, col3 = st.columns(3)
#                         with col1:
#                             st.metric(f"{metric} RMSE", f"{rmse:.2f}")
#                         with col2:
#                             st.metric(f"Training Records", f"{len(X_train)}")
#                         with col3:
#                             st.metric(f"Last {metric} Value", f"{training_data['Value'].iloc[-1]:.0f}")
                        
#                         # Create and display plot
#                         fig = plot_forecasts(
#                             training_data.tail(21),  # Show last 21 points (1 week at 3x/day)
#                             future_df,
#                             metric,
#                             hospital
#                         )
#                         st.plotly_chart(fig, use_container_width=True)
                        
#                         # Show forecast table
#                         with st.expander(f"📋 {metric} Forecast Details"):
#                             forecast_display = future_df[['Datetime', 'Predicted']].copy()
#                             forecast_display['Predicted'] = forecast_display['Predicted'].round(1)
#                             forecast_display['Date'] = forecast_display['Datetime'].dt.date
#                             forecast_display['Time'] = forecast_display['Datetime'].dt.strftime('%I:%M %p')
#                             st.dataframe(
#                                 forecast_display[['Date', 'Time', 'Predicted']],
#                                 use_container_width=True
#                             )
                        
#                         # Download button for forecasts
#                         csv_data = future_df[['Datetime', 'Hospital', 'Predicted']].copy()
#                         csv_data['Metric'] = metric
#                         st.download_button(
#                             f"📥 Download {metric} Forecast CSV",
#                             csv_data.to_csv(index=False),
#                             file_name=f"{hospital}_{metric}_forecast.csv",
#                             mime="text/csv",
#                             key=f"{hospital}_{metric}_download"
#                         )
                        
#                     except Exception as e:
#                         st.error(f"❌ Error training model for {metric} at {hospital}: {str(e)}")
#                         st.info("This might be due to insufficient data or data quality issues.")
                    
#                     st.divider()
        
#         # Add insights
#         add_forecasting_insights()
        
#     except Exception as e:
#         st.error(f"❌ Error processing file: {str(e)}")
#         st.info("Please check that your Excel file contains the required columns and data format.")

# else:
#     st.info("👆 Please upload an Excel file to begin forecasting")
    
#     # Show expected file format
#     with st.expander("📋 Expected File Format"):
#         st.markdown("""
#         **Required columns:**
#         - `Hospital Group Name`
#         - `Hospital` 
#         - `Date`
#         - `DayGAR`
#         - `Tracker8am`, `Tracker2pm`, `Tracker8pm` (ED counts)
#         - `TimeTotal_8am`, `TimeTotal_2pm`, `TimeTotal_8pm` (Trolley counts)
#         - `AdditionalCapacityOpen Morning` (Additional capacity)
        
#         **Data should contain:**
#         - Historical ED and trolley wait data
#         - Multiple hospitals (optional)
#         - At least 10-15 records per hospital-metric combination for basic forecasting
#         - At least 30+ records per hospital-metric combination for reliable forecasting
#         """)
