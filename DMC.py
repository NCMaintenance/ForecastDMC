# import streamlit as st
import pandas as pd
import numpy as np
import argparse
import itertools
import lightgbm as lgb
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit # Import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday
from pandas.tseries.offsets import DateOffset
from dateutil.rrule import MO
# import shap # Commented out as not installed by user's script and not essential for core MAE/forecast
# import matplotlib.pyplot as plt # Commented out
# import plotly.graph_objects as go # Commented out
# import plotly.express as px # Commented out
# from plotly.subplots import make_subplots # Commented out

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
        print(f"Warning: Very limited data ({len(df)} records). Using minimal features for {target_column}.")
        return df, []

    lag_features = []

    # Generate lag features for the target_column (e.g., 'ED Beds', 'Trolleys', 'Capacity')
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
        df[feature] = df[feature].ffill().bfill().fillna(0)

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
            except Exception: # Catch any errors during holiday calculation
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
    # Ensure these are initialized with data pertinent to the specific target_column
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
            print(f"Error in prediction step {idx} for {target_column}: {e}")
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

# def add_forecasting_insights():
#     """Displays insights and tips about the forecasting process."""
    # with st.expander("💡 Forecasting Insights & Tips", expanded=False):
    #     st.subheader("Data Requirements")
    #     st.markdown("""
    #     For accurate forecasting, you need:
    #     * **Minimum 30 records** per hospital-metric combination
    #     * **Consistent time intervals** (8am, 2pm, 8pm readings are ideal)
    #     * **Recent data** (within the last 6 months ideally for best relevance)
    #     * **Complete records** (avoid too many missing values, as they can hinder model performance)
    #     """)
    #
    #     st.subheader("Understanding Your Results")
    #     st.markdown("""
    #     * **MAE (Mean Absolute Error)**: Lower values indicate better model accuracy. This metric represents the average magnitude of the errors in your predictions.
    #     * **Historical vs. Forecast**: The generated chart clearly visualizes your past data patterns and the predicted future values, allowing for easy comparison.
    #     * **Validation**: The model's performance (MAE) is calculated on a subset of your historical data, showing how well it generalizes to unseen but similar data.
    #     """)

# --- Main execution flow for iterative testing ---
def main(file_path="sample_data.xlsx", forecast_days=7, hospital_option="All Hospitals"):
    # st.title("Emergency Department Forecasting (Ireland)")
    # st.markdown("Upload your ED Excel file, select hospital(s), and generate 7-day forecasts.")

    # Sidebar control for number of forecast days
    # forecast_days = st.sidebar.slider("Forecast Days", 1, 14, 7) # Hardcoded

    # File uploader widget
    # uploaded_file = st.file_uploader("Upload your ED Excel File", type=["xlsx"]) # Hardcoded
    uploaded_file = file_path # Use the provided file_path

    if uploaded_file:
        try:
            # Load and prepare data using the updated function
            df = pd.read_excel(uploaded_file, engine='openpyxl')
            df_processed = prepare_data(df) # Renamed to df_processed as it's no longer just 'long'

            # Show data loading success message and summary
            print(f"✅ Data loaded and processed successfully! {len(df_processed)} records found.")

            # Display a sample of the processed data to confirm the new structure
            # st.subheader("📋 Sample of Processed Data")
            # st.dataframe(df_processed[['Date', 'Time', 'ED Beds', 'Trolleys', 'Capacity']].head())
            print("\n📋 Sample of Processed Data:")
            print(df_processed[['Date', 'Time', 'ED Beds', 'Trolleys', 'Capacity']].head())

            # Get unique hospitals for selection in the sidebar
            hospitals = sorted(df_processed['Hospital'].unique())

            # Show data summary by Hospital and the new metrics
            data_summary = df_processed.groupby(['Hospital']).agg(
                ed_records=('ED Beds', 'count'),
                trolley_records=('Trolleys', 'count'),
                capacity_records=('Capacity', 'count') # Added capacity records to summary
            ).reset_index()
            # st.subheader("📊 Data Summary by Hospital")
            # st.dataframe(data_summary, use_container_width=True)
            print("\n📊 Data Summary by Hospital:")
            print(data_summary)

            # Hospital selection dropdown
            # st.sidebar.header("Forecast Settings")
            # hospital_option = st.sidebar.selectbox( # Hardcoded
            #     "Select Hospital:",
            #     options=["All Hospitals"] + hospitals
            # )

            # Run forecast button
            # run_forecast = st.sidebar.button("🚀 Run Forecast", type="primary") # Assume true for script execution
            run_forecast = True

            if run_forecast:
                # st.header("📊 Forecast Results")
                print("\n📊 Forecast Results")

                # Determine which hospitals to process based on user selection
                if hospital_option == "All Hospitals":
                    selected_hospitals = hospitals
                else:
                    selected_hospitals = [hospital_option]

                # TODO: Add iterative loop for hyperparameter tuning or data subset adjustment here
                # This loop could iterate over different model parameters, feature sets, or data windows.
                # For each iteration, the MAE would be calculated and stored.

                # Define base features that are common to all target columns
            base_features = [
                'Hour', 'DayOfWeek', 'DayOfMonth', 'Month', 'Quarter', 'WeekOfYear',
                'IsWeekend', 'IsMonday', 'IsFriday',
                'Hour_sin', 'Hour_cos', 'Day_sin', 'Day_cos', 'Month_sin', 'Month_cos',
                'IsHoliday', 'IsSummer', 'IsWinter', 'IsPeakHour', 'IsLowHour',
                'Hospital_Code' # Capacity is now a target, so remove from base_features if it's the target
            ]

            # Iterate through each selected hospital and metric ('ED Beds', 'Trolleys', 'Capacity')
            for hospital in selected_hospitals:
                # st.subheader(f"🏥 {hospital}")
                print(f"\n🏥 Processing Hospital: {hospital}")

                # Filter data for the current hospital
                hospital_data = df_processed[df_processed['Hospital'] == hospital].copy()

                # Extract hospital code and additional capacity (now 'Capacity') for future dates
                hospital_code = hospital_data['Hospital_Code'].iloc[0] if not hospital_data.empty else 0
                current_hospital_capacity_val = hospital_data['Capacity'].fillna(0).iloc[0] if not hospital_data.empty else 0


                # Get the last date from the historical data for creating future dates
                last_date = hospital_data['Datetime'].max().date() if not hospital_data.empty else datetime.now().date()

                # Process 'ED Beds', 'Trolleys', and 'Capacity' forecasts separately
                # TODO: This inner loop could also be part of the hyperparameter tuning,
                # potentially testing different parameters for each target column.
                for target_col_name in ['ED Beds', 'Trolleys', 'Capacity']: # Added 'Capacity' as a target
                    # Check if we have sufficient data for the current target column
                    if hospital_data[target_col_name].count() < 10:
                        print(f"Warning: ⚠️ Insufficient data for '{target_col_name}' at {hospital} ({hospital_data[target_col_name].count()} records). Need at least 10 records for meaningful forecasting.")
                        continue

                    print(f"Processing '{target_col_name}' for {hospital} ({hospital_data[target_col_name].count()} records)")

                    metric_specific_data = hospital_data.copy()

                    # Add lag and rolling features specifically for the current target column
                    data_with_lags, lag_features = add_lag_features_smart(metric_specific_data, target_col_name)

                    # Determine features for this specific model
                    model_features = base_features[:]
                    if target_col_name != 'Capacity':
                        model_features.append('Capacity')

                    all_features_for_model = model_features + lag_features
                    available_features = [f for f in all_features_for_model if f in data_with_lags.columns and f != target_col_name]
                    training_data = data_with_lags.dropna(subset=[target_col_name] + available_features)

                    if len(training_data) < 5:
                        print(f"Warning: ⚠️ After preprocessing, insufficient data for '{target_col_name}' at {hospital} ({len(training_data)} records). Need at least 5 records to train a model.")
                        continue

                    X = training_data[available_features]
                    y = training_data[target_col_name]

                    # --- Hyperparameter Tuning Loop ---
                    param_grid = {
                        'n_estimators': [100, 200, 300],
                        'learning_rate': [0.01, 0.05, 0.1],
                        'max_depth': [3, 4, 5],
                        'num_leaves': [20, 30, 40]
                    }
                    # Create all combinations of hyperparameters
                    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

                    best_mae_for_target = float('inf')
                    best_params_for_target = None
                    mae_achieved_lt_3 = False

                    # Default model in case no tuning is successful or data is too small for CV in iteration
                    final_model = lgb.LGBMRegressor( # Default parameters
                        n_estimators=min(600, len(X) * 2), learning_rate=0.1, max_depth=min(4, len(available_features) + 1),
                        num_leaves=min(40, 2 ** min(10, len(available_features) + 1) - 1), subsample=0.8, colsample_bytree=0.8,
                        reg_alpha=0.2, reg_lambda=0.2, verbose=-1, random_state=42, force_col_wise=True
                    )
                    avg_mae = float('nan') # Initialize avg_mae for the target


                    print(f"\n--- Starting Hyperparameter Tuning for {target_col_name} at {hospital} ---")
                    # Only run tuning if there's enough data to make it meaningful (e.g., for CV within iteration)
                    if len(X) >= 20: # Threshold for attempting tuning loop
                        for i, params in enumerate(all_params):
                            print(f"Iteration {i+1}/{len(all_params)} for {target_col_name} at {hospital}: Trying params {params}")
                            iteration_mae = run_forecasting_iteration(
                                training_data_full=training_data.copy(),
                                available_features=available_features,
                                target_col_name=target_col_name,
                                hospital_name=hospital,
                                forecast_days=forecast_days,
                                base_features=base_features,
                                hyperparams=params,
                                last_known_date=last_date,
                                hospital_code_val=hospital_code,
                                capacity_val=current_hospital_capacity_val
                            )

                            if pd.notna(iteration_mae):
                                print(f"MAE for iteration {i+1} with params {params}: {iteration_mae:.4f}")
                                if iteration_mae < best_mae_for_target:
                                    best_mae_for_target = iteration_mae
                                    best_params_for_target = params

                                if iteration_mae < 3:
                                    print(f"🎉 Success! MAE < 3 ({iteration_mae:.4f}) achieved for {target_col_name} at {hospital} with params: {params}")
                                    mae_achieved_lt_3 = True
                                    # Update best_mae and best_params one last time before breaking
                                    if iteration_mae < best_mae_for_target: # Should be redundant due to prior check, but safe
                                        best_mae_for_target = iteration_mae
                                        best_params_for_target = params
                                    break
                            else:
                                print(f"Warning: Iteration {i+1} with params {params} resulted in NaN MAE. Skipping.")

                        if best_params_for_target:
                            print(f"\nBest MAE for {target_col_name} at {hospital} after tuning: {best_mae_for_target:.4f} with parameters: {best_params_for_target}")
                            final_model.set_params(**best_params_for_target) # Set best params to the model
                            avg_mae = best_mae_for_target
                        else:
                            print(f"Warning: Hyperparameter tuning did not find suitable parameters for {target_col_name} at {hospital}. Using default model parameters.")
                            # avg_mae will remain NaN or be based on default model's simple test if data was too small for CV
                            # Need to calculate MAE for default model if tuning didn't run or find params
                            if len(X) > 0: # Ensure there's data
                                final_model.fit(X,y) # Fit default
                                y_pred_default_test = final_model.predict(X.tail(min(5,len(X))))
                                avg_mae = mean_absolute_error(y.tail(min(5,len(y))), y_pred_default_test)
                                print(f"MAE for default model (due to lack of tuning results or small data): {avg_mae:.4f}")
                            else:
                                avg_mae = float('nan')


                    else: # If not enough data for the tuning loop (e.g. < 20 records)
                        print(f"Info: Not enough data for hyperparameter tuning for '{target_col_name}' at {hospital} ({len(X)} records). Training with default parameters and simple evaluation.")
                        if len(X) > 0:
                            final_model.fit(X, y)
                            y_pred_test = final_model.predict(X.tail(min(5,len(X))))
                            avg_mae = mean_absolute_error(y.tail(min(5,len(y))), y_pred_test)
                            print(f"MAE with default parameters (simple eval): {avg_mae:.4f}")
                        else:
                             print(f"Warning: No data to train even a default model for '{target_col_name}' at {hospital}.")
                             final_model = None # Cannot proceed if no data
                             avg_mae = float('nan')

                    print(f"\n--- Hyperparameter Tuning Summary for {target_col_name} at {hospital} ---")
                    if pd.notna(avg_mae):
                        print(f"Final MAE for {target_col_name}: {avg_mae:.4f}")
                        if best_params_for_target:
                             print(f"Achieved with parameters: {best_params_for_target}")
                        else:
                             print("Achieved with default parameters (tuning skipped or unsuccessful).")
                        if mae_achieved_lt_3 or (pd.notna(avg_mae) and avg_mae < 3) : # Check if original success or default got it
                            print(f"Target MAE < 3 was achieved for {target_col_name} at {hospital}.")
                        else:
                            print(f"Target MAE < 3 was NOT achieved. Best MAE was {avg_mae:.4f}.")
                    else:
                        print(f"Warning: MAE could not be determined for {target_col_name} at {hospital}.")
                        final_model = None # Cannot proceed with forecasting


                    if final_model:
                        try:
                            # Train the final model on all available data (X,y) using the determined parameters
                            final_model.fit(X, y)
                            # Create future timestamps for which to generate forecasts
                            future_df = create_future_dates(
                                pd.to_datetime(last_date),
                                hospital,
                                hospital_code,
                                current_hospital_capacity_val, # This capacity is for creating features for other targets
                                days=forecast_days
                            )

                            # Generate predictions for future dates using the trained model
                            predictions = forecast_with_lags(final_model, training_data, future_df, available_features, target_col_name)
                            future_df['Predicted'] = predictions # Add predictions to the future DataFrame

                            # Display key metrics to console
                            print(f"\n--- Final Metrics for {target_col_name} at {hospital} (using determined params) ---")
                            print(f"{target_col_name} MAE: {avg_mae:.4f}")
                            print(f"Training Records: {len(X)}")
                            if best_params_for_target:
                                print(f"Used Hyperparameters: {final_model.get_params()}") # Print all params of the final_model
                            else:
                                print(f"Used Hyperparameters: Default")
                            print(f"Last {target_col_name} Value: {training_data[target_col_name].iloc[-1]:.0f}")

                            # Create and display the forecast plot (optional, can be commented out for speed)
                            # fig = plot_forecasts(
                            #     training_data.tail(21), # Show last 21 historical points (approx. 1 week at 3 readings/day)
                            #     future_df,
                            #     target_col_name, # Pass the metric name for plot title/labels
                            #     hospital
                            # )
                            # # st.plotly_chart(fig, use_container_width=True) # Commented out for non-Streamlit run

                            # Display detailed forecast data to console
                            print(f"\n📋 {target_col_name} Forecast Details for {hospital}:")
                            forecast_display_df = future_df[['Date', 'Time', 'Predicted']].copy()
                            forecast_display_df['Predicted'] = forecast_display_df['Predicted'].round(1)
                            print(forecast_display_df)

                            # Code for download button is fully commented out as it's Streamlit-specific
                            # # Provide a download button for the forecast data
                            # # csv_data = future_df[['Datetime', 'Hospital', 'Predicted']].copy()
                            # # csv_data['Metric'] = target_col_name # Identify the metric in the CSV
                            # # st.download_button( # Commented out
                            # #     f"📥 Download {target_col_name} Forecast CSV",
                            # #     csv_data.to_csv(index=False),
                            # #     file_name=f"{hospital}_{target_col_name.replace(' ', '_')}_forecast.csv", # Sanitize filename
                            # #     mime="text/csv",
                            # #     key=f"{hospital}_{target_col_name}_download"
                            # # )
                            # pass # No longer needed as print statements are active
                        except Exception as e:
                            print(f"❌ Error during final model training or forecast generation for '{target_col_name}' at {hospital}: {str(e)}")
                            print("Info: This might be due to issues with the dataset or model training for future predictions.")

                    # st.divider() # Separator for each metric's results
                    print("-" * 50) # Console separator

            # Display general forecasting insights after all forecasts are run
            # add_forecasting_insights() # Commented out

        except FileNotFoundError:
            print(f"❌ Error: The file '{uploaded_file}' was not found. Please check the path and try again.")
        except Exception as e:
            # Catch and display errors during file processing or initial data preparation
            print(f"❌ Error processing file: {str(e)}")
            print("Info: Please check that your Excel file contains the required columns and data format.")

    else:
        # Instructions for users when no file is uploaded
        print("👆 Please provide a valid Excel file path to begin forecasting.")
        # TODO: Add more detailed instructions on expected file format if needed, similar to the commented out Streamlit expander.
        # print("Expected file format details...")
        #     - `Hospital Group Name`: The group the hospital belongs to.
        #     - `Hospital`: Unique identifier for the hospital.
        #     - `Date`: The date of the observation (e.g., '30/05/2025').
        #     - `DayGAR`: Categorical day information (if available, e.g., 'Day X').
        #     - `Tracker8am`, `Tracker2pm`, `Tracker8pm`: ED bed counts at 8 AM, 2 PM, 8 PM respectively.
        #     - `TimeTotal_8am`, `TimeTotal_2pm`, `TimeTotal_8pm`: Trolley counts at 8 AM, 2 PM, 8 PM respectively.
        #     - `AdditionalCapacityOpen Morning`: Any additional capacity opened in the morning for that day.
        #
        #     **Data should contain:**
        #     - Historical ED and trolley count data.
        #     - Data for multiple hospitals (optional, but the app supports it).
        #     - At least **10-15 records** per hospital-metric combination for basic forecasting.
        #     - At least **30+ records** per hospital-metric combination for more reliable forecasting.
        #     """)

# Function to encapsulate model training and MAE calculation for one iteration
def run_forecasting_iteration(training_data_full, available_features, target_col_name, hospital_name, forecast_days, base_features, hyperparams, last_known_date, hospital_code_val, capacity_val):
    """
    Runs a single forecasting iteration with a given set of hyperparameters.
    Returns the calculated MAE.
    """
    X_iter = training_data_full[available_features]
    y_iter = training_data_full[target_col_name]

    # TODO: Implement data subsetting logic here if needed (e.g., use last N months)
    # For now, using all available historical data passed to this function.

    model_iter = lgb.LGBMRegressor(
        n_estimators=hyperparams['n_estimators'],
        learning_rate=hyperparams['learning_rate'],
        max_depth=hyperparams['max_depth'],
        num_leaves=hyperparams['num_leaves'],
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=0.2,
        verbose=-1,
        random_state=42,
        force_col_wise=True
    )

    current_mae = np.nan
    if len(X_iter) >= 20: # Perform CV if enough data
        tscv_iter = TimeSeriesSplit(n_splits=min(5, len(X_iter) // 10))
        fold_maes_iter = []
        for train_idx, test_idx in tscv_iter.split(X_iter):
            X_train_fold_iter, X_test_fold_iter = X_iter.iloc[train_idx], X_iter.iloc[test_idx]
            y_train_fold_iter, y_test_fold_iter = y_iter.iloc[train_idx], y_iter.iloc[test_idx]

            if len(X_train_fold_iter) > 0 and len(X_test_fold_iter) > 0:
                fold_model_iter = lgb.LGBMRegressor(**model_iter.get_params())
                fold_model_iter.fit(X_train_fold_iter, y_train_fold_iter)
                y_pred_fold_iter = fold_model_iter.predict(X_test_fold_iter)
                fold_maes_iter.append(mean_absolute_error(y_test_fold_iter, y_pred_fold_iter))
            else:
                # This case should ideally not happen if len(X_iter) >= 20 and splits are reasonable
                print(f"Warning: Skipping a CV fold due to insufficient data for {target_col_name} at {hospital_name} during hyperparameter tuning.")

        if fold_maes_iter:
            current_mae = np.mean(fold_maes_iter)
    elif len(X_iter) > 0: # If not enough for CV, train on all data and do a simple test
        model_iter.fit(X_iter, y_iter)
        # Predict on the last few points of the training set as a proxy (not ideal, but better than nothing for small sets)
        num_test_points = min(5, len(X_iter))
        y_pred_test_iter = model_iter.predict(X_iter.tail(num_test_points))
        current_mae = mean_absolute_error(y_iter.tail(num_test_points), y_pred_test_iter)
    else:
        print(f"Warning: Not enough data to train or evaluate model for {target_col_name} at {hospital_name} with current features.")
        return np.nan


    # The following lines for actual future forecasting can be kept if needed,
    # but for hyperparameter tuning, only MAE on historical data (CV or simple test) is typically used.
    # For this subtask, we are asked to print MAE, so we'll focus on that.
    # If full forecast generation per iteration is too slow, this part can be removed from the iteration function.
    # model_iter.fit(X_iter, y_iter) # Re-fit on full data for this iteration's params
    # future_df_iter = create_future_dates(
    #     pd.to_datetime(last_known_date),
    #     hospital_name,
    #     hospital_code_val,
    #     capacity_val,
    #     days=forecast_days
    # )
    # predictions_iter = forecast_with_lags(model_iter, training_data_full, future_df_iter, available_features, target_col_name)
    # future_df_iter['Predicted'] = predictions_iter
    # print(f"\n📋 Iteration Forecast Details for {target_col_name} at {hospital_name} with params {hyperparams}:")
    # print(future_df_iter[['Date', 'Time', 'Predicted']].round(1).head())


    return current_mae


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emergency Department Forecasting Tool")
    parser.add_argument("--file_path", type=str, default="sample_data.xlsx",
                        help="Path to the Excel file containing ED data.")
    # Add other arguments like forecast_days, hospital_option if they need to be CLI configurable
    # parser.add_argument("--forecast_days", type=int, default=7, help="Number of days to forecast.")
    # parser.add_argument("--hospital_option", type=str, default="All Hospitals", help="Specific hospital or 'All Hospitals'.")

    args = parser.parse_args()

    # Pass CLI arguments to main function
    # For now, main directly uses the args.file_path for simplicity as other params are still hardcoded/derived in main
    main(file_path=args.file_path) #, forecast_days=args.forecast_days, hospital_option=args.hospital_option)
