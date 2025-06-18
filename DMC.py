import streamlit as st
import pandas as pd
import numpy as np
# Import all required ML libraries
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.ensemble import GradientBoostingRegressor # A good scikit-learn equivalent
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday
from pandas.tseries.offsets import DateOffset
from dateutil.rrule import MO
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# --- Define Irish Bank Holidays ---
class IrishBankHolidays(AbstractHolidayCalendar):
    """
    Defines the rules for Irish Bank Holidays.
    Observance logic (nearest_workday, first Monday) is included.
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
    # This assumes Additional_Capacity is constant for a given Hospital and Date.
    df['Additional_Capacity'] = df.groupby(['Hospital', 'Date'])['Additional_Capacity'].transform('first')
    df['Additional_Capacity'] = df['Additional_Capacity'].fillna(0)

    # Define common id_vars for melting. DayGAR is removed as it's not used as a feature
    # and its future values are not easily predictable for forecasting.
    common_id_vars = ['Hospital Group Name', 'Hospital', 'Date', 'Additional_Capacity']

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
        on=['Hospital Group Name', 'Hospital', 'Date', 'Additional_Capacity', 'Time'], # DayGAR removed
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

def add_lag_features_smart(df, target_column):
    """
    Adds lag features intelligently to the DataFrame based on available data for a specific target column.
    Lag features are time-shifted values of the target, useful for capturing temporal dependencies.
    This function now applies lags and rolling features per hospital group.
    """
    df_copy = df.copy() # Work on a copy to avoid modifying the original DataFrame during groupby operations
    lag_features = []

    # Ensure the DataFrame is sorted by Hospital and Datetime before group-wise operations
    df_copy = df_copy.sort_values(by=['Hospital', 'Datetime'])

    # Apply lags and rolling means grouped by hospital
    # This is crucial to ensure lags are only calculated within each hospital's time series
    for hospital_name, hospital_group in df_copy.groupby('Hospital'):
        # Determine maximum safe lag based on the size of the filtered data for *this* hospital group
        max_safe_lag = min(7, len(hospital_group) // 4)
        if max_safe_lag < 1:
            st.warning(f"Very limited data ({len(hospital_group)} records) for {hospital_name}. Skipping lag features for {target_column}.")
            continue

        # Add lag features for the current target_column (e.g., 'ED Beds', 'Trolleys', 'Capacity')
        for i in range(1, max_safe_lag + 1):
            lag_col = f'Lag_{target_column}_{i}' # Lag column names are specific to the target (e.g., 'Lag_ED Beds_1')
            df_copy.loc[hospital_group.index, lag_col] = hospital_group[target_column].shift(i)
            if lag_col not in lag_features: # Add to list only once to avoid duplicates
                lag_features.append(lag_col)

        # Add rolling mean features if enough data exists for the current hospital group
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

    # Fill NaN values created by shifting/rolling operations.
    # It's better to fill NaNs after all group-wise operations are complete,
    # or handle them during model training (e.g., CatBoost can handle NaNs).
    # Here, we'll forward-fill, then back-fill, then default to 0 for robustness.
    for feature in lag_features:
        df_copy[feature] = df_copy[feature].fillna(method='ffill').fillna(method='bfill').fillna(0)

    return df_copy, lag_features


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
                # Check for holidays only within the specific future datetime
                holidays = calendar.holidays(start=future_datetime.normalize(), end=future_datetime.normalize())
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
    historical_data should already be filtered for the specific hospital and target column.
    Ensures predictions are non-negative and rounded to zero decimal places.
    """
    if historical_data.empty:
        st.error("Historical data for lag forecasting is empty. Cannot generate forecasts.")
        return [0] * len(future_df) # Return zeros if no historical data

    predictions = []

    # Get the last few actual values from historical data for initial lag features
    # Ensure to get at least 7 values or pad with 0s if not enough history
    last_values = historical_data[target_column].tail(7).values
    current_lags = list(reversed(last_values.tolist())) # Reverse to get most recent first
    # Pad with zeros if less than 7 historical points are available for initial lags
    current_lags = current_lags + [0] * (7 - len(current_lags))

    # Initialize rolling statistics with historical data's tail
    # Provide fallbacks if not enough data for actual rolling calculation
    historical_mean_3 = historical_data[target_column].tail(3).mean() if len(historical_data) >= 3 else historical_data[target_column].mean()
    historical_mean_7 = historical_data[target_column].tail(7).mean() if len(historical_data) >= 7 else historical_data[target_column].mean()


    for idx, row in future_df.iterrows():
        try:
            feature_values = [] # List to build the feature vector for the current prediction

            for feature in features:
                if feature.startswith(f'Lag_{target_column}_'): # Check if the feature is a lag for the current target
                    lag_num = int(feature.replace(f'Lag_{target_column}_', '')) - 1 # Extract lag order (0-indexed)
                    if lag_num < len(current_lags):
                        feature_values.append(current_lags[lag_num]) # Use the dynamically updated lag value
                    else:
                        feature_values.append(0) # Default to 0 if lag is not available (should be handled by padding)
                elif feature == f'Rolling_Mean_3_{target_column}': # Check for target-specific rolling mean 3
                    feature_values.append(historical_mean_3)
                elif feature == f'Rolling_Mean_7_{target_column}': # Check for target-specific rolling mean 7
                    feature_values.append(historical_mean_7)
                else:
                    # For all other (non-lag, non-rolling) features, use their values from the future_df row
                    if feature in row:
                        feature_values.append(row[feature])
                    else:
                        # This case indicates a potential mismatch between model features and future_df columns
                        st.warning(f"Feature '{feature}' not found in future_df row, defaulting to 0. Check feature consistency.")
                        feature_values.append(0)

            # Ensure the feature vector has the correct number of features as expected by the model
            if len(feature_values) != len(features):
                st.error(f"Feature vector length mismatch: Expected {len(features)}, got {len(feature_values)}. Skipping prediction.")
                predictions.append(historical_data[target_column].mean()) # Fallback
                continue

            # Make the prediction using the constructed feature vector
            feature_vector = np.array(feature_values).reshape(1, -1)
            pred = model.predict(feature_vector)[0]
            pred = max(0, pred) # Ensure predictions are non-negative
            pred = round(pred)   # Round to nearest whole number
            predictions.append(pred)

            # Update the lag values for the next prediction step
            # The most recent prediction becomes the first lag for the next step
            current_lags = [pred] + current_lags[:6] # Keep a window of 7 lags

            # Update rolling statistics based on the new prediction
            # Create a temporary list including current prediction and recent historicals
            # Ensure we have enough data points for the rolling calculations
            temp_rolling_data = list(reversed(predictions)) + list(reversed(historical_data[target_column].values))
            if len(temp_rolling_data) >= 3:
                historical_mean_3 = np.mean(temp_rolling_data[:3])
            if len(temp_rolling_data) >= 7:
                historical_mean_7 = np.mean(temp_rolling_data[:7])


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

def get_ml_model(model_name: str, X_train: pd.DataFrame, y_train: pd.Series):
    """
    Initializes and returns the selected machine learning model with appropriate hyperparameters.
    """
    if model_name == "CatBoost":
        model = cb.CatBoostRegressor(
            iterations=min(3000, len(X_train) * 3), # Dynamic iterations, capped at 1000
            learning_rate=0.08,
            depth=6,
            subsample=0.8,
            colsample_bylevel=0.8,
            l2_leaf_reg=3,
            verbose=False,
            random_state=42,
            allow_writing_files=False, # Essential for Streamlit deployment
            bagging_temperature=1,
            od_type='Iter',
            od_wait=50,
            loss_function='RMSE' # Specify RMSE loss
        )
    elif model_name == "LightGBM":
        model = lgb.LGBMRegressor(
            n_estimators=min(300, len(X_train) * 3), # Dynamic estimators, capped
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1, # No limit
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1, # Use all available cores
            objective='regression_l1' # MAE objective, often more robust to outliers
        )
    elif model_name == "XGBoost":
        model = xgb.XGBRegressor(
            n_estimators=min(500, len(X_train) * 3), # Dynamic estimators, capped
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1, # Use all available cores
            objective='reg:squarederror' # Standard RMSE objective
        )
    elif model_name == "GradientBoosting (Scikit-learn)":
        model = GradientBoostingRegressor(
            n_estimators=min(1000, len(X_train) * 2), # Fewer estimators for GB
            learning_rate=0.01,
            max_depth=5,
            subsample=0.8,
            random_state=42,
            loss='huber' # Huber loss is more robust to outliers than squared_error
        )
    else:
        st.error("Invalid model selected. Defaulting to CatBoost.")
        return get_ml_model("CatBoost", X_train, y_train) # Fallback

    return model

# --- Streamlit UI ---
st.title("Emergency Department Forecasting (Ireland)")
st.markdown("Upload your ED Excel file, select hospital(s), and generate 7-day forecasts.")

# Sidebar control for number of forecast days
forecast_days = st.sidebar.slider("Forecast Days", 1, 14, 7)

# Model selection dropdown in sidebar
st.sidebar.header("Model Settings")
model_option = st.sidebar.selectbox(
    "Select ML Model:",
    options=["CatBoost", "LightGBM", "XGBoost", "GradientBoosting (Scikit-learn)"]
)

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
            trolley_records=('Trolleys', 'count'),
            capacity_records=('Capacity', 'count') # Added capacity records to summary
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

            # Define base features that are common to all target columns
            base_features = [
                'Hour', 'DayOfWeek', 'DayOfMonth', 'Month', 'Quarter', 'WeekOfYear',
                'IsWeekend', 'IsMonday', 'IsFriday',
                'Hour_sin', 'Hour_cos', 'Day_sin', 'Day_cos', 'Month_sin', 'Month_cos',
                'IsHoliday', 'IsSummer', 'IsWinter', 'IsPeakHour', 'IsLowHour',
                'Hospital_Code'
            ]

            # Iterate through each selected hospital and metric ('ED Beds', 'Trolleys', 'Capacity')
            for hospital in selected_hospitals:
                st.subheader(f"🏥 {hospital}")

                # Filter data for the current hospital
                hospital_data = df_processed[df_processed['Hospital'] == hospital].copy()

                # Extract hospital code and current additional capacity (now 'Capacity') for future dates
                hospital_code = hospital_data['Hospital_Code'].iloc[0] if not hospital_data.empty else 0
                # Use the last known capacity value for forecasting future capacity or as a feature for other forecasts
                current_hospital_capacity_val = hospital_data['Capacity'].fillna(0).iloc[-1] if not hospital_data.empty else 0


                # Get the last date from the historical data for creating future dates
                last_date = hospital_data['Datetime'].max().date() if not hospital_data.empty else datetime.now().date()

                # Process 'ED Beds', 'Trolleys', and 'Capacity' forecasts separately
                for target_col_name in ['ED Beds', 'Trolleys', 'Capacity']: # Added 'Capacity' as a target
                    # Check if we have sufficient data for the current target column
                    if hospital_data[target_col_name].count() < 10:
                        st.warning(f"⚠️ Insufficient data for '{target_col_name}' at {hospital} ({hospital_data[target_col_name].count()} records). Need at least 10 records for meaningful forecasting.")
                        continue

                    st.info(f"Processing '{target_col_name}' for {hospital} using {model_option}")

                    # Add lag and rolling features specifically for the current target column,
                    # ensuring they are calculated within the hospital's data.
                    # This now correctly handles grouping internally.
                    data_with_lags, lag_features = add_lag_features_smart(hospital_data.copy(), target_col_name)

                    # Determine features for this specific model
                    model_features = base_features[:]
                    # If the target is not 'Capacity', then 'Capacity' can be a feature
                    if target_col_name != 'Capacity' and 'Capacity' in data_with_lags.columns:
                        model_features.append('Capacity')

                    all_features_for_model = model_features + lag_features
                    # Ensure features exist in the DataFrame and are not the target itself
                    available_features = [f for f in all_features_for_model if f in data_with_lags.columns and f != target_col_name]
                    training_data = data_with_lags.dropna(subset=[target_col_name] + available_features)

                    if len(training_data) < 5:
                        st.warning(f"⚠️ After preprocessing, insufficient data for '{target_col_name}' at {hospital} ({len(training_data)} records). Need at least 5 records to train a model.")
                        continue

                    X = training_data[available_features]
                    y = training_data[target_col_name]

                    # Initialize the selected model
                    model = get_ml_model(model_option, X, y)

                    # --- Time Series Cross-Validation ---
                    avg_rmse = np.nan # Initialize RMSE
                    if len(X) >= 20: # Only perform CV if enough data for at least 3 splits
                        # n_splits adjusted dynamically to prevent issues with small datasets
                        tscv = TimeSeriesSplit(n_splits=min(5, max(1, len(X) // 10)))
                        fold_rmses = []
                        for fold_idx, (train_index, test_index) in enumerate(tscv.split(X)):
                            X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
                            y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

                            if len(X_train_fold) > 0 and len(X_test_fold) > 0:
                                # Re-initialize model for each fold to ensure clean state
                                fold_model = get_ml_model(model_option, X_train_fold, y_train_fold)
                                fold_model.fit(X_train_fold, y_train_fold) # verbose=False is handled within get_ml_model
                                y_pred_fold = fold_model.predict(X_test_fold)
                                y_pred_fold = np.maximum(0, y_pred_fold).round(0) # Apply non-negative and rounding to fold predictions
                                fold_rmses.append(np.sqrt(mean_squared_error(y_test_fold, y_pred_fold)))
                            else:
                                st.warning(f"Skipping fold {fold_idx+1} due to insufficient data for '{target_col_name}' at {hospital}.")

                        if fold_rmses:
                            avg_rmse = np.mean(fold_rmses)
                            st.info(f"Cross-Validation RMSE for {target_col_name}: {avg_rmse:.2f} (Avg. over {len(fold_rmses)} folds)")
                        else:
                            st.warning(f"Could not perform cross-validation for {target_col_name} due to insufficient data or valid folds.")
                    else:
                        st.info(f"Not enough data for cross-validation for '{target_col_name}' ({len(X)} records). Training on all available data.")
                        # If not enough for CV, train on all data for the final forecast
                        model.fit(X, y) # verbose=False is handled within get_ml_model
                        if len(X) > 0:
                            y_pred_train = model.predict(X)
                            y_pred_train = np.maximum(0, y_pred_train).round(0) # Apply non-negative and rounding
                            avg_rmse = np.sqrt(mean_squared_error(y, y_pred_train))
                            st.info(f"Training RMSE for {target_col_name}: {avg_rmse:.2f} (Trained on all available data)")


                    try:
                        # Train the final model on all available data for the most accurate future forecast
                        # This ensures the final model is trained regardless of CV execution.
                        model = get_ml_model(model_option, X, y) # Re-initialize to ensure clean state for final training
                        model.fit(X, y) # verbose=False is handled within get_ml_model


                        # Create future timestamps for which to generate forecasts
                        future_df = create_future_dates(
                            pd.to_datetime(last_date),
                            hospital,
                            hospital_code,
                            current_hospital_capacity_val, # This capacity is for creating features for other targets
                            days=forecast_days
                        )

                        # Generate predictions for future dates using the trained model
                        # Ensure historical_data passed is specific to the hospital and target
                        predictions = forecast_with_lags(model, training_data, future_df, available_features, target_col_name)
                        future_df['Predicted'] = predictions # Add predictions to the future DataFrame

                        # Display key metrics using Streamlit columns
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(f"{target_col_name} RMSE", f"{avg_rmse:.2f}" if avg_rmse is not np.nan else "N/A") # Display average CV RMSE
                        with col2:
                            st.metric(f"Training Records", f"{len(X)}")
                        with col3:
                            last_val_display = f"{training_data[target_col_name].iloc[-1]:.0f}" if not training_data.empty else "N/A"
                            st.metric(f"Last {target_col_name} Value", last_val_display)

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
                            forecast_display['Predicted'] = forecast_display['Predicted'].round(0).astype(int) # Ensure int for display
                            st.dataframe(
                                forecast_display,
                                use_container_width=True
                            )

                        # Provide a download button for the forecast data
                        csv_data = future_df[['Datetime', 'Hospital', 'Predicted']].copy()
                        csv_data['Metric'] = target_col_name # Identify the metric in the CSV
                        csv_data['Predicted'] = csv_data['Predicted'].astype(int) # Ensure integer for CSV
                        st.download_button(
                            f"📥 Download {target_col_name} Forecast CSV",
                            csv_data.to_csv(index=False),
                            file_name=f"{hospital}_{target_col_name.replace(' ', '_')}_forecast.csv", # Sanitize filename
                            mime="text/csv",
                            key=f"{hospital}_{target_col_name}_download"
                        )

                    except Exception as e:
                        st.error(f"❌ Error during final model training or forecast generation for '{target_col_name}' at {hospital}: {str(e)}")
                        st.info("This might be due to issues with the dataset or model training for future predictions.")

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
        - `Tracker8am`, `Tracker2pm`, `Tracker8pm`: ED bed counts at 8 AM, 2 PM, 8 PM respectively.
        - `TimeTotal_8am`, `TimeTotal_2pm`, `TimeTotal_8pm`: Trolley counts at 8 AM, 2 PM, 8 PM respectively.
        - `AdditionalCapacityOpen Morning`: Any additional capacity opened in the morning for that day.

        **Data should contain:**
        - Historical ED and trolley count data.
        - Data for multiple hospitals (optional, but the app supports it).
        - At least **10-15 records** per hospital-metric combination for basic forecasting.
        - At least **30+ records** per hospital-metric combination for more reliable forecasting.
        """)

