import streamlit as st
import pandas as pd
from prophet import Prophet
import xgboost as xgb
import holidays
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import logging

# Suppress Prophet's verbose output
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING) # Suppress cmdstanpy warnings from Prophet

st.set_page_config(layout="wide", page_title="ED Metrics Hybrid Forecasting")

st.title("🏥 Emergency Department Metrics Hybrid Forecasting (Prophet + XGBoost)")
st.markdown("""
    Upload your raw ED data to forecast key metrics for the next 7 days using a hybrid Prophet-XGBoost model.
    This application performs essential feature engineering and combines the strengths of both models
    to provide robust and accurate predictions.
""")

# --- Data Upload Section ---
st.header("1. Upload Your Data")
uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])

if uploaded_file is not None:
    try:
        # Read Excel file [2]
        df = pd.read_excel(uploaded_file)
        st.success("File uploaded successfully!")
        st.subheader("Raw Data Preview:")
        st.dataframe(df.head())

        # Drop the first column as requested
        if df.shape[1] > 1:
            df = df.iloc[:, 1:]
            st.info("First column dropped as requested.")
            st.subheader("Data Preview after dropping first column:")
            st.dataframe(df.head())
        else:
            st.warning("File has only one column, skipping drop operation.")

        # Ensure 'Date' column exists and is in datetime format
        if 'Date' not in df.columns:
            st.error("Error: 'Date' column not found. Please ensure your Excel has a 'Date' column.")
            st.stop()
        df = pd.to_datetime(df)

        # Hospital Selection [3]
        st.header("2. Select Hospital for Forecasting")
        if 'Hospital' not in df.columns:
            st.error("Error: 'Hospital' column not found. Cannot filter by hospital. Please ensure your Excel has a 'Hospital' column.")
            st.stop()

        unique_hospitals = ['All Hospitals'] + sorted(df['Hospital'].unique().tolist())
        selected_hospital = st.selectbox("Select a Hospital:", unique_hospitals)

        df_filtered = df.copy()
        if selected_hospital!= 'All Hospitals':
            df_filtered = df_filtered[df_filtered['Hospital'] == selected_hospital]
            st.info(f"Data filtered for {selected_hospital}.")
        else:
            # Aggregate for 'All Hospitals' by summing the numerical metrics per day
            numerical_cols_to_sum =
            # Ensure these columns exist before summing
            existing_numerical_cols = [col for col in numerical_cols_to_sum if col in df_filtered.columns]
            if not existing_numerical_cols:
                st.error("Error: No valid metric columns found for aggregation. Please check your column names.")
                st.stop()

            df_filtered = df_filtered.groupby('Date')[existing_numerical_cols].sum().reset_index()
            st.info("Data aggregated for 'All Hospitals' by summing metrics per day.")

        if df_filtered.empty:
            st.warning("No data available for the selected hospital or after aggregation. Please check your data and selection.")
            st.stop()

        st.subheader("Data Preview after Hospital Selection/Aggregation:")
        st.dataframe(df_filtered.head())

        # Define the 7 target metrics and their corresponding time suffixes
        # Based on user's column names: Tracker8am, Tracker2pm, Tracker8pm, TimeTotal_8am, TimeTotal_2pm, TimeTotal_8pm, AdditionalCapacityOpen Morning
        target_metrics_info =

        # Check if all target columns exist in the filtered dataframe
        missing_targets = [info for info in target_metrics_info if info not in df_filtered.columns]
        if missing_targets:
            st.error(f"Error: The following expected metric columns are missing from your file: {', '.join(missing_targets)}. Please ensure your Excel contains these columns or adjust the column names in the code.")
            st.stop()

        st.header("3. Data Preprocessing & Feature Engineering")
        st.info("Performing data cleaning, time-based feature extraction, and creating lagged/rolling statistics.")

        all_forecasts = pd.DataFrame()
        all_metrics_df = pd.DataFrame()

        # Define the forecasting horizon
        forecast_horizon_days = 7
        forecast_horizon_periods_per_day = 3 # 8am, 2pm, 8pm
        # Generate future dates for 8am, 2pm, 8pm for the forecast horizon [4]
        last_date = df_filtered.max()
        future_dates_list =
        for i in range(1, forecast_horizon_days + 1):
            current_date = last_date + pd.Timedelta(days=i)
            future_dates_list.append(current_date.replace(hour=8, minute=0, second=0))
            future_dates_list.append(current_date.replace(hour=14, minute=0, second=0))
            future_dates_list.append(current_date.replace(hour=20, minute=0, second=0))

        future_df_base = pd.DataFrame({'ds': future_dates_list})

        # Add holiday features to future_df_base [5, 6]
        us_holidays = holidays.US() # Customize country as needed [7]
        future_df_base['is_holiday'] = future_df_base['ds'].dt.date.map(lambda x: 1 if x in us_holidays else 0)
        future_df_base['day_after_holiday'] = future_df_base['is_holiday'].shift(1).fillna(0) # Postponement effect [7]

        # Iterate through each metric to prepare data, train, and forecast
        for metric_name, time_suffix in target_metrics_info:
            st.write(f"--- Processing: **{metric_name}** ---")

            # Create 'ds' column for the current metric's historical data
            temp_df = df_filtered.copy()
            temp_df['ds'] = temp_df.dt.date.astype(str) + ' ' + time_suffix
            temp_df['ds'] = pd.to_datetime(temp_df['ds'])
            prophet_df_train = temp_df[['ds', metric_name]].rename(columns={metric_name: 'y'}).set_index('ds').sort_index().dropna()

            if prophet_df_train.empty:
                st.warning(f"No valid historical data for {metric_name}. Skipping forecast for this metric.")
                continue

            # Feature Engineering for the current metric's historical data [8, 4]
            prophet_df_train['year'] = prophet_df_train.index.year
            prophet_df_train['month'] = prophet_df_train.index.month
            prophet_df_train['day'] = prophet_df_train.index.day
            prophet_df_train['dayofweek'] = prophet_df_train.index.dayofweek
            prophet_df_train['hour'] = prophet_df_train.index.hour

            # Cyclical features [9]
            prophet_df_train['hour_sin'] = np.sin(2 * np.pi * prophet_df_train['hour'] / 24)
            prophet_df_train['hour_cos'] = np.cos(2 * np.pi * prophet_df_train['hour'] / 24)
            prophet_df_train['dayofweek_sin'] = np.sin(2 * np.pi * prophet_df_train['dayofweek'] / 7)
            prophet_df_train['dayofweek_cos'] = np.cos(2 * np.pi * prophet_df_train['dayofweek'] / 7)
            prophet_df_train['month_sin'] = np.sin(2 * np.pi * prophet_df_train['month'] / 12)
            prophet_df_train['month_cos'] = np.cos(2 * np.pi * prophet_df_train['month'] / 12)

            # Lagged variables [10, 11]
            prophet_df_train[f'y_lag1'] = prophet_df_train['y'].shift(1)
            prophet_df_train[f'y_lag3'] = prophet_df_train['y'].shift(3) # Previous day's same time (assuming 3 observations/day)
            prophet_df_train[f'y_lag21'] = prophet_df_train['y'].shift(21) # Previous week's same time and day (assuming 3 observations/day)

            # Rolling statistics [12, 13]
            prophet_df_train[f'y_rolling_mean_3'] = prophet_df_train['y'].rolling(window=3).mean().shift(1)
            prophet_df_train[f'y_rolling_mean_21'] = prophet_df_train['y'].rolling(window=21).mean().shift(1) # 7 days * 3 observations/day

            # Add holiday features to historical data
            prophet_df_train['is_holiday'] = prophet_df_train.index.dt.date.map(lambda x: 1 if x in us_holidays else 0)
            prophet_df_train['day_after_holiday'] = prophet_df_train['is_holiday'].shift(1).fillna(0)

            # Drop NaNs created by feature engineering
            prophet_df_train_cleaned = prophet_df_train.dropna().reset_index()

            if prophet_df_train_cleaned.empty:
                st.warning(f"Not enough data for {metric_name} after feature engineering. Skipping forecast.")
                continue

            # Prepare future_df for the current metric
            future_df_metric = future_df_base.copy()
            # Filter future_df_metric to only include relevant timestamps for this metric
            if time_suffix == '08:00:00':
                future_df_metric = future_df_metric[future_df_metric['ds'].dt.hour == 8]
            elif time_suffix == '14:00:00':
                future_df_metric = future_df_metric[future_df_metric['ds'].dt.hour == 14]
            elif time_suffix == '20:00:00':
                future_df_metric = future_df_metric[future_df_metric['ds'].dt.hour == 20]

            # Add engineered features to future_df_metric
            future_df_metric['year'] = future_df_metric['ds'].dt.year
            future_df_metric['month'] = future_df_metric['ds'].dt.month
            future_df_metric['day'] = future_df_metric['ds'].dt.day
            future_df_metric['dayofweek'] = future_df_metric['ds'].dt.dayofweek
            future_df_metric['hour'] = future_df_metric['ds'].dt.hour
            future_df_metric['hour_sin'] = np.sin(2 * np.pi * future_df_metric['hour'] / 24)
            future_df_metric['hour_cos'] = np.cos(2 * np.pi * future_df_metric['hour'] / 24)
            future_df_metric['dayofweek_sin'] = np.sin(2 * np.pi * future_df_metric['dayofweek'] / 7)
            future_df_metric['dayofweek_cos'] = np.cos(2 * np.pi * future_df_metric['dayofweek'] / 7)
            future_df_metric['month_sin'] = np.sin(2 * np.pi * future_df_metric['month'] / 12)
            future_df_metric['month_cos'] = np.cos(2 * np.pi * future_df_metric['month'] / 12)

            # For lagged and rolling features in future_df_metric, use last known values from training data [10]
            # This is a simplification. For production, consider recursive forecasting.
            for lag_col in [f'y_lag1', f'y_lag3', f'y_lag21', f'y_rolling_mean_3', f'y_rolling_mean_21']:
                if lag_col in prophet_df_train_cleaned.columns:
                    future_df_metric[lag_col] = prophet_df_train_cleaned[lag_col].iloc[-1]
                else:
                    future_df_metric[lag_col] = 0 # Fallback if feature not present

            # Define exogenous regressors for Prophet and XGBoost [9]
            prophet_exogenous_features = [
                'year', 'month', 'day', 'dayofweek', 'hour',
                'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos',
                'month_sin', 'month_cos', 'is_holiday', 'day_after_holiday',
                'y_lag1', 'y_lag3', 'y_lag21', 'y_rolling_mean_3', 'y_rolling_mean_21'
            ]
            prophet_exogenous_features = [f for f in prophet_exogenous_features if f in prophet_df_train_cleaned.columns]

            # 1. Train Prophet Model [14, 15]
            m = Prophet(
                seasonality_mode='additive',
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True
            )

            for feature in prophet_exogenous_features:
                m.add_regressor(feature)

            m.fit(prophet_df_train_cleaned[['ds', 'y'] + prophet_exogenous_features])

            # Create a combined dataframe for Prophet training and future prediction
            prophet_full_df_for_predict = pd.concat([
                prophet_df_train_cleaned[['ds'] + prophet_exogenous_features],
                future_df_metric[['ds'] + prophet_exogenous_features], ignore_index=True)

            prophet_forecast_full = m.predict(prophet_full_df_for_predict)

            # Extract Prophet's in-sample predictions for residuals calculation
            prophet_in_sample_yhat = prophet_forecast_full.set_index('ds').loc[prophet_df_train_cleaned['ds'], 'yhat']
            prophet_df_train_cleaned[f'prophet_yhat'] = prophet_in_sample_yhat.values # Align index
            prophet_df_train_cleaned[f'residuals'] = prophet_df_train_cleaned['y'] - prophet_df_train_cleaned[f'prophet_yhat']

            # 2. Train XGBoost Model on Residuals [16, 17]
            # Features for XGBoost: all engineered features that are known for the future [18]
            xgb_features_for_residuals = [
                'year', 'month', 'day', 'dayofweek', 'hour',
                'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos',
                'month_sin', 'month_cos', 'is_holiday', 'day_after_holiday',
                'y_lag1', 'y_lag3', 'y_lag21', 'y_rolling_mean_3', 'y_rolling_mean_21'
            ]
            xgb_features_for_residuals = [f for f in xgb_features_for_residuals if f in prophet_df_train_cleaned.columns]

            X_train_xgb = prophet_df_train_cleaned[xgb_features_for_residuals]
            y_train_xgb = prophet_df_train_cleaned[f'residuals']

            xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42)
            xgb_model.fit(X_train_xgb, y_train_xgb)

            # 3. Generate Hybrid Forecasts
            prophet_future_yhat = prophet_forecast_full.set_index('ds').loc[future_df_metric['ds'], 'yhat']

            X_future_xgb = future_df_metric[xgb_features_for_residuals]
            xgb_residuals_forecast = xgb_model.predict(X_future_xgb)

            hybrid_forecast_values = prophet_future_yhat.values + xgb_residuals_forecast

            # Store results
            forecast_metric_df = pd.DataFrame({
                'ds': future_df_metric['ds'],
                f'{metric_name}_forecast': hybrid_forecast_values,
                # For hybrid, confidence intervals are more complex. For simplicity, we'll use Prophet's CIs as a proxy
                # or calculate a simple range around the hybrid forecast.
                # For this demo, we'll just show the point forecast for hybrid.
                # In a real scenario, you'd need to combine uncertainties.
                f'{metric_name}_lower': prophet_forecast_full.set_index('ds').loc[future_df_metric['ds'], 'yhat_lower'].values,
                f'{metric_name}_upper': prophet_forecast_full.set_index('ds').loc[future_df_metric['ds'], 'yhat_upper'].values
            })
            forecast_metric_df['metric'] = metric_name
            all_forecasts = pd.concat([all_forecasts, forecast_metric_df])

            # Store actuals for plotting
            actuals_df = prophet_df_train_cleaned[['ds', 'y']].copy()
            actuals_df['metric'] = metric_name
            actuals_df.rename(columns={'y': 'actual_value'}, inplace=True)
            all_metrics_df = pd.concat([all_metrics_df, actuals_df])

        st.success("Hybrid forecasting complete for all selected metrics!")

        # --- Display Forecasts ---
        st.header("4. Next 7 Days Hybrid Forecasts")

        # Filter forecasts to only show the next 7 days (from the first future timestamp)
        display_forecasts = all_forecasts[all_forecasts['ds'] >= future_df_base['ds'].min()].copy()
        display_forecasts['ds_date'] = display_forecasts['ds'].dt.date
        display_forecasts['ds_time'] = display_forecasts['ds'].dt.time

        # Pivot for better display
        forecast_table_data = display_forecasts.pivot_table(
            index=['ds_date', 'ds_time'],
            columns='metric',
            values=[f'{m}_forecast' for m in target_metrics_info if f'{m}_forecast' in display_forecasts.columns]
        )
        st.dataframe(forecast_table_data)

        # --- Visualization ---
        st.header("5. Hybrid Forecast Visualizations")

        # Create subplots for each metric
        fig = make_subplots(rows=len(target_metrics_info), cols=1,
                            subplot_titles=[f'Hybrid Forecast for {info}' for info in target_metrics_info],
                            shared_xaxes=True, vertical_spacing=0.05)

        for i, (metric_name, _) in enumerate(target_metrics_info):
            actual_data = all_metrics_df[all_metrics_df['metric'] == metric_name]
            forecast_data = all_forecasts[all_forecasts['metric'] == metric_name]

            if actual_data.empty or forecast_data.empty:
                st.warning(f"Cannot plot {metric_name} due to missing data.")
                continue

            # Plot actuals
            fig.add_trace(go.Scatter(x=actual_data['ds'], y=actual_data['actual_value'],
                                     mode='lines', name=f'{metric_name} Actuals',
                                     line=dict(color='blue')),
                          row=i+1, col=1)

            # Plot forecasts
            fig.add_trace(go.Scatter(x=forecast_data['ds'], y=forecast_data[f'{metric_name}_forecast'],
                                     mode='lines', name=f'{metric_name} Hybrid Forecast',
                                     line=dict(color='red', dash='dash')),
                          row=i+1, col=1)

            # Plot confidence interval (from Prophet, as a proxy for hybrid uncertainty)
            fig.add_trace(go.Scatter(x=forecast_data['ds'], y=forecast_data[f'{metric_name}_upper'],
                                     mode='lines', line=dict(width=0),
                                     showlegend=False, name=f'{metric_name} Upper Bound'),
                          row=i+1, col=1)
            fig.add_trace(go.Scatter(x=forecast_data['ds'], y=forecast_data[f'{metric_name}_lower'],
                                     mode='lines', line=dict(width=0),
                                     fillcolor='rgba(255,0,0,0.1)', fill='tonexty',
                                     showlegend=False, name=f'{metric_name} Lower Bound'),
                          row=i+1, col=1)

        fig.update_layout(height=400 * len(target_metrics_info), title_text="ED Metrics Hybrid Forecasts", showlegend=False)
        fig.update_xaxes(title_text="Date and Time")
        fig.update_yaxes(title_text="Value")
        st.plotly_chart(fig, use_container_width=True)

        st.header("6. Considerations for Robustness and Overfitting")
        st.markdown("""
        This hybrid Prophet-XGBoost approach aims to combine the strengths of both models: Prophet for its robust handling of trend and seasonality [14, 9, 15], and XGBoost for its ability to capture complex non-linear relationships and leverage a wide array of engineered features to model the remaining patterns (residuals).[19, 16, 17]

        To ensure the robustness and prevent overfitting, as highlighted in the research:

        *   **Feature Engineering**: We've included various time-based features (hour, day of week, month, year, cyclical components) and basic lagged/rolling statistics. For a production system, more sophisticated lagged features (e.g., specific lags identified by autocorrelation plots [20, 10]), and rolling statistics (e.g., standard deviation, min/max over various windows) should be explored.[12, 13]
        *   **External Factors**: We've included a basic `is_holiday` and `day_after_holiday` feature. For higher accuracy, integrate real-time weather data, local event calendars, and disease outbreak information (e.g., flu season intensity) as exogenous variables.[21, 22, 23]
        *   **Cross-Validation**: For true robustness, implement time-series specific cross-validation (e.g., rolling window or expanding window validation) to rigorously evaluate model performance on unseen data and prevent data leakage.[16, 24]
        *   **Hyperparameter Tuning**: Systematically tune model hyperparameters for both Prophet and XGBoost using techniques like Grid Search, Random Search, or Bayesian Optimization to find the optimal model configuration.[16, 25]
        *   **Multivariate Approach**: The current setup trains independent hybrid models for each metric. For the interconnected ED metrics, a truly multivariate time series model (e.g., a multi-output LSTM or a VARMA-like approach) or a more sophisticated chained forecasting system (where forecasts of one metric inform another) would capture interdependencies more effectively, leading to more coherent predictions.[26, 27, 28]
        *   **"Additional Capacity Open Morning"**: If this is a binary indicator, it might be better modeled as a classification problem, predicting the probability of surge, rather than a continuous value.[29] The output of this classification could then be used as an exogenous variable for the other metrics.
        """)

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.error("Please ensure your Excel file is correctly formatted with a 'Date' column, 'Hospital' column, and the specified metric columns.")
        st.exception(e) # Display full exception for debugging

