import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
# from prophet.diagnostics import cross_validation, performance_metrics # Not used in current forecasting loop
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import plotly.graph_objs as go
import warnings
from scipy import stats # Not used directly in loop, but pearsonr is
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")
st.title("🏥 Enhanced Hospital Forecasting with Advanced Feature Engineering 📈")

# Sidebar
uploaded = st.sidebar.file_uploader("📂 Upload Excel file", type="xlsx")
hospitals = []
targets = [
    "Tracker8am", "Tracker2pm", "Tracker8pm",
    "AdditionalCapacityOpen Morning",
    "TimeTotal_8am", "TimeTotal_2pm", "TimeTotal_8pm"
]

if not uploaded:
    st.sidebar.info("ℹ️ Please upload your Excel file.")
    st.stop()

# Load data
df = pd.read_excel(uploaded)
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df = df.sort_values('Date')
hospitals = sorted(df['Hospital'].unique())

sel_hosp = st.sidebar.selectbox("🏨 Hospital", ["All"] + hospitals)
sel_target = st.sidebar.selectbox("🎯 Target", ["All"] + targets)
future_days = st.sidebar.slider("⏳ Forecast horizon (days ahead)", 7, 30, 14)
correlation_threshold = st.sidebar.slider("📊 Correlation threshold for feature selection", 0.0, 0.5, 0.1)
max_forecast_horizon_eval = st.sidebar.slider("📊 Horizons for Performance Evaluation (days)", 1, 7, 7, help="Evaluate model MAE at different forecast lead times. This is separate from the main 'Forecast horizon (days ahead)' used for final predictions.")
run_horizon_evaluation = st.sidebar.checkbox("🔬 Run Performance Horizon Evaluation", value=True, help="If checked, evaluates model MAE at different short-term forecast lead times. Uncheck to speed up if this evaluation is not needed.")
run = st.sidebar.button("▶️ Run Forecast")

if not run:
    st.sidebar.info("⚙️ Configure then click Run Forecast")
    st.stop()

def create_advanced_features(df_input, target_col, all_hospital_cols, max_lags=7):
    df_feat = df_input.copy()
    # Ensure 'ds' column exists for temporal features
    if 'ds' not in df_feat.columns:
        raise ValueError("Input DataFrame must contain a 'ds' column with datetime objects.")

    predictor_cols = [col for col in all_hospital_cols if col != target_col and col in df_feat.columns]
    # Use a common name for the target column in feature names, e.g., 'target'
    # but keep original target_col for accessing its data in df_feat

    all_feature_source_cols = predictor_cols + [target_col] # These are the raw columns to derive features from

    # Lags
    for col in all_feature_source_cols:
        col_name_in_feature = 'target' if col == target_col else col # Name used in the feature's column name
        for lag in range(1, max_lags + 1):
            df_feat[f'{col_name_in_feature}_lag_{lag}'] = df_feat[col].shift(lag)
    # Diffs
    for col in all_feature_source_cols:
        col_name_in_feature = 'target' if col == target_col else col
        for lag in [1, 2, 3, 7]: # Defined diff lags
            df_feat[f'{col_name_in_feature}_diff_{lag}'] = df_feat[col].diff(lag)
    # Rolling stats
    windows = [3, 7, 14, 21]
    for col in all_feature_source_cols:
        col_name_in_feature = 'target' if col == target_col else col
        for window in windows:
            # Shift(1) is crucial: rolling stats use data up to t-1 for predicting t
            df_feat[f'{col_name_in_feature}_roll_mean_{window}'] = df_feat[col].rolling(window=window, min_periods=1).mean().shift(1)
            df_feat[f'{col_name_in_feature}_roll_std_{window}'] = df_feat[col].rolling(window=window, min_periods=1).std().shift(1)
            df_feat[f'{col_name_in_feature}_roll_min_{window}'] = df_feat[col].rolling(window=window, min_periods=1).min().shift(1)
            df_feat[f'{col_name_in_feature}_roll_max_{window}'] = df_feat[col].rolling(window=window, min_periods=1).max().shift(1)
            df_feat[f'{col_name_in_feature}_roll_median_{window}'] = df_feat[col].rolling(window=window, min_periods=1).median().shift(1)
    # EWM
    for col in all_feature_source_cols:
        col_name_in_feature = 'target' if col == target_col else col
        for alpha in [0.1, 0.3, 0.5]: # Defined EWM alphas
            df_feat[f'{col_name_in_feature}_ewm_mean_{alpha}'] = df_feat[col].ewm(alpha=alpha, min_periods=1).mean().shift(1)
    # Cross-correlation (only between predictor_cols)
    for i, col1 in enumerate(predictor_cols):
        for j, col2 in enumerate(predictor_cols[i+1:], i+1): # Avoid redundant pairs and self-correlation
            for window in [7, 14]: # Defined cross-corr windows
                df_feat[f'{col1}_{col2}_rolling_corr_{window}'] = df_feat[col1].rolling(window=window, min_periods=1).corr(df_feat[col2]).shift(1)
            df_feat[f'{col1}_{col2}_ratio'] = (df_feat[col1] / (df_feat[col2] + 1e-8)).shift(1) # Add 1e-8 to avoid division by zero
    # Aggregate features (from predictor_cols)
    if len(predictor_cols) > 1:
        predictor_data = df_feat[predictor_cols] # Use original data, then shift
        df_feat['all_metrics_mean'] = predictor_data.mean(axis=1).shift(1)
        df_feat['all_metrics_std'] = predictor_data.std(axis=1).shift(1)
        df_feat['all_metrics_sum'] = predictor_data.sum(axis=1).shift(1)
        df_feat['all_metrics_max'] = predictor_data.max(axis=1).shift(1)
        df_feat['all_metrics_min'] = predictor_data.min(axis=1).shift(1)
    # Temporal features (derived from 'ds' column)
    df_feat['dow'] = df_feat['ds'].dt.dayofweek
    df_feat['month'] = df_feat['ds'].dt.month
    df_feat['quarter'] = df_feat['ds'].dt.quarter
    df_feat['week'] = df_feat['ds'].dt.isocalendar().week.astype(int)
    df_feat['dayofyear'] = df_feat['ds'].dt.dayofyear
    df_feat['is_weekend'] = (df_feat['dow'] >= 5).astype(int)
    df_feat['is_month_start'] = df_feat['ds'].dt.is_month_start.astype(int)
    df_feat['is_month_end'] = df_feat['ds'].dt.is_month_end.astype(int)
    # Cyclical features
    df_feat['dow_sin'] = np.sin(2 * np.pi * df_feat['dow'] / 7)
    df_feat['dow_cos'] = np.cos(2 * np.pi * df_feat['dow'] / 7)
    df_feat['month_sin'] = np.sin(2 * np.pi * df_feat['month'] / 12)
    df_feat['month_cos'] = np.cos(2 * np.pi * df_feat['month'] / 12)
    df_feat['dayofyear_sin'] = np.sin(2 * np.pi * df_feat['dayofyear'] / 365.25) # Use 365.25 for leap years
    df_feat['dayofyear_cos'] = np.cos(2 * np.pi * df_feat['dayofyear'] / 365.25)
    # Interaction features
    df_feat['dow_month_interaction'] = df_feat['dow'] * df_feat['month'] # Example interaction
    # Trend features
    df_feat['linear_trend'] = range(len(df_feat))
    df_feat['quadratic_trend'] = df_feat['linear_trend'] ** 2
    # Volatility (example: for target column, based on its own past)
    # This calculates absolute difference between y(t-1) and its rolling mean at t-1
    for window in [7, 14]:
        rolling_mean_target = df_feat[target_col].rolling(window=window, min_periods=1).mean().shift(1)
        df_feat[f'volatility_{window}'] = (df_feat[target_col].shift(1) - rolling_mean_target).abs()

    return df_feat

def select_features_by_correlation(X, y, threshold=0.1):
    correlations = {}
    # selected_features_init = [] # Not used
    for col in X.columns:
        if not X[col].isna().all(): # Ensure column is not all NaN
            # Align X[col] and y by dropping NaNs from both series based on their shared index
            temp_df = pd.DataFrame({'x_col': X[col], 'y_col': y}).dropna()
            if len(temp_df) < 2: # Pearson R needs at least 2 samples
                continue
            try:
                # Fill remaining NaNs in X[col] with 0 for correlation calculation AFTER alignment
                # This is a simplified strategy; mean/median imputation could be better
                corr, p_value = pearsonr(temp_df['x_col'], temp_df['y_col'])
                if abs(corr) >= threshold and p_value < 0.05: # Use >= for threshold
                    correlations[col] = abs(corr)
            except ValueError: # Handles cases like all constant values after dropna
                pass

    sorted_features_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

    selected_features_multicollinearity = []
    # Multicollinearity check
    for feature, corr_val in sorted_features_corr:
        add_feature = True
        for selected_feature_mc in selected_features_multicollinearity:
            # Align feature and selected_feature_mc
            temp_df_mc = pd.DataFrame({'feat1': X[feature], 'feat2': X[selected_feature_mc]}).dropna()
            if len(temp_df_mc) < 2:
                continue
            try:
                # Similar NaN handling as above for the pair being checked
                feature_corr_mc = pearsonr(temp_df_mc['feat1'], temp_df_mc['feat2'])[0]
                if abs(feature_corr_mc) > 0.8: # Threshold for multicollinearity
                    add_feature = False
                    break
            except ValueError:
                pass
        if add_feature:
            selected_features_multicollinearity.append(feature)

    return selected_features_multicollinearity, correlations


def evaluate_forecast_horizons(X_train, y_train, X_test, y_test, max_horizon=7):
    horizon_results = {}
    for horizon in range(1, max_horizon + 1):
        y_train_h = y_train.shift(-horizon + 1).dropna()
        X_train_h = X_train.iloc[:len(y_train_h)]

        if len(y_train_h) < 30: # Ensure enough samples for training
            st.warning(f"Skipping horizon {horizon} due to insufficient data ({len(y_train_h)} samples)")
            continue

        model = LGBMRegressor(n_estimators=100, learning_rate=0.1, num_leaves=15, max_depth=6, min_child_samples=20, random_state=42)
        tscv = TimeSeriesSplit(n_splits=3) # Use at least 3 splits if data allows
        cv_scores = []

        # Check if X_train_h is empty or too small for even one split
        if len(X_train_h) < tscv.get_n_splits() + 10: # Heuristic check
             st.warning(f"Skipping horizon {horizon} due to insufficient data for CV ({len(X_train_h)} samples)")
             continue

        for train_idx, val_idx in tscv.split(X_train_h):
            if len(train_idx) < 10 or len(val_idx) < 1: # Ensure splits are meaningful
                continue
            X_tr, X_val = X_train_h.iloc[train_idx], X_train_h.iloc[val_idx]
            y_tr, y_val = y_train_h.iloc[train_idx], y_train_h.iloc[val_idx]

            model.fit(X_tr, y_tr)
            pred_val = model.predict(X_val)
            cv_scores.append(mean_absolute_error(y_val, pred_val))

        if cv_scores: # Ensure scores were actually calculated
            horizon_results[horizon] = np.mean(cv_scores)
        else:
            horizon_results[horizon] = np.nan # Indicate failure for this horizon

    return horizon_results

# Main processing loop
h_list = hospitals if sel_hosp == "All" else [sel_hosp]
t_list = targets if sel_target == "All" else [sel_target]
results = [] # To store summary results if needed

# --- MODEL TRAINING AND HISTORICAL PREDICTION SETUP (Simplified for brevity) ---
# This part needs to be robustly handled in the actual script.
# We assume that by the time we reach the future forecasting loop, we have:
# - lgb_model: a trained LGBMRegressor model
# - scaler: a fitted StandardScaler
# - selected_features: list of feature names used for training
# - train_data, test_data: pandas DataFrames
# - df_feat_hist: The DataFrame created by create_advanced_features on historical data (df2)
# - test_pred: Predictions on the test set (y_test portion of historical data)

for hosp in h_list:
    st.header(f"🏥 Hospital: {hosp}")
    df_h = df[df['Hospital'] == hosp].reset_index(drop=True)

    # --- BEGIN MODIFICATION: Display Correlation Matrix ---
    # This section calculates and displays the correlation matrix for the
    # primary target metrics for the currently selected hospital.
    # It helps in understanding relationships between these key metrics.
    # Display correlation matrix for target columns for this hospital
    hospital_target_cols = [col for col in targets if col in df_h.columns and not df_h[col].isna().all()] # Ensure column has some data
    if len(hospital_target_cols) > 1: # Need at least 2 columns to make a correlation matrix
        correlation_matrix = df_h[hospital_target_cols].corr()

        st.subheader(f"Correlation Matrix of Target Metrics for {hosp}")
        st.dataframe(correlation_matrix.style.background_gradient(cmap='coolwarm', axis=None).format("{:.2f}"))
        st.markdown("---") # Add a separator
    elif len(hospital_target_cols) <= 1 and hosp != "All": # Avoid message if "All" hospitals selected and some just don't have data
         st.info(f"Not enough target columns with sufficient data to display a correlation matrix for {hosp}.")
         st.markdown("---") # Add a separator
    # --- END MODIFICATION: Display Correlation Matrix ---


    for tgt in t_list:
        st.subheader(f"🎯 Target: {tgt}")
        if df_h[tgt].isna().any():
            st.warning(f"⚠️ Skipping {tgt} due to null values in target")
            continue

        available_targets_for_hospital = [col for col in targets if col in df_h.columns and not df_h[col].isna().all()]

        # df2 contains 'ds' and all available raw metrics for the hospital, including the current target (but not yet named 'y')
        df2 = df_h[['Date'] + available_targets_for_hospital].rename(columns={'Date': 'ds'})
        # Now, rename the current target column in df2 to 'y' for feature engineering convenience
        df2_renamed_target = df2.rename(columns={tgt: 'y'})

        # Create features on historical data (df2_renamed_target which has 'ds' and 'y')
        # Pass available_targets_for_hospital to know which other metrics can be used for features
        df_feat_hist = create_advanced_features(df2_renamed_target.copy(), 'y', available_targets_for_hospital, max_lags=7)

        # Drop rows with too many NaNs after feature creation (mostly at the beginning due to lags/rolling windows)
        df_feat_hist = df_feat_hist.dropna(thresh=int(len(df_feat_hist.columns) * 0.7)) # Keep rows with at least 70% non-NaNs

        if len(df_feat_hist) < 50: # Check after NaN drop
            st.warning(f"⚠️ Insufficient data for {tgt} after feature engineering (less than 50 rows). Skipping.")
            continue

        n = len(df_feat_hist)
        train_size = int(0.75 * n) # Or another split ratio

        train_df = df_feat_hist.iloc[:train_size].copy()
        test_df = df_feat_hist.iloc[train_size:].copy()

        # Identify feature columns for selection (exclude 'ds', 'y', and original raw data columns)
        # Original raw data columns (like 'Tracker8am', etc.) should not be directly used if they are targets or 'y'
        # and their derived features (lags, rolling of 'Tracker8am') are what we want.
        potential_feature_cols = [col for col in df_feat_hist.columns if col not in ['ds', 'y'] + available_targets_for_hospital]

        selected_features, feature_correlations = select_features_by_correlation(
            train_df[potential_feature_cols].fillna(0), # Use .fillna(0) here as a precaution for correlation
            train_df['y'],
            threshold=correlation_threshold
        )

        if feature_correlations:
            st.subheader(f"Top 20 Correlated Features for {tgt} (Absolute Correlation with Target)")
            sorted_correlations = sorted(feature_correlations.items(), key=lambda item: item[1], reverse=True)
            top_20_features = sorted_correlations[:20]

            if top_20_features:
                df_top_corr_features = pd.DataFrame(top_20_features, columns=['Feature', 'Absolute Correlation'])
                # Format correlation to 4 decimal places for better readability
                st.dataframe(df_top_corr_features.style.format({'Absolute Correlation': "{:.4f}"}))
            else:
                # This case might occur if feature_correlations is not empty but all values were below the threshold
                # or if feature_correlations was empty to begin with (covered by outer else)
                st.info(f"No features met the correlation threshold for {tgt} to display in top 20.")
        else:
            st.info(f"No feature correlations were computed for {tgt} (e.g., due to insufficient data or all features failing criteria).")

        if not selected_features:
            st.warning(f"⚠️ No features selected for {tgt} based on correlation threshold (and multicollinearity check). Skipping model training.")
            continue

        X_train = train_df[selected_features]
        y_train = train_df['y']
        X_test = test_df[selected_features]
        y_test = test_df['y']

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.fillna(0)) # Fill NaNs before scaling
        X_test_scaled = scaler.transform(X_test.fillna(0))     # Fill NaNs before scaling

        # Evaluate performance across different horizons
        if run_horizon_evaluation:
            if len(X_train) > 30 and len(y_train) > 30 and max_forecast_horizon_eval > 0:
                st.write("Evaluating performance for different short-term forecast horizons...")
                # Pass X_train (selected features, unscaled), y_train.
                # X_test, y_test are part of the function signature but not directly used in its CV loop.
                horizon_results = evaluate_forecast_horizons(X_train, y_train, X_test, y_test, max_horizon=max_forecast_horizon_eval)
                if horizon_results:
                    st.subheader("Performance across different short-term forecast horizons (Lower MAE is better):")
                    # Filter out NaN results before creating DataFrame
                    filtered_horizon_results = {h:mae for h, mae in horizon_results.items() if pd.notna(mae)}
                    if filtered_horizon_results:
                        horizon_df = pd.DataFrame.from_dict(filtered_horizon_results, orient='index', columns=['Mean CV MAE'])
                        horizon_df.index.name = 'Forecast Horizon (Days)'
                        st.dataframe(horizon_df.sort_index())
                    else:
                        st.info("Could not evaluate performance across different horizons (e.g., results were NaN for all horizons).")
                else:
                    st.info("Could not evaluate performance across different horizons (e.g., due to insufficient data for any horizon).")
            elif max_forecast_horizon_eval <= 0 : # If main checkbox is true, but slider is 0
                 st.info("Performance Horizon Evaluation was enabled, but 'Horizons for Performance Evaluation (days)' is 0. Set to > 0 to run.")
            # Implicitly, if data is insufficient, the inner 'if' handles it or the function itself warns.

        # Train LightGBM model
        lgb_model = LGBMRegressor(n_estimators=150, learning_rate=0.05, num_leaves=20, max_depth=7, min_child_samples=15, random_state=42, n_jobs=-1)
        lgb_model.fit(X_train_scaled, y_train)

        # Predictions on test set (for historical evaluation and plotting)
        test_pred = lgb_model.predict(X_test_scaled)
        test_mae = mean_absolute_error(y_test, test_pred)
        st.write(f"Test MAE for {tgt}: {test_mae:.2f}")

        # --- Plotting historical actuals, train predictions, test predictions ---
        # (Assuming this part exists and uses y_train, y_test, predictions on train (if any), test_pred)
        # For example:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train_df['ds'], y=y_train, mode='lines', name='Train Actual'))
        fig.add_trace(go.Scatter(x=test_df['ds'], y=y_test, mode='lines', name='Test Actual'))
        fig.add_trace(go.Scatter(x=test_df['ds'], y=test_pred, mode='lines', name='Test Forecast (LGBM)'))
        # (Prophet baseline could also be added here if implemented)
        # fig.update_layout(title=f"{hosp} - {tgt}: Historical vs Forecast", xaxis_title="Date", yaxis_title=tgt)
        # st.plotly_chart(fig)

        # --- BEGIN MODIFICATION: Refactored Future Forecasting Loop ---
        # The following loop generates future forecasts day by day.
        # For each future day, it meticulously recalculates all selected features
        # based on the most up-to-date historical data (including previous forecasts).
        # This ensures that features like lags, rolling means, etc., are accurately
        # derived without using stale data from the initial historical feature engineering step.
        st.write("🔮 Generating future forecasts...")

        # Initialization for the future forecasting loop
        # master_data_for_iteration holds 'ds' and original values of ALL available_targets_for_hospital (including current target, NOT YET RENAMED 'y')
        # It is NOT modified during the loop. Used as a source for non-target metric history.
        master_data_for_iteration = df2.copy() # df2 has 'ds' and original target names

        # history_plus_forecasts_y: Starts with historical 'y' (train actuals + test predictions)
        # and gets appended with new future_y_predictions.
        # Note: Using actual y_train and y_test might be better if available and makes sense for "true" history.
        # However, if we want features based on what model *would have seen*, using test_pred is more aligned.
        # For this refactor, let's use train_df['y'] (actuals) and test_pred (model's view of test period).
        # Or, more simply, use the 'y' column from df_feat_hist which was used for training/testing.

        # y_history_for_features contains actual historical values of 'y' up to the end of the test period.
        # df_feat_hist['y'] contains the values of 'tgt' after it was renamed to 'y'.
        y_history_for_features = df_feat_hist['y'].tolist() # This is historical 'y' used for training/testing features

        future_y_predictions = [] # Stores only the final predicted 'y' values for the future_days horizon

        last_known_date_overall = df_feat_hist['ds'].iloc[-1] # Last date from historical features

        # Constants for feature generation (mirroring create_advanced_features)
        MAX_LAGS = 7
        ROLLING_WINDOWS = [3, 7, 14, 21]
        EWM_ALPHAS = [0.1, 0.3, 0.5]
        DIFF_LAGS = [1, 2, 3, 7] # From create_advanced_features
        CROSS_CORR_WINDOWS = [7, 14] # From create_advanced_features
        VOLATILITY_WINDOWS = [7, 14] # From create_advanced_features

        # The main forecasting loop
        for day_idx in range(future_days):
            next_future_date = last_known_date_overall + pd.Timedelta(days=day_idx + 1)
            current_step_features = pd.Series(index=selected_features, dtype=np.float64)

            # For calculating features for 'next_future_date':
            # - Target-related features (lags, rolling of 'y') use 'y_history_for_features'.
            # - Other metric features use their historical series from 'master_data_for_iteration'.
            # - Temporal features use 'next_future_date'.

            # --- Temporal and Cyclical Features ---
            temp_date_series = pd.Series([next_future_date])
            current_step_features['dow'] = temp_date_series.dt.dayofweek.iloc[0]
            current_step_features['month'] = temp_date_series.dt.month.iloc[0]
            current_step_features['quarter'] = temp_date_series.dt.quarter.iloc[0]
            current_step_features['week'] = temp_date_series.dt.isocalendar().week.astype(int).iloc[0]
            current_step_features['dayofyear'] = temp_date_series.dt.dayofyear.iloc[0]
            current_step_features['is_weekend'] = (current_step_features['dow'] >= 5).astype(int)
            current_step_features['is_month_start'] = temp_date_series.dt.is_month_start.astype(int).iloc[0]
            current_step_features['is_month_end'] = temp_date_series.dt.is_month_end.astype(int).iloc[0]
            current_step_features['dow_sin'] = np.sin(2 * np.pi * current_step_features['dow'] / 7)
            current_step_features['dow_cos'] = np.cos(2 * np.pi * current_step_features['dow'] / 7)
            current_step_features['month_sin'] = np.sin(2 * np.pi * current_step_features['month'] / 12)
            current_step_features['month_cos'] = np.cos(2 * np.pi * current_step_features['month'] / 12)
            current_step_features['dayofyear_sin'] = np.sin(2 * np.pi * current_step_features['dayofyear'] / 365.25)
            current_step_features['dayofyear_cos'] = np.cos(2 * np.pi * current_step_features['dayofyear'] / 365.25)
            current_step_features['dow_month_interaction'] = current_step_features['dow'] * current_step_features['month']

            # --- Trend Features ---
            # Linear trend continues from the end of historical features
            linear_trend_val = len(df_feat_hist) + day_idx
            current_step_features['linear_trend'] = linear_trend_val
            current_step_features['quadratic_trend'] = linear_trend_val ** 2

            # --- Iterate over selected_features to calculate them ---
            for feature_name in selected_features:
                if feature_name in current_step_features and pd.notna(current_step_features[feature_name]):
                    continue # Already calculated (e.g. temporal, trend)

                # --- Target-related features (lags, diffs, rolling, ewm, volatility of 'y') ---
                # These use 'y_history_for_features' which contains y(t-1), y(t-2), ...
                y_series_for_calc = pd.Series(y_history_for_features) # Convert list to Series for .diff, .rolling, .ewm

                if feature_name.startswith('target_lag_'):
                    lag = int(feature_name.split('_')[-1])
                    if lag <= len(y_series_for_calc):
                        current_step_features[feature_name] = y_series_for_calc.iloc[-lag]
                    else: current_step_features[feature_name] = np.nan

                elif feature_name.startswith('target_diff_'):
                    lag = int(feature_name.split('_')[-1])
                    if len(y_series_for_calc) >= lag : # .diff(lag) needs at least lag+1 points to produce a non-NaN value at the end
                        current_step_features[feature_name] = y_series_for_calc.diff(lag).iloc[-1]
                    else: current_step_features[feature_name] = np.nan

                elif feature_name.startswith('target_roll_'): # mean, std, min, max, median
                    parts = feature_name.split('_')
                    stat = parts[2]
                    window = int(parts[3])
                    # shift(1) was used in create_advanced_features, meaning rolling stats are based on t-1 data.
                    # y_series_for_calc already represents data up to t-1.
                    if len(y_series_for_calc) >= 1 : # Need at least 1 point for rolling
                        if stat == 'mean': current_step_features[feature_name] = y_series_for_calc.rolling(window=window, min_periods=1).mean().iloc[-1]
                        elif stat == 'std': current_step_features[feature_name] = y_series_for_calc.rolling(window=window, min_periods=1).std().iloc[-1]
                        elif stat == 'min': current_step_features[feature_name] = y_series_for_calc.rolling(window=window, min_periods=1).min().iloc[-1]
                        elif stat == 'max': current_step_features[feature_name] = y_series_for_calc.rolling(window=window, min_periods=1).max().iloc[-1]
                        elif stat == 'median': current_step_features[feature_name] = y_series_for_calc.rolling(window=window, min_periods=1).median().iloc[-1]
                    else: current_step_features[feature_name] = np.nan

                elif feature_name.startswith('target_ewm_mean_'):
                    alpha = float(feature_name.split('_')[-1])
                    if len(y_series_for_calc) >=1:
                        current_step_features[feature_name] = y_series_for_calc.ewm(alpha=alpha, min_periods=1).mean().iloc[-1]
                    else: current_step_features[feature_name] = np.nan

                elif feature_name.startswith('volatility_'): # Target volatility
                    window = int(feature_name.split('_')[-1])
                    # Original: (df_feat[target_col].shift(1) - df_feat[target_col].rolling(window=W, min_periods=1).mean().shift(1)).abs()
                    # This means volatility for predicting time t is:
                    # abs( y(t-1) - mean_of_y_values_from_t-W-1_to_t-2 ) where the mean is taken at t-1.
                    # y_series_for_calc is history up to y(t-1).
                    if len(y_series_for_calc) >= 1: # Need y(t-1)
                        y_t_minus_1 = y_series_for_calc.iloc[-1]
                        if len(y_series_for_calc) >= 2 : # Need at least one value for the shifted rolling mean (history up to t-2)
                            # Series for rolling mean should end at y(t-2)
                            series_for_rolling_mean = y_series_for_calc.iloc[:-1]
                            if not series_for_rolling_mean.empty:
                                rolling_mean_shifted = series_for_rolling_mean.rolling(window=window, min_periods=1).mean().iloc[-1]
                                current_step_features[feature_name] = abs(y_t_minus_1 - rolling_mean_shifted)
                            else: # Not enough history for shifted rolling mean
                                current_step_features[feature_name] = np.nan
                        else: # Not enough history for shifted rolling mean
                             current_step_features[feature_name] = np.nan
                    else: # Not enough history for y(t-1)
                        current_step_features[feature_name] = np.nan

                # --- Other Metric Features (lags, diffs, rolling, ewm from 'master_data_for_iteration') ---
                # And Cross-Correlation and Aggregate features
                else:
                    # Check if feature involves other metrics
                    found_other_metric_feature = False
                    for other_metric_original_name in available_targets_for_hospital:
                        if other_metric_original_name == tgt: continue # Skip 'y' itself

                        # Check for lags, diffs, rolling, ewm of other_metric_original_name
                        # These use the historical series from master_data_for_iteration[other_metric_original_name]
                        # The shift(1) in create_advanced_features means these are based on data up to t-1.
                        metric_series_hist = master_data_for_iteration[other_metric_original_name] # Full history of this metric

                        if feature_name.startswith(f'{other_metric_original_name}_lag_'):
                            lag = int(feature_name.split('_')[-1])
                            if lag <= len(metric_series_hist): current_step_features[feature_name] = metric_series_hist.iloc[-lag]
                            else: current_step_features[feature_name] = np.nan
                            found_other_metric_feature = True; break

                        elif feature_name.startswith(f'{other_metric_original_name}_diff_'):
                            lag = int(feature_name.split('_')[-1])
                            # .diff(lag) needs at least lag points in history. iloc[-1] gets the latest diff.
                            if len(metric_series_hist) > lag : current_step_features[feature_name] = metric_series_hist.diff(lag).iloc[-1]
                            else: current_step_features[feature_name] = np.nan
                            found_other_metric_feature = True; break

                        # FIX: Replaced error-prone parsing with a robust method
                        elif feature_name.startswith(f'{other_metric_original_name}_roll_'):
                            # Robustly parse stat and window from feature name, accounting for underscores in metric names
                            prefix_to_remove = f"{other_metric_original_name}_roll_"
                            stat_and_window_str = feature_name[len(prefix_to_remove):]
                            parts = stat_and_window_str.split('_') # Should give ['stat', 'window']
                            
                            stat = parts[0]
                            window = int(parts[1]) # This is now safe

                            if len(metric_series_hist) >=1: # shift(1) in original means use data up to t-1
                                if stat == 'mean': current_step_features[feature_name] = metric_series_hist.rolling(window=window, min_periods=1).mean().iloc[-1]
                                elif stat == 'std': current_step_features[feature_name] = metric_series_hist.rolling(window=window, min_periods=1).std().iloc[-1]
                                elif stat == 'min': current_step_features[feature_name] = metric_series_hist.rolling(window=window, min_periods=1).min().iloc[-1]
                                elif stat == 'max': current_step_features[feature_name] = metric_series_hist.rolling(window=window, min_periods=1).max().iloc[-1]
                                elif stat == 'median': current_step_features[feature_name] = metric_series_hist.rolling(window=window, min_periods=1).median().iloc[-1]
                            else: current_step_features[feature_name] = np.nan
                            found_other_metric_feature = True; break
                        # END FIX

                        elif feature_name.startswith(f'{other_metric_original_name}_ewm_mean_'):
                            alpha = float(feature_name.split('_')[-1])
                            if len(metric_series_hist) >= 1:
                                current_step_features[feature_name] = metric_series_hist.ewm(alpha=alpha, min_periods=1).mean().iloc[-1]
                            else: current_step_features[feature_name] = np.nan
                            found_other_metric_feature = True; break

                    if found_other_metric_feature: continue

                    # --- Cross-Correlation and Ratio Features ---
                    # Example: col1_col2_rolling_corr_W or col1_col2_ratio
                    # These use historical series of col1 and col2 from master_data_for_iteration
                    # The shift(1) in original means use data up to t-1.
                    is_cross_feature = False
                    for i_cf, col1_cf_name in enumerate(available_targets_for_hospital):
                        if col1_cf_name == tgt: continue
                        for j_cf, col2_cf_name in enumerate(available_targets_for_hospital): # Check all pairs for ratio, distinct pairs for corr
                            if col2_cf_name == tgt: continue
                            if col1_cf_name == col2_cf_name and not feature_name.endswith("_ratio"): continue # Corr is between distinct

                            col1_series_hist = master_data_for_iteration[col1_cf_name]
                            col2_series_hist = master_data_for_iteration[col2_cf_name]

                            if feature_name.startswith(f'{col1_cf_name}_{col2_cf_name}_rolling_corr_'):
                                window = int(feature_name.split('_')[-1])
                                if len(col1_series_hist) >= 1 and len(col2_series_hist) >=1:
                                     current_step_features[feature_name] = col1_series_hist.rolling(window=window, min_periods=1).corr(col2_series_hist).iloc[-1]
                                else: current_step_features[feature_name] = np.nan
                                is_cross_feature = True; break

                            if feature_name == f'{col1_cf_name}_{col2_cf_name}_ratio':
                                if len(col1_series_hist) >=1 and len(col2_series_hist) >=1:
                                    current_step_features[feature_name] = col1_series_hist.iloc[-1] / (col2_series_hist.iloc[-1] + 1e-8)
                                else: current_step_features[feature_name] = np.nan
                                is_cross_feature = True; break
                        if is_cross_feature: break
                    if is_cross_feature: continue

                    # --- Aggregate Features --- (all_metrics_mean, etc.)
                    # These are based on predictor_cols from master_data_for_iteration at t-1
                    predictor_metric_names = [col for col in available_targets_for_hospital if col != tgt]
                    if len(predictor_metric_names) > 1:
                        current_predictors_values = master_data_for_iteration[predictor_metric_names].iloc[-1] # Values at t-1
                        if feature_name == 'all_metrics_mean': current_step_features[feature_name] = current_predictors_values.mean()
                        elif feature_name == 'all_metrics_std': current_step_features[feature_name] = current_predictors_values.std()
                        elif feature_name == 'all_metrics_sum': current_step_features[feature_name] = current_predictors_values.sum()
                        elif feature_name == 'all_metrics_max': current_step_features[feature_name] = current_predictors_values.max()
                        elif feature_name == 'all_metrics_min': current_step_features[feature_name] = current_predictors_values.min()
                    # No 'else' here, if it's an agg feature not listed, it remains unhandled and will be NaN -> 0

            # Fill any NaNs that resulted from feature calculation (e.g. insufficient history for a lag/window)
            current_step_features_filled = current_step_features.fillna(0) # Or use training data means/medians

            # Assemble feature vector for prediction, ensuring correct order
            X_pred_df = pd.DataFrame([current_step_features_filled])
            X_pred_df = X_pred_df[selected_features] # Ensure column order matches training

            # Scale and Predict
            X_pred_scaled = scaler.transform(X_pred_df)
            predicted_y = lgb_model.predict(X_pred_scaled)[0]

            future_y_predictions.append(predicted_y)
            y_history_for_features.append(predicted_y) # Add current prediction to history for next step's features

        # --- Future forecast plotting ---
        if future_y_predictions:
            last_historical_date = df_feat_hist['ds'].iloc[-1]
            future_dates = pd.date_range(start=last_historical_date + pd.Timedelta(days=1), periods=future_days)
            fig.add_trace(go.Scatter(x=future_dates, y=future_y_predictions, mode='lines', name=f'Future Forecast (LGBM) - {tgt}', line=dict(dash='dot')))

        fig.update_layout(title=f"{hosp} - {tgt}: Forecast", xaxis_title="Date", yaxis_title=tgt)
        st.plotly_chart(fig, use_container_width=True)

        # --- END MODIFICATION: Refactored Future Forecasting Loop ---

# (Original summary results part could be added here if needed)
# if results:
#     results_df = pd.DataFrame(results)
#     st.subheader("📊 Overall Model Performance Summary")
#     st.dataframe(results_df)

st.sidebar.success("✅ Forecasting complete!")
