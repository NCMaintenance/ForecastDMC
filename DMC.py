import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import plotly.graph_objs as go
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")
st.title("📈 Improved Hospital Forecasting with Reduced Overfitting")

# Sidebar
uploaded = st.sidebar.file_uploader("Upload Excel file", type="xlsx")
hospitals = []
targets = [
    "Tracker8am","Tracker2pm","Tracker8pm",
    "AdditionalCapacityOpen Morning",
    "TimeTotal_8am","TimeTotal_2pm","TimeTotal_8pm"
]

if not uploaded:
    st.sidebar.info("Please upload your Excel file.")
    st.stop()

# Load data
df = pd.read_excel(uploaded)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
hospitals = sorted(df['Hospital'].unique())
sel_hosp = st.sidebar.selectbox("Hospital", ["All"]+hospitals)
sel_target = st.sidebar.selectbox("Target", ["All"]+targets)

# Fixed at 7 days for iterative forecasting
future_days = 7
st.sidebar.info(f"Forecast horizon: {future_days} days (iterative)")

run = st.sidebar.button("🚀 Run Forecast")
if not run:
    st.sidebar.info("Configure then click Run Forecast")
    st.stop()

# Filters
h_list = hospitals if sel_hosp=="All" else [sel_hosp]
t_list = targets if sel_target=="All" else [sel_target]

results = []

def create_features(df_input, target_col, is_future=False):
    """Create features with reduced complexity to prevent overfitting"""
    df_feat = df_input.copy()
    
    # Basic lag features (multiple lags for comparison)
    df_feat['y_lag1'] = df_feat[target_col].shift(1)
    df_feat['y_lag2'] = df_feat[target_col].shift(2)
    df_feat['y_lag3'] = df_feat[target_col].shift(3)
    df_feat['y_lag7'] = df_feat[target_col].shift(7)  # Weekly pattern
    
    # Percentage changes vs original
    df_feat['pct_change_1d'] = ((df_feat[target_col] - df_feat['y_lag1']) / df_feat['y_lag1'] * 100).round(2)
    df_feat['pct_change_2d'] = ((df_feat[target_col] - df_feat['y_lag2']) / df_feat['y_lag2'] * 100).round(2)
    df_feat['pct_change_3d'] = ((df_feat[target_col] - df_feat['y_lag3']) / df_feat['y_lag3'] * 100).round(2)
    df_feat['pct_change_7d'] = ((df_feat[target_col] - df_feat['y_lag7']) / df_feat['y_lag7'] * 100).round(2)
    
    # Absolute differences
    df_feat['abs_diff_1d'] = (df_feat[target_col] - df_feat['y_lag1']).round(2)
    df_feat['abs_diff_2d'] = (df_feat[target_col] - df_feat['y_lag2']).round(2)
    df_feat['abs_diff_3d'] = (df_feat[target_col] - df_feat['y_lag3']).round(2)
    df_feat['abs_diff_7d'] = (df_feat[target_col] - df_feat['y_lag7']).round(2)
    
    # Sequential day-to-day changes (lag vs previous lag)
    df_feat['pct_change_lag1_vs_lag2'] = ((df_feat['y_lag1'] - df_feat['y_lag2']) / df_feat['y_lag2'] * 100).round(2)
    df_feat['pct_change_lag2_vs_lag3'] = ((df_feat['y_lag2'] - df_feat['y_lag3']) / df_feat['y_lag3'] * 100).round(2)
    
    # Volatility measures
    df_feat['volatility_3d'] = df_feat[target_col].rolling(window=3, min_periods=1).std().round(2)
    df_feat['volatility_7d'] = df_feat[target_col].rolling(window=7, min_periods=1).std().round(2)
    
    # Simple rolling statistics (shorter windows)
    df_feat['roll_mean3'] = df_feat[target_col].rolling(window=3, min_periods=1).mean()
    df_feat['roll_mean7'] = df_feat[target_col].rolling(window=7, min_periods=1).mean()
    
    # Percentage vs rolling means
    df_feat['pct_vs_mean3'] = ((df_feat[target_col] - df_feat['roll_mean3']) / df_feat['roll_mean3'] * 100).round(2)
    df_feat['pct_vs_mean7'] = ((df_feat[target_col] - df_feat['roll_mean7']) / df_feat['roll_mean7'] * 100).round(2)
    
    # Time features (simplified)
    df_feat['dow'] = df_feat['Date'].dt.dayofweek
    df_feat['month'] = df_feat['Date'].dt.month
    df_feat['is_weekend'] = (df_feat['dow'] >= 5).astype(int)
    
    # Cyclical encoding for day of week only
    df_feat['dow_sin'] = np.sin(2 * np.pi * df_feat['dow'] / 7)
    df_feat['dow_cos'] = np.cos(2 * np.pi * df_feat['dow'] / 7)
    
    # Trend component (simple)
    df_feat['trend'] = np.arange(len(df_feat))
    
    return df_feat

def iterative_forecast(model, last_data, prophet_model, scaler, n_days=7):
    """Forecast iteratively, one day at a time"""
    forecasts = []
    current_data = last_data.copy()
    
    for day in range(n_days):
        # Get next date
        next_date = current_data['Date'].iloc[-1] + pd.Timedelta(days=1)
        
        # Get Prophet prediction for this date
        prophet_pred = prophet_model.predict(pd.DataFrame({'ds': [next_date]}))
        prophet_value = prophet_pred['yhat'].iloc[0]
        
        # Get last few values for calculations
        last_vals = current_data['y'].iloc[-7:].values
        last_val = last_vals[-1]
        
        # Create features for next day with all the new comparison features
        next_row = pd.DataFrame({
            'Date': [next_date],
            'y': [np.nan],  # Unknown target
            'y_lag1': [last_val],
            'y_lag2': [last_vals[-2] if len(last_vals) >= 2 else last_val],
            'y_lag3': [last_vals[-3] if len(last_vals) >= 3 else last_val],
            'y_lag7': [last_vals[0] if len(last_vals) >= 7 else last_val],
            'roll_mean3': [current_data['y'].iloc[-3:].mean()],
            'roll_mean7': [current_data['y'].iloc[-7:].mean()],
            'volatility_3d': [current_data['y'].iloc[-3:].std()],
            'volatility_7d': [current_data['y'].iloc[-7:].std()],
            'dow': [next_date.dayofweek],
            'month': [next_date.month],
            'is_weekend': [1 if next_date.dayofweek >= 5 else 0],
            'dow_sin': [np.sin(2 * np.pi * next_date.dayofweek / 7)],
            'dow_cos': [np.cos(2 * np.pi * next_date.dayofweek / 7)],
            'trend': [current_data['trend'].iloc[-1] + 1],
            'prophet_pred': [prophet_value]
        })
        
        # Calculate percentage changes (using last available values)
        lag1_val = next_row['y_lag1'].iloc[0]
        lag2_val = next_row['y_lag2'].iloc[0]
        lag3_val = next_row['y_lag3'].iloc[0]
        lag7_val = next_row['y_lag7'].iloc[0]
        mean3_val = next_row['roll_mean3'].iloc[0]
        mean7_val = next_row['roll_mean7'].iloc[0]
        
        # Add calculated features (using previous day's value as estimate for current)
        next_row['pct_change_1d'] = [0.0]  # Will be updated after prediction
        next_row['pct_change_2d'] = [((lag1_val - lag2_val) / lag2_val * 100) if lag2_val != 0 else 0.0]
        next_row['pct_change_3d'] = [((lag1_val - lag3_val) / lag3_val * 100) if lag3_val != 0 else 0.0]
        next_row['pct_change_7d'] = [((lag1_val - lag7_val) / lag7_val * 100) if lag7_val != 0 else 0.0]
        next_row['pct_vs_mean3'] = [((lag1_val - mean3_val) / mean3_val * 100) if mean3_val != 0 else 0.0]
        next_row['pct_vs_mean7'] = [((lag1_val - mean7_val) / mean7_val * 100) if mean7_val != 0 else 0.0]
        
        # Features for prediction (core features only to prevent overfitting)
        feature_cols = ['y_lag1', 'y_lag2', 'y_lag7', 'roll_mean3', 'roll_mean7', 
                       'volatility_3d', 'dow', 'month', 'is_weekend', 'dow_sin', 'dow_cos', 
                       'trend', 'prophet_pred']
        
        # Scale features
        next_features_scaled = scaler.transform(next_row[feature_cols])
        
        # Make prediction
        pred = model.predict(next_features_scaled)[0]
        forecasts.append(pred)
        
        # Update current_data with prediction and recalculate features
        next_row['y'] = pred
        next_row = create_features(next_row, 'y')
        current_data = pd.concat([current_data, next_row], ignore_index=True)
    
    return forecasts

for hosp in h_list:
    st.header(f"🏥 {hosp}")
    df_h = df[df['Hospital']==hosp].reset_index(drop=True)

    for tgt in t_list:
        st.subheader(f"🎯 {tgt}")
        if df_h[tgt].isna().any():
            st.warning("Skipping (nulls present)")
            continue

        # Prepare data
        df2 = df_h[['Date', tgt]].rename(columns={tgt:'y'})
        
        # Create features
        df2 = create_features(df2, 'y')
        
        # Train Prophet model for additional features
        prophet_data = df2[['Date', 'y']].rename(columns={'Date':'ds'})
        
        # Simplified Prophet model to reduce overfitting
        m_prophet = Prophet(
            yearly_seasonality=True, 
            weekly_seasonality=True, 
            daily_seasonality=False,
            seasonality_mode='additive',
            changepoint_prior_scale=0.01,  # Reduced to prevent overfitting
            seasonality_prior_scale=1.0
        )
        m_prophet.add_country_holidays(country_name='IE')
        m_prophet.fit(prophet_data)
        
        # Add Prophet predictions as features
        prophet_pred = m_prophet.predict(prophet_data[['ds']])
        df2['prophet_pred'] = prophet_pred['yhat'].values
        
        # Remove rows with NaN values
        df2 = df2.dropna().reset_index(drop=True)
        
        if len(df2) < 30:  # Need sufficient data
            st.warning("Insufficient data for reliable forecasting")
            continue
        
        # Split data (use more recent data for testing)
        n = len(df2)
        split = max(30, int(0.85 * n))  # Use more data for training, but keep reasonable test set
        train = df2.iloc[:split].reset_index(drop=True)
        test = df2.iloc[split:].reset_index(drop=True)
        
        # Feature selection (core features to prevent overfitting but include key comparisons)
        feature_cols = ['y_lag1', 'y_lag2', 'y_lag7', 'roll_mean3', 'roll_mean7', 
                       'volatility_3d', 'dow', 'month', 'is_weekend', 'dow_sin', 'dow_cos', 
                       'trend', 'prophet_pred']
        
        X_train, y_train = train[feature_cols], train['y']
        X_test, y_test = test[feature_cols], test['y']
        
        # Scale features to improve model stability
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Cross-validation with more realistic setup
        tscv = TimeSeriesSplit(n_splits=3, test_size=max(7, len(train)//10))  # Fewer splits, larger test sizes
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X_train_scaled):
            # Regularized LightGBM model
            lgb_model = LGBMRegressor(
                n_estimators=100,  # Reduced from 200
                learning_rate=0.1,  # Increased for faster convergence
                num_leaves=15,     # Reduced from 31
                reg_alpha=1.0,     # L1 regularization
                reg_lambda=1.0,    # L2 regularization
                min_child_samples=20,  # Increased minimum samples
                random_state=42,
                verbose=-1
            )
            
            lgb_model.fit(X_train_scaled[train_idx], y_train.iloc[train_idx])
            val_pred = lgb_model.predict(X_train_scaled[val_idx])
            cv_scores.append(mean_absolute_error(y_train.iloc[val_idx], val_pred))
        
        cv_mae = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        # Train final model
        final_model = LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=15,
            reg_alpha=1.0,
            reg_lambda=1.0,
            min_child_samples=20,
            random_state=42,
            verbose=-1
        )
        
        final_model.fit(X_train_scaled, y_train)
        
        # Test set predictions
        test_pred = final_model.predict(X_test_scaled)
        test_mae = mean_absolute_error(y_test, test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        # Iterative forecasting for 7 days
        forecast_values = iterative_forecast(final_model, train, m_prophet, scaler, n_days=7)
        
        # Create forecast dates
        last_date = df2['Date'].iloc[-1]
        forecast_dates = [last_date + pd.Timedelta(days=i+1) for i in range(7)]
        
        # Plotting
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=train['Date'], 
            y=train['y'], 
            mode='lines', 
            name='Historical', 
            line=dict(color='blue', width=1)
        ))
        
        # Test data
        fig.add_trace(go.Scatter(
            x=test['Date'], 
            y=test['y'], 
            mode='lines+markers', 
            name='Actual (Test)', 
            line=dict(color='green', width=2)
        ))
        
        # Test predictions
        fig.add_trace(go.Scatter(
            x=test['Date'], 
            y=test_pred, 
            mode='lines+markers', 
            name='Predicted (Test)', 
            line=dict(color='orange', width=2)
        ))
        
        # Future forecast
        fig.add_trace(go.Scatter(
            x=forecast_dates, 
            y=forecast_values, 
            mode='lines+markers', 
            name='7-Day Forecast', 
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=f"{hosp} • {tgt}<br>CV MAE: {cv_mae:.2f}±{cv_std:.2f} | Test MAE: {test_mae:.2f} | Test RMSE: {test_rmse:.2f}",
            xaxis_title="Date", 
            yaxis_title=tgt, 
            template="plotly_white",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Model performance metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Cross-Validation Results**")
            st.metric("CV MAE", f"{cv_mae:.2f}")
            st.metric("CV Std", f"{cv_std:.2f}")
            
        with col2:
            st.markdown("**Test Set Results**")
            st.metric("Test MAE", f"{test_mae:.2f}")
            st.metric("Test RMSE", f"{test_rmse:.2f}")
        
        # Test set results table with comprehensive comparisons
        st.markdown("**Test Set Predictions with Lag Comparisons**")
        
        # Create comprehensive comparison dataframe
        test_comparison = pd.DataFrame({
            "Date": test['Date'],
            "Actual": test['y'],
            "Predicted": test_pred,
            "Abs_Error": np.abs(test['y'] - test_pred),
            "Rel_Error_%": (np.abs(test['y'] - test_pred) / test['y'] * 100),
            
            # Lag comparisons
            "Lag_1d": test['y_lag1'],
            "Lag_2d": test['y_lag2'], 
            "Lag_3d": test['y_lag3'],
            "Lag_7d": test['y_lag7'],
            
            # Percentage changes vs lags
            "Vs_1d_%": test['pct_change_1d'],
            "Vs_2d_%": test['pct_change_2d'],
            "Vs_3d_%": test['pct_change_3d'],
            "Vs_7d_%": test['pct_change_7d'],
            
            # Sequential lag comparisons
            "Lag1_vs_Lag2_%": test['pct_change_lag1_vs_lag2'],
            "Lag2_vs_Lag3_%": test['pct_change_lag2_vs_lag3'],
            
            # Vs rolling means
            "Vs_Mean3d_%": test['pct_vs_mean3'],
            "Vs_Mean7d_%": test['pct_vs_mean7'],
            
            # Volatility
            "Volatility_3d": test['volatility_3d'],
            "Volatility_7d": test['volatility_7d']
        }).round(2)
        
        # Add color coding for percentage changes
        def highlight_changes(val):
            if pd.isna(val):
                return ''
            if abs(val) > 20:
                return 'background-color: #ffcccc'  # Light red for high changes
            elif abs(val) > 10:
                return 'background-color: #ffffcc'  # Light yellow for medium changes
            elif abs(val) > 5:
                return 'background-color: #ccffcc'  # Light green for small changes
            else:
                return ''
        
        # Apply styling to percentage columns
        pct_columns = [col for col in test_comparison.columns if col.endswith('_%')]
        styled_df = test_comparison.style.applymap(highlight_changes, subset=pct_columns)
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Summary statistics for comparisons
        st.markdown("**Lag Comparison Summary Statistics**")
        comparison_stats = pd.DataFrame({
            'Metric': ['Mean % Change vs 1d', 'Mean % Change vs 2d', 'Mean % Change vs 3d', 'Mean % Change vs 7d',
                      'Std % Change vs 1d', 'Std % Change vs 2d', 'Std % Change vs 3d', 'Std % Change vs 7d',
                      'Mean Volatility 3d', 'Mean Volatility 7d'],
            'Value': [
                test['pct_change_1d'].mean(), test['pct_change_2d'].mean(), 
                test['pct_change_3d'].mean(), test['pct_change_7d'].mean(),
                test['pct_change_1d'].std(), test['pct_change_2d'].std(),
                test['pct_change_3d'].std(), test['pct_change_7d'].std(),
                test['volatility_3d'].mean(), test['volatility_7d'].mean()
            ]
        }).round(2)
        st.dataframe(comparison_stats, use_container_width=True)
        
        # Future forecast table with enhanced comparisons
        st.markdown("**7-Day Iterative Forecast with Trend Analysis**")
        
        # Calculate forecast comparisons
        forecast_comparison = []
        for i, (date, forecast_val) in enumerate(zip(forecast_dates, forecast_values)):
            # Get reference values for comparison
            last_actual = train['y'].iloc[-1]
            last_week_actual = train['y'].iloc[-7] if len(train) >= 7 else last_actual
            recent_mean = train['y'].iloc[-7:].mean()
            
            comparison_row = {
                "Date": date,
                "Day": f"Day {i+1}",
                "Day_of_Week": date.strftime("%A"),
                "Forecast": round(forecast_val, 2),
                
                # Comparisons vs recent actuals
                "Vs_Last_Actual": round(forecast_val - last_actual, 2),
                "Vs_Last_Actual_%": round((forecast_val - last_actual) / last_actual * 100, 2) if last_actual != 0 else 0,
                
                "Vs_Week_Ago": round(forecast_val - last_week_actual, 2),
                "Vs_Week_Ago_%": round((forecast_val - last_week_actual) / last_week_actual * 100, 2) if last_week_actual != 0 else 0,
                
                "Vs_Recent_Mean": round(forecast_val - recent_mean, 2),
                "Vs_Recent_Mean_%": round((forecast_val - recent_mean) / recent_mean * 100, 2) if recent_mean != 0 else 0,
            }
            
            # Day-to-day forecast changes
            if i > 0:
                prev_forecast = forecast_values[i-1]
                comparison_row["Vs_Prev_Day"] = round(forecast_val - prev_forecast, 2)
                comparison_row["Vs_Prev_Day_%"] = round((forecast_val - prev_forecast) / prev_forecast * 100, 2) if prev_forecast != 0 else 0
            else:
                comparison_row["Vs_Prev_Day"] = 0
                comparison_row["Vs_Prev_Day_%"] = 0
                
            forecast_comparison.append(comparison_row)
        
        forecast_df = pd.DataFrame(forecast_comparison)
        
        # Style the forecast table
        def highlight_forecast_changes(val):
            if pd.isna(val):
                return ''
            if abs(val) > 15:
                return 'background-color: #ffcccc; font-weight: bold'  # High change
            elif abs(val) > 8:
                return 'background-color: #ffffcc'  # Medium change
            elif abs(val) > 3:
                return 'background-color: #ccffcc'  # Small change
            else:
                return ''
        
        forecast_pct_cols = [col for col in forecast_df.columns if col.endswith('_%')]
        styled_forecast = forecast_df.style.applymap(highlight_forecast_changes, subset=forecast_pct_cols)
        
        st.dataframe(styled_forecast, use_container_width=True)
        
        # Forecast trend analysis
        st.markdown("**Forecast Trend Analysis**")
        trend_analysis = pd.DataFrame({
            'Metric': [
                'Average Daily Change (%)',
                'Max Daily Change (%)', 
                'Min Daily Change (%)',
                'Forecast Volatility',
                'Overall 7-Day Change (%)',
                'Strongest Growth Day',
                'Strongest Decline Day'
            ],
            'Value': [
                np.mean([row["Vs_Prev_Day_%"] for row in forecast_comparison[1:]]),
                np.max([row["Vs_Prev_Day_%"] for row in forecast_comparison[1:]]),
                np.min([row["Vs_Prev_Day_%"] for row in forecast_comparison[1:]]),
                np.std(forecast_values),
                (forecast_values[-1] - forecast_values[0]) / forecast_values[0] * 100,
                forecast_dates[np.argmax([row["Vs_Prev_Day_%"] for row in forecast_comparison[1:]]) + 1].strftime("%A"),
                forecast_dates[np.argmin([row["Vs_Prev_Day_%"] for row in forecast_comparison[1:]]) + 1].strftime("%A")
            ]
        })
        
        # Format numeric values
        for i in range(len(trend_analysis) - 2):  # Skip last two rows (day names)
            if isinstance(trend_analysis.iloc[i]['Value'], (int, float)):
                trend_analysis.at[i, 'Value'] = round(trend_analysis.iloc[i]['Value'], 2)
        
        st.dataframe(trend_analysis, use_container_width=True)
        
        # Feature importance
        importance = final_model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        st.markdown("**Feature Importance**")
        st.bar_chart(importance_df.set_index('Feature')['Importance'])
        
        results.append({
            "Hospital": hosp, 
            "Target": tgt, 
            "CV MAE": round(cv_mae, 2),
            "CV Std": round(cv_std, 2),
            "Test MAE": round(test_mae, 2),
            "Test RMSE": round(test_rmse, 2),
            "Overfitting Risk": "Low" if test_mae <= cv_mae * 1.2 else "Medium" if test_mae <= cv_mae * 1.5 else "High"
        })

st.subheader("📊 Model Performance Summary")
results_df = pd.DataFrame(results)
st.dataframe(results_df, use_container_width=True)

# Summary statistics
if len(results_df) > 0:
    st.subheader("📈 Overall Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Average CV MAE", f"{results_df['CV MAE'].mean():.2f}")
    with col2:
        st.metric("Average Test MAE", f"{results_df['Test MAE'].mean():.2f}")
    with col3:
        st.metric("Low Risk Models", f"{sum(results_df['Overfitting Risk'] == 'Low')}/{len(results_df)}")
    with col4:
        avg_generalization = results_df['Test MAE'].mean() / results_df['CV MAE'].mean()
        st.metric("Generalization Ratio", f"{avg_generalization:.2f}")
