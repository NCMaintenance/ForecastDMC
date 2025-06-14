import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import plotly.graph_objs as go
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")
st.title("📈 Enhanced Hospital Forecasting with Correlation-Based Features")

# Sidebar
uploaded = st.sidebar.file_uploader("Upload Excel file", type="xlsx")
hospitals = []
targets = [
    "Tracker8am", "Tracker2pm", "Tracker8pm",
    "AdditionalCapacityOpen Morning",
    "TimeTotal_8am", "TimeTotal_2pm", "TimeTotal_8pm"
]

# Define correlation groups based on your description
left_group = ["Tracker8am", "Tracker2pm", "Tracker8pm"]
middle_overlap = ["AdditionalCapacityOpen Morning"]
right_group = ["TimeTotal_8am", "TimeTotal_2pm", "TimeTotal_8pm"]

if not uploaded:
    st.sidebar.info("Please upload your Excel file.")
    st.stop()

# Load data
df = pd.read_excel(uploaded)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
hospitals = sorted(df['Hospital'].unique())
sel_hosp = st.sidebar.selectbox("Hospital", ["All"] + hospitals)
sel_target = st.sidebar.selectbox("Target", ["All"] + targets)

future_days = st.sidebar.slider("Forecast horizon (days ahead)", 1, 30, 7)
use_iterative = st.sidebar.checkbox("Use iterative daily forecasting", value=True)
correlation_threshold = st.sidebar.slider("Correlation threshold for feature selection", 0.1, 0.9, 0.3)

run = st.sidebar.button("🚀 Run Forecast")
if not run:
    st.sidebar.info("Configure then click Run Forecast")
    st.stop()

# Helper functions
def get_correlated_features(target_col, df_hospital, threshold=0.3):
    """Get features correlated with target based on correlation groups"""
    correlations = df_hospital[targets].corr()[target_col].abs()
    correlated_features = correlations[correlations >= threshold].index.tolist()
    
    # Remove the target itself
    if target_col in correlated_features:
        correlated_features.remove(target_col)
    
    # Apply correlation group logic
    if target_col in left_group:
        # For left group, exclude right group unless correlation is very high
        correlated_features = [f for f in correlated_features 
                             if f not in right_group or correlations[f] > 0.7]
    elif target_col in right_group:
        # For right group, exclude left group unless correlation is very high
        correlated_features = [f for f in correlated_features 
                             if f not in left_group or correlations[f] > 0.7]
    # Middle group can use all correlated features
    
    return correlated_features

def create_features(df_data, target_col, correlated_cols, lookback_days=7):
    """Create engineered features including lags and rolling statistics"""
    df_feat = df_data.copy()
    
    # Target lags (up to 7 days)
    for lag in range(1, min(lookback_days + 1, len(df_feat))):
        df_feat[f'{target_col}_lag{lag}'] = df_feat[target_col].shift(lag)
    
    # Target differences
    df_feat[f'{target_col}_diff1'] = df_feat[target_col] - df_feat[target_col].shift(1)
    df_feat[f'{target_col}_diff7'] = df_feat[target_col] - df_feat[target_col].shift(7)
    
    # Rolling statistics for target
    for window in [3, 7, 14]:
        if len(df_feat) >= window:
            df_feat[f'{target_col}_roll_mean{window}'] = df_feat[target_col].rolling(window=window).mean()
            df_feat[f'{target_col}_roll_std{window}'] = df_feat[target_col].rolling(window=window).std()
            df_feat[f'{target_col}_roll_min{window}'] = df_feat[target_col].rolling(window=window).min()
            df_feat[f'{target_col}_roll_max{window}'] = df_feat[target_col].rolling(window=window).max()
    
    # Correlated features lags (limited to most recent)
    for col in correlated_cols:
        if col in df_feat.columns:
            df_feat[f'{col}_lag1'] = df_feat[col].shift(1)
            df_feat[f'{col}_lag3'] = df_feat[col].shift(3)
            df_feat[f'{col}_roll_mean7'] = df_feat[col].rolling(window=7).mean()
    
    # Time-based features
    df_feat['dow'] = df_feat['Date'].dt.weekday
    df_feat['month'] = df_feat['Date'].dt.month
    df_feat['week'] = df_feat['Date'].dt.isocalendar().week.astype(int)
    df_feat['dayofyear'] = df_feat['Date'].dt.dayofyear
    df_feat['is_weekend'] = (df_feat['dow'] >= 5).astype(int)
    
    # Cyclical encoding
    df_feat['dow_sin'] = np.sin(2 * np.pi * df_feat['dow'] / 7)
    df_feat['dow_cos'] = np.cos(2 * np.pi * df_feat['dow'] / 7)
    df_feat['month_sin'] = np.sin(2 * np.pi * df_feat['month'] / 12)
    df_feat['month_cos'] = np.cos(2 * np.pi * df_feat['month'] / 12)
    
    return df_feat

def iterative_forecast(model, df_base, prophet_model, target_col, correlated_cols, future_days, feature_cols):
    """Perform iterative daily forecasting with proper feature alignment"""
    forecast_data = df_base.copy()
    predictions = []
    
    # Get the exact feature names that the model was trained on
    if hasattr(model, 'feature_name_'):
        model_features = model.feature_name_
    else:
        model_features = feature_cols
    
    for day in range(future_days):
        try:
            # Get the last available data
            last_row = forecast_data.iloc[-1]
            next_date = last_row['Date'] + pd.Timedelta(days=1)
            
            # Get Prophet prediction for this date
            try:
                prophet_pred = prophet_model.predict(pd.DataFrame({'ds': [next_date]}))['yhat'].values[0]
            except:
                prophet_pred = last_row.get('prophet_yhat', 0)
            
            # Initialize new row with all model features set to 0
            new_row = {feat: 0.0 for feat in model_features}
            
            # Time-based features
            new_row['dow'] = float(next_date.weekday())
            new_row['month'] = float(next_date.month)
            new_row['week'] = float(next_date.isocalendar().week)
            new_row['dayofyear'] = float(next_date.dayofyear)
            new_row['is_weekend'] = float(1 if next_date.weekday() >= 5 else 0)
            new_row['dow_sin'] = np.sin(2 * np.pi * next_date.weekday() / 7)
            new_row['dow_cos'] = np.cos(2 * np.pi * next_date.weekday() / 7)
            new_row['month_sin'] = np.sin(2 * np.pi * next_date.month / 12)
            new_row['month_cos'] = np.cos(2 * np.pi * next_date.month / 12)
            
            # Target lags
            for lag in range(1, 8):
                lag_col = f'{target_col}_lag{lag}'
                if lag_col in model_features and len(forecast_data) >= lag:
                    new_row[lag_col] = float(forecast_data[target_col].iloc[-lag])
            
            # Target differences
            diff1_col = f'{target_col}_diff1'
            if diff1_col in model_features and len(forecast_data) >= 2:
                new_row[diff1_col] = float(forecast_data[target_col].iloc[-1] - forecast_data[target_col].iloc[-2])
            
            diff7_col = f'{target_col}_diff7'
            if diff7_col in model_features and len(forecast_data) >= 7:
                new_row[diff7_col] = float(forecast_data[target_col].iloc[-1] - forecast_data[target_col].iloc[-7])
            
            # Rolling statistics
            for window in [3, 7, 14]:
                if len(forecast_data) >= window:
                    recent_values = forecast_data[target_col].iloc[-window:]
                    
                    mean_col = f'{target_col}_roll_mean{window}'
                    if mean_col in model_features:
                        new_row[mean_col] = float(recent_values.mean())
                    
                    std_col = f'{target_col}_roll_std{window}'
                    if std_col in model_features:
                        std_val = recent_values.std()
                        new_row[std_col] = float(std_val if not np.isnan(std_val) else 0)
                    
                    min_col = f'{target_col}_roll_min{window}'
                    if min_col in model_features:
                        new_row[min_col] = float(recent_values.min())
                    
                    max_col = f'{target_col}_roll_max{window}'
                    if max_col in model_features:
                        new_row[max_col] = float(recent_values.max())
            
            # Correlated features lags
            for col in correlated_cols:
                if col in forecast_data.columns:
                    lag1_col = f'{col}_lag1'
                    if lag1_col in model_features:
                        new_row[lag1_col] = float(forecast_data[col].iloc[-1])
                    
                    lag3_col = f'{col}_lag3'
                    if lag3_col in model_features and len(forecast_data) >= 3:
                        new_row[lag3_col] = float(forecast_data[col].iloc[-3])
                    
                    roll_col = f'{col}_roll_mean7'
                    if roll_col in model_features and len(forecast_data) >= 7:
                        new_row[roll_col] = float(forecast_data[col].iloc[-7:].mean())
            
            # Prophet features
            if 'prophet_yhat' in model_features:
                new_row['prophet_yhat'] = float(prophet_pred)
            
            if 'prophet_yhat_lag1' in model_features:
                if 'prophet_yhat' in forecast_data.columns and len(forecast_data) > 0:
                    new_row['prophet_yhat_lag1'] = float(forecast_data['prophet_yhat'].iloc[-1])
                else:
                    new_row['prophet_yhat_lag1'] = float(prophet_pred)
            
            # Create prediction input ensuring exact feature order and types
            X_pred = np.array([[new_row[feat] for feat in model_features]], dtype=np.float64)
            
            # Make prediction
            pred = model.predict(X_pred)[0]
            
            # Create the complete new row for appending to forecast_data
            complete_row = {
                'Date': next_date,
                target_col: float(pred),
                'prophet_yhat': float(prophet_pred)
            }
            
            # Add all original columns with appropriate values
            for col in forecast_data.columns:
                if col not in complete_row:
                    if col in new_row:
                        complete_row[col] = new_row[col]
                    elif col in correlated_cols:
                        # For correlated features, use last known value or prediction if it's the same target
                        complete_row[col] = forecast_data[col].iloc[-1]
                    else:
                        complete_row[col] = 0.0
            
            forecast_data = pd.concat([forecast_data, pd.DataFrame([complete_row])], ignore_index=True)
            predictions.append(float(pred))
            
        except Exception as e:
            st.error(f"Error in day {day+1} of forecasting: {str(e)}")
            # Use last prediction or mean as fallback
            fallback_pred = predictions[-1] if predictions else forecast_data[target_col].mean()
            predictions.append(fallback_pred)
            break
    
    return predictions, forecast_data

# Main processing
h_list = hospitals if sel_hosp == "All" else [sel_hosp]
t_list = targets if sel_target == "All" else [sel_target]

results = []

for hosp in h_list:
    st.header(f"🏥 {hosp}")
    df_h = df[df['Hospital'] == hosp].reset_index(drop=True)
    
    # Show correlation matrix for this hospital
    if len(t_list) > 1:
        st.subheader("📊 Target Correlations")
        corr_matrix = df_h[targets].corr()
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu_r',
            zmid=0
        ))
        fig_corr.update_layout(title="Target Variable Correlations")
        st.plotly_chart(fig_corr, use_container_width=True)

    for tgt in t_list:
        st.subheader(f"🎯 {tgt}")
        
        if df_h[tgt].isna().sum() > len(df_h) * 0.1:  # More than 10% nulls
            st.warning(f"Skipping {tgt} (too many nulls: {df_h[tgt].isna().sum()}/{len(df_h)})")
            continue
        
        # Get correlated features for this target
        correlated_features = get_correlated_features(tgt, df_h, correlation_threshold)
        st.info(f"Using correlated features: {correlated_features}")
        
        # Prepare data
        df_work = df_h[['Date', tgt] + [col for col in correlated_features if col in df_h.columns]].copy()
        df_work = df_work.dropna().reset_index(drop=True)
        
        if len(df_work) < 30:  # Need minimum data
            st.warning(f"Insufficient data for {tgt} (only {len(df_work)} rows)")
            continue
        
        # Create features
        df_feat = create_features(df_work, tgt, correlated_features)
        
        # Prophet model for additional features
        df_prophet = df_feat[['Date', tgt]].rename(columns={'Date': 'ds', tgt: 'y'})
        
        m_prophet = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='additive'
        )
        m_prophet.add_country_holidays(country_name='IE')
        m_prophet.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        m_prophet.fit(df_prophet)
        
        # Add Prophet predictions as features
        prophet_preds = m_prophet.predict(df_prophet[['ds']])
        df_feat['prophet_yhat'] = prophet_preds['yhat'].values
        df_feat['prophet_yhat_lag1'] = df_feat['prophet_yhat'].shift(1)
        
        # Remove rows with NaN values
        df_feat = df_feat.dropna().reset_index(drop=True)
        
        if len(df_feat) < 20:
            st.warning(f"Insufficient data after feature engineering for {tgt}")
            continue
        
        # Define feature columns (exclude Date and target)
        feature_cols = [col for col in df_feat.columns 
                       if col not in ['Date', tgt] and not col.startswith('Unnamed')]
        
        # Train-test split
        n = len(df_feat)
        split = int(0.8 * n)
        train_data = df_feat.iloc[:split].copy()
        test_data = df_feat.iloc[split:].copy()
        
        # Train LightGBM model
        X_train = train_data[feature_cols].fillna(0)
        y_train = train_data[tgt]
        X_test = test_data[feature_cols].fillna(0)
        y_test = test_data[tgt]
        
        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=min(5, len(X_train) // 10))
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X_train):
            lgb_model = LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42,
                verbose=-1
            )
            
            # Ensure consistent data types for CV
            X_train_fold = X_train.iloc[train_idx].astype(np.float64)
            y_train_fold = y_train.iloc[train_idx].astype(np.float64)
            X_val_fold = X_train.iloc[val_idx].astype(np.float64)
            y_val_fold = y_train.iloc[val_idx].astype(np.float64)
            
            lgb_model.fit(X_train_fold, y_train_fold)
            val_pred = lgb_model.predict(X_val_fold)
            cv_scores.append(mean_absolute_error(y_val_fold, val_pred))
        
        cv_mae = np.mean(cv_scores) if cv_scores else np.inf
        
        # Train final model
        final_model = LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbose=-1
        )
        
        # Ensure consistent data types
        X_train = X_train.astype(np.float64)
        y_train = y_train.astype(np.float64)
        X_test = X_test.astype(np.float64)
        
        final_model.fit(X_train, y_train)
        
        # Test predictions
        test_pred = final_model.predict(X_test)
        test_mae = mean_absolute_error(y_test, test_pred)
        
        # Future forecasting
        try:
            if use_iterative:
                future_preds, forecast_df = iterative_forecast(
                    final_model, df_feat, m_prophet, tgt, correlated_features, 
                    future_days, feature_cols
                )
                method = "LightGBM (Iterative)"
            else:
                # Simple Prophet-only approach for comparison
                future_dates_prophet = pd.date_range(
                    start=df_feat['Date'].iloc[-1] + pd.Timedelta(days=1),
                    periods=future_days,
                    freq='D'
                )
                prophet_future = pd.DataFrame({'ds': future_dates_prophet})
                prophet_forecast = m_prophet.predict(prophet_future)
                future_preds = prophet_forecast['yhat'].values.tolist()
                method = "Prophet Only"
        except Exception as e:
            st.error(f"Error in forecasting: {str(e)}")
            # Fallback to simple approach
            future_preds = [y_test.iloc[-1]] * future_days
            method = "Fallback (Last Value)"
        
        # Plotting
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=test_data['Date'],
            y=y_test,
            mode='lines+markers',
            name='Actual',
            line=dict(color='blue')
        ))
        
        # Test predictions
        fig.add_trace(go.Scatter(
            x=test_data['Date'],
            y=test_pred,
            mode='lines+markers',
            name='Test Predictions',
            line=dict(color='orange')
        ))
        
        # Future forecasts
        future_dates = pd.date_range(
            start=df_feat['Date'].iloc[-1] + pd.Timedelta(days=1),
            periods=future_days,
            freq='D'
        )
        
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_preds,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title=f"{hosp} • {tgt} | {method} | Test MAE: {test_mae:.2f} | CV MAE: {cv_mae:.2f}",
            xaxis_title="Date",
            yaxis_title=tgt,
            template="plotly_white",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show feature importance
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': final_model.feature_importances_
        }).sort_values('Importance', ascending=False).head(10)
        
        st.markdown("**Top 10 Most Important Features**")
        st.dataframe(feature_importance, use_container_width=True)
        
        # Test results
        st.markdown("**Test-set Results**")
        test_results = pd.DataFrame({
            "Date": test_data['Date'],
            "Actual": y_test,
            "Predicted": test_pred,
            "Error": np.abs(y_test - test_pred)
        }).round(2)
        st.dataframe(test_results, use_container_width=True)
        
        # Future forecast
        st.markdown("**Future Forecast**")
        future_results = pd.DataFrame({
            "Date": future_dates,
            "Forecast": np.round(future_preds, 2)
        })
        st.dataframe(future_results, use_container_width=True)
        
        results.append({
            "Hospital": hosp,
            "Target": tgt,
            "Method": method,
            "Test MAE": round(test_mae, 2),
            "CV MAE": round(cv_mae, 2),
            "Features Used": len(feature_cols),
            "Correlated Targets": len(correlated_features)
        })

# Summary
st.subheader("📊 Summary")
if results:
    summary_df = pd.DataFrame(results)
    st.dataframe(summary_df, use_container_width=True)
    
    # Average performance by target
    if len(results) > 1:
        avg_performance = summary_df.groupby('Target').agg({
            'Test MAE': 'mean',
            'CV MAE': 'mean',
            'Features Used': 'mean'
        }).round(2).reset_index()
        
        st.markdown("**Average Performance by Target**")
        st.dataframe(avg_performance, use_container_width=True)
else:
    st.warning("No successful forecasts generated. Check your data and parameters.")
