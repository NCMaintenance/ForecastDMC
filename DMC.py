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
    
    # Basic lag features (reduced from multiple lags)
    df_feat['y_lag1'] = df_feat[target_col].shift(1)
    df_feat['y_lag7'] = df_feat[target_col].shift(7)  # Weekly pattern
    
    # Simple rolling statistics (shorter windows)
    df_feat['roll_mean3'] = df_feat[target_col].rolling(window=3, min_periods=1).mean()
    df_feat['roll_mean7'] = df_feat[target_col].rolling(window=7, min_periods=1).mean()
    
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

def iterative_forecast(model, last_data, prophet_model, n_days=7):
    """Forecast iteratively, one day at a time"""
    forecasts = []
    current_data = last_data.copy()
    
    for day in range(n_days):
        # Get next date
        next_date = current_data['Date'].iloc[-1] + pd.Timedelta(days=1)
        
        # Get Prophet prediction for this date
        prophet_pred = prophet_model.predict(pd.DataFrame({'ds': [next_date]}))
        prophet_value = prophet_pred['yhat'].iloc[0]
        
        # Create features for next day
        next_row = pd.DataFrame({
            'Date': [next_date],
            'y': [np.nan],  # Unknown target
            'y_lag1': [current_data['y'].iloc[-1]],
            'y_lag7': [current_data['y'].iloc[-7] if len(current_data) >= 7 else current_data['y'].iloc[-1]],
            'roll_mean3': [current_data['y'].iloc[-3:].mean()],
            'roll_mean7': [current_data['y'].iloc[-7:].mean()],
            'dow': [next_date.dayofweek],
            'month': [next_date.month],
            'is_weekend': [1 if next_date.dayofweek >= 5 else 0],
            'dow_sin': [np.sin(2 * np.pi * next_date.dayofweek / 7)],
            'dow_cos': [np.cos(2 * np.pi * next_date.dayofweek / 7)],
            'trend': [current_data['trend'].iloc[-1] + 1],
            'prophet_pred': [prophet_value]
        })
        
        # Features for prediction
        feature_cols = ['y_lag1', 'y_lag7', 'roll_mean3', 'roll_mean7', 'dow', 'month', 
                       'is_weekend', 'dow_sin', 'dow_cos', 'trend', 'prophet_pred']
        
        # Make prediction
        pred = model.predict(next_row[feature_cols])[0]
        forecasts.append(pred)
        
        # Update current_data with prediction
        next_row['y'] = pred
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
        
        # Feature selection (reduced set to prevent overfitting)
        feature_cols = ['y_lag1', 'y_lag7', 'roll_mean3', 'roll_mean7', 'dow', 'month', 
                       'is_weekend', 'dow_sin', 'dow_cos', 'trend', 'prophet_pred']
        
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
        forecast_values = iterative_forecast(final_model, train, m_prophet, n_days=7)
        
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
        
        # Test set results table
        st.markdown("**Test Set Predictions**")
        test_results = pd.DataFrame({
            "Date": test['Date'],
            "Actual": test['y'],
            "Predicted": test_pred,
            "Absolute Error": np.abs(test['y'] - test_pred),
            "Relative Error (%)": (np.abs(test['y'] - test_pred) / test['y'] * 100).round(1)
        }).round(2)
        st.dataframe(test_results, use_container_width=True)
        
        # Future forecast table
        st.markdown("**7-Day Iterative Forecast**")
        forecast_df = pd.DataFrame({
            "Date": forecast_dates,
            "Day": [f"Day {i+1}" for i in range(7)],
            "Forecast": np.round(forecast_values, 2),
            "Day of Week": [date.strftime("%A") for date in forecast_dates]
        })
        st.dataframe(forecast_df, use_container_width=True)
        
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
