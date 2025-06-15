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
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")
st.title("🏥 Enhanced Hospital Forecasting with Advanced Feature Engineering & Prophet Integration 📈")

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Configuration")
uploaded = st.sidebar.file_uploader("📂 Upload Excel file", type="xlsx")
hospitals = []
targets = [
    "Tracker8am", "Tracker2pm", "Tracker8pm",
    "AdditionalCapacityOpen Morning",
    "TimeTotal_8am", "TimeTotal_2pm", "TimeTotal_8pm"
]

if not uploaded:
    st.info("👋 Welcome! Please upload your Excel file using the sidebar to begin.")
    st.stop()

# --- Data Loading and Initial Processing ---
@st.cache_data
def load_data(uploaded_file):
    df = pd.read_excel(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    return df

df = load_data(uploaded)
hospitals = sorted(df['Hospital'].unique())

sel_hosp = st.sidebar.selectbox("🏨 Hospital", ["All"] + hospitals)
sel_target = st.sidebar.selectbox("🎯 Target", ["All"] + targets)
future_days = st.sidebar.slider("⏳ Forecast horizon (days ahead)", 7, 30, 14)
correlation_threshold = st.sidebar.slider("📊 Correlation threshold for feature selection", 0.0, 0.5, 0.1, 0.05)
max_forecast_horizon = st.sidebar.slider("🔍 Max forecast horizon to test (days)", 1, 7, 7)
run = st.sidebar.button("▶️ Run Forecast")

if not run:
    st.info("ℹ️ Configure your forecast parameters in the sidebar and click 'Run Forecast'.")
    st.stop()

# --- Feature Engineering ---
def create_advanced_features(df_input, target_col, all_hospital_cols, max_lags=7):
    """
    Create comprehensive feature set.
    This function is now robust and checks for column existence before creating features.
    """
    df_feat = df_input.copy().rename(columns={'ds': 'Date'}) # Use 'Date' internally
    
    # Combine predictor columns and the target column for feature generation
    all_feature_cols = all_hospital_cols + [target_col]
    
    for col in all_feature_cols:
        # --- FIX: Check if the source column exists before creating features from it ---
        if col in df_feat.columns:
            col_name = 'target' if col == target_col else col
            
            # Lag features
            for lag in range(1, max_lags + 1):
                df_feat[f'{col_name}_lag_{lag}'] = df_feat[col].shift(lag)
            
            # Rolling statistics
            windows = [3, 7, 14]
            for window in windows:
                df_feat[f'{col_name}_roll_mean_{window}'] = df_feat[col].shift(1).rolling(window=window, min_periods=1).mean()
                df_feat[f'{col_name}_roll_std_{window}'] = df_feat[col].shift(1).rolling(window=window, min_periods=1).std()

            # Exponentially Weighted Mean
            df_feat[f'{col_name}_ewm_mean_0.3'] = df_feat[col].shift(1).ewm(alpha=0.3).mean()

    # Date-based features (always created)
    df_feat['dow'] = df_feat['Date'].dt.dayofweek
    df_feat['month'] = df_feat['Date'].dt.month
    df_feat['quarter'] = df_feat['Date'].dt.quarter
    df_feat['weekofyear'] = df_feat['Date'].dt.isocalendar().week.astype(int)
    df_feat['dayofyear'] = df_feat['Date'].dt.dayofyear
    df_feat['is_weekend'] = (df_feat['dow'] >= 5).astype(int)
    
    # Cyclical encoding
    df_feat['dow_sin'] = np.sin(2 * np.pi * df_feat['dow'] / 7)
    df_feat['dow_cos'] = np.cos(2 * np.pi * df_feat['dow'] / 7)
    df_feat['month_sin'] = np.sin(2 * np.pi * df_feat['month'] / 12)
    df_feat['month_cos'] = np.cos(2 * np.pi * df_feat['month'] / 12)

    return df_feat.rename(columns={'Date': 'ds'})

# --- Feature Selection ---
def select_features_by_correlation(X, y, threshold=0.1):
    """Select features based on Pearson correlation with the target."""
    correlations = {}
    for col in X.columns:
        if not X[col].isna().all() and pd.api.types.is_numeric_dtype(X[col]):
            corr, _ = pearsonr(X[col].fillna(0), y)
            if abs(corr) >= threshold:
                correlations[col] = abs(corr)
    
    selected_features = list(correlations.keys())
    return selected_features, correlations

# --- Prophet Forecasting ---
def create_prophet_forecasts(prophet_df, train_size, forecast_days=14):
    """Create Prophet forecasts with confidence intervals."""
    m = Prophet(
        yearly_seasonality=True, 
        weekly_seasonality=True, 
        daily_seasonality=False,
        interval_width=0.95,
        seasonality_mode='multiplicative'
    )
    m.add_country_holidays(country_name='IE')
    
    train_prophet = prophet_df.iloc[:train_size]
    m.fit(train_prophet)
    
    future = m.make_future_dataframe(periods=len(prophet_df) - train_size + forecast_days)
    forecast = m.predict(future)
    
    return m, forecast

# --- Main Application Logic ---
all_results = []
h_list = hospitals if sel_hosp == "All" else [sel_hosp]
t_list = targets if sel_target == "All" else [sel_target]

for hosp in h_list:
    st.header(f"🏥 Hospital: {hosp}", divider='rainbow')
    df_h = df[df['Hospital'] == hosp].reset_index(drop=True)
    
    for tgt in t_list:
        st.subheader(f"🎯 Target: {tgt}")
        
        if tgt not in df_h.columns or df_h[tgt].isna().all():
            st.warning("⚠️ Target data not available or is all null. Skipping.")
            continue

        # Get all other available metrics to use as features
        available_targets = [col for col in targets if col in df_h.columns and not df_h[col].isna().all()]
        df2 = df_h[['Date'] + available_targets].rename(columns={'Date': 'ds', tgt: 'y'})
        
        # --- Feature Engineering & Data Splitting ---
        df_feat = create_advanced_features(df2, 'y', available_targets, max_lags=7)
        df_feat = df_feat.dropna(subset=['y'])

        n = len(df_feat)
        if n < 60:
            st.warning("⚠️ Insufficient data for this target after processing. Skipping.")
            continue
            
        train_size = int(0.8 * n)
        train_df = df_feat.iloc[:train_size].copy()
        test_df = df_feat.iloc[train_size:].copy()

        # --- Feature Selection ---
        feature_cols = [col for col in df_feat.columns if col not in ['ds', 'y'] and pd.api.types.is_numeric_dtype(df_feat[col])]
        
        selected_features, feature_corrs = select_features_by_correlation(
            train_df[feature_cols].fillna(0), train_df['y'], threshold=correlation_threshold
        )
        
        if not selected_features:
            st.warning("⚠️ No features met the correlation threshold. Skipping.")
            continue
        
        with st.expander(f"📊 Selected {len(selected_features)} features. Click to see details."):
            top_features = sorted(feature_corrs.items(), key=lambda x: x[1], reverse=True)[:15]
            st.write("🔝 Top 15 Features by Correlation:")
            st.dataframe(pd.DataFrame(top_features, columns=['Feature', 'Correlation']))

        X_train = train_df[selected_features].fillna(0)
        y_train = train_df['y']
        X_test = test_df[selected_features].fillna(0)
        y_test = test_df['y']
        
        # --- Model Training (LightGBM) ---
        lgb_model = LGBMRegressor(random_state=42)
        lgb_model.fit(X_train, y_train)
        
        test_pred_lgb = lgb_model.predict(X_test)
        test_mae_lgb = mean_absolute_error(y_test, test_pred_lgb)
        test_rmse_lgb = np.sqrt(mean_squared_error(y_test, test_pred_lgb))
        
        # --- Prophet Forecasting ---
        try:
            prophet_model, prophet_forecast = create_prophet_forecasts(
                df2[['ds', 'y']], train_size, forecast_days=future_days
            )
            prophet_test_pred = prophet_forecast['yhat'][train_size:n].values
            test_mae_prophet = mean_absolute_error(y_test, prophet_test_pred)
            prophet_future_forecast = prophet_forecast.iloc[-future_days:]
        except Exception as e:
            st.error(f"⚠️ Prophet forecasting failed: {e}")
            test_mae_prophet = float('inf')
            prophet_model, prophet_future_forecast = None, None

        # --- Results Visualization (Test Period) ---
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test_df['ds'], y=y_test, mode='lines', name='Actual', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=test_df['ds'], y=test_pred_lgb, mode='lines', name='LightGBM Forecast', line=dict(color='red')))
        
        if test_mae_prophet != float('inf'):
            fig.add_trace(go.Scatter(x=test_df['ds'], y=prophet_test_pred, mode='lines', name='Prophet Forecast', line=dict(color='green', dash='dash')))
            
        fig.update_layout(title=f"Model Performance on Test Data", xaxis_title="Date", yaxis_title="Value", legend_title="Legend")
        st.plotly_chart(fig, use_container_width=True)

        # --- Performance Metrics ---
        col1, col2, col3 = st.columns(3)
        col1.metric("LightGBM Test MAE", f"{test_mae_lgb:.2f}")
        col2.metric("LightGBM Test RMSE", f"{test_rmse_lgb:.2f}")
        if test_mae_prophet != float('inf'):
            improvement = ((test_mae_prophet - test_mae_lgb) / test_mae_prophet) * 100
            col3.metric("Prophet Test MAE", f"{test_mae_prophet:.2f}", f"{improvement:.1f}% vs LGBM")

        # --- Feature Importance ---
        with st.expander("💡 LightGBM Feature Importance"):
            importance_df = pd.DataFrame({
                'feature': selected_features,
                'importance': lgb_model.feature_importances_
            }).sort_values('importance', ascending=False).head(20)
            
            fig_imp = go.Figure(go.Bar(x=importance_df['importance'], y=importance_df['feature'], orientation='h'))
            fig_imp.update_layout(title="Top 20 Feature Importances", yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_imp, use_container_width=True)

        # --- Future Forecasting ---
        st.write(f"🔮 **Generating Future Forecasts for the next {future_days} days...**")
        
        # Train final model on all available data
        full_train_X = df_feat[selected_features].fillna(0)
        full_train_y = df_feat['y']
        lgb_model.fit(full_train_X, full_train_y)

        # Create a template for future dates
        last_known_date = df_feat['ds'].max()
        future_dates = pd.to_datetime(pd.date_range(start=last_known_date + pd.Timedelta(days=1), periods=future_days))
        future_df_template = pd.DataFrame({'ds': future_dates})

        # Use the robust create_advanced_features to generate date-based features
        future_features_df = create_advanced_features(future_df_template, 'y', [], max_lags=1)

        # Carry forward the last known values for non-date based features
        for col in selected_features:
            if col not in future_features_df.columns:
                future_features_df[col] = full_train_X[col].iloc[-1]
        
        # Ensure column order is the same as training
        future_features_df = future_features_df[selected_features].fillna(0)

        future_preds_lgb = lgb_model.predict(future_features_df)
        
        # --- Future Forecast Visualization & Table ---
        fig_future = go.Figure()
        fig_future.add_trace(go.Scatter(x=df_feat['ds'].tail(30), y=df_feat['y'].tail(30), mode='lines', name='Historical Data', line=dict(color='grey')))
        fig_future.add_trace(go.Scatter(x=future_dates, y=future_preds_lgb, mode='lines+markers', name='LightGBM Forecast', line=dict(color='red')))
        
        if prophet_future_forecast is not None:
            fig_future.add_trace(go.Scatter(x=prophet_future_forecast['ds'], y=prophet_future_forecast['yhat'], mode='lines', name='Prophet Forecast', line=dict(color='green', dash='dot')))
            fig_future.add_trace(go.Scatter(x=prophet_future_forecast['ds'], y=prophet_future_forecast['yhat_upper'], fill=None, mode='lines', line_color='rgba(0,176,246,0.2)', showlegend=False))
            fig_future.add_trace(go.Scatter(x=prophet_future_forecast['ds'], y=prophet_future_forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(0,176,246,0.2)', name='Prophet 95% CI'))

        fig_future.update_layout(title=f"Future Forecast: {hosp} - {tgt}", xaxis_title="Date", yaxis_title="Value")
        st.plotly_chart(fig_future, use_container_width=True)

        # --- Forecast Table ---
        forecast_summary = pd.DataFrame({'Date': future_dates, 'LightGBM Forecast': np.round(future_preds_lgb, 2)})
        if prophet_future_forecast is not None:
            forecast_summary['Prophet Forecast'] = np.round(prophet_future_forecast['yhat'].values, 2)
            forecast_summary['Prophet Lower 95%'] = np.round(prophet_future_forecast['yhat_lower'].values, 2)
            forecast_summary['Prophet Upper 95%'] = np.round(prophet_future_forecast['yhat_upper'].values, 2)
        
        st.dataframe(forecast_summary.set_index('Date'), use_container_width=True)
        all_results.append({
            'Hospital': hosp, 'Target': tgt, **forecast_summary.set_index('Date').to_dict('list')
        })

        # --- Prophet Components Plot ---
        if prophet_model:
            with st.expander("🔍 Prophet Forecast Components"):
                fig_prophet_comp = prophet_model.plot_components(prophet_forecast)
                st.pyplot(fig_prophet_comp)

# --- Consolidated Summary ---
if len(all_results) > 1 and (sel_hosp == "All" or sel_target == "All"):
    st.header("📋 Consolidated Future Forecast Summary", divider='rainbow')
    
    summary_list = []
    for res in all_results:
        days = [1, 3, 7]
        row = {'Hospital': res['Hospital'], 'Target': res['Target']}
        for day in days:
            if 'LightGBM Forecast' in res and day <= len(res['LightGBM Forecast']):
                row[f'Day {day} Forecast (LGBM)'] = res['LightGBM Forecast'][day-1]
        summary_list.append(row)
    
    summary_df = pd.DataFrame(summary_list)
    st.dataframe(summary_df, use_container_width=True)

