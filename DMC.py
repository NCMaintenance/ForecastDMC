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
correlation_threshold = st.sidebar.slider("Correlation threshold for feature selection", 0.1, 0.9, 0.3)
run = st.sidebar.button("🚀 Run Forecast")

if not run:
    st.sidebar.info("Configure then click Run Forecast")
    st.stop()

def get_correlated_features(df_hospital, target_col, targets_list, threshold=0.3):
    """
    Determine which target columns to use as features based on correlation
    """
    # Calculate correlations between target and all other targets
    correlations = {}
    target_data = df_hospital[target_col].dropna()
    
    for other_target in targets_list:
        if other_target != target_col and other_target in df_hospital.columns:
            other_data = df_hospital[other_target].dropna()
            # Find common dates for correlation calculation
            common_dates = df_hospital[df_hospital[target_col].notna() & df_hospital[other_target].notna()]['Date']
            if len(common_dates) > 10:  # Need sufficient data points
                target_common = df_hospital[df_hospital['Date'].isin(common_dates)][target_col]
                other_common = df_hospital[df_hospital['Date'].isin(common_dates)][other_target]
                corr = target_common.corr(other_common)
                if not np.isnan(corr):
                    correlations[other_target] = abs(corr)
    
    # Select features above threshold
    selected_features = [col for col, corr in correlations.items() if corr >= threshold]
    
    return selected_features, correlations

def create_lagged_features(df, target_col, feature_cols, lookback_days=7):
    """
    Create lagged features for target and correlated columns
    """
    df_feat = df[['Date', target_col] + feature_cols].copy()
    df_feat = df_feat.rename(columns={'Date': 'ds', target_col: 'y'})
    
    # Target lags
    for lag in range(1, lookback_days + 1):
        df_feat[f'y_lag{lag}'] = df_feat['y'].shift(lag)
    
    # Feature column lags
    for col in feature_cols:
        for lag in range(1, min(4, lookback_days + 1)):  # Limit feature lags to avoid overfitting
            df_feat[f'{col}_lag{lag}'] = df_feat[col].shift(lag)
    
    # Differencing
    df_feat['y_diff1'] = df_feat['y'] - df_feat['y'].shift(1)
    df_feat['y_diff7'] = df_feat['y'] - df_feat['y'].shift(7)
    
    # Rolling statistics
    for window in [3, 7]:
        df_feat[f'y_roll_mean{window}'] = df_feat['y'].rolling(window=window).mean()
        df_feat[f'y_roll_std{window}'] = df_feat['y'].rolling(window=window).std()
        df_feat[f'y_roll_min{window}'] = df_feat['y'].rolling(window=window).min()
        df_feat[f'y_roll_max{window}'] = df_feat['y'].rolling(window=window).max()
    
    # Time features
    df_feat['dow'] = df_feat['ds'].dt.weekday
    df_feat['month'] = df_feat['ds'].dt.month
    df_feat['quarter'] = df_feat['ds'].dt.quarter
    df_feat['week'] = df_feat['ds'].dt.isocalendar().week.astype(int)
    df_feat['dayofyear'] = df_feat['ds'].dt.dayofyear
    
    # Cyclical encoding
    df_feat['dow_sin'] = np.sin(2 * np.pi * df_feat['dow'] / 7)
    df_feat['dow_cos'] = np.cos(2 * np.pi * df_feat['dow'] / 7)
    df_feat['month_sin'] = np.sin(2 * np.pi * df_feat['month'] / 12)
    df_feat['month_cos'] = np.cos(2 * np.pi * df_feat['month'] / 12)
    
    return df_feat

def iterative_forecast(model, df_base, feature_cols, prophet_model, forecast_days=7):
    """
    Perform iterative 1-day ahead forecasting
    """
    forecasts = []
    current_df = df_base.copy()
    
    for day in range(forecast_days):
        # Get the last row for creating next day features
        last_row = current_df.iloc[-1].copy()
        next_date = last_row['ds'] + pd.Timedelta(days=1)
        
        # Get Prophet prediction for next day
        prophet_pred = prophet_model.predict(pd.DataFrame({'ds': [next_date]}))
        yhat_prophet = prophet_pred['yhat'].values[0]
        
        # Create feature row for next day
        new_row = {'ds': next_date, 'y': np.nan}
        
        # Add lagged features
        lookback_days = 7
        for lag in range(1, lookback_days + 1):
            if len(current_df) >= lag:
                new_row[f'y_lag{lag}'] = current_df.iloc[-lag]['y']
            else:
                new_row[f'y_lag{lag}'] = current_df.iloc[-1]['y']
        
        # Add feature column lags
        for col in [c for c in current_df.columns if any(fc in c for fc in feature_cols)]:
            if col.endswith('_lag1') or col.endswith('_lag2') or col.endswith('_lag3'):
                base_col = col.rsplit('_lag', 1)[0]
                lag_num = int(col.rsplit('_lag', 1)[1])
                if f'{base_col}_lag{lag_num-1}' in current_df.columns:
                    new_row[col] = current_df.iloc[-1][f'{base_col}_lag{lag_num-1}']
                elif base_col in current_df.columns:
                    new_row[col] = current_df.iloc[-1][base_col]
                else:
                    new_row[col] = 0
        
        # Add differencing
        new_row['y_diff1'] = new_row['y_lag1'] - new_row.get('y_lag2', new_row['y_lag1'])
        if len(current_df) >= 7:
            new_row['y_diff7'] = new_row['y_lag1'] - new_row.get('y_lag7', new_row['y_lag1'])
        else:
            new_row['y_diff7'] = 0
        
        # Add rolling statistics
        recent_values = current_df['y'].dropna().iloc[-7:].values
        for window in [3, 7]:
            window_values = recent_values[-window:] if len(recent_values) >= window else recent_values
            new_row[f'y_roll_mean{window}'] = np.mean(window_values)
            new_row[f'y_roll_std{window}'] = np.std(window_values) if len(window_values) > 1 else 0
            new_row[f'y_roll_min{window}'] = np.min(window_values)
            new_row[f'y_roll_max{window}'] = np.max(window_values)
        
        # Add time features
        new_row['dow'] = next_date.weekday()
        new_row['month'] = next_date.month
        new_row['quarter'] = next_date.quarter
        new_row['week'] = next_date.isocalendar().week
        new_row['dayofyear'] = next_date.dayofyear
        new_row['dow_sin'] = np.sin(2 * np.pi * next_date.weekday() / 7)
        new_row['dow_cos'] = np.cos(2 * np.pi * next_date.weekday() / 7)
        new_row['month_sin'] = np.sin(2 * np.pi * next_date.month / 12)
        new_row['month_cos'] = np.cos(2 * np.pi * next_date.month / 12)
        
        # Make prediction
        feature_row = pd.DataFrame([new_row])
        available_features = [col for col in feature_cols if col in feature_row.columns]
        
        if available_features:
            pred = model.predict(feature_row[available_features])[0]
        else:
            pred = yhat_prophet  # Fallback to Prophet if no features available
        
        # Store forecast
        forecasts.append({'Date': next_date, 'Forecast': pred})
        
        # Update the dataframe with the new prediction
        new_row['y'] = pred
        current_df = pd.concat([current_df, pd.DataFrame([new_row])], ignore_index=True)
    
    return forecasts

# Filters
h_list = hospitals if sel_hosp == "All" else [sel_hosp]
t_list = targets if sel_target == "All" else [sel_target]

results = []

for hosp in h_list:
    st.header(f"🏥 {hosp}")
    df_h = df[df['Hospital'] == hosp].reset_index(drop=True)
    
    for tgt in t_list:
        st.subheader(f"🎯 {tgt}")
        
        if df_h[tgt].isna().all():
            st.warning("Skipping (all values are null)")
            continue
        
        # Get correlated features for this target
        correlated_features, correlations = get_correlated_features(df_h, tgt, targets, correlation_threshold)
        
        st.info(f"**Correlated features selected:** {', '.join(correlated_features) if correlated_features else 'None'}")
        if correlations:
            st.write("**Correlation scores:**")
            corr_df = pd.DataFrame(list(correlations.items()), columns=['Feature', 'Correlation'])
            corr_df = corr_df.sort_values('Correlation', ascending=False)
            st.dataframe(corr_df, use_container_width=True)
        
        # Create features
        df_feat = create_lagged_features(df_h, tgt, correlated_features, lookback_days=7)
        df_feat = df_feat.dropna().reset_index(drop=True)
        
        if len(df_feat) < 30:
            st.warning("Insufficient data after feature creation")
            continue
        
        # Train/test split
        n = len(df_feat)
        split = int(0.8 * n)
        train = df_feat.iloc[:split].reset_index(drop=True)
        test = df_feat.iloc[split:].reset_index(drop=True)
        
        # Prepare features
        feature_columns = [col for col in df_feat.columns if col not in ['ds', 'y'] and not col.startswith('Unnamed')]
        
        # Train Prophet model
        m_full = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='additive'
        )
        m_full.add_country_holidays(country_name='IE')
        m_full.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        m_full.fit(df_feat[['ds', 'y']])
        
        # Train LightGBM model
        X_train, y_train = train[feature_columns], train['y']
        X_test, y_test = test[feature_columns], test['y']
        
        # Cross-validation for LightGBM
        tscv = TimeSeriesSplit(n_splits=3)
        lgb_maes = []
        
        for train_idx, val_idx in tscv.split(X_train):
            lgb_model = LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42
            )
            lgb_model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            val_pred = lgb_model.predict(X_train.iloc[val_idx])
            lgb_maes.append(mean_absolute_error(y_train.iloc[val_idx], val_pred))
        
        mae_lgb = np.mean(lgb_maes)
        
        # Train final LightGBM model
        lgb_final = LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42
        )
        lgb_final.fit(X_train, y_train)
        
        # Test set predictions
        lgb_test_pred = lgb_final.predict(X_test)
        
        # Prophet cross-validation
        try:
            cv_results = cross_validation(
                m_full,
                initial='90 days',
                period='15 days',
                horizon='15 days',
                parallel=None
            )
            mae_prophet = performance_metrics(cv_results)['mae'].mean()
        except:
            mae_prophet = np.inf
        
        # Combine predictions
        if np.isinf(mae_prophet):
            test_pred = lgb_test_pred
            method = "LightGBM only"
        else:
            w_prophet = 1 / mae_prophet
            w_lgb = 1 / mae_lgb
            total_weight = w_prophet + w_lgb
            
            prophet_test_pred = m_full.predict(test[['ds']])['yhat'].values
            test_pred = (w_prophet * prophet_test_pred + w_lgb * lgb_test_pred) / total_weight
            method = f"Hybrid (Prophet: {w_prophet/total_weight:.2f}, LGB: {w_lgb/total_weight:.2f})"
        
        mae_test = mean_absolute_error(y_test, test_pred)
        
        # Future forecasting using iterative approach
        future_forecasts = iterative_forecast(
            lgb_final, df_feat, feature_columns, m_full, future_days
        )
        
        # Plotting
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=train['ds'],
            y=train['y'],
            mode='lines',
            name='Training Data',
            line=dict(color='blue')
        ))
        
        # Test data
        fig.add_trace(go.Scatter(
            x=test['ds'],
            y=test['y'],
            mode='lines+markers',
            name='Actual (Test)',
            line=dict(color='green')
        ))
        
        # Test predictions
        fig.add_trace(go.Scatter(
            x=test['ds'],
            y=test_pred,
            mode='lines+markers',
            name='Predicted (Test)',
            line=dict(color='red')
        ))
        
        # Future forecasts
        forecast_dates = [f['Date'] for f in future_forecasts]
        forecast_values = [f['Forecast'] for f in future_forecasts]
        
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_values,
            mode='lines+markers',
            name='Future Forecast',
            line=dict(color='orange', dash='dash')
        ))
        
        fig.update_layout(
            title=f"{hosp} • {tgt} | {method} | Test MAE: {mae_test:.2f}",
            xaxis_title="Date",
            yaxis_title=tgt,
            template="plotly_white",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Results tables
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Test Set Results**")
            test_results_df = pd.DataFrame({
                "Date": test['ds'],
                "Actual": test['y'],
                "Predicted": test_pred,
                "Error": np.abs(test['y'] - test_pred)
            }).round(2)
            st.dataframe(test_results_df, use_container_width=True)
        
        with col2:
            st.markdown("**Future Forecasts**")
            forecast_df = pd.DataFrame(future_forecasts).round(2)
            st.dataframe(forecast_df, use_container_width=True)
        
        results.append({
            "Hospital": hosp,
            "Target": tgt,
            "Method": method,
            "Test MAE": round(mae_test, 2),
            "Features Used": len(correlated_features),
            "Forecast Days": future_days
        })

# Summary
st.subheader("📊 Summary")
results_df = pd.DataFrame(results)
st.dataframe(results_df, use_container_width=True)

# Feature importance (if available)
if 'lgb_final' in locals():
    st.subheader("🔍 Feature Importance (Last Model)")
    try:
        importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': lgb_final.feature_importances_
        }).sort_values('Importance', ascending=False).head(20)
        
        fig_imp = go.Figure(go.Bar(
            x=importance_df['Importance'],
            y=importance_df['Feature'],
            orientation='h'
        ))
        fig_imp.update_layout(
            title="Top 20 Feature Importances",
            xaxis_title="Importance",
            yaxis_title="Features",
            template="plotly_white",
            height=600
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    except:
        st.info("Feature importance not available for this model configuration.")
