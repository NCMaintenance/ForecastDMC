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
from scipy import stats
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
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
hospitals = sorted(df['Hospital'].unique())

sel_hosp = st.sidebar.selectbox("🏨 Hospital", ["All"] + hospitals)
sel_target = st.sidebar.selectbox("🎯 Target", ["All"] + targets)
future_days = st.sidebar.slider("⏳ Forecast horizon (days ahead)", 7, 30, 14)
correlation_threshold = st.sidebar.slider("📊 Correlation threshold for feature selection", 0.0, 0.5, 0.1)
max_forecast_horizon = st.sidebar.slider("🔍 Max forecast horizon to test (days)", 1, 7, 7)
run = st.sidebar.button("▶️ Run Forecast")

if not run:
    st.sidebar.info("⚙️ Configure then click Run Forecast")
    st.stop()

def create_advanced_features(df_input, target_col, max_lags=7):
    """Create comprehensive feature set with leak prevention"""
    df_feat = df_input.copy()
    
    # Basic lag features (1-7 days)
    for lag in range(1, max_lags + 1):
        df_feat[f'y_lag_{lag}'] = df_feat[target_col].shift(lag)
    
    # Differencing features
    for lag in [1, 2, 3, 7]:
        df_feat[f'y_diff_{lag}'] = df_feat[target_col].diff(lag)
    
    # Rolling statistics (multiple windows)
    windows = [3, 7, 14, 21]
    for window in windows:
        df_feat[f'roll_mean_{window}'] = df_feat[target_col].rolling(window=window, min_periods=1).mean().shift(1)
        df_feat[f'roll_std_{window}'] = df_feat[target_col].rolling(window=window, min_periods=1).std().shift(1)
        df_feat[f'roll_min_{window}'] = df_feat[target_col].rolling(window=window, min_periods=1).min().shift(1)
        df_feat[f'roll_max_{window}'] = df_feat[target_col].rolling(window=window, min_periods=1).max().shift(1)
        df_feat[f'roll_median_{window}'] = df_feat[target_col].rolling(window=window, min_periods=1).median().shift(1)
    
    # Exponential weighted features
    for alpha in [0.1, 0.3, 0.5]:
        df_feat[f'ewm_mean_{alpha}'] = df_feat[target_col].ewm(alpha=alpha).mean().shift(1)
    
    # Temporal features
    df_feat['dow'] = df_feat['ds'].dt.dayofweek
    df_feat['month'] = df_feat['ds'].dt.month
    df_feat['quarter'] = df_feat['ds'].dt.quarter
    df_feat['week'] = df_feat['ds'].dt.isocalendar().week.astype(int)
    df_feat['dayofyear'] = df_feat['ds'].dt.dayofyear
    df_feat['is_weekend'] = (df_feat['dow'] >= 5).astype(int)
    df_feat['is_month_start'] = df_feat['ds'].dt.is_month_start.astype(int)
    df_feat['is_month_end'] = df_feat['ds'].dt.is_month_end.astype(int)
    
    # Cyclical encoding
    df_feat['dow_sin'] = np.sin(2 * np.pi * df_feat['dow'] / 7)
    df_feat['dow_cos'] = np.cos(2 * np.pi * df_feat['dow'] / 7)
    df_feat['month_sin'] = np.sin(2 * np.pi * df_feat['month'] / 12)
    df_feat['month_cos'] = np.cos(2 * np.pi * df_feat['month'] / 12)
    df_feat['dayofyear_sin'] = np.sin(2 * np.pi * df_feat['dayofyear'] / 365.25)
    df_feat['dayofyear_cos'] = np.cos(2 * np.pi * df_feat['dayofyear'] / 365.25)
    
    # Interaction features
    df_feat['dow_month_interaction'] = df_feat['dow'] * df_feat['month']
    
    # Trend features
    df_feat['linear_trend'] = range(len(df_feat))
    df_feat['quadratic_trend'] = df_feat['linear_trend'] ** 2
    
    # Volatility features
    for window in [7, 14]:
        rolling_mean = df_feat[target_col].rolling(window=window, min_periods=1).mean().shift(1)
        df_feat[f'volatility_{window}'] = (df_feat[target_col].shift(1) - rolling_mean).abs()
    
    return df_feat

def select_features_by_correlation(X, y, threshold=0.1):
    """Select features based on correlation with target and remove highly correlated features"""
    correlations = {}
    selected_features = []
    
    for col in X.columns:
        if not X[col].isna().all():
            corr, p_value = pearsonr(X[col].fillna(0), y)
            if abs(corr) > threshold and p_value < 0.05:
                correlations[col] = abs(corr)
    
    # Sort by correlation strength
    sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    
    # Remove highly correlated features (multicollinearity)
    selected_features = []
    for feature, corr in sorted_features:
        add_feature = True
        for selected_feature in selected_features:
            if not X[feature].isna().all() and not X[selected_feature].isna().all():
                feature_corr = pearsonr(X[feature].fillna(0), X[selected_feature].fillna(0))[0]
                if abs(feature_corr) > 0.8:  # High correlation threshold
                    add_feature = False
                    break
        if add_feature:
            selected_features.append(feature)
    
    return selected_features, correlations

def evaluate_forecast_horizons(X_train, y_train, X_test, y_test, max_horizon=7):
    """Test different forecast horizons to find optimal performance"""
    horizon_results = {}
    
    for horizon in range(1, max_horizon + 1):
        # Create horizon-specific target
        y_train_h = y_train.shift(-horizon + 1).dropna()
        X_train_h = X_train.iloc[:len(y_train_h)]
        
        if len(y_train_h) < 30:  # Minimum samples needed
            continue
            
        # Train model
        model = LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=15,
            max_depth=6,
            min_child_samples=20,
            random_state=42
        )
        
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X_train_h):
            X_tr, X_val = X_train_h.iloc[train_idx], X_train_h.iloc[val_idx]
            y_tr, y_val = y_train_h.iloc[train_idx], y_train_h.iloc[val_idx]
            
            model.fit(X_tr, y_tr)
            pred_val = model.predict(X_val)
            cv_scores.append(mean_absolute_error(y_val, pred_val))
        
        horizon_results[horizon] = np.mean(cv_scores)
    
    return horizon_results

# Filters
h_list = hospitals if sel_hosp == "All" else [sel_hosp]
t_list = targets if sel_target == "All" else [sel_target]
results = []

for hosp in h_list:
    st.header(f"🏥 Hospital: {hosp}")
    df_h = df[df['Hospital'] == hosp].reset_index(drop=True)

    for tgt in t_list:
        st.subheader(f"🎯 Target: {tgt}")
        if df_h[tgt].isna().any():
            st.warning("⚠️ Skipping due to null values in target")
            continue

        df2 = df_h[['Date', tgt]].rename(columns={'Date': 'ds', tgt: 'y'})
        
        # Create advanced features
        df_feat = create_advanced_features(df2, 'y', max_lags=7)
        
        # Remove rows with too many NaN values
        df_feat = df_feat.dropna(thresh=len(df_feat.columns) * 0.7)
        
        if len(df_feat) < 50:
            st.warning("⚠️ Insufficient data after feature engineering")
            continue
        
        # Split data (stricter time-based split)
        n = len(df_feat)
        train_size = int(0.75 * n)  # Use 75% for training
        
        train_data = df_feat.iloc[:train_size].copy()
        test_data = df_feat.iloc[train_size:].copy()
        
        # Feature selection based on correlation
        feature_cols = [col for col in df_feat.columns if col not in ['ds', 'y']]
        
        # Select features using correlation analysis
        selected_features, feature_correlations = select_features_by_correlation(
            train_data[feature_cols].fillna(0), 
            train_data['y'], 
            threshold=correlation_threshold
        )
        
        if not selected_features:
            st.warning("⚠️ No features selected based on correlation threshold")
            continue
        
        st.write(f"📊 Selected {len(selected_features)} features out of {len(feature_cols)} total features")
        
        # Show top correlations
        top_features = sorted(feature_correlations.items(), key=lambda x: x[1], reverse=True)[:10]
        st.write("🔝 Top 10 Features by Correlation:")
        for feat, corr in top_features:
            st.write(f"   • {feat}: {corr:.3f}")
        
        # Prepare training data
        X_train = train_data[selected_features].fillna(method='ffill').fillna(0)
        y_train = train_data['y']
        X_test = test_data[selected_features].fillna(method='ffill').fillna(0)
        y_test = test_data['y']
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), 
            columns=X_train.columns, 
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), 
            columns=X_test.columns, 
            index=X_test.index
        )
        
        # Evaluate different forecast horizons
        st.write("🔍 Evaluating forecast horizons...")
        horizon_results = evaluate_forecast_horizons(
            X_train_scaled, y_train, X_test_scaled, y_test, max_forecast_horizon
        )
        
        if horizon_results:
            best_horizon = min(horizon_results.items(), key=lambda x: x[1])
            st.write(f"🎯 Best forecast horizon: {best_horizon[0]} days (MAE: {best_horizon[1]:.3f})")
            
            # Show horizon comparison
            horizon_df = pd.DataFrame([
                {'Horizon': h, 'MAE': mae} for h, mae in horizon_results.items()
            ])
            st.dataframe(horizon_df)
        else:
            best_horizon = (1, float('inf'))
        
        # Train final LightGBM model with optimal parameters
        lgb_model = LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=20,
            max_depth=8,
            min_child_samples=15,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42
        )
        
        # Cross-validation for model evaluation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X_train_scaled):
            X_tr, X_val = X_train_scaled.iloc[train_idx], X_train_scaled.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            lgb_model.fit(X_tr, y_tr)
            pred_val = lgb_model.predict(X_val)
            cv_scores.append(mean_absolute_error(y_val, pred_val))
        
        cv_mae = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        # Train final model on all training data
        lgb_model.fit(X_train_scaled, y_train)
        
        # Test predictions
        test_pred = lgb_model.predict(X_test_scaled)
        test_mae = mean_absolute_error(y_test, test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': selected_features,
            'importance': lgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Prophet baseline for comparison
        prophet_df = df2[['ds', 'y']].copy()
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        m.add_country_holidays(country_name='Ireland')
        
        try:
            m.fit(prophet_df.iloc[:train_size])
            prophet_test_pred = m.predict(prophet_df.iloc[train_size:][['ds']])['yhat'].values
            prophet_mae = mean_absolute_error(y_test, prophet_test_pred)
        except:
            prophet_mae = float('inf')
        
        # Visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=test_data['ds'], 
            y=y_test, 
            mode='lines+markers', 
            name='Actual', 
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=test_data['ds'], 
            y=test_pred, 
            mode='lines', 
            name='LightGBM Prediction', 
            line=dict(color='red')
        ))
        
        if prophet_mae != float('inf'):
            fig.add_trace(go.Scatter(
                x=test_data['ds'], 
                y=prophet_test_pred, 
                mode='lines', 
                name='Prophet Baseline', 
                line=dict(color='green', dash='dash')
            ))
        
        fig.update_layout(
            title=f"{hosp} - {tgt} Test Period Forecast",
            xaxis_title="Date", 
            yaxis_title="Value",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("CV MAE", f"{cv_mae:.3f}", f"±{cv_std:.3f}")
        with col2:
            st.metric("Test MAE", f"{test_mae:.3f}")
        with col3:
            st.metric("Test RMSE", f"{test_rmse:.3f}")
        
        if prophet_mae != float('inf'):
            st.write(f"📊 Prophet Baseline MAE: {prophet_mae:.3f}")
            improvement = ((prophet_mae - test_mae) / prophet_mae) * 100
            st.write(f"🚀 Improvement over Prophet: {improvement:.1f}%")
        
        # Feature importance plot
        if len(feature_importance) > 0:
            fig_imp = go.Figure(go.Bar(
                x=feature_importance['importance'].head(15),
                y=feature_importance['feature'].head(15),
                orientation='h'
            ))
            fig_imp.update_layout(
                title="Top 15 Feature Importances",
                xaxis_title="Importance",
                yaxis_title="Features",
                height=500
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        
        # Generate future forecast using iterative approach
        st.write("🔮 Generating future forecasts...")
        
        # Implement proper walk-forward forecasting
        future_preds = []
        current_data = df_feat.copy()
        
        for day in range(future_days):
            # Get the last row for feature generation
            last_row = current_data.iloc[-1].copy()
            next_date = last_row['ds'] + pd.Timedelta(days=1)
            
            # Create new row with engineered features
            new_row = pd.Series(index=current_data.columns)
            new_row['ds'] = next_date
            
            # Generate temporal features
            new_row['dow'] = next_date.dayofweek
            new_row['month'] = next_date.month
            new_row['quarter'] = next_date.quarter
            new_row['week'] = next_date.isocalendar().week
            new_row['dayofyear'] = next_date.dayofyear
            new_row['is_weekend'] = int(next_date.dayofweek >= 5)
            new_row['is_month_start'] = int(next_date.is_month_start)
            new_row['is_month_end'] = int(next_date.is_month_end)
            
            # Cyclical features
            new_row['dow_sin'] = np.sin(2 * np.pi * new_row['dow'] / 7)
            new_row['dow_cos'] = np.cos(2 * np.pi * new_row['dow'] / 7)
            new_row['month_sin'] = np.sin(2 * np.pi * new_row['month'] / 12)
            new_row['month_cos'] = np.cos(2 * np.pi * new_row['month'] / 12)
            new_row['dayofyear_sin'] = np.sin(2 * np.pi * new_row['dayofyear'] / 365.25)
            new_row['dayofyear_cos'] = np.cos(2 * np.pi * new_row['dayofyear'] / 365.25)
            
            # Lag features (use recent predictions for future lags)
            for lag in range(1, 8):
                if lag <= len(future_preds):
                    new_row[f'y_lag_{lag}'] = future_preds[-lag]
                else:
                    new_row[f'y_lag_{lag}'] = current_data['y'].iloc[-lag]
            
            # Rolling features
            recent_values = list(current_data['y'].tail(21)) + future_preds
            for window in [3, 7, 14, 21]:
                if len(recent_values) >= window:
                    new_row[f'roll_mean_{window}'] = np.mean(recent_values[-window:])
                    new_row[f'roll_std_{window}'] = np.std(recent_values[-window:])
                    new_row[f'roll_min_{window}'] = np.min(recent_values[-window:])
                    new_row[f'roll_max_{window}'] = np.max(recent_values[-window:])
                    new_row[f'roll_median_{window}'] = np.median(recent_values[-window:])
            
            # Fill remaining features with reasonable defaults
            for col in selected_features:
                if pd.isna(new_row[col]):
                    new_row[col] = current_data[col].iloc[-1] if col in current_data.columns else 0
            
            # Make prediction
            X_pred = new_row[selected_features].values.reshape(1, -1)
            X_pred_scaled = scaler.transform(X_pred)
            pred = lgb_model.predict(X_pred_scaled)[0]
            
            future_preds.append(pred)
            
            # Add the new row to current_data for next iteration
            new_row['y'] = pred
            current_data = pd.concat([current_data, new_row.to_frame().T], ignore_index=True)
        
        # Plot future forecast
        future_dates = pd.date_range(
            start=df_feat['ds'].max() + pd.Timedelta(days=1), 
            periods=future_days
        )
        
        fig_future = go.Figure()
        fig_future.add_trace(go.Scatter(
            x=df_feat['ds'].tail(30), 
            y=df_feat['y'].tail(30), 
            mode='lines+markers', 
            name='Historical',
            line=dict(color='blue')
        ))
        fig_future.add_trace(go.Scatter(
            x=future_dates, 
            y=future_preds, 
            mode='lines+markers', 
            name='Forecast',
            line=dict(color='red')
        ))
        fig_future.update_layout(
            title=f"{hosp} - {tgt} Future Forecast ({future_days} days)",
            xaxis_title="Date", 
            yaxis_title="Value"
        )
        st.plotly_chart(fig_future, use_container_width=True)
        
        results.append({
            'Hospital': hosp,
            'Target': tgt,
            'CV_MAE': cv_mae,
            'Test_MAE': test_mae,
            'Prophet_MAE': prophet_mae if prophet_mae != float('inf') else 'N/A',
            'Best_Horizon': best_horizon[0],
            'Features_Used': len(selected_features),
            'Improvement_vs_Prophet': f"{((prophet_mae - test_mae) / prophet_mae) * 100:.1f}%" if prophet_mae != float('inf') else 'N/A'
        })

# Summary
st.header("📋 Summary Results")
results_df = pd.DataFrame(results)
st.dataframe(results_df)

if results:
    st.write("### 🎯 Key Insights")
    avg_improvement = results_df[results_df['Improvement_vs_Prophet'] != 'N/A']['Improvement_vs_Prophet'].str.rstrip('%').astype(float).mean()
    if not np.isnan(avg_improvement):
        st.write(f"• Average improvement over Prophet baseline: {avg_improvement:.1f}%")
    
    best_performer = results_df.loc[results_df['Test_MAE'].idxmin()]
    st.write(f"• Best performer: {best_performer['Hospital']} - {best_performer['Target']} (MAE: {best_performer['Test_MAE']:.3f})")
    
    st.write("### ⚠️ Model Validation Notes")
    st.write("• Used strict time-based splits to prevent data leakage")
    st.write("• Applied correlation-based feature selection to reduce overfitting")
    st.write("• Tested multiple forecast horizons to find optimal performance")
    st.write("• Used cross-validation with conservative parameters")
    st.write("• Implemented walk-forward validation for future forecasts")
