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
    try:
        df = pd.read_excel(uploaded_file)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])  # Remove rows with invalid dates
        df = df.sort_values('Date')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data(uploaded)
if df is None:
    st.stop()

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
    Create comprehensive feature set with robust error handling.
    """
    try:
        df_feat = df_input.copy()
        
        # Ensure we have the date column
        if 'ds' in df_feat.columns:
            df_feat = df_feat.rename(columns={'ds': 'Date'})
        
        # Validate date column
        if 'Date' not in df_feat.columns:
            raise ValueError("Date column not found")
            
        # Combine predictor columns and the target column for feature generation
        all_feature_cols = []
        
        # Only include columns that actually exist and have data
        for col in all_hospital_cols + [target_col]:
            if col in df_feat.columns and not df_feat[col].isna().all():
                all_feature_cols.append(col)
        
        # Create lag and rolling features
        for col in all_feature_cols:
            if col in df_feat.columns and pd.api.types.is_numeric_dtype(df_feat[col]):
                col_name = 'target' if col == target_col else col
                
                # Lag features (with error handling)
                try:
                    for lag in range(1, min(max_lags + 1, len(df_feat))):
                        df_feat[f'{col_name}_lag_{lag}'] = df_feat[col].shift(lag)
                except Exception as e:
                    st.warning(f"Error creating lag features for {col}: {e}")
                    continue
                
                # Rolling statistics (with minimum periods check)
                try:
                    windows = [w for w in [3, 7, 14] if w < len(df_feat)]
                    for window in windows:
                        df_feat[f'{col_name}_roll_mean_{window}'] = df_feat[col].shift(1).rolling(
                            window=window, min_periods=1
                        ).mean()
                        df_feat[f'{col_name}_roll_std_{window}'] = df_feat[col].shift(1).rolling(
                            window=window, min_periods=1
                        ).std()
                except Exception as e:
                    st.warning(f"Error creating rolling features for {col}: {e}")
                    continue

                # Exponentially Weighted Mean
                try:
                    df_feat[f'{col_name}_ewm_mean_0.3'] = df_feat[col].shift(1).ewm(alpha=0.3).mean()
                except Exception as e:
                    st.warning(f"Error creating EWM features for {col}: {e}")
                    continue

        # Date-based features (always created)
        try:
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
        except Exception as e:
            st.warning(f"Error creating date features: {e}")

        return df_feat.rename(columns={'Date': 'ds'})
    
    except Exception as e:
        st.error(f"Critical error in feature engineering: {e}")
        return df_input

# --- Feature Selection ---
def select_features_by_correlation(X, y, threshold=0.1):
    """Select features based on Pearson correlation with the target."""
    correlations = {}
    
    if len(y) == 0:
        return [], {}
    
    for col in X.columns:
        try:
            if not X[col].isna().all() and pd.api.types.is_numeric_dtype(X[col]):
                # Handle infinite values
                col_clean = X[col].replace([np.inf, -np.inf], np.nan).fillna(0)
                y_clean = pd.Series(y).replace([np.inf, -np.inf], np.nan).fillna(0)
                
                if len(col_clean.unique()) > 1 and len(y_clean.unique()) > 1:
                    corr, p_value = pearsonr(col_clean, y_clean)
                    if not np.isnan(corr) and abs(corr) >= threshold:
                        correlations[col] = abs(corr)
        except Exception as e:
            st.warning(f"Error calculating correlation for {col}: {e}")
            continue
    
    selected_features = list(correlations.keys())
    return selected_features, correlations

# --- Prophet Forecasting ---
def create_prophet_forecasts(prophet_df, train_size, forecast_days=14):
    """Create Prophet forecasts with confidence intervals."""
    try:
        # Validate input data
        if len(prophet_df) < 10:
            raise ValueError("Insufficient data for Prophet")
        
        m = Prophet(
            yearly_seasonality=True, 
            weekly_seasonality=True, 
            daily_seasonality=False,
            interval_width=0.95,
            seasonality_mode='multiplicative'
        )
        
        # Add holidays with error handling
        try:
            m.add_country_holidays(country_name='IE')
        except:
            pass  # Continue without holidays if not available
        
        train_prophet = prophet_df.iloc[:train_size].copy()
        
        # Ensure no infinite values
        train_prophet['y'] = train_prophet['y'].replace([np.inf, -np.inf], np.nan)
        train_prophet = train_prophet.dropna()
        
        if len(train_prophet) < 10:
            raise ValueError("Insufficient training data after cleaning")
        
        m.fit(train_prophet)
        
        future = m.make_future_dataframe(periods=len(prophet_df) - train_size + forecast_days)
        forecast = m.predict(future)
        
        return m, forecast
    
    except Exception as e:
        st.warning(f"Prophet forecasting failed: {e}")
        return None, None

# --- Main Application Logic ---
all_results = []
h_list = hospitals if sel_hosp == "All" else [sel_hosp]
t_list = targets if sel_target == "All" else [sel_target]

for hosp in h_list:
    st.header(f"🏥 Hospital: {hosp}", divider='rainbow')
    df_h = df[df['Hospital'] == hosp].copy().reset_index(drop=True)
    
    if len(df_h) == 0:
        st.warning(f"⚠️ No data found for hospital {hosp}")
        continue
    
    for tgt in t_list:
        st.subheader(f"🎯 Target: {tgt}")
        
        try:
            # Validate target column
            if tgt not in df_h.columns:
                st.warning(f"⚠️ Target column '{tgt}' not found in data. Skipping.")
                continue
                
            if df_h[tgt].isna().all():
                st.warning(f"⚠️ Target '{tgt}' has no valid data. Skipping.")
                continue

            # Get all other available metrics to use as features
            available_targets = [col for col in targets 
                               if col in df_h.columns and not df_h[col].isna().all() and col != tgt]
            
            df2 = df_h[['Date'] + [tgt] + available_targets].copy()
            df2 = df2.rename(columns={'Date': 'ds', tgt: 'y'})
            
            # Remove rows with missing target values
            df2 = df2.dropna(subset=['y'])
            
            # Handle infinite values in target
            df2['y'] = df2['y'].replace([np.inf, -np.inf], np.nan)
            df2 = df2.dropna(subset=['y'])
            
            if len(df2) < 30:
                st.warning(f"⚠️ Insufficient data for target '{tgt}' (only {len(df2)} valid rows). Skipping.")
                continue

            # --- Feature Engineering & Data Splitting ---
            df_feat = create_advanced_features(df2, 'y', available_targets, max_lags=min(7, len(df2)//4))
            
            # Final cleanup
            df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
            initial_len = len(df_feat)
            df_feat = df_feat.dropna(subset=['y'])
            
            if len(df_feat) < initial_len * 0.7:
                st.warning(f"⚠️ Too much data lost during feature engineering for '{tgt}'. Skipping.")
                continue

            n = len(df_feat)
            if n < 60:
                st.warning(f"⚠️ Insufficient data for '{tgt}' after processing ({n} rows). Need at least 60. Skipping.")
                continue
                
            train_size = max(30, int(0.8 * n))  # Ensure minimum training size
            train_df = df_feat.iloc[:train_size].copy()
            test_df = df_feat.iloc[train_size:].copy()

            # --- Feature Selection ---
            feature_cols = [col for col in df_feat.columns 
                          if col not in ['ds', 'y'] and pd.api.types.is_numeric_dtype(df_feat[col])]
            
            if not feature_cols:
                st.warning(f"⚠️ No valid features available for '{tgt}'. Skipping.")
                continue
            
            selected_features, feature_corrs = select_features_by_correlation(
                train_df[feature_cols].fillna(0), train_df['y'], threshold=correlation_threshold
            )
            
            if not selected_features:
                st.warning(f"⚠️ No features met the correlation threshold for '{tgt}'. Trying with lower threshold.")
                selected_features, feature_corrs = select_features_by_correlation(
                    train_df[feature_cols].fillna(0), train_df['y'], threshold=0.01
                )
                
                if not selected_features:
                    st.warning(f"⚠️ Still no valid features found for '{tgt}'. Skipping.")
                    continue
            
            with st.expander(f"📊 Selected {len(selected_features)} features. Click to see details."):
                top_features = sorted(feature_corrs.items(), key=lambda x: x[1], reverse=True)[:15]
                st.write("🔝 Top Features by Correlation:")
                st.dataframe(pd.DataFrame(top_features, columns=['Feature', 'Correlation']))

            # Prepare training data
            X_train = train_df[selected_features].fillna(0)
            y_train = train_df['y']
            X_test = test_df[selected_features].fillna(0)
            y_test = test_df['y']
            
            # Additional validation for LightGBM
            if len(X_train) == 0 or len(y_train) == 0:
                st.warning(f"⚠️ Empty training data for '{tgt}'. Skipping.")
                continue
                
            if y_train.std() == 0:
                st.warning(f"⚠️ Target has no variation for '{tgt}'. Skipping.")
                continue
            
            # --- Model Training (LightGBM) ---
            try:
                lgb_model = LGBMRegressor(
                    random_state=42,
                    n_estimators=100,  # Reduce for faster training
                    learning_rate=0.1,
                    num_leaves=31,
                    feature_fraction=0.8,
                    verbosity=-1  # Suppress warnings
                )
                
                lgb_model.fit(X_train, y_train)
                
                # Make predictions
                test_pred_lgb = lgb_model.predict(X_test)
                
                # Handle potential prediction issues
                test_pred_lgb = np.nan_to_num(test_pred_lgb, nan=y_train.mean())
                
                test_mae_lgb = mean_absolute_error(y_test, test_pred_lgb)
                test_rmse_lgb = np.sqrt(mean_squared_error(y_test, test_pred_lgb))
                
            except Exception as e:
                st.error(f"⚠️ LightGBM training failed for '{tgt}': {e}")
                continue
            
            # --- Prophet Forecasting ---
            prophet_model, prophet_forecast = create_prophet_forecasts(
                df2[['ds', 'y']], train_size, forecast_days=future_days
            )
            
            if prophet_model is not None and prophet_forecast is not None:
                try:
                    prophet_test_pred = prophet_forecast['yhat'][train_size:n].values
                    if len(prophet_test_pred) == len(y_test):
                        test_mae_prophet = mean_absolute_error(y_test, prophet_test_pred)
                        prophet_future_forecast = prophet_forecast.iloc[-future_days:]
                    else:
                        test_mae_prophet = float('inf')
                        prophet_future_forecast = None
                except Exception as e:
                    st.warning(f"Prophet prediction extraction failed: {e}")
                    test_mae_prophet = float('inf')
                    prophet_future_forecast = None
            else:
                test_mae_prophet = float('inf')
                prophet_future_forecast = None

            # --- Results Visualization (Test Period) ---
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=test_df['ds'], y=y_test, mode='lines', name='Actual', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=test_df['ds'], y=test_pred_lgb, mode='lines', name='LightGBM Forecast', line=dict(color='red')))
            
            if test_mae_prophet != float('inf'):
                fig.add_trace(go.Scatter(x=test_df['ds'], y=prophet_test_pred, mode='lines', name='Prophet Forecast', line=dict(color='green', dash='dash')))
                
            fig.update_layout(title=f"Model Performance on Test Data - {tgt}", xaxis_title="Date", yaxis_title="Value", legend_title="Legend")
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
                try:
                    importance_df = pd.DataFrame({
                        'feature': selected_features,
                        'importance': lgb_model.feature_importances_
                    }).sort_values('importance', ascending=False).head(20)
                    
                    fig_imp = go.Figure(go.Bar(x=importance_df['importance'], y=importance_df['feature'], orientation='h'))
                    fig_imp.update_layout(title="Top 20 Feature Importances", yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_imp, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not display feature importance: {e}")

            # --- Future Forecasting ---
            st.write(f"🔮 **Generating Future Forecasts for the next {future_days} days...**")
            
            try:
                # Train final model on all available data
                full_train_X = df_feat[selected_features].fillna(0)
                full_train_y = df_feat['y']
                
                lgb_model_final = LGBMRegressor(
                    random_state=42,
                    n_estimators=100,
                    learning_rate=0.1,
                    num_leaves=31,
                    feature_fraction=0.8,
                    verbosity=-1
                )
                lgb_model_final.fit(full_train_X, full_train_y)

                # Create future predictions with recursive forecasting
                last_known_date = df_feat['ds'].max()
                future_dates = pd.date_range(start=last_known_date + pd.Timedelta(days=1), periods=future_days)
                
                # Create extended dataframe for recursive forecasting
                extended_df = df_feat.copy()
                future_preds_lgb = []
                
                for i in range(future_days):
                    current_date = future_dates[i]
                    
                    # Create a row for the future date with date-based features
                    future_row = pd.DataFrame({'ds': [current_date]})
                    future_row['dow'] = future_row['ds'].dt.dayofweek
                    future_row['month'] = future_row['ds'].dt.month
                    future_row['quarter'] = future_row['ds'].dt.quarter
                    future_row['weekofyear'] = future_row['ds'].dt.isocalendar().week.astype(int)
                    future_row['dayofyear'] = future_row['ds'].dt.dayofyear
                    future_row['is_weekend'] = (future_row['dow'] >= 5).astype(int)
                    future_row['dow_sin'] = np.sin(2 * np.pi * future_row['dow'] / 7)
                    future_row['dow_cos'] = np.cos(2 * np.pi * future_row['dow'] / 7)
                    future_row['month_sin'] = np.sin(2 * np.pi * future_row['month'] / 12)
                    future_row['month_cos'] = np.cos(2 * np.pi * future_row['month'] / 12)
                    
                    # Calculate lag and rolling features based on extended history
                    for feature in selected_features:
                        if feature in future_row.columns:
                            continue  # Already calculated (date features)
                            
                        # Handle lag features
                        if '_lag_' in feature:
                            lag_num = int(feature.split('_lag_')[-1])
                            base_col = feature.replace(f'_lag_{lag_num}', '')
                            
                            if base_col == 'target':
                                # Use target values from extended_df
                                if len(extended_df) >= lag_num:
                                    future_row[feature] = extended_df['y'].iloc[-lag_num]
                                else:
                                    future_row[feature] = extended_df['y'].mean()
                            else:
                                # Use other column values
                                if base_col in extended_df.columns and len(extended_df) >= lag_num:
                                    future_row[feature] = extended_df[base_col].iloc[-lag_num]
                                else:
                                    future_row[feature] = 0
                        
                        # Handle rolling mean features
                        elif '_roll_mean_' in feature:
                            window = int(feature.split('_roll_mean_')[-1])
                            base_col = feature.replace(f'_roll_mean_{window}', '')
                            
                            if base_col == 'target':
                                if len(extended_df) >= window:
                                    future_row[feature] = extended_df['y'].tail(window).mean()
                                else:
                                    future_row[feature] = extended_df['y'].mean()
                            else:
                                if base_col in extended_df.columns and len(extended_df) >= window:
                                    future_row[feature] = extended_df[base_col].tail(window).mean()
                                else:
                                    future_row[feature] = 0
                        
                        # Handle rolling std features
                        elif '_roll_std_' in feature:
                            window = int(feature.split('_roll_std_')[-1])
                            base_col = feature.replace(f'_roll_std_{window}', '')
                            
                            if base_col == 'target':
                                if len(extended_df) >= window:
                                    future_row[feature] = extended_df['y'].tail(window).std()
                                else:
                                    future_row[feature] = extended_df['y'].std()
                            else:
                                if base_col in extended_df.columns and len(extended_df) >= window:
                                    future_row[feature] = extended_df[base_col].tail(window).std()
                                else:
                                    future_row[feature] = 0
                        
                        # Handle EWM features
                        elif '_ewm_mean_' in feature:
                            base_col = feature.replace('_ewm_mean_0.3', '')
                            
                            if base_col == 'target':
                                if len(extended_df) > 0:
                                    future_row[feature] = extended_df['y'].ewm(alpha=0.3).mean().iloc[-1]
                                else:
                                    future_row[feature] = extended_df['y'].mean()
                            else:
                                if base_col in extended_df.columns and len(extended_df) > 0:
                                    future_row[feature] = extended_df[base_col].ewm(alpha=0.3).mean().iloc[-1]
                                else:
                                    future_row[feature] = 0
                        
                        # For any other features, use the last known value or zero
                        else:
                            if feature in extended_df.columns:
                                future_row[feature] = extended_df[feature].iloc[-1]
                            else:
                                future_row[feature] = 0
                    
                    # Ensure all selected features are present
                    for feature in selected_features:
                        if feature not in future_row.columns:
                            future_row[feature] = 0
                    
                    # Make prediction
                    feature_values = future_row[selected_features].fillna(0)
                    pred = lgb_model_final.predict(feature_values)[0]
                    future_preds_lgb.append(pred)
                    
                    # Add this prediction to the extended dataframe for next iteration
                    new_row = future_row.copy()
                    new_row['y'] = pred
                    extended_df = pd.concat([extended_df, new_row], ignore_index=True)
                
                future_preds_lgb = np.array(future_preds_lgb)
                
                # --- Future Forecast Visualization & Table ---
                fig_future = go.Figure()
                recent_data = df_feat.tail(30)
                fig_future.add_trace(go.Scatter(x=recent_data['ds'], y=recent_data['y'], mode='lines', name='Historical Data', line=dict(color='grey')))
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
                
            except Exception as e:
                st.error(f"Future forecasting failed for '{tgt}': {e}")

            # --- Prophet Components Plot ---
            if prophet_model:
                with st.expander("🔍 Prophet Forecast Components"):
                    try:
                        fig_prophet_comp = prophet_model.plot_components(prophet_forecast)
                        st.pyplot(fig_prophet_comp)
                    except Exception as e:
                        st.warning(f"Could not display Prophet components: {e}")
            
        except Exception as e:
            st.error(f"Critical error processing target '{tgt}' for hospital '{hosp}': {e}")
            continue

# --- Consolidated Summary ---
if len(all_results) > 1 and (sel_hosp == "All" or sel_target == "All"):
    st.header("📋 Consolidated Future Forecast Summary", divider='rainbow')
    
    try:
        summary_list = []
        for res in all_results:
            days = [1, 3, 7]
            row = {'Hospital': res['Hospital'], 'Target': res['Target']}
            for day in days:
                if 'LightGBM Forecast' in res and day <= len(res['LightGBM Forecast']):
                    row[f'Day {day} Forecast (LGBM)'] = res['LightGBM Forecast'][day-1]
            summary_list.append(row)
        
        if summary_list:
            summary_df = pd.DataFrame(summary_list)
            st.dataframe(summary_df, use_container_width=True)
        else:
            st.warning("No successful forecasts to summarize.")
            
    except Exception as e:
        st.error(f"Error creating summary: {e}")

st.success("✅ Forecasting process completed!")
