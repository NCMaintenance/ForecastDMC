import streamlit as st
import pandas as pd
import numpy as np
import optuna
from prophet import Prophet
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import json
from datetime import datetime
import plotly.graph_objs as go
import plotly.express as px

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")
st.title("🔬 Optuna Hyperparameter Optimization for Hospital Forecasting")

# Sidebar configuration
uploaded = st.sidebar.file_uploader("Upload Excel file", type="xlsx")
if not uploaded:
    st.sidebar.info("Please upload your Excel file.")
    st.stop()

# Load data
df = pd.read_excel(uploaded)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

hospitals = sorted(df['Hospital'].unique())
targets = [
    "Tracker8am", "Tracker2pm", "Tracker8pm",
    "AdditionalCapacityOpen Morning",
    "TimeTotal_8am", "TimeTotal_2pm", "TimeTotal_8pm"
]

sel_hosp = st.sidebar.selectbox("Hospital", hospitals)
sel_target = st.sidebar.selectbox("Target", targets)

# Optimization settings
n_trials = st.sidebar.slider("Number of trials", 50, 500, 200)
forecast_horizon = 7  # Fixed at 7 days
validation_days = st.sidebar.slider("Validation window (days)", 14, 60, 30)

# Start optimization
run_optimization = st.sidebar.button("🚀 Start Optimization")

if not run_optimization:
    st.sidebar.info("Configure settings and click Start Optimization")
    st.stop()

# Filter data
df_work = df[df['Hospital'] == sel_hosp].copy()
if df_work[sel_target].isna().sum() > len(df_work) * 0.1:
    st.error(f"Too many missing values in {sel_target}")
    st.stop()

df_work = df_work.dropna(subset=[sel_target]).reset_index(drop=True)

# Global variables for optimization
optimization_results = []
best_params = {}
best_score = float('inf')

def create_features(data, target_col, config):
    """Create features based on configuration"""
    df_feat = data.copy()
    
    # Basic time features
    df_feat['dow'] = df_feat['Date'].dt.dayofweek
    df_feat['month'] = df_feat['Date'].dt.month
    df_feat['quarter'] = df_feat['Date'].dt.quarter
    df_feat['dayofyear'] = df_feat['Date'].dt.dayofyear
    df_feat['is_weekend'] = (df_feat['dow'] >= 5).astype(int)
    
    # Cyclical encoding
    if config.get('use_cyclical', True):
        df_feat['dow_sin'] = np.sin(2 * np.pi * df_feat['dow'] / 7)
        df_feat['dow_cos'] = np.cos(2 * np.pi * df_feat['dow'] / 7)
        df_feat['month_sin'] = np.sin(2 * np.pi * df_feat['month'] / 12)
        df_feat['month_cos'] = np.cos(2 * np.pi * df_feat['month'] / 12)
    
    # Lag features
    max_lag = config.get('max_lag', 7)
    for lag in range(1, max_lag + 1):
        df_feat[f'{target_col}_lag{lag}'] = df_feat[target_col].shift(lag)
    
    # Difference features
    if config.get('use_differences', True):
        df_feat[f'{target_col}_diff1'] = df_feat[target_col].diff(1)
        df_feat[f'{target_col}_diff7'] = df_feat[target_col].diff(7)
    
    # Rolling statistics
    windows = config.get('rolling_windows', [3, 7, 14])
    for window in windows:
        if len(df_feat) >= window:
            df_feat[f'{target_col}_roll_mean{window}'] = df_feat[target_col].rolling(window).mean()
            df_feat[f'{target_col}_roll_std{window}'] = df_feat[target_col].rolling(window).std()
            df_feat[f'{target_col}_roll_min{window}'] = df_feat[target_col].rolling(window).min()
            df_feat[f'{target_col}_roll_max{window}'] = df_feat[target_col].rolling(window).max()
            
            if config.get('use_advanced_rolling', False):
                df_feat[f'{target_col}_roll_median{window}'] = df_feat[target_col].rolling(window).median()
                df_feat[f'{target_col}_roll_skew{window}'] = df_feat[target_col].rolling(window).skew()
    
    # Exponential moving averages
    if config.get('use_ema', False):
        for alpha in [0.1, 0.3, 0.5]:
            df_feat[f'{target_col}_ema_{alpha}'] = df_feat[target_col].ewm(alpha=alpha).mean()
    
    # Fourier features for seasonality
    if config.get('use_fourier', False):
        for k in range(1, config.get('fourier_order', 3) + 1):
            df_feat[f'sin_{k}_week'] = np.sin(2 * np.pi * k * df_feat['dow'] / 7)
            df_feat[f'cos_{k}_week'] = np.cos(2 * np.pi * k * df_feat['dow'] / 7)
    
    return df_feat

def train_prophet_model(data, target_col, config):
    """Train Prophet model with configuration"""
    df_prophet = data[['Date', target_col]].rename(columns={'Date': 'ds', target_col: 'y'})
    
    model = Prophet(
        yearly_seasonality=config.get('yearly_seasonality', True),
        weekly_seasonality=config.get('weekly_seasonality', True),
        daily_seasonality=config.get('daily_seasonality', False),
        seasonality_mode=config.get('seasonality_mode', 'additive'),
        changepoint_prior_scale=config.get('changepoint_prior_scale', 0.05),
        seasonality_prior_scale=config.get('seasonality_prior_scale', 10.0),
        n_changepoints=config.get('n_changepoints', 25)
    )
    
    if config.get('add_country_holidays', True):
        model.add_country_holidays(country_name='IE')
    
    if config.get('add_monthly_seasonality', True):
        model.add_seasonality(name='monthly', period=30.5, fourier_order=config.get('monthly_fourier', 5))
    
    model.fit(df_prophet)
    return model

def iterative_forecast_optimized(model, data, prophet_model, target_col, days, config, feature_cols):
    """Optimized iterative forecasting"""
    forecast_data = data.copy()
    predictions = []
    
    model_features = feature_cols if hasattr(model, 'predict') else None
    
    for day in range(days):
        try:
            last_date = forecast_data['Date'].iloc[-1]
            next_date = last_date + pd.Timedelta(days=1)
            
            # Prophet prediction
            try:
                if prophet_model and config.get('use_prophet_features', True):
                    prophet_pred = prophet_model.predict(pd.DataFrame({'ds': [next_date]}))['yhat'].values[0]
                else:
                    prophet_pred = 0
            except:
                prophet_pred = 0
            
            # Create feature row
            new_row = {'Date': next_date}
            
            # Time features
            new_row['dow'] = next_date.dayofweek
            new_row['month'] = next_date.month
            new_row['quarter'] = next_date.quarter
            new_row['dayofyear'] = next_date.dayofyear
            new_row['is_weekend'] = 1 if next_date.dayofweek >= 5 else 0
            
            # Cyclical features
            if config.get('use_cyclical', True):
                new_row['dow_sin'] = np.sin(2 * np.pi * next_date.dayofweek / 7)
                new_row['dow_cos'] = np.cos(2 * np.pi * next_date.dayofweek / 7)
                new_row['month_sin'] = np.sin(2 * np.pi * next_date.month / 12)
                new_row['month_cos'] = np.cos(2 * np.pi * next_date.month / 12)
            
            # Lag features
            max_lag = config.get('max_lag', 7)
            for lag in range(1, max_lag + 1):
                if len(forecast_data) >= lag:
                    new_row[f'{target_col}_lag{lag}'] = forecast_data[target_col].iloc[-lag]
                else:
                    new_row[f'{target_col}_lag{lag}'] = 0
            
            # Difference features
            if config.get('use_differences', True):
                if len(forecast_data) >= 2:
                    new_row[f'{target_col}_diff1'] = forecast_data[target_col].iloc[-1] - forecast_data[target_col].iloc[-2]
                else:
                    new_row[f'{target_col}_diff1'] = 0
                    
                if len(forecast_data) >= 7:
                    new_row[f'{target_col}_diff7'] = forecast_data[target_col].iloc[-1] - forecast_data[target_col].iloc[-7]
                else:
                    new_row[f'{target_col}_diff7'] = 0
            
            # Rolling features
            windows = config.get('rolling_windows', [3, 7, 14])
            for window in windows:
                if len(forecast_data) >= window:
                    recent = forecast_data[target_col].iloc[-window:]
                    new_row[f'{target_col}_roll_mean{window}'] = recent.mean()
                    new_row[f'{target_col}_roll_std{window}'] = recent.std() if recent.std() > 0 else 0
                    new_row[f'{target_col}_roll_min{window}'] = recent.min()
                    new_row[f'{target_col}_roll_max{window}'] = recent.max()
                    
                    if config.get('use_advanced_rolling', False):
                        new_row[f'{target_col}_roll_median{window}'] = recent.median()
                        new_row[f'{target_col}_roll_skew{window}'] = recent.skew() if abs(recent.skew()) < 10 else 0
            
            # EMA features
            if config.get('use_ema', False):
                for alpha in [0.1, 0.3, 0.5]:
                    ema_val = forecast_data[target_col].ewm(alpha=alpha).mean().iloc[-1]
                    new_row[f'{target_col}_ema_{alpha}'] = ema_val
            
            # Fourier features
            if config.get('use_fourier', False):
                for k in range(1, config.get('fourier_order', 3) + 1):
                    new_row[f'sin_{k}_week'] = np.sin(2 * np.pi * k * next_date.dayofweek / 7)
                    new_row[f'cos_{k}_week'] = np.cos(2 * np.pi * k * next_date.dayofweek / 7)
            
            # Prophet features
            if config.get('use_prophet_features', True):
                new_row['prophet_yhat'] = prophet_pred
            
            # Ensure all required features are present
            for feat in model_features:
                if feat not in new_row:
                    new_row[feat] = 0
            
            # Make prediction
            if model_features:
                X_pred = np.array([[new_row[feat] for feat in model_features]], dtype=np.float64)
                pred = model.predict(X_pred)[0]
            else:
                pred = prophet_pred
            
            # Add to forecast data
            new_row[target_col] = pred
            forecast_data = pd.concat([forecast_data, pd.DataFrame([new_row])], ignore_index=True)
            predictions.append(pred)
            
        except Exception as e:
            # Fallback prediction
            fallback = predictions[-1] if predictions else forecast_data[target_col].iloc[-1]
            predictions.append(fallback)
    
    return predictions

def objective(trial):
    """Optuna objective function"""
    global best_score, best_params
    
    # Model selection
    model_type = trial.suggest_categorical('model_type', ['lightgbm', 'prophet', 'hybrid', 'rf', 'ridge'])
    
    # Feature engineering configuration
    config = {
        'max_lag': trial.suggest_int('max_lag', 3, 14),
        'use_differences': trial.suggest_categorical('use_differences', [True, False]),
        'use_cyclical': trial.suggest_categorical('use_cyclical', [True, False]),
        'rolling_windows': trial.suggest_categorical('rolling_windows', [[3, 7], [7, 14], [3, 7, 14], [3, 7, 14, 21]]),
        'use_advanced_rolling': trial.suggest_categorical('use_advanced_rolling', [True, False]),
        'use_ema': trial.suggest_categorical('use_ema', [True, False]),
        'use_fourier': trial.suggest_categorical('use_fourier', [True, False]),
        'fourier_order': trial.suggest_int('fourier_order', 2, 5) if trial.suggest_categorical('use_fourier_temp', [True, False]) else 3,
        'use_prophet_features': trial.suggest_categorical('use_prophet_features', [True, False]) if model_type in ['lightgbm', 'hybrid', 'rf', 'ridge'] else False
    }
    
    # Prophet-specific parameters
    if model_type in ['prophet', 'hybrid'] or config['use_prophet_features']:
        config.update({
            'yearly_seasonality': trial.suggest_categorical('yearly_seasonality', [True, False]),
            'weekly_seasonality': trial.suggest_categorical('weekly_seasonality', [True, False]),
            'seasonality_mode': trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative']),
            'changepoint_prior_scale': trial.suggest_float('changepoint_prior_scale', 0.001, 0.5, log=True),
            'seasonality_prior_scale': trial.suggest_float('seasonality_prior_scale', 0.01, 10, log=True),
            'n_changepoints': trial.suggest_int('n_changepoints', 10, 50),
            'add_monthly_seasonality': trial.suggest_categorical('add_monthly_seasonality', [True, False]),
            'monthly_fourier': trial.suggest_int('monthly_fourier', 3, 10)
        })
    
    # Model-specific parameters
    if model_type == 'lightgbm':
        config.update({
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100)
        })
    elif model_type == 'rf':
        config.update({
            'n_estimators': trial.suggest_int('rf_n_estimators', 50, 500),
            'max_depth': trial.suggest_int('rf_max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 10)
        })
    elif model_type == 'ridge':
        config.update({
            'alpha': trial.suggest_float('ridge_alpha', 0.1, 100, log=True)
        })
    
    try:
        # Create features
        df_feat = create_features(df_work, sel_target, config)
        df_feat = df_feat.dropna().reset_index(drop=True)
        
        if len(df_feat) < 50:
            return float('inf')
        
        # Time series split for validation
        split_point = len(df_feat) - validation_days - forecast_horizon
        if split_point < 30:
            return float('inf')
        
        train_data = df_feat.iloc[:split_point].copy()
        val_data = df_feat.iloc[split_point:split_point + validation_days].copy()
        
        # Prepare features
        feature_cols = [col for col in df_feat.columns if col not in ['Date', sel_target]]
        X_train = train_data[feature_cols].astype(np.float64)
        y_train = train_data[sel_target].astype(np.float64)
        
        # Train Prophet model if needed
        prophet_model = None
        if model_type in ['prophet', 'hybrid'] or config.get('use_prophet_features', False):
            prophet_model = train_prophet_model(train_data, sel_target, config)
        
        # Train main model
        if model_type == 'lightgbm':
            model = LGBMRegressor(
                n_estimators=config['n_estimators'],
                learning_rate=config['learning_rate'],
                num_leaves=config['num_leaves'],
                feature_fraction=config['feature_fraction'],
                bagging_fraction=config['bagging_fraction'],
                min_child_samples=config['min_child_samples'],
                random_state=42,
                verbose=-1
            )
            model.fit(X_train, y_train)
        elif model_type == 'rf':
            model = RandomForestRegressor(
                n_estimators=config['rf_n_estimators'],
                max_depth=config['rf_max_depth'],
                min_samples_split=config['rf_min_samples_split'],
                min_samples_leaf=config['rf_min_samples_leaf'],
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
        elif model_type == 'ridge':
            model = Ridge(alpha=config['ridge_alpha'])
            model.fit(X_train, y_train)
        elif model_type == 'prophet':
            model = prophet_model
        elif model_type == 'hybrid':
            # Train LightGBM for hybrid
            model = LGBMRegressor(
                n_estimators=config.get('n_estimators', 200),
                learning_rate=config.get('learning_rate', 0.1),
                num_leaves=config.get('num_leaves', 31),
                random_state=42,
                verbose=-1
            )
            model.fit(X_train, y_train)
        
        # Validate with iterative forecasting
        mae_scores = []
        
        # Multiple validation windows
        for i in range(min(5, validation_days - forecast_horizon)):
            val_start = split_point + i
            val_end = val_start + forecast_horizon
            
            if val_end > len(df_feat):
                break
            
            # Get validation period
            val_period = df_feat.iloc[val_start:val_end]
            
            # Forecast from point before validation
            forecast_base = df_feat.iloc[:val_start]
            
            if model_type == 'prophet':
                # Pure Prophet forecast
                future_df = prophet_model.make_future_dataframe(periods=forecast_horizon, include_history=False)
                forecast_result = prophet_model.predict(future_df)
                predictions = forecast_result['yhat'].values
            else:
                # Iterative forecast
                predictions = iterative_forecast_optimized(
                    model, forecast_base, prophet_model, sel_target, 
                    forecast_horizon, config, feature_cols
                )
            
            # Calculate MAE
            actual = val_period[sel_target].values
            mae = mean_absolute_error(actual, predictions[:len(actual)])
            mae_scores.append(mae)
        
        final_score = np.mean(mae_scores) if mae_scores else float('inf')
        
        # Update best parameters
        if final_score < best_score:
            best_score = final_score
            best_params = config.copy()
            best_params['model_type'] = model_type
            best_params['score'] = final_score
        
        # Store result
        result = {
            'trial': trial.number,
            'model_type': model_type,
            'score': final_score,
            'config': config.copy()
        }
        optimization_results.append(result)
        
        return final_score
        
    except Exception as e:
        return float('inf')

# Run optimization
st.info(f"Starting optimization with {n_trials} trials...")
progress_bar = st.progress(0)
results_placeholder = st.empty()

study = optuna.create_study(direction='minimize')

def callback(study, trial):
    progress_bar.progress(trial.number / n_trials)
    
    if trial.number % 10 == 0:
        # Show intermediate results
        with results_placeholder.container():
            st.write(f"Trial {trial.number}/{n_trials}")
            st.write(f"Best score so far: {best_score:.4f}")
            if best_params:
                st.write("Best parameters:")
                st.json(best_params)

study.optimize(objective, n_trials=n_trials, callbacks=[callback])

# Final results
st.success("Optimization completed!")

st.header("🏆 Best Results")
st.write(f"**Best Score (MAE): {best_score:.4f}**")
st.write("**Best Configuration:**")
st.json(best_params)

# Results analysis
st.header("📊 Results Analysis")

# Convert results to DataFrame
results_df = pd.DataFrame([
    {
        'Trial': r['trial'],
        'Model': r['model_type'],
        'Score': r['score'],
        'Max_Lag': r['config'].get('max_lag', 0),
        'Use_Differences': r['config'].get('use_differences', False),
        'Use_Cyclical': r['config'].get('use_cyclical', False),
        'Use_Prophet': r['config'].get('use_prophet_features', False),
        'Rolling_Windows': str(r['config'].get('rolling_windows', [])),
        'Use_EMA': r['config'].get('use_ema', False),
        'Use_Fourier': r['config'].get('use_fourier', False)
    }
    for r in optimization_results
    if r['score'] != float('inf')
])

if len(results_df) > 0:
    # Top 10 results
    st.subheader("🔝 Top 10 Results")
    top_results = results_df.nsmallest(10, 'Score')
    st.dataframe(top_results)
    
    # Model type performance
    st.subheader("📈 Model Type Performance")
    model_performance = results_df.groupby('Model')['Score'].agg(['mean', 'std', 'count']).round(4)
    st.dataframe(model_performance)
    
    # Feature importance analysis
    st.subheader("🔍 Feature Analysis")
    
    # Box plots for different configurations
    fig = px.box(results_df, x='Model', y='Score', title='Score Distribution by Model Type')
    st.plotly_chart(fig, use_container_width=True)
    
    # Parameter correlation
    param_cols = ['Max_Lag', 'Use_Differences', 'Use_Cyclical', 'Use_Prophet', 'Use_EMA', 'Use_Fourier']
    param_data = results_df[param_cols + ['Score']].copy()
    
    # Convert boolean to numeric
    for col in param_cols:
        if col != 'Max_Lag':
            param_data[col] = param_data[col].astype(int)
    
    correlation_matrix = param_data.corr()
    
    fig_corr = px.imshow(correlation_matrix, text_auto=True, title='Parameter Correlation Matrix')
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Export results
    st.subheader("💾 Export Results")
    
    # Create detailed export
    export_data = {
        'best_params': best_params,
        'best_score': best_score,
        'top_10_results': top_results.to_dict('records'),
        'model_performance': model_performance.to_dict(),
        'optimization_summary': {
            'total_trials': n_trials,
            'successful_trials': len(results_df),
            'target': sel_target,
            'hospital': sel_hosp,
            'validation_days': validation_days,
            'forecast_horizon': forecast_horizon,
            'timestamp': datetime.now().isoformat()
        }
    }
    
    st.download_button(
        label="📥 Download Results (JSON)",
        data=json.dumps(export_data, indent=2, default=str),
        file_name=f"optimization_results_{sel_hosp}_{sel_target}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )
    
    # Summary recommendations
    st.subheader("💡 Recommendations")
    
    best_model = results_df.loc[results_df['Score'].idxmin(), 'Model']
    st.write(f"- **Best Model Type**: {best_model}")
    st.write(f"- **Optimal Max Lag**: {best_params.get('max_lag', 'N/A')} days")
    st.write(f"- **Use Differences**: {'Yes' if best_params.get('use_differences') else 'No'}")
    st.write(f"- **Use Cyclical Features**: {'Yes' if best_params.get('use_cyclical') else 'No'}")
    st.write(f"- **Use Prophet Features**: {'Yes' if best_params.get('use_prophet_features') else 'No'}")
    st.write(f"- **Rolling Windows**: {best_params.get('rolling_windows', 'N/A')}")
    
    if best_model == 'lightgbm':
        st.write("**LightGBM Parameters:**")
        st.write(f"  - Learning Rate: {best_params.get('learning_rate', 'N/A')}")
        st.write(f"  - N Estimators: {best_params.get('n_estimators', 'N/A')}")
        st.write(f"  - Num Leaves: {best_params.get('num_leaves', 'N/A')}")

else:
    st.error("No successful trials completed. Please check your data and try again.")

st.info("Use these results to configure your production forecasting model!")
