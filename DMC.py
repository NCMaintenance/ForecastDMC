import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression, SelectFromModel
import plotly.graph_objs as go
import warnings
from scipy.stats import pearsonr
from datetime import datetime, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings("ignore")

# Current system information
CURRENT_UTC = "2025-06-15 10:09:12"
CURRENT_USER = "NCMaintenance"

st.set_page_config(layout="wide")
st.title("🏥 Advanced Hospital Forecasting with Auto-Optimization 📈")

# Helper Functions
def calculate_feature_importance_score(df, feature, target):
    """Calculate feature importance score based on multiple metrics"""
    try:
        correlation = abs(pearsonr(df[feature], df[target])[0])
        mi_score = mutual_info_regression(df[[feature]], df[target], random_state=42)[0]
        return (correlation + mi_score) / 2
    except:
        return 0

def optimize_lookback_period(df, target_col, min_lookback=7, max_lookback=90):
    """Find optimal lookback period"""
    best_mae = float('inf')
    best_lookback = min_lookback
    
    for lookback in range(min_lookback, max_lookback + 1, 7):
        try:
            df_feat = create_comprehensive_features(df, target_col, max_lags=lookback)
            train_size = int(len(df_feat) * 0.8)
            y_train = df_feat['y'][:train_size]
            y_test = df_feat['y'][train_size:]
            ma_pred = df_feat['y'].rolling(window=lookback).mean()[train_size:]
            mae = mean_absolute_error(y_test, ma_pred)
            
            if mae < best_mae:
                best_mae = mae
                best_lookback = lookback
        except Exception as e:
            continue
    
    return best_lookback

def safe_predict_with_fallback(model, X, fallback_value):
    """Make predictions with fallback on error"""
    try:
        return model.predict(X)
    except Exception as e:
        st.warning(f"Prediction failed: {e}. Using fallback.")
        return np.full(len(X), fallback_value)

def create_robust_ensemble(predictions_dict, weights_dict, y_true=None):
    """Create ensemble predictions with automatic weight adjustment"""
    valid_predictions = {}
    valid_weights = {}
    
    for model_name, pred in predictions_dict.items():
        if pred is not None and len(pred) == len(next(iter(predictions_dict.values()))):
            valid_predictions[model_name] = pred
            valid_weights[model_name] = weights_dict[model_name]
    
    weight_sum = sum(valid_weights.values())
    if weight_sum > 0:
        valid_weights = {k: v/weight_sum for k, v in valid_weights.items()}
    
    ensemble_pred = np.zeros(len(next(iter(valid_predictions.values()))))
    for model_name, pred in valid_predictions.items():
        ensemble_pred += valid_weights[model_name] * pred
    
    return ensemble_pred

def create_comprehensive_features(df_input, target_col, all_hospital_cols, max_lags=30):
    try:
        df_feat = df_input.copy()
        
        if 'ds' in df_feat.columns:
            df_feat = df_feat.rename(columns={'ds': 'Date'})
        
        if 'Date' not in df_feat.columns:
            raise ValueError("Date column not found")
        
        all_feature_cols = []
        for col in all_hospital_cols + [target_col]:
            if col in df_feat.columns and not df_feat[col].isna().all():
                all_feature_cols.append(col)
        
        for col in all_feature_cols:
            if col in df_feat.columns and pd.api.types.is_numeric_dtype(df_feat[col]):
                col_name = 'target' if col == target_col else col
                
                # Enhanced lag features
                for lag in [1, 2, 3, 7, 14, 21, 28, 35, 42]:
                    if lag < len(df_feat):
                        df_feat[f'{col_name}_lag_{lag}'] = df_feat[col].shift(lag)
                
                # Enhanced rolling statistics
                windows = [3, 7, 14, 21, 28, 42]
                for window in windows:
                    if window < len(df_feat):
                        df_feat[f'{col_name}_roll_mean_{window}'] = df_feat[col].shift(1).rolling(
                            window=window, min_periods=max(1, window//3)
                        ).mean()
                        df_feat[f'{col_name}_roll_std_{window}'] = df_feat[col].shift(1).rolling(
                            window=window, min_periods=max(1, window//3)
                        ).std()
                        df_feat[f'{col_name}_roll_min_{window}'] = df_feat[col].shift(1).rolling(
                            window=window, min_periods=max(1, window//3)
                        ).min()
                        df_feat[f'{col_name}_roll_max_{window}'] = df_feat[col].shift(1).rolling(
                            window=window, min_periods=max(1, window//3)
                        ).max()
                        df_feat[f'{col_name}_roll_median_{window}'] = df_feat[col].shift(1).rolling(
                            window=window, min_periods=max(1, window//3)
                        ).median()
                
                # Enhanced EWM features
                alphas = [0.1, 0.3, 0.5, 0.7]
                for alpha in alphas:
                    df_feat[f'{col_name}_ewm_mean_{alpha}'] = df_feat[col].shift(1).ewm(
                        alpha=alpha, min_periods=3
                    ).mean()
                
                # Trend features
                for window in [7, 14, 28]:
                    if window < len(df_feat):
                        df_feat[f'{col_name}_trend_{window}d'] = df_feat[col].rolling(
                            window=window+1, min_periods=2
                        ).apply(
                            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0,
                            raw=True
                        )
                
                # Momentum features
                for period in [7, 14, 28]:
                    if period < len(df_feat):
                        df_feat[f'{col_name}_momentum_{period}d'] = (
                            df_feat[col] - df_feat[col].shift(period)
                        ) / (df_feat[col].shift(period) + 1e-8)
                
                # Volatility features
                for window in [7, 14, 28]:
                    if window < len(df_feat):
                        rolling_std = df_feat[col].rolling(window=window, min_periods=2).std()
                        rolling_mean = df_feat[col].rolling(window=window, min_periods=2).mean()
                        df_feat[f'{col_name}_volatility_{window}d'] = rolling_std / (rolling_mean + 1e-8)
                
                # Try seasonal decomposition
                try:
                    if len(df_feat) >= 14:  # Minimum length for seasonal decomposition
                        decomposition = seasonal_decompose(
                            df_feat[col].fillna(method='ffill').fillna(method='bfill'),
                            period=7,
                            extrapolate_trend='freq'
                        )
                        df_feat[f'{col_name}_seasonal'] = decomposition.seasonal
                        df_feat[f'{col_name}_trend'] = decomposition.trend
                        df_feat[f'{col_name}_resid'] = decomposition.resid
                except Exception as e:
                    pass
        
        # Enhanced date features
        df_feat['dow'] = df_feat['Date'].dt.dayofweek
        df_feat['month'] = df_feat['Date'].dt.month
        df_feat['quarter'] = df_feat['Date'].dt.quarter
        df_feat['year'] = df_feat['Date'].dt.year
        df_feat['weekofyear'] = df_feat['Date'].dt.isocalendar().week.astype(int)
        df_feat['dayofyear'] = df_feat['Date'].dt.dayofyear
        
        # Cyclical encodings
        df_feat['dow_sin'] = np.sin(2 * np.pi * df_feat['dow'] / 7)
        df_feat['dow_cos'] = np.cos(2 * np.pi * df_feat['dow'] / 7)
        df_feat['month_sin'] = np.sin(2 * np.pi * df_feat['month'] / 12)
        df_feat['month_cos'] = np.cos(2 * np.pi * df_feat['month'] / 12)
        df_feat['dayofyear_sin'] = np.sin(2 * np.pi * df_feat['dayofyear'] / 365.25)
        df_feat['dayofyear_cos'] = np.cos(2 * np.pi * df_feat['dayofyear'] / 365.25)
        
        return df_feat.rename(columns={'Date': 'ds'})
    
    except Exception as e:
        st.error(f"Error in feature engineering: {e}")
        return df_input

class AutoForecaster:
    def __init__(self, df, target_col, max_iterations=5):
        self.df = df
        self.target_col = target_col
        self.max_iterations = max_iterations
        self.best_mae = float('inf')
        self.best_config = None
        self.improvement_threshold = 0.01
    
    def optimize(self):
        configs = self._generate_configs()
        
        for i, config in enumerate(configs, 1):
            current_mae = self._evaluate_config(config)
            
            if current_mae < self.best_mae * (1 - self.improvement_threshold):
                self.best_mae = current_mae
                self.best_config = config
                st.success(f"✨ Found better configuration! MAE improved to: {current_mae:.2f}")
            
            if i >= self.max_iterations:
                break
        
        return self.best_config
    
    def _generate_configs(self):
        base_lookback = optimize_lookback_period(self.df, self.target_col)
        
        configs = [
            {
                'lookback': base_lookback,
                'feature_correlation_threshold': 0.05,
                'max_features': 50,
                'ensemble_weights': {'lgb': 0.4, 'prophet': 0.3, 'arima': 0.3}
            },
            {
                'lookback': int(base_lookback * 1.5),
                'feature_correlation_threshold': 0.03,
                'max_features': 75,
                'ensemble_weights': {'lgb': 0.5, 'prophet': 0.25, 'arima': 0.25}
            },
            {
                'lookback': int(base_lookback * 2),
                'feature_correlation_threshold': 0.02,
                'max_features': 100,
                'ensemble_weights': {'lgb': 0.6, 'prophet': 0.2, 'arima': 0.2}
            },
            {
                'lookback': int(base_lookback * 0.75),
                'feature_correlation_threshold': 0.07,
                'max_features': 40,
                'ensemble_weights': {'lgb': 0.45, 'prophet': 0.35, 'arima': 0.2}
            }
        ]
        
        return configs
    
    def _evaluate_config(self, config):
        try:
            df_feat = create_comprehensive_features(
                self.df, 
                self.target_col,
                [],
                max_lags=config['lookback']
            )
            
            train_size = int(len(df_feat) * 0.8)
            train_df = df_feat.iloc[:train_size]
            test_df = df_feat.iloc[train_size:]
            
            feature_cols = [col for col in df_feat.columns 
                          if col not in ['ds', 'y'] 
                          and pd.api.types.is_numeric_dtype(df_feat[col])]
            
            X_train = train_df[feature_cols]
            y_train = train_df['y']
            X_test = test_df[feature_cols]
            y_test = test_df['y']
            
            # LightGBM
            lgb_model = LGBMRegressor(random_state=42)
            lgb_model.fit(X_train, y_train)
            lgb_pred = lgb_model.predict(X_test)
            
            # Prophet
            prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
            prophet_model.fit(train_df[['ds', 'y']])
            prophet_forecast = prophet_model.predict(test_df[['ds']])
            prophet_pred = prophet_forecast['yhat'].values
            
            # ARIMA
            try:
                arima_model = SARIMAX(y_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
                arima_results = arima_model.fit(disp=False)
                arima_pred = arima_results.forecast(len(y_test))
            except:
                arima_pred = None
            
            # Ensemble
            predictions = {
                'lgb': lgb_pred,
                'prophet': prophet_pred,
                'arima': arima_pred
            }
            
            final_pred = create_robust_ensemble(predictions, config['ensemble_weights'])
            mae = mean_absolute_error(y_test, final_pred)
            
            return mae
        
        except Exception as e:
            st.warning(f"Error evaluating config: {e}")
            return float('inf')
