import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
from scipy.stats import pearsonr
from datetime import datetime, timedelta
import plotly.graph_objs as go

warnings.filterwarnings("ignore")

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide")
st.title("🏥 Advanced Hospital Forecasting with Auto-Optimization 📈")

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

# --- Data Loading Function ---
@st.cache_data
def load_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file)
        if 'Date' not in df.columns or 'Hospital' not in df.columns:
            st.error("❌ The uploaded file must contain 'Date' and 'Hospital' columns.")
            return None
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df = df.sort_values('Date')
        return df
    except Exception as e:
        st.error(f"⚠️ Error loading data: {e}")
        return None

df = load_data(uploaded)
if df is None:
    st.stop()

hospitals = sorted(df['Hospital'].unique())
sel_hosp = st.sidebar.selectbox("🏨 Hospital", ["All"] + hospitals)
sel_target = st.sidebar.selectbox("🎯 Target", ["All"] + targets)
future_days = st.sidebar.slider("⏳ Forecast horizon (days ahead)", 7, 90, 30)

# --- Helper Functions ---
def calculate_optimal_lookback(df_length):
    """Calculate optimal lookback period based on data length and seasonal patterns."""
    if df_length >= 730:  # 2+ years
        return min(90, df_length // 8)  # Up to 90 days lookback
    elif df_length >= 365:  # 1+ year
        return min(60, df_length // 6)  # Up to 60 days lookback
    elif df_length >= 180:  # 6+ months
        return min(45, df_length // 4)  # Up to 45 days lookback
    else:
        return min(30, df_length // 3)  # Minimum viable lookback

def create_comprehensive_features(df_input, target_col, max_lags=30):
    """Create comprehensive feature set with lagged, rolling, and seasonal data."""
    try:
        df_feat = df_input.copy()
        if 'Date' not in df_feat.columns:
            raise ValueError("Date column not found")
        
        # Lag Features
        for lag in range(1, max_lags + 1):
            df_feat[f'{target_col}_lag_{lag}'] = df_feat[target_col].shift(lag)
        
        # Rolling Features
        for window in [7, 14, 30]:
            df_feat[f'{target_col}_roll_mean_{window}'] = df_feat[target_col].shift(1).rolling(window).mean()
            df_feat[f'{target_col}_roll_std_{window}'] = df_feat[target_col].shift(1).rolling(window).std()
        
        # Date-based Features
        df_feat['dow'] = df_feat['Date'].dt.dayofweek
        df_feat['month'] = df_feat['Date'].dt.month
        df_feat['quarter'] = df_feat['Date'].dt.quarter
        df_feat['year'] = df_feat['Date'].dt.year
        
        return df_feat.dropna()
    except Exception as e:
        st.error(f"Error creating features: {e}")
        return None

# --- Main Forecasting Logic ---
if sel_hosp == "All":
    selected_hospitals = hospitals
else:
    selected_hospitals = [sel_hosp]

if sel_target == "All":
    selected_targets = targets
else:
    selected_targets = [sel_target]

results = []

for hosp in selected_hospitals:
    st.header(f"🏨 Hospital: {hosp}")
    df_hosp = df[df['Hospital'] == hosp].copy()
    
    for target in selected_targets:
        if target not in df_hosp.columns:
            st.warning(f"⚠️ Target '{target}' not found for hospital '{hosp}'. Skipping.")
            continue
        
        st.subheader(f"🎯 Target: {target}")
        df_target = df_hosp[['Date', target]].dropna().rename(columns={target: 'y'})
        df_target = create_comprehensive_features(df_target, 'y', max_lags=calculate_optimal_lookback(len(df_target)))
        
        if df_target is None or len(df_target) < 50:
            st.warning(f"⚠️ Not enough data to forecast '{target}' for hospital '{hosp}'. Skipping.")
            continue
        
        # Split Data
        train_size = int(len(df_target) * 0.8)
        train_df = df_target.iloc[:train_size]
        test_df = df_target.iloc[train_size:]
        
        # Train LightGBM
        lgb_model = LGBMRegressor(random_state=42)
        feature_cols = [col for col in train_df.columns if col not in ['Date', 'y']]
        lgb_model.fit(train_df[feature_cols], train_df['y'])
        
        # Predictions
        lgb_pred = lgb_model.predict(test_df[feature_cols])
        mse = mean_squared_error(test_df['y'], lgb_pred)
        rmse = np.sqrt(mse)  # Fixed RMSE calculation
        mae = mean_absolute_error(test_df['y'], lgb_pred)
        
        # Results Visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test_df['Date'], y=test_df['y'], mode='lines+markers', name='Actual'))
        fig.add_trace(go.Scatter(x=test_df['Date'], y=lgb_pred, mode='lines+markers', name='Predicted'))
        fig.update_layout(title=f"Forecast for {target} ({hosp})", xaxis_title="Date", yaxis_title="Value")
        st.plotly_chart(fig)
        
        results.append({'Hospital': hosp, 'Target': target, 'MAE': mae, 'RMSE': rmse})

# --- Results Summary ---
if results:
    st.header("📊 Summary of Results")
    results_df = pd.DataFrame(results)
    st.dataframe(results_df)
else:
    st.info("No valid results to display.")
