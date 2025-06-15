import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from timesfm import TimesFM  # Using TimesFM instead of LightGBM
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import plotly.graph_objs as go
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")
st.title("📈 Hospital Forecasting with Prophet & TimesFM")

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

future_days = st.sidebar.slider("Forecast horizon (days ahead)", 7, 30, 14)
run = st.sidebar.button("🚀 Run Forecast")
if not run:
    st.sidebar.info("Configure then click Run Forecast")
    st.stop()

# Filters
h_list = hospitals if sel_hosp == "All" else [sel_hosp]
t_list = targets if sel_target == "All" else [sel_target]

results = []

# Main loop for hospitals and targets
for hosp in h_list:
    st.header(f"🏥 {hosp}")
    df_h = df[df['Hospital'] == hosp].reset_index(drop=True)
    
    for tgt in t_list:
        st.subheader(f"🎯 {tgt}")
        
        # Data preparation
        try:
            # Drop non-correlated variables
            drop_cols = []
            if tgt in ["Tracker8am", "Tracker2pm", "Tracker8pm"]:
                drop_cols = ["TimeTotal_8am", "TimeTotal_2pm", "TimeTotal_8pm"]
            elif tgt in ["TimeTotal_8am", "TimeTotal_2pm", "TimeTotal_8pm"]:
                drop_cols = ["Tracker8am", "Tracker2pm", "Tracker8pm"]
            drop_cols = [col for col in drop_cols if col in df_h.columns and col != "AdditionalCapacityOpen Morning"]
            df_h_clean = df_h.drop(columns=drop_cols, errors='ignore')

            if df_h_clean[tgt].isna().any():
                st.warning("Skipping (nulls present)")
                continue

            # Feature engineering
            df2 = df_h_clean[['Date', tgt]].rename(columns={'Date': 'ds', tgt: 'y'})
            
            # Add time-based features
            for lag in range(1, 8):  # Extending lag features
                df2[f'y_lag{lag}'] = df2['y'].shift(lag)
            df2['y_diff1'] = df2['y'] - df2['y'].shift(1)
            df2['y_diff7'] = df2['y'] - df2['y'].shift(7)  # Weekly difference

            # Enhanced seasonal features
            df2['dow'] = df2['ds'].dt.weekday
            df2['month'] = df2['ds'].dt.month
            df2['week'] = df2['ds'].dt.isocalendar().week.astype(int)
            df2['dayofyear'] = df2['ds'].dt.dayofyear
            
            # Cyclical features
            for period, name in [(7, 'dow'), (12, 'month'), (52, 'week')]:
                df2[f'{name}_sin'] = np.sin(2 * np.pi * df2[name] / period)
                df2[f'{name}_cos'] = np.cos(2 * np.pi * df2[name] / period)

            # Rolling statistics
            for window in [7, 14, 30]:  # Multiple windows
                df2[f'roll_mean{window}'] = df2['y'].rolling(window=window).mean()
                df2[f'roll_std{window}'] = df2['y'].rolling(window=window).std()
                df2[f'roll_max{window}'] = df2['y'].rolling(window=window).max()
                df2[f'roll_min{window}'] = df2['y'].rolling(window=window).min()

            df2 = df2.dropna().reset_index(drop=True)

            # Prophet model for feature generation
            m_feat = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode='additive'
            )
            m_feat.add_country_holidays(country_name='IE')
            m_feat.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            m_feat.fit(df2[['ds', 'y']])
            prophet_feat = m_feat.predict(df2[['ds']])[['ds', 'yhat']]
            df2 = df2.merge(prophet_feat, on='ds')

            # Split data
            n = len(df2)
            split = int(0.8 * n)
            train = df2.iloc[:split].reset_index(drop=True)
            test = df2.iloc[split:].reset_index(drop=True)

            # Define features for TimesFM
            feats = [col for col in df2.columns if col not in ['ds', 'y']]
            
            # TimesFM model
            tfm = TimesFM(
                time_col='ds',
                target_col='y',
                frequency='D',
                max_lags=7,
                seasonal_period=[7, 30]  # Weekly and monthly seasonality
            )
            
            # Cross-validation for TimesFM
            tscv = TimeSeriesSplit(n_splits=5)
            tfm_maes = []
            
            for ti, vi in tscv.split(train):
                tfm.fit(train.iloc[ti])
                tfm_pred = tfm.predict(train.iloc[vi][['ds']])
                tfm_maes.append(mean_absolute_error(train.iloc[vi]['y'], tfm_pred))
            
            mae_tfm = np.mean(tfm_maes)

            # Final TimesFM model
            tfm_final = TimesFM(
                time_col='ds',
                target_col='y',
                frequency='D',
                max_lags=7,
                seasonal_period=[7, 30]
            )
            tfm_final.fit(train)
            t_test = tfm_final.predict(test[['ds']])

            # Prophet cross-validation
            try:
                cv = cross_validation(m_feat, initial='180 days', period='30 days', horizon='30 days', parallel=None)
                mae_prophet = performance_metrics(cv)['mae'].mean()
            except:
                mae_prophet = np.inf

            # Hybrid prediction
            if np.isinf(mae_prophet):
                pred_test = t_test
                method = "TimesFM only"
            else:
                w_p = 1 / mae_prophet
                w_t = 1 / mae_tfm
                S = w_p + w_t
                pred_test = (w_p * m_feat.predict(test[['ds']])['yhat'].values + w_t * t_test) / S
                method = f"Hybrid (P:{w_p/S:.2f},T:{w_t/S:.2f})"

            # Future predictions
            fut_pd = m_feat.make_future_dataframe(periods=future_days, include_history=False)
            p_prop_fut = m_feat.predict(fut_pd)['yhat'].values
            t_fut = tfm_final.predict(fut_pd)
            
            pred_fut = (w_p * p_prop_fut + w_t * t_fut) / S if not np.isinf(mae_prophet) else t_fut

            # Calculate metrics
            mae_test = mean_absolute_error(test['y'], pred_test)

            # Plotting
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=test['ds'], y=test['y'], mode='lines+markers', name='Actual'))
            fig.add_trace(go.Scatter(x=test['ds'], y=pred_test, mode='lines+markers', name='Pred (test)'))
            fig.add_trace(go.Scatter(x=fut_pd['ds'], y=pred_fut, mode='lines+markers', name='Forecast', line=dict(dash='dash')))
            
            fig.update_layout(
                title=f"{hosp} • {tgt} | {method} | MAE: {mae_test:.2f}",
                xaxis_title="Date",
                yaxis_title=tgt,
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Display results
            st.markdown("**Test‑set Results**")
            st.dataframe(pd.DataFrame({
                "Date": test['ds'],
                "Actual": test['y'],
                "Predicted": pred_test,
                "Error": np.abs(test['y'] - pred_test)
            }).round(2), use_container_width=True)

            st.markdown("**Future Forecast**")
            st.dataframe(pd.DataFrame({
                "Date": fut_pd['ds'],
                "Forecast": pred_fut
            }).round(2), use_container_width=True)

            results.append({
                "Hospital": hosp,
                "Target": tgt,
                "Method": method,
                "Test MAE": round(mae_test, 2)
            })
            
        except Exception as e:
            st.error(f"Error processing {hosp} - {tgt}: {str(e)}")
            continue

# Display summary
st.subheader("📊 Summary")
st.dataframe(pd.DataFrame(results), use_container_width=True)
