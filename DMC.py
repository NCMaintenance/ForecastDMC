import streamlit as st
import pandas as pd
import numpy as np
import shap
import plotly.graph_objs as go
import warnings

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")
st.set_page_config(layout="wide")
st.title("📈 Hospital Forecasting with SHAP Explanations")

# Upload
uploaded = st.sidebar.file_uploader("Upload Excel file", type="xlsx")
if not uploaded:
    st.sidebar.info("Please upload your Excel file.")
    st.stop()

# Load data
df = pd.read_excel(uploaded)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
hospitals = sorted(df['Hospital'].unique())

# Sidebar
sel_hosp = st.sidebar.selectbox("Hospital", ["All"] + hospitals)
targets = [
    "Tracker8am", "Tracker2pm", "Tracker8pm",
    "AdditionalCapacityOpen Morning",
    "TimeTotal_8am", "TimeTotal_2pm", "TimeTotal_8pm"
]
sel_target = st.sidebar.selectbox("Target", ["All"] + targets)
future_days = st.sidebar.slider("Forecast horizon (days ahead)", 7, 30, 14)
run = st.sidebar.button("🚀 Run Forecast")

if not run:
    st.sidebar.info("Configure then click Run Forecast")
    st.stop()

h_list = hospitals if sel_hosp == "All" else [sel_hosp]
t_list = targets if sel_target == "All" else [sel_target]
results = []

for hosp in h_list:
    st.header(f"🏥 {hosp}")
    df_h = df[df['Hospital'] == hosp].reset_index(drop=True)

    for tgt in t_list:
        st.subheader(f"🎯 {tgt}")
        if df_h[tgt].isna().any():
            st.warning("Skipping (nulls present)")
            continue

        # Feature engineering
        df2 = df_h[['Date', tgt]].rename(columns={'Date': 'ds', tgt: 'y'})
        df2['y_lag1'] = df2['y'].shift(1)
        df2['y_lag2'] = df2['y'].shift(2)
        df2['y_diff1'] = df2['y'] - df2['y'].shift(1)
        df2['dow'] = df2['ds'].dt.weekday
        df2['month'] = df2['ds'].dt.month
        df2['dow_sin'] = np.sin(2 * np.pi * df2['dow'] / 7)
        df2['dow_cos'] = np.cos(2 * np.pi * df2['dow'] / 7)
        df2['roll_mean7'] = df2['y'].rolling(window=7).mean()
        df2['roll_std7'] = df2['y'].rolling(window=7).std()
        df2 = df2.dropna().reset_index(drop=True)

        # Prophet features
        m_feat = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        m_feat.add_country_holidays(country_name='IE')
        m_feat.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        m_feat.fit(df2[['ds', 'y']])
        prophet_feat = m_feat.predict(df2[['ds']])[['ds', 'yhat']]
        df2 = df2.merge(prophet_feat, on='ds')
        df2['yhat_lag1'] = df2['yhat'].shift(1)
        df2['yhat_lag7'] = df2['yhat'].shift(7)
        df2 = df2.dropna().reset_index(drop=True)

        feats = [
            'y_lag1', 'y_lag2', 'y_diff1', 'roll_mean7', 'roll_std7',
            'dow', 'month', 'dow_sin', 'dow_cos', 'yhat', 'yhat_lag1', 'yhat_lag7'
        ]

        # Train/test split
        n = len(df2)
        split = int(0.8 * n)
        train = df2.iloc[:split]
        test = df2.iloc[split:]

        X_tr, y_tr = train[feats], train['y']

        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        lgb_maes = []
        for ti, vi in tscv.split(X_tr):
            mdl = LGBMRegressor(n_estimators=200, learning_rate=0.05)
            mdl.fit(X_tr.iloc[ti], y_tr.iloc[ti])
            lgb_maes.append(mean_absolute_error(y_tr.iloc[vi], mdl.predict(X_tr.iloc[vi])))
        mae_lgb = np.mean(lgb_maes)

        # Final model
        lgb_final = LGBMRegressor(n_estimators=200, learning_rate=0.05)
        lgb_final.fit(X_tr, y_tr)

        # SHAP
        explainer = shap.Explainer(lgb_final)
        shap_values = explainer(X_tr)
        st.markdown("### 🔍 SHAP Feature Impact")
        st.pyplot(shap.summary_plot(shap_values, X_tr, plot_type="bar", show=False))

        # Recursive forecast setup
        future_preds_lgb = []
        future_preds_prophet = []
        lag_df = df2.copy()

        m_full = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        m_full.add_country_holidays(country_name='IE')
        m_full.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        m_full.fit(df2[['ds', 'y']])

        for _ in range(future_days):
            last = lag_df.iloc[-1]
            next_date = last['ds'] + pd.Timedelta(days=1)
            yhat_prophet = m_full.predict(pd.DataFrame({'ds': [next_date]}))['yhat'].values[0]

            new_row = {
                'ds': next_date,
                'y': np.nan,
                'y_lag1': last['y'],
                'y_lag2': last['y_lag1'],
                'y_diff1': last['y_lag1'] - last['y_lag2'],
                'roll_mean7': lag_df['y'].iloc[-7:].mean(),
                'roll_std7': lag_df['y'].iloc[-7:].std(),
                'dow': next_date.weekday(),
                'month': next_date.month,
                'dow_sin': np.sin(2 * np.pi * next_date.weekday() / 7),
                'dow_cos': np.cos(2 * np.pi * next_date.weekday() / 7),
                'yhat': yhat_prophet,
                'yhat_lag1': last['yhat'],
                'yhat_lag7': lag_df['yhat'].iloc[-7] if len(lag_df) >= 7 else last['yhat']
            }

            lag_df = pd.concat([lag_df, pd.DataFrame([new_row])], ignore_index=True)
            X_new = lag_df[feats].iloc[[-1]]
            y_pred_lgb = lgb_final.predict(X_new)[0]
            future_preds_lgb.append(y_pred_lgb)
            future_preds_prophet.append(yhat_prophet)
            lag_df.at[len(lag_df)-1, 'y'] = y_pred_lgb

        # Blend LGBM + Prophet
        try:
            cv = cross_validation(m_full, initial='180 days', period='30 days', horizon='30 days')
            mae_prophet = performance_metrics(cv)['mae'].mean()
        except:
            mae_prophet = np.inf

        if np.isinf(mae_prophet):
            pred_test = lgb_final.predict(test[feats])
            method = "LGBM only"
            pred_fut = np.array(future_preds_lgb)
        else:
            w_p = 1 / mae_prophet
            w_l = 1 / mae_lgb
            S = w_p + w_l
            pred_test = (w_p * m_feat.predict(test[['ds']])['yhat'].values +
                         w_l * lgb_final.predict(test[feats])) / S
            pred_fut = (w_p * np.array(future_preds_prophet) + w_l * np.array(future_preds_lgb)) / S
            method = f"Hybrid (P:{w_p/S:.2f},L:{w_l/S:.2f})"

        mae_test = mean_absolute_error(test['y'], pred_test)

        # Plot
        fut_dates = pd.date_range(start=df2['ds'].max() + pd.Timedelta(days=1), periods=future_days)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test['ds'], y=test['y'], mode='lines+markers', name='Actual'))
        fig.add_trace(go.Scatter(x=test['ds'], y=pred_test, mode='lines+markers', name='Pred (test)'))
        fig.add_trace(go.Scatter(x=fut_dates, y=pred_fut, mode='lines+markers',
                                 name='Forecast', line=dict(dash='dash')))
        fig.update_layout(title=f"{hosp} • {tgt} | {method} | MAE: {mae_test:.2f}",
                          xaxis_title="Date", yaxis_title=tgt)
        st.plotly_chart(fig, use_container_width=True)

        # Tables
        st.markdown("**Test‑set Results**")
        st.dataframe(pd.DataFrame({
            "Date": test['ds'],
            "Actual": test['y'],
            "Predicted": pred_test,
            "Error": np.abs(test['y'] - pred_test)
        }).round(2), use_container_width=True)

        st.markdown("**Future Forecast**")
        st.dataframe(pd.DataFrame({
            "Date": fut_dates,
            "Forecast": pred_fut
        }).round(2), use_container_width=True)

        results.append({
            "Hospital": hosp,
            "Target": tgt,
            "Method": method,
            "Test MAE": round(mae_test, 2)
        })

# Summary
st.subheader("📊 Summary")
st.dataframe(pd.DataFrame(results), use_container_width=True)
