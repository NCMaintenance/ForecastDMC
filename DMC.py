# Hospital Forecasting with Prophet & LightGBM
import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance
import plotly.graph_objs as go
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")
st.title("Hospital Forecasting with Prophet & LightGBM")

# Sidebar
uploaded = st.sidebar.file_uploader("Upload Excel file", type="xlsx")
targets = [
    "Tracker8am", "Tracker2pm", "Tracker8pm",
    "AdditionalCapacityOpen Morning", "TimeTotal_8am", "TimeTotal_2pm", "TimeTotal_8pm"
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
auto_remove_neg = st.sidebar.checkbox("Auto-remove harmful features", value=True)
run = st.sidebar.button("Run Forecast")

if not run:
    st.sidebar.info("Configure then click Run Forecast")
    st.stop()

h_list = hospitals if sel_hosp == "All" else [sel_hosp]
t_list = targets if sel_target == "All" else [sel_target]

for hosp in h_list:
    st.header(f"Hospital: {hosp}")
    df_h = df[df['Hospital'] == hosp].reset_index(drop=True)

    for tgt in t_list:
        try:
            st.subheader(f"Target: {tgt}")
            if df_h[tgt].isna().any():
                st.warning("Skipping (nulls present)")
                continue

            # Feature engineering
            df2 = df_h[['Date', tgt]].rename(columns={'Date': 'ds', tgt: 'y'})
            df2['y_lag1'] = df2['y'].shift(1)
            df2['y_lag2'] = df2['y'].shift(2)
            df2['y_lag3'] = df2['y'].shift(3)
            df2['y_lag7'] = df2['y'].shift(7)
            df2['y_diff1'] = df2['y'] - df2['y'].shift(1)
            df2['dow'] = df2['ds'].dt.weekday
            df2['month'] = df2['ds'].dt.month
            df2['dow_sin'] = np.sin(2 * np.pi * df2['dow'] / 7)
            df2['dow_cos'] = np.cos(2 * np.pi * df2['dow'] / 7)

            for window in [3, 7]:
                for stat in ['mean', 'std', 'min', 'max', 'median', 'skew', 'kurt']:
                    df2[f'roll_{stat}_{window}'] = df2['y'].rolling(window=window).agg(stat)

            df2 = df2.dropna().reset_index(drop=True)

            if len(df2) < 2:
                st.warning(f"Skipping {hosp} - {tgt}: Not enough data after feature engineering")
                continue

            # Prophet model for features
            m_feat = Prophet(yearly_seasonality=True, weekly_seasonality=True, seasonality_mode='additive')
            m_feat.add_country_holidays(country_name='Ireland')
            m_feat.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            m_feat.fit(df2[['ds', 'y']])
            prophet_feat = m_feat.predict(df2[['ds']])[['ds', 'yhat']]
            df2 = df2.merge(prophet_feat, on='ds')
            df2['yhat_lag1'] = df2['yhat'].shift(1)
            df2['yhat_lag3'] = df2['yhat'].shift(3)
            df2['yhat_lag7'] = df2['yhat'].shift(7)
            df2 = df2.dropna().reset_index(drop=True)

            if len(df2) < 2:
                st.warning(f"Skipping {hosp} - {tgt}: Not enough data after adding prophet features")
                continue

            # Split data
            n = len(df2)
            split = int(0.8 * n)
            train = df2.iloc[:split].reset_index(drop=True)
            test = df2.iloc[split:].reset_index(drop=True)
            feats = [col for col in df2.columns if col not in ['ds', 'y']]

            X_tr, y_tr = train[feats], train['y']
            tscv = TimeSeriesSplit(n_splits=5)
            lgb_maes = []
            for ti, vi in tscv.split(X_tr):
                mdl = LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=31)
                mdl.fit(X_tr.iloc[ti], y_tr.iloc[ti])
                lgb_maes.append(mean_absolute_error(y_tr.iloc[vi], mdl.predict(X_tr.iloc[vi])))
            mae_lgb = np.mean(lgb_maes)

            lgb_final = LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=31)
            lgb_final.fit(X_tr, y_tr)

            perm = permutation_importance(lgb_final, test[feats], test['y'], scoring='neg_mean_absolute_error', n_repeats=5, random_state=42)
            perm_imp = pd.Series(perm.importances_mean, index=feats)
            harmful_feats = perm_imp[perm_imp < 0].index.tolist()
            if auto_remove_neg:
                feats = [f for f in feats if f not in harmful_feats]
                st.markdown(f"Removed {len(harmful_feats)} features with negative impact")

            l_test = lgb_final.predict(test[feats])

            # Prophet full model for forecasting
            m_full = Prophet(yearly_seasonality=True, weekly_seasonality=True, seasonality_mode='additive')
            m_full.add_country_holidays(country_name='Ireland')
            m_full.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            m_full.fit(df2[['ds', 'y']])
            fut_pd = m_full.make_future_dataframe(periods=future_days, include_history=False)
            fut_res = m_full.predict(fut_pd)
            p_prop_fut = fut_res['yhat'].values

            # Prepare features for future LGBM forecast
            lag_df = df2.copy()
            for _ in range(future_days):
                last = lag_df.iloc[-1]
                nd = last['ds'] + pd.Timedelta(days=1)
                yhat = m_full.predict(pd.DataFrame({'ds': [nd]}))['yhat'].values[0]
                row = {
                    'ds': nd, 'y': np.nan,
                    'y_lag1': last['y'], 'y_lag2': last['y_lag1'],
                    'y_lag3': last['y_lag2'], 'y_lag7': last['y_lag3'],
                    'y_diff1': last['y_lag1'] - last['y_lag2'],
                    'dow': nd.weekday(), 'month': nd.month,
                    'dow_sin': np.sin(2 * np.pi * nd.weekday() / 7),
                    'dow_cos': np.cos(2 * np.pi * nd.weekday() / 7),
                    'yhat': yhat, 'yhat_lag1': last['yhat'],
                    'yhat_lag3': last['yhat_lag1'], 'yhat_lag7': last['yhat_lag3']
                }
                for window in [3, 7]:
                    for stat in ['mean', 'std', 'min', 'max', 'median', 'skew', 'kurt']:
                        key = f'roll_{stat}_{window}'
                        row[key] = lag_df['y'].iloc[-window:].agg(stat)
                lag_df = pd.concat([lag_df, pd.DataFrame([row])], ignore_index=True)

            p_lgb_fut = lgb_final.predict(lag_df[feats].iloc[-future_days:])

            # Prophet cross-validation MAE
            try:
                cv = cross_validation(m_full, initial='180 days', period='30 days', horizon='30 days', parallel=None)
                mae_prophet = performance_metrics(cv)['mae'].mean()
            except Exception:
                mae_prophet = np.inf

            # Combine forecasts
            if np.isinf(mae_prophet):
                pred_test = l_test
                method = "LGBM only"
                pred_fut = p_lgb_fut
            else:
                w_p = 1 / mae_prophet
                w_l = 1 / mae_lgb
                S = w_p + w_l
                pred_test = (w_p * m_feat.predict(test[['ds']])['yhat'].values + w_l * l_test) / S
                pred_fut = (w_p * p_prop_fut + w_l * p_lgb_fut) / S
                method = f"Hybrid (P:{w_p/S:.2f},L:{w_l/S:.2f})"

            mae_test = mean_absolute_error(test['y'], pred_test)

            # Plot results
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=test['ds'], y=test['y'], mode='lines+markers', name='Actual'))
            fig.add_trace(go.Scatter(x=test['ds'], y=pred_test, mode='lines+markers', name='Pred (test)'))
            fig.add_trace(go.Scatter(x=fut_pd['ds'], y=pred_fut, mode='lines+markers', name='Forecast', line=dict(dash='dash')))
            fig.update_layout(title=f"{hosp} • {tgt} | {method} | MAE: {mae_test:.2f}",
                              xaxis_title="Date", yaxis_title=tgt, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Test-set Results**")
            df_test_results = test[['ds', 'y']].copy()
            df_test_results['Prediction'] = pred_test
            df_test_results.columns = ['Date', 'Actual', 'Prediction']
            st.dataframe(df_test_results)

        except Exception as e:
            st.error(f"Error processing {hosp} - {tgt}: {e}")
