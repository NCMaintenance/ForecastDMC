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
st.title("Hospital Forecasting with Prophet & LightGBM")

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
run = st.sidebar.button("Run Forecast")

if not run:
    st.sidebar.info("Configure then click Run Forecast")
    st.stop()

# Filters
h_list = hospitals if sel_hosp == "All" else [sel_hosp]
t_list = targets if sel_target == "All" else [sel_target]
results = []

for hosp in h_list:
    st.header(f"Hospital: {hosp}")
    df_h = df[df['Hospital'] == hosp].reset_index(drop=True)

    for tgt in t_list:
        st.subheader(f"Target: {tgt}")
        if df_h[tgt].isna().any():
            st.warning("Skipping (nulls present)")
            continue

        df2 = df_h[['Date', tgt]].rename(columns={'Date': 'ds', tgt: 'y'})
        df2['y_lag1'] = df2['y'].shift(1)
        df2['y_lag2'] = df2['y'].shift(2)
        df2['y_diff1'] = df2['y'] - df2['y'].shift(1)

        df2['dow'] = df2['ds'].dt.weekday
        df2['month'] = df2['ds'].dt.month
        df2['week'] = df2['ds'].dt.isocalendar().week.astype(int)
        df2['dayofyear'] = df2['ds'].dt.dayofyear
        df2['dow_sin'] = np.sin(2 * np.pi * df2['dow'] / 7)
        df2['dow_cos'] = np.cos(2 * np.pi * df2['dow'] / 7)
        df2['roll_mean7'] = df2['y'].rolling(window=7).mean()
        df2['roll_std7'] = df2['y'].rolling(window=7).std()

        df2 = df2.dropna().reset_index(drop=True)

        # Fit Prophet model for feature generation
        m_feat = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                         daily_seasonality=False, seasonality_mode='additive')
        m_feat.add_country_holidays(country_name='Ireland')
        m_feat.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        m_feat.fit(df2[['ds', 'y']])
        prophet_feat = m_feat.predict(df2[['ds']])[['ds', 'yhat']]
        df2 = df2.merge(prophet_feat, on='ds')
        df2['yhat_lag1'] = df2['yhat'].shift(1)
        df2['yhat_lag7'] = df2['yhat'].shift(7)
        df2 = df2.dropna().reset_index(drop=True)

        # Add Prophet multi-day ahead forecasts as features for each row
        def add_prophet_multi_ahead_feats(row, model):
            future_pd = pd.DataFrame({'ds': [row['ds'] + pd.Timedelta(days=d) for d in range(1, 8)]})
            fut_pred = model.predict(future_pd)
            for i, val in enumerate(fut_pred['yhat'].values, 1):
                row[f'yhat_plus_{i}'] = val
            return row

        df2 = df2.apply(lambda row: add_prophet_multi_ahead_feats(row, m_feat), axis=1)

        n = len(df2)
        split = int(0.8 * n)
        train = df2.iloc[:split].reset_index(drop=True)
        test = df2.iloc[split:].reset_index(drop=True)

        feats = ['y_lag1', 'y_lag2', 'y_diff1', 'roll_mean7', 'roll_std7',
                 'dow', 'month', 'dow_sin', 'dow_cos', 'yhat',
                 'yhat_lag1', 'yhat_lag7',
                 'yhat_plus_1', 'yhat_plus_2', 'yhat_plus_3',
                 'yhat_plus_4', 'yhat_plus_5', 'yhat_plus_6', 'yhat_plus_7']

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
        l_test = lgb_final.predict(test[feats])

        # Fit full Prophet model on entire data for forecasting
        m_full = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                         daily_seasonality=False, seasonality_mode='additive')
        m_full.add_country_holidays(country_name='Ireland')
        m_full.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        m_full.fit(df2[['ds', 'y']])

        # Prepare future dataframe for Prophet base forecast (used for weighting)
        fut_pd = m_full.make_future_dataframe(periods=future_days, include_history=False)
        fut_res = m_full.predict(fut_pd)
        p_prop_fut_base = fut_res['yhat'].values  # base Prophet forecasts (1-step only)

        # Rolling forecast loop with full feature engineering each step (up to future_days)
        lag_df = df2.copy()

        for i in range(future_days):
            last = lag_df.iloc[-1]
            nd = last['ds'] + pd.Timedelta(days=1)

            # Prophet multi-day forecasts for next 7 days starting from nd
            future_pd_7 = pd.DataFrame({'ds': [nd + pd.Timedelta(days=d) for d in range(7)]})
            fut_res_7 = m_full.predict(future_pd_7)
            yhat_plus = fut_res_7['yhat'].values

            # Compute lags and rolling features for new day
            y_lag1 = last['y'] if not np.isnan(last['y']) else lag_df['y'].iloc[-1]
            y_lag2 = lag_df['y_lag1'].iloc[-1] if 'y_lag1' in lag_df.columns else np.nan
            y_diff1 = y_lag1 - y_lag2 if not (np.isnan(y_lag1) or np.isnan(y_lag2)) else 0

            recent_y = lag_df['y'].iloc[-7:].copy()
            roll_mean7 = recent_y.mean()
            roll_std7 = recent_y.std()

            dow = nd.weekday()
            month = nd.month
            dow_sin = np.sin(2 * np.pi * dow / 7)
            dow_cos = np.cos(2 * np.pi * dow / 7)

            new_row = {
                'ds': nd,
                'y': np.nan,
                'y_lag1': y_lag1,
                'y_lag2': y_lag2,
                'y_diff1': y_diff1,
                'roll_mean7': roll_mean7,
                'roll_std7': roll_std7,
                'dow': dow,
                'month': month,
                'dow_sin': dow_sin,
                'dow_cos': dow_cos,
                'yhat': yhat_plus[0],
                'yhat_lag1': last['yhat'] if 'yhat' in last else yhat_plus[0],
                'yhat_lag7': lag_df['yhat'].iloc[-7] if len(lag_df) >= 7 else yhat_plus[0],
                'yhat_plus_1': yhat_plus[0],
                'yhat_plus_2': yhat_plus[1],
                'yhat_plus_3': yhat_plus[2],
                'yhat_plus_4': yhat_plus[3],
                'yhat_plus_5': yhat_plus[4],
                'yhat_plus_6': yhat_plus[5],
                'yhat_plus_7': yhat_plus[6],
            }

            lag_df = pd.concat([lag_df, pd.DataFrame([new_row])], ignore_index=True)

            X_pred = lag_df[feats].iloc[-1:].fillna(0)

            y_pred = lgb_final.predict(X_pred)[0]

            lag_df.at[lag_df.index[-1], 'y'] = y_pred

        p_lgb_fut = lag_df['y'].iloc[-future_days:].values

        # Evaluate Prophet CV MAE for weighting
        try:
            cv = cross_validation(m_full, initial='180 days', period='30 days',
                                  horizon='30 days', parallel=None)
            mae_prophet = performance_metrics(cv)['mae'].mean()
        except Exception:
            mae_prophet = np.inf

        if np.isinf(mae_prophet):
            pred_test = l_test
            method = "LGBM only"
        else:
            w_p = 1 / mae_prophet
            w_l = 1 / mae_lgb
            S = w_p + w_l
            pred_test = (w_p * m_feat.predict(test[['ds']])['yhat'].values + w_l * l_test)

if np.isinf(mae_prophet):
            pred_test = l_test
            method = "LGBM only"
        else:
            w_p = 1 / mae_prophet
            w_l = 1 / mae_lgb
            S = w_p + w_l
            pred_test = (w_p * m_feat.predict(test[['ds']])['yhat'].values + w_l * l_test) / S
            method = f"Weighted Prophet + LGBM (weights: {w_p:.2f}, {w_l:.2f})"

        # Weighted future forecast between Prophet base and LGBM rolling
        if np.isinf(mae_prophet):
            p_fut_final = p_lgb_fut
        else:
            p_fut_final = (w_p * p_prop_fut_base + w_l * p_lgb_fut) / S

        # Show test plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test['ds'], y=test['y'], mode='lines+markers', name='Actual Test'))
        fig.add_trace(go.Scatter(x=test['ds'], y=pred_test, mode='lines', name='Prediction Test'))
        fig.update_layout(title=f"{hosp} - {tgt} Test Period Forecast ({method})",
                          xaxis_title="Date", yaxis_title="Value")
        st.plotly_chart(fig, use_container_width=True)

        # Show future forecast plot
        future_dates = pd.date_range(start=df2['ds'].max() + pd.Timedelta(days=1), periods=future_days)
        fut_df = pd.DataFrame({'ds': future_dates})
        fut_df['Prophet Base'] = p_prop_fut_base
        fut_df['LGBM Rolling'] = p_lgb_fut
        fut_df['Weighted Forecast'] = p_fut_final

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=fut_df['ds'], y=fut_df['Prophet Base'], mode='lines', name='Prophet Base'))
        fig2.add_trace(go.Scatter(x=fut_df['ds'], y=fut_df['LGBM Rolling'], mode='lines', name='LGBM Rolling'))
        fig2.add_trace(go.Scatter(x=fut_df['ds'], y=fut_df['Weighted Forecast'], mode='lines+markers', name='Weighted Forecast'))
        fig2.update_layout(title=f"{hosp} - {tgt} Future Forecast ({future_days} days)",
                           xaxis_title="Date", yaxis_title="Value")
        st.plotly_chart(fig2, use_container_width=True)

        # Summary stats
        st.write(f"MAE LightGBM CV: {mae_lgb:.3f}")
        st.write(f"MAE Prophet CV: {mae_prophet if not np.isinf(mae_prophet) else 'Unavailable'}")
        st.write(f"Forecast Method Used: {method}")

        results.append({
            'Hospital': hosp,
            'Target': tgt,
            'MAE_LGBM': mae_lgb,
            'MAE_Prophet': mae_prophet,
            'Method': method
        })

st.sidebar.write("### Summary Results")
st.sidebar.dataframe(pd.DataFrame(results))
