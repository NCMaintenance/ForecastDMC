import streamlit as st 
import pandas as pd 
import numpy as np 
from prophet import Prophet 
from prophet.diagnostics 
import cross_validation, performance_metrics 
from lightgbm import LGBMRegressor 
from sklearn.model_selection 
import TimeSeriesSplit 
from sklearn.metrics import mean_absolute_error 
import plotly.graph_objs as go 
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide") st.title("📈 Hospital Forecasting with Prophet & LightGBM")

Sidebar

uploaded = st.sidebar.file_uploader("Upload Excel file", type="xlsx") hospitals = [] targets   = [ "Tracker8am","Tracker2pm","Tracker8pm", "AdditionalCapacityOpen Morning" ]

if not uploaded: st.sidebar.info("Please upload your Excel file.") st.stop()

Load data

df = pd.read_excel(uploaded) df['Date'] = pd.to_datetime(df['Date']) df = df.sort_values('Date') hospitals = sorted(df['Hospital'].unique())

sel_hosp   = st.sidebar.selectbox("Hospital", ["All"]+hospitals) sel_target = st.sidebar.selectbox("Target",   ["All"]+targets)

future_days = st.sidebar.slider("Forecast horizon (days ahead)", 7, 30, 14) run = st.sidebar.button("🚀 Run Forecast") if not run: st.sidebar.info("Configure then click Run Forecast") st.stop()

Filters

h_list = hospitals if sel_hosp=="All" else [sel_hosp] t_list = targets   if sel_target=="All" else [sel_target]

results = []

for hosp in h_list: st.header(f"🏥 {hosp}") df_h = df[df['Hospital']==hosp].reset_index(drop=True)

for tgt in t_list:
    st.subheader(f"🎯 {tgt}")
    if df_h[tgt].isna().any():
        st.warning("Skipping (nulls present)")
        continue

    df2 = df_h[['Date', tgt]].rename(columns={'Date':'ds', tgt:'y'})
    df2['y_lag1']  = df2['y'].shift(1)
    df2['y_lag2']  = df2['y'].shift(2)
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

    m_feat = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, seasonality_mode='additive')
    m_feat.add_country_holidays(country_name='IE')
    m_feat.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    m_feat.fit(df2[['ds', 'y']])
    prophet_feat = m_feat.predict(df2[['ds']])[['ds', 'yhat']]
    df2 = df2.merge(prophet_feat, on='ds')
    df2['yhat_lag1'] = df2['yhat'].shift(1)
    df2['yhat_lag7'] = df2['yhat'].shift(7)
    df2 = df2.dropna().reset_index(drop=True)

    n = len(df2)
    split = int(0.8 * n)
    train = df2.iloc[:split].reset_index(drop=True)
    test = df2.iloc[split:].reset_index(drop=True)

    feats = ['y_lag1', 'y_lag2', 'y_diff1', 'roll_mean7', 'roll_std7', 'dow', 'month', 'dow_sin', 'dow_cos', 'yhat', 'yhat_lag1', 'yhat_lag7']

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

    m_full = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, seasonality_mode='additive')
    m_full.add_country_holidays(country_name='IE')
    m_full.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    m_full.fit(df2[['ds', 'y']])

    lag_df = df2.copy()
    preds_lgb = []
    preds_prophet = []
    fut_dates = []

    for _ in range(future_days):
        last = lag_df.iloc[-1]
        nd = last['ds'] + pd.Timedelta(days=1)
        yhat = m_full.predict(pd.DataFrame({'ds': [nd]}))['yhat'].values[0]

        row = {
            'ds': nd,
            'y': np.nan,
            'y_lag1': last['y'],
            'y_lag2': last['y_lag1'],
            'y_diff1': last['y_lag1'] - last['y_lag2'],
            'roll_mean7': lag_df['y'].iloc[-7:].mean(),
            'roll_std7': lag_df['y'].iloc[-7:].std(),
            'dow': nd.weekday(),
            'month': nd.month,
            'dow_sin': np.sin(2 * np.pi * nd.weekday() / 7),
            'dow_cos': np.cos(2 * np.pi * nd.weekday() / 7),
            'yhat': yhat,
            'yhat_lag1': last['yhat'],
            'yhat_lag7': lag_df['yhat'].iloc[-7] if len(lag_df) >= 7 else last['yhat']
        }

        row_df = pd.DataFrame([row])
        pred_lgb = lgb_final.predict(row_df[feats])[0]
        row['y'] = pred_lgb

        preds_lgb.append(pred_lgb)
        preds_prophet.append(yhat)
        fut_dates.append(nd)

        lag_df = pd.concat([lag_df, pd.DataFrame([row])], ignore_index=True)

    try:
        cv = cross_validation(m_full, initial='180 days', period='30 days', horizon='30 days', parallel=None)
        mae_prophet = performance_metrics(cv)['mae'].mean()
    except:
        mae_prophet = np.inf

    if np.isinf(mae_prophet):
        pred_test = l_test
        pred_fut = preds_lgb
        method = "LGBM only"
    else:
        w_p = 1 / mae_prophet
        w_l = 1 / mae_lgb
        S = w_p + w_l
        pred_test = (w_p * m_feat.predict(test[['ds']])['yhat'].values + w_l * l_test) / S
        pred_fut = [(w_p * p + w_l * l) / S for p, l in zip(preds_prophet, preds_lgb)]
        method = f"Hybrid (P:{w_p/S:.2f},L:{w_l/S:.2f})"

    mae_test = mean_absolute_error(test['y'], pred_test)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test['ds'], y=test['y'], mode='lines+markers', name='Actual'))
    fig.add_trace(go.Scatter(x=test['ds'], y=pred_test, mode='lines+markers', name='Pred (test)'))
    fig.add_trace(go.Scatter(x=fut_dates, y=pred_fut, mode='lines+markers', name='Forecast', line=dict(dash='dash')))
    fig.update_layout(title=f"{hosp} • {tgt} | {method} | MAE: {mae_test:.2f}",
                      xaxis_title="Date", yaxis_title=tgt, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

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

    results.append({"Hospital": hosp, "Target": tgt, "Method": method, "Test MAE": round(mae_test, 2)})

st.subheader("📊 Summary") st.dataframe(pd.DataFrame(results), use_container_width=True)
