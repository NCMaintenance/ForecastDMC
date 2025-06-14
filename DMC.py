import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")
st.title("Hospital Forecasting with Prophet & LightGBM")

# Sidebar
uploaded = st.sidebar.file_uploader("Upload Excel file", type="xlsx")
targets = [
    "Tracker8am", "Tracker2pm", "Tracker8pm",
    "AdditionalCapacityOpen Morning", "TimeTotal_8am",
    "TimeTotal_2pm", "TimeTotal_8pm"
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

h_list = hospitals if sel_hosp == "All" else [sel_hosp]
t_list = targets if sel_target == "All" else [sel_target]

results = []

for hosp in h_list:
    st.header(f"Hospital: {hosp}")
    df_h = df[df['Hospital'] == hosp].reset_index(drop=True)

    for tgt in t_list:
        try:
            st.subheader(f"Target: {tgt}")
            if df_h[tgt].isna().any():
                st.warning("Skipping (nulls present)")
                continue

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
            l_test = lgb_final.predict(test[feats])

            m_full = Prophet(yearly_seasonality=True, weekly_seasonality=True, seasonality_mode='additive')
            m_full.add_country_holidays(country_name='Ireland')
            m_full.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            m_full.fit(df2[['ds', 'y']])
            fut_pd = m_full.make_future_dataframe(periods=future_days, include_history=False)
            fut_res = m_full.predict(fut_pd)
            p_prop_fut = fut_res['yhat'].values

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

            try:
                cv = cross_validation(m_full, initial='180 days', period='30 days', horizon='30 days', parallel=None)
                mae_prophet = performance_metrics(cv)['mae'].mean()
            except:
                mae_prophet = np.inf

            if np.isinf(mae_prophet):
                pred_test = l_test
                method = "LGBM only"
            else:
                w_p = 1 / mae_prophet
                w_l = 1 / mae_lgb
                S = w_p + w_l
                pred_test = (w_p * m_feat.predict(test[['ds']])['yhat'].values + w_l * l_test) / S
                method = f"Hybrid (P:{w_p/S:.2f},L:{w_l/S:.2f})"

            mae_test = mean_absolute_error(test['y'], pred_test)
            pred_fut = (w_p * p_prop_fut + w_l * p_lgb_fut) / S if not np.isinf(mae_prophet) else p_lgb_fut

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=test['ds'], y=test['y'], mode='lines+markers', name='Actual'))
            fig.add_trace(go.Scatter(x=test['ds'], y=pred_test, mode='lines+markers', name='Pred (test)'))
            fig.add_trace(go.Scatter(x=fut_pd['ds'], y=pred_fut, mode='lines+markers', name='Forecast', line=dict(dash='dash')))
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
                "Date": fut_pd['ds'],
                "Forecast": pred_fut
            }).round(2), use_container_width=True)

            st.markdown("**Top 20 Feature Correlations with Target**")
            corr = df2[['y'] + feats].corr().abs()['y'].drop('y').sort_values(ascending=False).head(20)
            fig_corr, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=corr.values, y=corr.index, palette="coolwarm", ax=ax)
            ax.set_title("Top 20 Feature Correlations with Target")
            ax.set_xlabel("Correlation")
            ax.set_ylabel("Feature")
            st.pyplot(fig_corr)

            results.append({"Hospital": hosp, "Target": tgt, "Method": method, "Test MAE": round(mae_test, 2)})

        except Exception as e:
            st.error(f"Error for {hosp} - {tgt}: {str(e)}")

st.subheader("Summary")
st.dataframe(pd.DataFrame(results), use_container_width=True)


To make the app display correctly and enhance the correlation heatmap with the Top 20 features most correlated with the target, you’ll need several fixes and improvements:


---

✅ Fixes Required:

1. Incorrect formatting: Your code had improper line breaks, markdown usage, and a backtick (`) block mid-code.


2. Feature correlation improvement: Show only the top 20 features most correlated with the target variable y.




---

✅ Enhanced Code Block

Here's a corrected and enhanced version of your correlation heatmap block (just replace your current section for correlation plot with this):

# Correlation heatmap (Top 20 correlated features)
st.markdown("**Top 20 Feature Correlations with Target**")
try:
    corr = df2[['y'] + feats].corr()
    corr_y = corr['y'].drop('y').abs().sort_values(ascending=False)
    top_feats = corr_y.head(20).index.tolist()
    top_corr = df2[['y'] + top_feats].corr()

    fig_corr, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(top_corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, cbar=True, square=True)
    ax.set_title("Top 20 Correlated Features with Target")
    st.pyplot(fig_corr)

except Exception as e:
    st.warning(f"Correlation heatmap failed: {str(e)}")


---

🛠 Optional: Additional Fixes You Should Make

To avoid display/logic errors in your full Streamlit app:

Ensure that your import block is not mid-cell or merged with other statements.

Wrap long code blocks properly—no stray backticks (`) in the middle.

Break up logic into reusable functions (e.g., feature engineering, forecasting, plotting).

Ensure you check that feats actually exists and has enough features for top 20 selection.



---

✅ Summary Table at End (already mostly correct)

Make sure your end-of-app section is clearly defined:

st.subheader("Summary of Forecasting Results")
summary_df = pd.DataFrame(results)
st.dataframe(summary_df, use_container_width=True)
