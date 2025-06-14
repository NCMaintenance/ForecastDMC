import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import optuna
import plotly.graph_objs as go
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")
st.title("📈 Hospital Forecasting with Prophet & LightGBM (7-day holdout tuning)")

# 📂 Sidebar
uploaded = st.sidebar.file_uploader("📤 Upload Excel file", type="xlsx")
hospitals = []
targets = [
    "Tracker8am", "Tracker2pm", "Tracker8pm",
    "AdditionalCapacityOpen Morning"
]

if not uploaded:
    st.sidebar.info("📄 Please upload your Excel file.")
    st.stop()

# 📊 Load data
df = pd.read_excel(uploaded)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
hospitals = sorted(df['Hospital'].unique())

sel_hosp = st.sidebar.selectbox("🏥 Select Hospital", ["All"] + hospitals)
sel_target = st.sidebar.selectbox("🎯 Select Target", ["All"] + targets)

future_days = st.sidebar.slider("🔮 Forecast horizon (days ahead)", 7, 30, 14)
run = st.sidebar.button("🚀 Run Forecast")
if not run:
    st.sidebar.info("⚙️ Configure then click Run Forecast")
    st.stop()

# Filters
h_list = hospitals if sel_hosp == "All" else [sel_hosp]
t_list = targets if sel_target == "All" else [sel_target]

results = []

def create_features(df2):
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
    return df2

def train_prophet(df2):
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, seasonality_mode='additive')
    m.add_country_holidays(country_name='IE')
    m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    m.fit(df2[['ds', 'y']])
    return m

def add_prophet_preds(m, df2):
    prophet_feat = m.predict(df2[['ds']])[['ds', 'yhat']]
    df2 = df2.merge(prophet_feat, on='ds')
    df2['yhat_lag1'] = df2['yhat'].shift(1)
    df2['yhat_lag7'] = df2['yhat'].shift(7)
    return df2.dropna().reset_index(drop=True)

def lgb_objective(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
        'num_leaves': trial.suggest_int('num_leaves', 20, 50),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 10.0),
        'random_state': 42,
    }
    tscv = TimeSeriesSplit(n_splits=3)
    maes = []
    for ti, vi in tscv.split(X):
        model = LGBMRegressor(**params)
        model.fit(X.iloc[ti], y.iloc[ti])
        preds = model.predict(X.iloc[vi])
        maes.append(mean_absolute_error(y.iloc[vi], preds))
    return np.mean(maes)

for hosp in h_list:
    st.header(f"🏥 {hosp}")
    df_h = df[df['Hospital'] == hosp].reset_index(drop=True)

    for tgt in t_list:
        st.subheader(f"🎯 {tgt}")
        if df_h[tgt].isna().any():
            st.warning("⛔ Skipping due to null values")
            continue

        df2 = df_h[['Date', tgt]].rename(columns={'Date': 'ds', tgt: 'y'})
        df2 = create_features(df2)
        df2 = df2.dropna().reset_index(drop=True)

        # Fit prophet on full data (for final preds later)
        m_prophet_full = train_prophet(df2)
        df2 = add_prophet_preds(m_prophet_full, df2)

        feats = ['y_lag1', 'y_lag2', 'y_diff1', 'roll_mean7', 'roll_std7',
                 'dow', 'month', 'dow_sin', 'dow_cos', 'yhat', 'yhat_lag1', 'yhat_lag7']

        # We will do 7 rolling holdouts of 7 days each, from end going backwards
        n = len(df2)
        window_size = 7
        if n < 50 + window_size:
            st.warning("⚠️ Not enough data to perform 7 rolling 7-day holdouts reliably.")
            continue

        all_params = []
        all_maes = []
        all_models = []
        for i in range(7):
            # Define train/test split for this window
            test_start = n - window_size - i
            test_end = test_start + window_size
            train_df = df2.iloc[:test_start].reset_index(drop=True)
            test_df = df2.iloc[test_start:test_end].reset_index(drop=True)

            X_train, y_train = train_df[feats], train_df['y']
            X_test, y_test = test_df[feats], test_df['y']

            study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(lambda trial: lgb_objective(trial, X_train, y_train), n_trials=30, show_progress_bar=False)

            best_params = study.best_params
            best_params['random_state'] = 42
            best_params['n_estimators'] = int(best_params['n_estimators'])

            model = LGBMRegressor(**best_params)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mae = mean_absolute_error(y_test, preds)

            all_params.append(best_params)
            all_maes.append(mae)
            all_models.append(model)

            st.write(f"Window {i+1} holdout 7-day MAE: {mae:.3f}")

        # Summary of 7 holdouts
        df_param_mae = pd.DataFrame(all_params)
        df_param_mae['7-day MAE'] = all_maes
        st.markdown("### 🧮 Hyperparameter tuning results (7 rolling 7-day holdouts)")
        st.dataframe(df_param_mae)

        # Choose best model overall by lowest average MAE
        best_idx = np.argmin(all_maes)
        best_params_final = all_params[best_idx]
        st.markdown(f"### 🎯 Best hyperparameters from holdouts (lowest 7-day MAE: {all_maes[best_idx]:.3f}):")
        st.write(best_params_final)

        # Train final model on full data with best params
        X_full, y_full = df2[feats], df2['y']
        final_model = LGBMRegressor(**best_params_final)
        final_model.fit(X_full, y_full)

        # Forecast future_days ahead with final model + prophet features
        lag_df = df2.copy()
        preds_lgb = []
        preds_prophet = []
        fut_dates = []

        for _ in range(future_days):
            last = lag_df.iloc[-1]
            nd = last['ds'] + pd.Timedelta(days=1)
            yhat = m_prophet_full.predict(pd.DataFrame({'ds': [nd]}))['yhat'].values[0]

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
            pred_lgb = final_model.predict(row_df[feats])[0]
            row['y'] = pred_lgb

            preds_lgb.append(pred_lgb)
            preds_prophet.append(yhat)
            fut_dates.append(nd)

            lag_df = pd.concat([lag_df, pd.DataFrame([row])], ignore_index=True)

        # For comparison, compute final 7-day MAE on last 7 days (true holdout)
        if n >= window_size:
            final_holdout_true = df2.iloc[-window_size:]
            final_holdout_pred = final_model.predict(final_holdout_true[feats])
            final_mae = mean_absolute_error(final_holdout_true['y'], final_holdout_pred)
            st.markdown(f"### 📉 Final model 7-day holdout MAE on last 7 days: {final_mae:.3f}")
        else:
            final_mae = np.nan

        # Plot results on last 30 days + forecast
        last30 = df2.iloc[-30:].copy()
        last30_pred = final_model.predict(last30[feats])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=last30['ds'], y=last30['y'], mode='lines+markers', name='Actual'))
        fig.add_trace(go.Scatter(x=last30['ds'], y=last30_pred, mode='lines+markers', name='Predicted (last 30 days)'))
        fig.add_trace(go.Scatter(x=fut_dates, y=preds_lgb, mode='lines+markers', name='Forecast', line=dict(dash='dash')))
        fig.update_layout(title=f"{hosp} • {tgt} | Final Model Forecast | 📉 7-day MAE: {final_mae:.3f}",
                          xaxis_title="📅 Date", yaxis_title=tgt, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        # Show future forecast table
        st.markdown("**🔮 Future Forecast**")
        st.dataframe(pd.DataFrame({
            "📅 Date": fut_dates,
            "Forecast": preds_lgb
        }).round(2), use_container_width=True)

        results.append({
            "Hospital": hosp,
            "Target": tgt,
            "Best 7-day MAE": round(final_mae, 3) if not np.isnan(final_mae) else "N/A"
        })

# Summary of all
st.subheader("📊 Summary of final 7-day MAEs")
st.dataframe(pd.DataFrame(results), use_container_width=True)
