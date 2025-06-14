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
st.title("📈 Hospital Forecasting with Prophet & LightGBM + Optuna Tuning")

uploaded = st.sidebar.file_uploader("📤 Upload Excel file", type="xlsx")
targets = [
    "Tracker8am", "Tracker2pm", "Tracker8pm",
    "AdditionalCapacityOpen Morning"
]

if not uploaded:
    st.sidebar.info("📄 Please upload your Excel file.")
    st.stop()

df = pd.read_excel(uploaded)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
hospitals = sorted(df['Hospital'].unique())

sel_hosp = st.sidebar.selectbox("🏥 Select Hospital", ["All"] + hospitals)
sel_target = st.sidebar.selectbox("🎯 Select Target", ["All"] + targets)

future_days = st.sidebar.slider("🔮 Forecast horizon (days ahead)", 7, 14, 14)
run = st.sidebar.button("🚀 Run Forecast")
if not run:
    st.sidebar.info("⚙️ Configure then click Run Forecast")
    st.stop()

h_list = hospitals if sel_hosp == "All" else [sel_hosp]
t_list = targets if sel_target == "All" else [sel_target]

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
    return df2.dropna().reset_index(drop=True)

def get_prophet_features(df2):
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, seasonality_mode='additive')
    m.add_country_holidays(country_name='IE')
    m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    m.fit(df2[['ds', 'y']])
    prophet_pred = m.predict(df2[['ds']])[['ds', 'yhat']]
    df2 = df2.merge(prophet_pred, on='ds')
    df2['yhat_lag1'] = df2['yhat'].shift(1)
    df2['yhat_lag7'] = df2['yhat'].shift(7)
    return df2.dropna().reset_index(drop=True), m

def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.1),
        "num_leaves": trial.suggest_int("num_leaves", 20, 50),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
        "subsample": trial.suggest_uniform("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-4, 1.0),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-4, 1.0),
        "random_state": 42
    }
    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    return mae

results = []

for hosp in h_list:
    st.header(f"🏥 {hosp}")
    df_h = df[df['Hospital'] == hosp].reset_index(drop=True)

    for tgt in t_list:
        st.subheader(f"🎯 {tgt}")

        if df_h[tgt].isna().any():
            st.warning("⛔ Skipping due to null values")
            continue

        # Prepare data with features + prophet
        df2 = df_h[['Date', tgt]].rename(columns={'Date': 'ds', tgt: 'y'})
        df2 = create_features(df2)
        df2, prophet_model = get_prophet_features(df2)

        feats = ['y_lag1', 'y_lag2', 'y_diff1', 'roll_mean7', 'roll_std7', 'dow', 'month',
                 'dow_sin', 'dow_cos', 'yhat', 'yhat_lag1', 'yhat_lag7']

        # Rolling windows tuning: 7 windows, 1-day horizon validation
        window_size = 180  # training days
        val_horizon = 1    # 1 day validation horizon
        total_len = len(df2)

        if total_len < window_size + val_horizon + 7:
            st.warning("⚠️ Not enough data for rolling tuning")
            continue

        tuning_results = []
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))

        def optuna_objective(trial):
            maes = []
            # Run rolling validation for the trial params
            for start in range(total_len - window_size - val_horizon - 6, total_len - window_size - val_horizon + 1):
                train_idx = range(start, start + window_size)
                val_idx = [start + window_size]

                X_train = df2.loc[train_idx, feats]
                y_train = df2.loc[train_idx, 'y']
                X_val = df2.loc[val_idx, feats]
                y_val = df2.loc[val_idx, 'y']

                mae = objective(trial, X_train, y_train, X_val, y_val)
                maes.append(mae)

            return np.mean(maes)

        # Run tuning for 50 trials (adjustable)
        study.optimize(optuna_objective, n_trials=50, show_progress_bar=True)

        best_params = study.best_params
        st.write(f"✅ Best hyperparameters found: {best_params}")

        # Train final model on all data except last future_days (to validate)
        train_final = df2.iloc[:-future_days].reset_index(drop=True)
        test_final = df2.iloc[-future_days:].reset_index(drop=True)

        X_train_final = train_final[feats]
        y_train_final = train_final['y']
        X_test_final = test_final[feats]
        y_test_final = test_final['y']

        model_final = LGBMRegressor(**best_params)
        model_final.fit(X_train_final, y_train_final)

        # Iterative 1-day ahead forecasting for future_days
        lag_df = df2.iloc[:-future_days].copy()
        preds_iterative = []

        for day in range(future_days):
            last = lag_df.iloc[-1]
            next_date = last['ds'] + pd.Timedelta(days=1)

            # Prophet prediction for next day
            prophet_pred = prophet_model.predict(pd.DataFrame({'ds': [next_date]}))['yhat'].values[0]

            # Build features row for next day
            row = {
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
                'yhat': prophet_pred,
                'yhat_lag1': last['yhat'],
                'yhat_lag7': lag_df['yhat'].iloc[-7] if len(lag_df) >= 7 else last['yhat']
            }
            row_df = pd.DataFrame([row])
            # Predict with LightGBM
            pred_lgb = model_final.predict(row_df[feats])[0]
            row['y'] = pred_lgb

            preds_iterative.append(pred_lgb)
            lag_df = pd.concat([lag_df, pd.DataFrame([row])], ignore_index=True)

        mae_test_final = mean_absolute_error(test_final['y'], preds_iterative)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test_final['ds'], y=test_final['y'], mode='lines+markers', name='Actual'))
        fig.add_trace(go.Scatter(x=test_final['ds'], y=preds_iterative, mode='lines+markers', name='Forecast (Iterative)'))
        fig.update_layout(title=f"{hosp} • {tgt} | Optuna tuned LGBM | 📉 MAE (7-day unseen): {mae_test_final:.2f}",
                          xaxis_title="📅 Date", yaxis_title=tgt, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**📉 7-day Unseen Test‑set Results**")
        st.dataframe(pd.DataFrame({
            "📅 Date": test_final['ds'],
            "Actual": test_final['y'],
            "Predicted": preds_iterative,
            "Error": np.abs(test_final['y'] - preds_iterative)
        }).round(2), use_container_width=True)

        # Final forecast forward beyond available data (optional)
        # Extend lag_df to forecast future_days ahead (after test_final)
        forecast_start = df2['ds'].max()
        forecast_dates = [forecast_start + pd.Timedelta(days=i+1) for i in range(future_days)]
        preds_future = []

        for nd in forecast_dates:
            last = lag_df.iloc[-1]
            prophet_pred = prophet_model.predict(pd.DataFrame({'ds': [nd]}))['yhat'].values[0]

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
                'yhat': prophet_pred,
                'yhat_lag1': last['yhat'],
                'yhat_lag7': lag_df['yhat'].iloc[-7] if len(lag_df) >= 7 else last['yhat']
            }
            row_df = pd.DataFrame([row])
            pred_lgb = model_final.predict(row_df[feats])[0]
            row['y'] = pred_lgb
            preds_future.append(pred_lgb)
            lag_df = pd.concat([lag_df, pd.DataFrame([row])], ignore_index=True)

        st.markdown("**🔮 Final Future Forecast**")
        st.dataframe(pd.DataFrame({
            "📅 Date": forecast_dates,
            "Forecast": preds_future
        }).round(2), use_container_width=True)

        results.append({"Hospital": hosp, "Target": tgt,
                        "Best Hyperparams": best_params,
                        "📉 7-day MAE": round(mae_test_final, 2)})

st.subheader("📊 Summary")
summary_df = pd.DataFrame(results)
st.dataframe(summary_df, use_container_width=True)
