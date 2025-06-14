import streamlit as st
import pandas as pd
import numpy as np
import optuna
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import warnings

warnings.filterwarnings("ignore")

# ========== Feature Engineering ==========

def create_features(df, target_column, params):
    df = df.copy().sort_values("date")
    lags = params.get("lags", 3)
    rolling = params.get("rolling", 3)

    for lag in range(1, lags + 1):
        df[f"{target_column}_lag{lag}"] = df[target_column].shift(lag)

    for r in range(2, rolling + 1):
        df[f"{target_column}_rolling{r}"] = df[target_column].shift(1).rolling(r).mean()

    df = df.dropna()
    return df

# ========== Model Evaluation ==========

def evaluate_forecast(df, model_name, params):
    tscv = TimeSeriesSplit(n_splits=3)
    mae_scores = []

    features = [col for col in df.columns if col not in ["date", "target", "hospital_id"]]
    X = df[features]
    y = df["target"]

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if model_name == "ridge":
            model = Ridge(alpha=params.get("alpha", 1.0))
        elif model_name == "random_forest":
            model = RandomForestRegressor(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", 5),
                random_state=42,
            )
        else:
            raise ValueError("Unsupported model")

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae_scores.append(mean_absolute_error(y_test, preds))

    return np.mean(mae_scores)

# ========== Objective Function ==========

def objective(trial):
    model_name = st.session_state.model_name
    sel_target = st.session_state.sel_target
    df_work = st.session_state.df_work

    if model_name == "ridge":
        params = {
            "alpha": trial.suggest_float("alpha", 0.001, 10.0, log=True),
            "lags": trial.suggest_int("lags", 2, 7),
            "rolling": trial.suggest_int("rolling", 2, 7),
        }
    elif model_name == "random_forest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "lags": trial.suggest_int("lags", 2, 7),
            "rolling": trial.suggest_int("rolling", 2, 7),
        }
    else:
        raise ValueError("Unsupported model")

    mae_list = []
    hospitals = df_work["hospital_id"].unique()

    for hospital in hospitals:
        df_h = df_work[df_work["hospital_id"] == hospital].copy()

        # Ensure minimum length and proper ordering
        if len(df_h) < 20:
            continue
        df_h = df_h.sort_values("date")
        df_h = df_h.reset_index(drop=True)

        # Create target column
        if st.session_state.sel_target not in df_h.columns:
            continue

        df_feat = create_features(df_h, st.session_state.sel_target, params)
        if df_feat.empty:
            continue

        df_feat["target"] = df_feat[st.session_state.sel_target]

        try:
            mae = evaluate_forecast(df_feat, model_name, params)
            mae_list.append(mae)
        except Exception:
            continue

    if not mae_list:
        return float("inf")

    return np.mean(mae_list)

# ========== Streamlit App ==========

st.title("📈 Hospital Forecast Optimization")

# Upload or use sample data
if "df_work" not in st.session_state:
    st.subheader("Demo Mode: Using Random Sample Data")
    np.random.seed(42)
    date_range = pd.date_range("2023-01-01", "2023-12-31")
    hospitals = ["HOSP_A", "HOSP_B", "HOSP_C"]
    df_demo = pd.DataFrame({
        "date": np.tile(date_range, len(hospitals)),
        "hospital_id": np.repeat(hospitals, len(date_range)),
        "target": np.random.normal(loc=100, scale=15, size=len(date_range) * len(hospitals)),
    })
    st.session_state.df_work = df_demo

df = st.session_state.df_work
st.session_state.sel_target = "target"
st.session_state.model_name = st.selectbox("Choose model", ["ridge", "random_forest"])

# Show data overview
st.markdown("### 📊 Data Preview")
st.dataframe(df.head())

# Run Optimization
if st.button("🚀 Start Optimization"):
    with st.spinner("Running Optuna optimization across hospitals..."):
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=25)

        st.success("✅ Optimization completed!")

        st.subheader("🏆 Best Results")
        st.write(f"**Best Score (Average MAE):** `{study.best_value:.2f}`")
        st.write("**Best Parameters:**")
        st.json(study.best_params)
