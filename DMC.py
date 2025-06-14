import streamlit as st
import optuna
import pandas as pd
import numpy as np

# Example feature creation function
def create_features(df, target_col, params):
    try:
        df = df.copy()
        df['lag1'] = df[target_col].shift(1)
        df['rolling_mean_3'] = df[target_col].rolling(window=3).mean()
        df = df.dropna()
        return df
    except Exception as e:
        raise RuntimeError(f"Feature creation failed: {e}")

# Example evaluation function
def iterative_forecast_evaluation(df, model_name, params):
    from sklearn.linear_model import Ridge
    from lightgbm import LGBMRegressor
    from sklearn.metrics import mean_absolute_error

    df = df.copy()
    if 'target' not in df.columns:
        raise ValueError("No target column found in dataframe")

    X = df.drop(columns=['target'])
    y = df['target']

    if X.empty or y.empty:
        raise ValueError("Empty features or target")

    if model_name == "Ridge":
        model = Ridge(**params)
    elif model_name == "LightGBM":
        model = LGBMRegressor(**params)
    else:
        raise ValueError("Unsupported model")

    model.fit(X, y)
    preds = model.predict(X)
    return mean_absolute_error(y, preds)

# Main Optuna objective function
def objective(trial):
    try:
        model_name = trial.suggest_categorical("model", ["LightGBM", "Ridge"])
        
        if model_name == "LightGBM":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            }

        elif model_name == "Ridge":
            params = {
                "alpha": trial.suggest_float("alpha", 0.1, 10.0)
            }

        else:
            raise ValueError("Unknown model")

        # Simulated input dataframe
        df_work = st.session_state.get("df_work", pd.DataFrame({
            'target': np.random.randn(100)
        }))
        sel_target = 'target'

        # Feature engineering
        df_feat = create_features(df_work, sel_target, params)

        if df_feat.empty:
            raise ValueError("No features after processing")

        if sel_target not in df_feat.columns:
            df_feat['target'] = df_work[sel_target].iloc[-len(df_feat):].values
        else:
            df_feat['target'] = df_feat[sel_target]

        score = iterative_forecast_evaluation(df_feat, model_name, params)

        if not np.isfinite(score):
            raise ValueError("Score was not finite")

        return score

    except Exception as e:
        st.warning(f"Trial {trial.number} failed: {str(e)}")
        return float("inf")


# Streamlit UI
st.title("📈 Time Series Optimization with Optuna")

# Simulate session data if missing
if "df_work" not in st.session_state:
    st.session_state.df_work = pd.DataFrame({
        'target': np.random.randn(100)
    })

# Run optimization
if st.button("Run Optimization"):
    study = optuna.create_study(direction="minimize")
    with st.spinner("Starting optimization with 20 trials..."):
        study.optimize(objective, n_trials=20)

    st.success("Optimization completed!")

    if study.best_trial.value != float("inf"):
        st.markdown("### 🏆 Best Results")
        st.write(f"Best Score (MAE): {study.best_trial.value:.4f}")
        st.json(study.best_trial.params)
    else:
        st.error("No successful trials completed. Please check your data and try again.")
