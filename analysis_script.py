import pandas as pd
import numpy as np
import lightgbm as lgb
import json

# --- Load Data ---
print("Loading data...")
try:
    df = pd.read_csv('processed_may_data.csv')
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    print(f"Data loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print("Error: processed_may_data.csv not found. Please ensure it was generated in the previous step.")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- Correlation Analysis ---
print("Starting Correlation Analysis...")
# Define features for correlation: use all numeric columns except pure IDs or direct date parts if already encoded.
# The preprocessed data should have features like 'Hour_sin', 'IsHoliday', etc.
# and lag features like 'Lag_ED Beds_1'.

# Identify columns that are definitely not features or are raw dates/IDs
cols_to_exclude_corr = ['Hospital Group Name', 'Hospital', 'Date', 'Time', 'Datetime', 'DayGAR']
potential_corr_features = [col for col in df.columns if col not in cols_to_exclude_corr]

# Create a DataFrame for correlation, ensuring all data is numeric
numeric_df_for_corr = df[potential_corr_features].copy()

for col in list(numeric_df_for_corr.columns): # Iterate over a copy of column list for safe removal
    if not pd.api.types.is_numeric_dtype(numeric_df_for_corr[col]):
        try:
            # This might be risky if a category column is forced to numeric
            # However, most features from prepare_data should be numeric or convertible
            numeric_df_for_corr[col] = pd.to_numeric(numeric_df_for_corr[col])
            print(f"Column {col} converted to numeric for correlation.")
        except ValueError:
            print(f"Warning: Column {col} could not be converted to numeric for correlation. Dropping it.")
            numeric_df_for_corr.drop(columns=[col], inplace=True)

# Add back target variables for correlation calculation
if 'ED Beds' in df.columns: numeric_df_for_corr['ED Beds'] = df['ED Beds']
if 'Trolleys' in df.columns: numeric_df_for_corr['Trolleys'] = df['Trolleys']
if 'Capacity' in df.columns: numeric_df_for_corr['Capacity'] = df['Capacity']

correlations = {}
if 'ED Beds' in numeric_df_for_corr.columns:
    correlations['ED Beds'] = numeric_df_for_corr.corr()['ED Beds'].sort_values(ascending=False)
if 'Trolleys' in numeric_df_for_corr.columns:
    correlations['Trolleys'] = numeric_df_for_corr.corr()['Trolleys'].sort_values(ascending=False)
if 'Capacity' in numeric_df_for_corr.columns:
    correlations['Capacity'] = numeric_df_for_corr.corr()['Capacity'].sort_values(ascending=False)


with open('correlation_analysis.txt', 'w') as f:
    if 'ED Beds' in correlations:
        f.write("--- ED Beds Correlations ---\n")
        f.write(str(correlations['ED Beds']))
        f.write("\n\n")
    if 'Trolleys' in correlations:
        f.write("--- Trolleys Correlations ---\n")
        f.write(str(correlations['Trolleys']))
        f.write("\n\n")
    if 'Capacity' in correlations:
        f.write("--- Capacity Correlations ---\n")
        f.write(str(correlations['Capacity']))
        f.write("\n")
print("Correlation analysis saved to correlation_analysis.txt")

# --- Feature Importance Analysis ---
print("Starting Feature Importance Analysis...")

# Load hyperparameters
try:
    with open('best_hyperparameters.json', 'r') as f_hyper:
        hyperparams = json.load(f_hyper)['parameters']
    print("Loaded hyperparameters from best_hyperparameters.json")
except FileNotFoundError:
    print("Error: best_hyperparameters.json not found. Using default LightGBM parameters.")
    # Define some safe defaults if file not found, though it should exist from prev step
    hyperparams = {'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42, 'verbose': -1}
except Exception as e:
    print(f"Error loading hyperparameters: {e}. Using default LightGBM parameters.")
    hyperparams = {'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42, 'verbose': -1}


models_importances = {}

# Base features definition (should exist in processed_may_data.csv)
base_features_for_model = [
    'Hour_sin', 'Hour_cos', 'Day_sin', 'Day_cos', 'Month_sin', 'Month_cos',
    'IsWeekend', 'IsMonday', 'IsFriday', 'IsHoliday',
    'IsSummer', 'IsWinter', 'IsPeakHour', 'IsLowHour', 'Hospital_Code',
    # Raw time features can also be included if they showed importance or low collinearity with cyclical
    'Hour', 'DayOfWeek', 'DayOfMonth', 'Month', 'Quarter', 'WeekOfYear'
]
# Ensure base features are present in the DataFrame
base_features_for_model = [f for f in base_features_for_model if f in df.columns]


for target_col_name in ['ED Beds', 'Trolleys', 'Capacity']:
    print(f"  Training model for feature importance of: {target_col_name}")

    current_model_base_features = base_features_for_model[:]

    # Add 'Capacity' as a feature for 'ED Beds' and 'Trolleys' models
    if target_col_name in ['ED Beds', 'Trolleys']:
        if 'Capacity' in df.columns and 'Capacity' not in current_model_base_features:
            current_model_base_features.append('Capacity')
    # If 'Capacity' is the target, it should not be a feature.
    elif target_col_name == 'Capacity':
        if 'Capacity' in current_model_base_features:
            current_model_base_features.remove('Capacity')

    # Identify lag/rolling features specific to the current target
    # These were created in the first step (e.g., Lag_ED Beds_1, Rolling_Mean_3_ED Beds)
    target_specific_lag_roll_features = [
        col for col in df.columns if
        (col.startswith(f'Lag_{target_col_name}_') or col.startswith(f'Rolling_Mean_3_{target_col_name}') or col.startswith(f'Rolling_Mean_7_{target_col_name}'))
        and target_col_name in col # Double check to avoid issues with similar names like "Lag_Capacity_Multiplier" if it existed
    ]

    final_features_for_model = current_model_base_features + target_specific_lag_roll_features

    # Ensure all selected features are unique, exist in the dataframe, and are not the target itself
    final_features_for_model = sorted(list(set(f for f in final_features_for_model if f in df.columns and f != target_col_name)))

    print(f"    Features for {target_col_name} model ({len(final_features_for_model)}): {final_features_for_model[:5]}... (first 5 shown)")

    # Drop rows where target or any selected feature is NaN for this specific model
    # This is important because different models might have different sets of lag features
    # leading to different NaN patterns if not handled carefully in preprocessing.
    # processed_may_data.csv should have NaNs from lags filled, but target NaNs are possible.
    temp_df = df.dropna(subset=[target_col_name] + final_features_for_model)

    X = temp_df[final_features_for_model]
    y = temp_df[target_col_name]

    if X.empty or y.empty or len(X) < 5: # Check for empty data or too few samples
        print(f"    Skipping feature importance for {target_col_name} due to insufficient data after pre-processing for model training (records: {len(X)}).")
        continue

    # Adjust LightGBM params if data is very small to avoid errors
    current_hyperparams = hyperparams.copy()
    if len(X) < 20: # Example threshold for very small data
        current_hyperparams['num_leaves'] = min(hyperparams.get('num_leaves', 31), len(X) -1 if len(X) >1 else 2)
        current_hyperparams['max_depth'] = max(1, min(hyperparams.get('max_depth', -1), 3)) # Cap depth for small data
        if current_hyperparams['num_leaves'] < 2 : current_hyperparams['num_leaves'] = 2


    model = lgb.LGBMRegressor(**current_hyperparams)

    try:
        model.fit(X, y)
        importances = pd.Series(model.feature_importances_, index=X.columns)
        models_importances[target_col_name] = importances.sort_values(ascending=False)
        print(f"    Successfully trained model for {target_col_name} and extracted importances.")
    except Exception as e:
        print(f"    Error training model or getting importances for {target_col_name}: {e}")


with open('feature_importance_analysis.txt', 'w') as f:
    f.write("--- LightGBM Feature Importances (Top 20 for each target) ---\n")
    if models_importances:
        for target, importance_series in models_importances.items():
            f.write(f"\n-- Target: {target} --\n")
            f.write(str(importance_series.head(20)))
            f.write("\n")
    else:
        f.write("No feature importances were calculated.\n")
print("Feature importance analysis saved to feature_importance_analysis.txt")
print("SHAP analysis was skipped for this subtask.")
print("Analysis script finished.")
