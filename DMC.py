import pandas as pd
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt

def analyze_data(df, hospital, target):
    print(f"Analyzing {hospital} - {target}...")

    df_h = df[df['Hospital'] == hospital].sort_values('Date').reset_index(drop=True)
    df_h['Date'] = pd.to_datetime(df_h['Date'])
    
    # 1. Check missing dates
    date_range = pd.date_range(df_h['Date'].min(), df_h['Date'].max())
    missing_dates = date_range.difference(df_h['Date'])
    if len(missing_dates) > 0:
        print(f"⚠️ Missing dates: {missing_dates}")

    # 2. Check missing values in target
    missing_vals = df_h[target].isna().sum()
    print(f"Missing values in target: {missing_vals}")

    # 3. Check duplicates
    dupes = df_h.duplicated(subset=['Date'])
    print(f"Duplicate rows: {dupes.sum()}")

    # 4. Outliers detection
    df_h['zscore'] = zscore(df_h[target].fillna(method='ffill'))
    outliers = df_h[(df_h['zscore'].abs() > 3)]
    print(f"Outliers found: {len(outliers)}")

    # 5. Feature check for lag1
    df_h['y_lag1'] = df_h[target].shift(1)
    if not df_h[target].iloc[:-1].equals(df_h['y_lag1'].iloc[1:]):
        print("⚠️ Lag 1 misalignment!")

    # 6. Rolling mean check
    df_h['roll_mean7'] = df_h[target].shift(1).rolling(window=7).mean()
    normal_roll = df_h[target].rolling(window=7).mean()
    if not np.allclose(df_h['roll_mean7'].dropna(), normal_roll.shift(1).dropna(), equal_nan=True):
        print("⚠️ Rolling mean might use future info!")

    # 7. Visualize actuals and rolling mean
    plt.figure(figsize=(12,5))
    plt.plot(df_h['Date'], df_h[target], label='Actual')
    plt.plot(df_h['Date'], df_h['roll_mean7'], label='Rolling mean 7 (shifted)')
    plt.legend()
    plt.title(f"{hospital} - {target} Actual vs Rolling Mean")
    plt.show()

# Example usage
analyze_data(df, 'Hospital A', 'Tracker8am')
