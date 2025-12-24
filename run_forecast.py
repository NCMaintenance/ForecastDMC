import sys
import subprocess
import os
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import warnings
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday
from pandas.tseries.offsets import DateOffset
from dateutil.rrule import MO
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# --- 1. Auto-Install Dependencies (Colab/Local Safe) ---
def install(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except Exception as e:
        print(f"  Could not install {package}: {e}")

print("Checking dependencies...")
try:
    import neuralforecast
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATSx
    from neuralforecast.losses.pytorch import MAE
    print("✅ NeuralForecast is installed.")
except ImportError:
    install("neuralforecast")
    # Also install scipy if it's missing, as it's a critical dependency often missed
    try:
        import scipy
    except ImportError:
        install("scipy")
    import neuralforecast
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATSx
    from neuralforecast.losses.pytorch import MAE

try:
    import meteostat
    from meteostat import Point, Hourly
    print("✅ Meteostat is installed.")
except ImportError:
    install("meteostat")
    from meteostat import Point, Hourly

# --- 2. Holiday Calendar (Ireland) ---
class IrishBankHolidays(AbstractHolidayCalendar):
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=nearest_workday),
        Holiday("St. Brigid's Day", month=2, day=1, observance=nearest_workday),
        Holiday("St. Patrick's Day", month=3, day=17, observance=nearest_workday),
        Holiday("May Day", month=5, day=1, offset=DateOffset(weekday=MO(1))),
        Holiday("June Bank Holiday", month=6, day=1, offset=DateOffset(weekday=MO(1))),
        Holiday("October Bank Holiday", month=10, day=1, offset=DateOffset(weekday=MO(-1))),
        Holiday("Christmas Day", month=12, day=25),
        Holiday("St. Stephen's Day", month=12, day=26),
    ]

# --- 3. Data Loading & Reshaping ---
def load_and_unpivot_all_data(file_path):
    """
    Loads Excel file and converts it from Wide (8am/2pm/8pm columns) to Long format.
    Handles multiple hospitals.
    """
    print(f"\n--- Loading Data from {file_path} ---")
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"❌ Error reading Excel file: {e}")
        return pd.DataFrame()

    # Rename columns to standard internal names
    col_map = {
        'Tracker8am': 'ED_8am', 'Tracker2pm': 'ED_2pm', 'Tracker8pm': 'ED_8pm',
        'TimeTotal_8am': 'Trolley_8am', 'TimeTotal_2pm': 'Trolley_2pm', 'TimeTotal_8pm': 'Trolley_8pm',
        'AdditionalCapacityOpen Morning': 'Additional_Capacity'
    }
    df = df.rename(columns=col_map)

    # Normalize Date
    df['Date'] = pd.to_datetime(df['Date'])

    # Fill daily capacity forward/backward for the group
    if 'Additional_Capacity' in df.columns:
        df['Additional_Capacity'] = df.groupby(['Hospital', 'Date'])['Additional_Capacity'].transform('first').fillna(0)

    # Unpivot: Convert 3 daily columns into rows
    time_map = {'8am': '08:00', '2pm': '14:00', '8pm': '20:00'}
    rows = []

    for _, row in df.iterrows():
        for time_key, time_val in time_map.items():
            ed_val = pd.to_numeric(row.get(f'ED_{time_key}'), errors='coerce')
            trolley_val = pd.to_numeric(row.get(f'Trolley_{time_key}'), errors='coerce')

            rows.append({
                'Hospital': row['Hospital'],
                'Date': row['Date'],
                'Time': time_val,
                'Capacity': row.get('Additional_Capacity', 0),
                'ED Emissions': ed_val,
                'Trolleys': trolley_val
            })

    df_long = pd.DataFrame(rows)
    df_long['Datetime'] = pd.to_datetime(df_long['Date'].astype(str) + ' ' + df_long['Time'])
    df_long = df_long.sort_values(['Hospital', 'Datetime']).reset_index(drop=True)

    print(f"✅ Data processed. Rows: {len(df_long)} | Hospitals found: {df_long['Hospital'].unique()}")
    return df_long

# --- 4. Advanced Weather Processing (The "Up To" Logic) ---
def get_advanced_weather_features(start_date, end_date, location_name="Cork"):
    """
    Downloads hourly weather and aggregates it into 8am, 2pm, 8pm slots.
    Calculates rolling max/min/mean for the previous X hours leading up to the slot.
    """
    print(f"\n--- Fetching Weather Data for {location_name} ---")

    # Coordinates
    locs = {
        "Cork": Point(51.8979, -8.4706, 25),
        "Dublin": Point(53.3498, -6.2603, 20),
        "Galway": Point(53.2707, -9.0568, 25),
        "Kerry": Point(52.2704, -9.7026, 34)
    }

    point = locs.get(location_name, locs["Cork"])

    try:
        # Fetch hourly data
        data = Hourly(point, start_date, end_date).fetch()
        if data.empty:
            print("⚠️ No weather data found.")
            return None

        data = data.reset_index().rename(columns={'time': 'Datetime'})

        # Ensure we have a continuous index for rolling calc
        data = data.set_index('Datetime')

        # --- Rolling Aggregations (The "Up To" Logic) ---
        # We want stats for the 6 hours leading up to the timestamp
        # e.g., 8am row contains max windspeed from 2am to 8am.

        rolling_6h = data.rolling('6h')

        data['Max_Wspd_6h'] = rolling_6h['wspd'].max()
        data['Min_Temp_6h'] = rolling_6h['temp'].min()
        data['Max_Temp_6h'] = rolling_6h['temp'].max()
        data['Mean_Rain_6h'] = rolling_6h['prcp'].sum() # Accumulate rain
        data['Mean_Pres_6h'] = rolling_6h['pres'].mean()

        # Spot values (values exactly at that hour)
        data['Spot_Temp'] = data['temp']
        data['Spot_Wspd'] = data['wspd']

        # Reset index to filter for specific times
        data = data.reset_index()

        # Filter only the times we care about: 08:00, 14:00, 20:00
        # (Assuming your hospital data is only at these times)
        data = data[data['Datetime'].dt.hour.isin([8, 14, 20])].copy()

        cols_to_keep = ['Datetime', 'Max_Wspd_6h', 'Min_Temp_6h', 'Max_Temp_6h',
                        'Mean_Rain_6h', 'Mean_Pres_6h', 'Spot_Temp', 'Spot_Wspd']

        print("✅ Weather features (rolling max/min/sum) calculated.")
        return data[cols_to_keep]

    except Exception as e:
        print(f"❌ Weather fetch failed: {e}")
        return None

# --- 5. Virus Data ---
def download_virus_data():
    url = "https://respiratoryvirus.hpsc.ie/datasets/dbd10a497fd84e44954b858249919997_0.csv"
    print("\n--- Downloading Virus Data ---")
    try:
        df = pd.read_csv(url)
        df['Date'] = pd.to_datetime(df['floor']).dt.tz_localize(None).dt.normalize()
        pivot = df.pivot_table(index='Date', columns='disease', values='weekly_cases', aggfunc='sum').fillna(0)
        pivot.columns = [f"Virus_{c.replace(' ','_')}" for c in pivot.columns]
        return pivot.reset_index()
    except:
        print("⚠️ Could not download virus data. Proceeding without.")
        return pd.DataFrame()

# --- 6. Cross-Hospital Feature Engineering ---
def generate_cross_hospital_table(df_long):
    """
    Pivots ALL hospitals to create a wide master table.
    Then shifts the data to create valid lagged regressors for prediction.
    """
    print("\n--- Generating Cross-Hospital Lags (The Large Table) ---")

    # Pivot to get columns like: Cork_ED_Emissions, Mater_Trolleys, etc.
    df_pivot = df_long.pivot_table(
        index='Datetime',
        columns='Hospital',
        values=['ED Emissions', 'Trolleys']
    )

    # Flatten MultiIndex columns
    df_pivot.columns = [f"{col[1].replace(' ', '_')}_{col[0].replace(' ', '_')}" for col in df_pivot.columns]

    # Create Lags (We can't use current values of other hospitals to predict current target, so we lag)
    # Lag 1 = Previous time slot (e.g., predict 2pm using 8am data)
    # Lag 3 = Previous day same time (approx)
    lagged_df = pd.DataFrame(index=df_pivot.index)

    for col in df_pivot.columns:
        lagged_df[f"Lag1_{col}"] = df_pivot[col].shift(1)
        lagged_df[f"Lag24h_{col}"] = df_pivot[col].shift(3) # Assuming 3 slots per day

    return lagged_df.fillna(0).reset_index()

# --- 7. Main Pipeline ---
def prepare_nbeats_dataset(file_path, target_hospital, weather_location="Cork"):

    # 1. Load Main Data
    df_main = load_and_unpivot_all_data(file_path)
    if df_main.empty: return None

    # 2. Generate "Large Table" of Other Hospitals
    df_cross_features = generate_cross_hospital_table(df_main)

    # 3. Filter for Target Hospital (Rows)
    df_target = df_main[df_main['Hospital'] == target_hospital].copy()

    # 4. Merge Cross-Hospital Features
    df_target = pd.merge(df_target, df_cross_features, on='Datetime', how='left')

    # 5. Fetch and Merge Advanced Weather
    min_date = df_target['Datetime'].min() - timedelta(days=1) # Buffer for rolling calc
    max_date = df_target['Datetime'].max()

    weather_df = get_advanced_weather_features(min_date, max_date, weather_location)

    if weather_df is not None:
        df_target = pd.merge(df_target, weather_df, on='Datetime', how='left')
        # Fill missing weather (if any) with interpolation
        w_cols = [c for c in weather_df.columns if c != 'Datetime']
        df_target[w_cols] = df_target[w_cols].interpolate(method='linear').bfill().ffill()

    # 6. Virus Data
    virus_df = download_virus_data()
    if not virus_df.empty:
        df_target['DateOnly'] = df_target['Datetime'].dt.normalize()
        df_target = pd.merge_asof(
            df_target.sort_values('DateOnly'),
            virus_df.sort_values('Date'),
            left_on='DateOnly', right_on='Date', direction='backward'
        )
        df_target.drop(columns=['DateOnly', 'Date_y'], inplace=True, errors='ignore')

    # 7. Time Features
    df_target['Hour'] = df_target['Datetime'].dt.hour
    df_target['DayOfWeek'] = df_target['Datetime'].dt.dayofweek
    df_target['IsWeekend'] = df_target['Datetime'].dt.dayofweek.isin([5, 6]).astype(int)

    cal = IrishBankHolidays()
    hols = cal.holidays(start=df_target['Datetime'].min(), end=df_target['Datetime'].max())
    df_target['IsHoliday'] = df_target['Datetime'].dt.normalize().isin(hols).astype(int)

    # 8. Clean up for NeuralForecast
    # NeuralForecast needs 'unique_id', 'ds', 'y'
    # We will rename accordingly before return in the main execution block

    return df_target

def run_training_and_evaluation(df, target_col):
    """
    Trains NBEATSx model and calculates MAE and R^2.
    """
    print(f"\n--- Training NBEATSx for Target: {target_col} ---")

    # Prepare DataFrame for NeuralForecast
    nf_df = df.copy()
    nf_df = nf_df.rename(columns={'Datetime': 'ds', target_col: 'y', 'Hospital': 'unique_id'})

    # Select static and historic exogenous features
    # All numeric columns except 'y' and 'ds' can be used as features
    exclude_cols = ['ds', 'y', 'unique_id', 'Date', 'Time', 'Date_x', 'DateOnly']
    numeric_cols = nf_df.select_dtypes(include=[np.number]).columns.tolist()
    futr_exog_list = [c for c in numeric_cols if c not in exclude_cols]

    # Split Data (Last 7 days for test)
    # Assuming 3 observations per day
    horizon = 7 * 3

    # Instantiate Model
    # Note: NeuralForecast models expect specific input configs
    models = [
        NBEATSx(
            h=horizon,
            input_size=horizon * 2,
            loss=MAE(),
            scaler_type='standard',
            max_steps=100, # Kept low for demo speed, increase for production
            futr_exog_list=futr_exog_list
        )
    ]

    nf = NeuralForecast(models=models, freq='H') # Frequency is roughly hourly/sub-daily

    # Cross-validation / Backtesting would be ideal, but simple train/test split for now
    # We need to manually split if we want to calculate MAE/R2 on a holdout set
    # NeuralForecast cross_validation returns the forecasts

    print(f"Running Cross-Validation (Horizon={horizon} steps)...")
    try:
        cv_df = nf.cross_validation(df=nf_df, n_windows=1)
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return

    # Calculate Metrics
    y_true = cv_df['y']
    y_pred = cv_df['NBEATSx']

    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n📊 Results for {target_col}:")
    print(f"   MAE: {mae:.4f}")
    print(f"   R² : {r2:.4f}")

    return cv_df

if __name__ == "__main__":
    # --- Configuration ---
    FILE_PATH = "hospital_data.xlsx"
    TARGET_HOSPITAL = "Cork University Hospital"
    LOCATION = "Cork"

    # Check if file exists
    if not os.path.exists(FILE_PATH):
        print(f"⚠️ {FILE_PATH} not found. Please place your Excel file in the directory.")
        # Attempt to list files to help debug what is available
        print(f"Current directory files: {os.listdir('.')}")
    else:
        final_df = prepare_nbeats_dataset(FILE_PATH, TARGET_HOSPITAL, LOCATION)

        if final_df is not None:
            # We have two targets: 'ED Emissions' and 'Trolleys'
            # Train and Eval for ED Emissions
            run_training_and_evaluation(final_df, 'ED Emissions')

            # Train and Eval for Trolleys
            run_training_and_evaluation(final_df, 'Trolleys')
