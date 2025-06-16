import pandas as pd
import numpy as np

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes the raw hospital data:
    - Renames columns
    - Melts the dataset to long format
    - Extracts datetime features (hour, dayofweek, weekend, cyclical)
    - Adds lag features for each metric
    - Adds rolling window features for each metric
    """
    try:
        # Rename columns for clarity
        df = df.rename(columns={
            'Tracker8am': 'ED_8am',
            'Tracker2pm': 'ED_2pm',
            'Tracker8pm': 'ED_8pm',
            'AdditionalCapacityOpen Morning': 'Additional_Surgery_Capacity',
            'TimeTotal_8am': 'Trolley_8am',
            'TimeTotal_2pm': 'Trolley_2pm',
            'TimeTotal_8pm': 'Trolley_8pm'
        })

        # Ensure 'Date' column is datetime
        df['Date'] = pd.to_datetime(df['Date'])

        # Melt the dataset into long format
        id_vars = ['Hospital Group Name', 'Hospital', 'Date', 'Additional_Surgery_Capacity', 'DayGAR']
        value_vars = ['ED_8am', 'ED_2pm', 'ED_8pm', 'Trolley_8am', 'Trolley_2pm', 'Trolley_8pm']

        # Filter out columns that might not exist in a user's upload
        actual_id_vars = [col for col in id_vars if col in df.columns]
        actual_value_vars = [col for col in value_vars if col in df.columns]

        if not actual_value_vars:
            raise ValueError("No value_vars found in the DataFrame after renaming. Check original column names.")

        df_long = pd.melt(
            df,
            id_vars=actual_id_vars,
            value_vars=actual_value_vars,
            var_name='Metric_Time',
            value_name='Value'
        )

        # Extract Metric (ED or Trolley) and Time (e.g., 8am, 2pm)
        df_long[['Metric', 'TimeLabel']] = df_long['Metric_Time'].str.extract(r'(\w+)_([\d]+[ap]m)')

        # Map time labels to 24-hour time strings
        time_map = {'8am': '08:00', '2pm': '14:00', '8pm': '20:00'}
        df_long['TimeStr'] = df_long['TimeLabel'].map(time_map)

        # Combine date and time into full datetime
        # Ensure 'Date' is string before concatenation to avoid timezone issues or mixed types
        df_long['Datetime'] = pd.to_datetime(df_long['Date'].dt.strftime('%Y-%m-%d') + ' ' + df_long['TimeStr'])


        # Extract time-based features
        df_long['Hour'] = df_long['Datetime'].dt.hour
        df_long['DayOfWeek'] = df_long['Datetime'].dt.dayofweek # Monday=0, Sunday=6
        df_long['IsWeekend'] = df_long['DayOfWeek'].isin([5, 6]).astype(int)

        # Add cyclical features for hour of day
        df_long['Hour_sin'] = np.sin(2 * np.pi * df_long['Hour'] / 24)
        df_long['Hour_cos'] = np.cos(2 * np.pi * df_long['Hour'] / 24)

        # Add cyclical features for day of week
        df_long['Day_sin'] = np.sin(2 * np.pi * df_long['DayOfWeek'] / 7)
        df_long['Day_cos'] = np.cos(2 * np.pi * df_long['DayOfWeek'] / 7)

        # Sort by Hospital, Metric, and Datetime to ensure correct lag/rolling calculations
        df_long = df_long.sort_values(by=['Hospital', 'Metric', 'Datetime'])

        # Add Lag Features (value from 1 day ago for the same hospital, metric, and time of day)
        # We group by Hospital, Metric, and the original TimeLabel (8am, 2pm, 8pm)
        # then shift. This assumes we want to compare 8am to 8am, 2pm to 2pm etc. on consecutive days.
        df_long['Lag_1Day_Value'] = df_long.groupby(['Hospital', 'Metric', 'TimeLabel'])['Value'].shift(1)

        # Add Rolling Window Features (e.g., 7-day rolling mean)
        # The rolling window is calculated over the 'Value' for each group.
        # The window size is 7, min_periods is 1 to output a value even if less than 7 days of data are available.
        # .shift(1) is used to ensure the rolling mean uses data *before* the current day, not including current day.
        df_long['Rolling_Mean_7D_Value'] = df_long.groupby(['Hospital', 'Metric', 'TimeLabel'])['Value']\
                                               .rolling(window=7, min_periods=1)\
                                               .mean()\
                                               .reset_index(level=[0,1,2], drop=True)\
                                               .shift(1) # Use data prior to the current observation


        # Final cleanup
        df_long = df_long.drop(columns=['Metric_Time', 'TimeStr']) # TimeLabel might be useful

        return df_long

    except Exception as e:
        print(f"Error during data processing: {e}")
        # Optionally, re-raise the exception if you want the Streamlit app to handle it
        # raise
        # Or return an empty/error DataFrame
        return pd.DataFrame()

if __name__ == '__main__':
    # Example Usage (for testing purposes)
    # Create a sample DataFrame similar to what an Excel upload might look like
    data = {
        'Hospital Group Name': ['Group A', 'Group A', 'Group B', 'Group B'] * 6,
        'Hospital': ['Hospital1', 'Hospital1', 'Hospital2', 'Hospital2'] * 6,
        'Date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-01', '2023-01-01',
                                '2023-01-02', '2023-01-02', '2023-01-02', '2023-01-02',
                                '2023-01-03', '2023-01-03', '2023-01-03', '2023-01-03',
                                '2023-01-04', '2023-01-04', '2023-01-04', '2023-01-04',
                                '2023-01-05', '2023-01-05', '2023-01-05', '2023-01-05',
                                '2023-01-06', '2023-01-06', '2023-01-06', '2023-01-06',
                                ]),
        'Tracker8am': np.random.randint(50, 100, 24),
        'Tracker2pm': np.random.randint(60, 110, 24),
        'Tracker8pm': np.random.randint(70, 120, 24),
        'AdditionalCapacityOpen Morning': [True, False, True, False] * 6,
        'DayGAR': ['GAR1', 'GAR2', 'GAR1', 'GAR2'] * 6,
        'TimeTotal_8am': np.random.randint(5, 20, 24),
        'TimeTotal_2pm': np.random.randint(10, 25, 24),
        'TimeTotal_8pm': np.random.randint(15, 30, 24)
    }
    sample_df = pd.DataFrame(data)

    print("Original Sample DataFrame:")
    print(sample_df.head())

    processed_df = process_data(sample_df.copy()) # Use .copy() to avoid modifying original sample_df

    if not processed_df.empty:
        print("\nProcessed DataFrame head:")
        print(processed_df.head())
        print("\nProcessed DataFrame columns:")
        print(processed_df.columns)
        print("\nInfo for a specific group (Hospital1, ED, 8am) to check lag/roll features:")
        # Filter for a specific group to check lag/rolling features more easily
        example_group = processed_df[
            (processed_df['Hospital'] == 'Hospital1') &
            (processed_df['Metric'] == 'ED') &
            (processed_df['TimeLabel'] == '8am')
        ]
        print(example_group[['Date', 'Datetime', 'Value', 'Lag_1Day_Value', 'Rolling_Mean_7D_Value']])
    else:
        print("\nProcessing failed or returned empty DataFrame.")
