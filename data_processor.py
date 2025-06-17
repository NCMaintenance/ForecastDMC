import pandas as pd
import numpy as np
from datetime import timedelta # Added for holiday calculations

def process_data(df: pd.DataFrame, irish_holidays_calendar: callable) -> pd.DataFrame:
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
            'AdditionalCapacityOpen Morning': 'Additional_Capacity',
            'TimeTotal_8am': 'Trolley_8am',
            'TimeTotal_2pm': 'Trolley_2pm',
            'TimeTotal_8pm': 'Trolley_8pm'
        })

        # Ensure 'Date' column is datetime
        df['Date'] = pd.to_datetime(df['Date'])

        # Fill Additional Capacity across the day - important to do this before melt
        if 'Additional_Capacity' in df.columns and 'Hospital' in df.columns and 'Date' in df.columns:
            df['Additional_Capacity'] = df.groupby(['Hospital', 'Date'])['Additional_Capacity'].transform('first')
        else:
            # Handle cases where these columns might be missing, perhaps by logging or setting a default
            # For now, if 'Additional_Capacity' is there, we assume it's already properly filled or doesn't need this transformation.
            pass


        # Melt the dataset into long format
        id_vars = ['Hospital Group Name', 'Hospital', 'Date', 'Additional_Capacity', 'DayGAR']
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
        df_long['DayOfWeek'] = df_long['Datetime'].dt.dayofweek  # Monday=0, Sunday=6
        df_long['DayOfMonth'] = df_long['Datetime'].dt.day
        df_long['Month'] = df_long['Datetime'].dt.month
        df_long['Quarter'] = df_long['Datetime'].dt.quarter
        # Ensure WeekOfYear is integer
        df_long['WeekOfYear'] = df_long['Datetime'].dt.isocalendar().week.astype(int)
        df_long['IsWeekend'] = df_long['DayOfWeek'].isin([5, 6]).astype(int)
        df_long['IsMonday'] = (df_long['DayOfWeek'] == 0).astype(int)
        df_long['IsFriday'] = (df_long['DayOfWeek'] == 4).astype(int)

        # Add cyclical features
        df_long['Hour_sin'] = np.sin(2 * np.pi * df_long['Hour'] / 24)
        df_long['Hour_cos'] = np.cos(2 * np.pi * df_long['Hour'] / 24)
        df_long['Day_sin'] = np.sin(2 * np.pi * df_long['DayOfWeek'] / 7)
        df_long['Day_cos'] = np.cos(2 * np.pi * df_long['DayOfWeek'] / 7)
        df_long['Month_sin'] = np.sin(2 * np.pi * df_long['Month'] / 12)
        df_long['Month_cos'] = np.cos(2 * np.pi * df_long['Month'] / 12)
        df_long['Week_sin'] = np.sin(2 * np.pi * df_long['WeekOfYear'] / 52)
        df_long['Week_cos'] = np.cos(2 * np.pi * df_long['WeekOfYear'] / 52)

        # Add Irish holidays
        # irish_holidays_calendar should be an instance of IrishBankHolidays passed to the function
        holidays = irish_holidays_calendar.holidays(start=df_long['Datetime'].min(), end=df_long['Datetime'].max() + timedelta(days=30))
        df_long['IsHoliday'] = df_long['Datetime'].dt.normalize().isin(holidays).astype(int)

        # Holiday proximity features
        df_long['DaysToHoliday'] = 0
        df_long['DaysFromHoliday'] = 0

        # This loop can be slow for large datasets. Consider vectorizing if performance is an issue.
        # For now, keeping it as is to match DMC.py's logic.
        for idx, row in df_long.iterrows():
            date = row['Datetime'].normalize()
            future_holidays = [h for h in holidays if h >= date]
            past_holidays = [h for h in holidays if h < date]

            if future_holidays:
                df_long.loc[idx, 'DaysToHoliday'] = (min(future_holidays) - date).days
            else:
                df_long.loc[idx, 'DaysToHoliday'] = 365  # Far future

            if past_holidays:
                df_long.loc[idx, 'DaysFromHoliday'] = (date - max(past_holidays)).days
            else:
                df_long.loc[idx, 'DaysFromHoliday'] = 365  # Far past

        # Seasonal indicators
        df_long['IsSummer'] = df_long['Month'].isin([6, 7, 8]).astype(int)
        df_long['IsWinter'] = df_long['Month'].isin([12, 1, 2]).astype(int)
        df_long['IsSpring'] = df_long['Month'].isin([3, 4, 5]).astype(int)

        # Peak hour indicators
        df_long['IsPeakHour'] = df_long['Hour'].isin([20]).astype(int)
        df_long['IsLowHour'] = df_long['Hour'].isin([8]).astype(int)

        # Encode hospital if 'Hospital' column exists
        if 'Hospital' in df_long.columns:
            df_long['Hospital_Code'] = df_long['Hospital'].astype('category').cat.codes
        else:
            df_long['Hospital_Code'] = 0 # Default or handle as per requirements

        # Sort by Hospital, Metric, and Datetime (important for consistency and some subsequent operations in DMC.py)
        # Ensure 'Hospital' and 'Metric' are present before sorting by them.
        sort_by_cols = []
        if 'Hospital' in df_long.columns:
            sort_by_cols.append('Hospital')
        if 'Metric' in df_long.columns:
            sort_by_cols.append('Metric')
        sort_by_cols.append('Datetime')
        df_long = df_long.sort_values(by=sort_by_cols)

        # Final cleanup
        # We will keep TimeLabel as it might be useful for analysis or direct use in DMC.py
        # Metric_Time and TimeStr are intermediate and can be dropped.
        df_long = df_long.drop(columns=['Metric_Time', 'TimeStr'], errors='ignore')

        return df_long

    except Exception as e:
        print(f"Error during data processing: {e}")
        # Optionally, re-raise the exception if you want the Streamlit app to handle it
        # raise
        # Or return an empty/error DataFrame
        return pd.DataFrame()

if __name__ == '__main__':
    # Example Usage (for testing purposes)
    # --- Define a dummy Irish Bank Holidays for testing ---
    from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday
    from pandas.tseries.offsets import DateOffset # Required if using offset in Holiday rules
    from dateutil.rrule import MO # Required if using MO in Holiday rules

    class DummyIrishBankHolidays(AbstractHolidayCalendar):
        rules = [
            Holiday("New Year's Day", month=1, day=1), # Simple holiday for testing
            Holiday("St. Patrick's Day", month=3, day=17),
        ]

    data = {
        'Hospital Group Name': ['Group A', 'Group A', 'Group B', 'Group B'] * 6,
        'Hospital': ['Hospital1', 'Hospital1', 'Hospital2', 'Hospital2'] * 6,
        'Date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-01', '2023-01-01',
                                '2023-03-16', '2023-03-16', '2023-03-17', '2023-03-17', # Include dates around a holiday
                                '2023-03-18', '2023-03-18', '2023-03-18', '2023-03-18',
                                '2023-01-04', '2023-01-04', '2023-01-04', '2023-01-04',
                                '2023-01-05', '2023-01-05', '2023-01-05', '2023-01-05',
                                '2023-01-06', '2023-01-06', '2023-01-06', '2023-01-06',
                                ]),
        'Tracker8am': np.random.randint(50, 100, 24),
        'Tracker2pm': np.random.randint(60, 110, 24),
        'Tracker8pm': np.random.randint(70, 120, 24),
        'AdditionalCapacityOpen Morning': [True, False, True, False, True, True, False, False] * 3,
        'DayGAR': ['GAR1', 'GAR2', 'GAR1', 'GAR2'] * 6,
        'TimeTotal_8am': np.random.randint(5, 20, 24),
        'TimeTotal_2pm': np.random.randint(10, 25, 24),
        'TimeTotal_8pm': np.random.randint(15, 30, 24)
    }
    sample_df = pd.DataFrame(data)

    print("Original Sample DataFrame:")
    print(sample_df.head())

    holiday_calendar_instance = DummyIrishBankHolidays()

    # Pass the holiday calendar to process_data
    processed_df = process_data(sample_df.copy(), irish_holidays_calendar=holiday_calendar_instance)

    if not processed_df.empty:
        print("\nProcessed DataFrame head:")
        print(processed_df.head(10)) # Show more rows
        print("\nProcessed DataFrame columns:")
        print(processed_df.columns)
        print("\nInfo for a specific group (Hospital1, ED, 8am) to check features including holiday features:")

        # Ensure the columns for filtering exist before attempting to filter
        if all(col in processed_df.columns for col in ['Hospital', 'Metric', 'TimeLabel']):
            example_group_df = processed_df[
                (processed_df['Hospital'] == 'Hospital1') &
                (processed_df['Metric'] == 'ED') &
                (processed_df['TimeLabel'] == '8am') # Ensure this TimeLabel is generated
            ]
            if not example_group_df.empty:
                relevant_cols = ['Date', 'Datetime', 'Value', 'Hour', 'DayOfWeek', 'IsWeekend',
                                 'IsHoliday', 'DaysToHoliday', 'DaysFromHoliday', 'Hospital_Code',
                                 'Month_sin', 'Week_cos', 'IsSummer', 'IsPeakHour', 'Additional_Capacity']
                # Filter relevant_cols to only those that actually exist in example_group_df
                display_cols = [col for col in relevant_cols if col in example_group_df.columns]
                print(example_group_df[display_cols].head(10)) # Show more rows of the example
            else:
                print("Could not find the specific example group (Hospital1, ED, 8am).")
        else:
            print("One or more columns required for filtering the example group ('Hospital', 'Metric', 'TimeLabel') are missing.")

    else:
        print("\nProcessing failed or returned empty DataFrame.")
