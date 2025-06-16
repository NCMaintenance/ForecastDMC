# Hospital Data Analysis Platform

This application allows users to upload hospital metrics data from an Excel file, processes it to generate additional features (like lag and rolling averages), and provides visualizations to explore the data.

## Features

*   **Data Upload**: Upload hospital data via an Excel file.
*   **Data Processing**:
    *   Renames columns for clarity.
    *   Melts data into a long format.
    *   Extracts time-based features (hour, day of week, cyclical features).
    *   Generates lag features (previous day's values).
    *   Generates rolling window statistics (e.g., 7-day rolling mean).
*   **Data Display**:
    *   View the processed data table (head or full).
    *   Display summary statistics.
*   **Interactive Visualizations**:
    *   Line charts showing metric values over time.
    *   Filters for Hospital Group, Hospital, Metric, and Time Label.
    *   Comparison chart for actual values vs. lag and rolling mean values.

## Project Structure

```
.
├── app.py                          # Main Streamlit application script
├── data_processor.py               # Script for data processing and feature engineering
├── requirements.txt                # Python dependencies
├── sample_data.xlsx                # Sample Excel data file
└── README.md                       # This file
```

## Setup and Installation

1.  **Clone the repository (if applicable) or download the files.**

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    Make sure you are in the project's root directory (where `requirements.txt` is located).
    ```bash
    pip install -r requirements.txt
    ```

## How to Run the Application

1.  **Ensure your virtual environment is activated** (if you created one).

2.  **Navigate to the project's root directory** in your terminal.

3.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    This will typically open the application in your default web browser.

## Input Excel File Format

The application expects an Excel file (`.xlsx` or `.xls`) with a sheet containing data similar to the `sample_data.xlsx` provided. Key columns expected are:

*   `Hospital Group Name`: (Text) Name of the hospital group.
*   `Hospital`: (Text) Name of the hospital.
*   `Date`: (Date/Text) Date of the record (e.g., YYYY-MM-DD or recognized by pandas).
*   `Tracker8am`: (Numeric) Tracker value at 8 AM.
*   `Tracker2pm`: (Numeric) Tracker value at 2 PM.
*   `Tracker8pm`: (Numeric) Tracker value at 8 PM.
*   `AdditionalCapacityOpen Morning`: (Boolean/Text e.g., TRUE/FALSE) Additional capacity status.
*   `DayGAR`: (Text) Day GAR value.
*   `TimeTotal_8am`: (Numeric) Time total at 8 AM.
*   `TimeTotal_2pm`: (Numeric) Time total at 2 PM.
*   `TimeTotal_8pm`: (Numeric) Time total at 8 PM.

The script `data_processor.py` will rename these columns as follows for internal processing:
*   `Tracker8am` -> `ED_8am`
*   `Tracker2pm` -> `ED_2pm`
*   `Tracker8pm` -> `ED_8pm`
*   `AdditionalCapacityOpen Morning` -> `Additional_Surgery_Capacity`
*   `TimeTotal_8am` -> `Trolley_8am`
*   `TimeTotal_2pm` -> `Trolley_2pm`
*   `TimeTotal_8pm` -> `Trolley_8pm`

If your column names differ, you might need to adjust the renaming map in `data_processor.py` or rename your columns before uploading.

## Developed By

AI Agent Jules
