# Hospital Forecasting - Local Execution

This repository contains scripts to forecast hospital metrics (ED Beds, Trolleys) using an ensemble of ARIMAX, Gradient Boosting, and Prophet models.

## Usage

1.  Ensure `hospital_data.xlsx` is in the root directory.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the forecast script:
    ```bash
    python run_forecast.py
    ```
    This will generate a CSV file with the forecast for the next 7 days.

## Models

The forecasting pipeline uses:
*   **Daily ARIMAX**: Forecasts daily aggregates using virus data and weather features.
*   **Hourly Gradient Boosting (Sklearn)**: Forecasts hourly metrics (8am, 2pm, 8pm).
    *   **2pm Anchor**: The 2pm metric is predicted first and used as a feature for other time slots.
    *   **3-Horizon Strategy**: Separate models are trained for Day 1, Days 2-4, and Days 5-7 to optimize accuracy for different forecast horizons.
*   **Prophet**: Combines the GBM predictions with seasonality and holiday effects for the final output.

## Dependencies

*   Python 3.8+
*   See `requirements.txt` for package versions. Note that `meteostat==1.6.5` and `numpy==1.26.4` are pinned for compatibility.
