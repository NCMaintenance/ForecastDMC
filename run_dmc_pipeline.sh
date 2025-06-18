#!/bin/bash

# Step 1: Define the input file URL and local name
RAW_FILE_URL="https://raw.githubusercontent.com/NCMaintenance/ForecastDMC/feat/forecast-additional-capacity/may.xlsx"
INPUT_FILE="may.xlsx" # Local name to save the file

echo "INFO: Attempting to download ${INPUT_FILE} from ${RAW_FILE_URL}..."
curl -L -o "${INPUT_FILE}" "${RAW_FILE_URL}"
echo "INFO: Download command finished."

# Step 2: Check if download was successful
DOWNLOAD_SUCCESSFUL=false
if [ -f "${INPUT_FILE}" ] && [ -s "${INPUT_FILE}" ]; then
    DOWNLOAD_SUCCESSFUL=true
    echo "SUCCESS: File ${INPUT_FILE} downloaded successfully."
    echo "INFO: File size: $(wc -c < ${INPUT_FILE}) bytes."
else
    echo "ERROR: Failed to download ${INPUT_FILE} from the URL, or the file is empty."
    if [ -f "${INPUT_FILE}" ]; then
        echo "INFO: Downloaded file ${INPUT_FILE} exists but might be empty or incorrect."
        echo "INFO: File size: $(wc -c < ${INPUT_FILE}) bytes."
        echo "INFO: Content of the downloaded file (first 10 lines):"
        head -n 10 "${INPUT_FILE}"
        if grep -q -i "html" "${INPUT_FILE}"; then
            echo "ERROR: The downloaded file appears to be an HTML page."
            if grep -q -i "404: Not Found" "${INPUT_FILE}"; then
                 echo "ERROR: This looks like a 404 error. Please double check the branch name and file path."
            fi
        fi
    else
        echo "ERROR: Downloaded file ${INPUT_FILE} does not exist."
    fi
fi

if [ "$DOWNLOAD_SUCCESSFUL" = true ]; then
    # Step 3: Modify DMC.py to address warnings
    echo "INFO: Modifying DMC.py to address fillna FutureWarning..."
    # Corrected sed command:
    # The original line is: df[feature] = df[feature].fillna(method='ffill').fillna(method='bfill').fillna(0)
    # The target is:      df[feature] = df[feature].ffill().bfill().fillna(0)
    if [ -f "DMC.py" ]; then
        sed -i "s/df\[feature\].fillna(method='ffill').fillna(method='bfill').fillna(0)/df[feature].ffill().bfill().fillna(0)/g" DMC.py
        echo "INFO: Applied fix for fillna FutureWarning to DMC.py."
    else
        echo "WARNING: DMC.py not found in the current directory. Cannot apply sed fix."
    fi

    # Step 4: Ensure core dependencies are installed
    echo "COMMAND: Installing dependencies: pandas, openpyxl, lightgbm, scikit-learn..."
    pip install pandas openpyxl lightgbm scikit-learn
    echo "COMMAND: Dependency installation command finished."

    # Step 5: Run the DMC.py script with the downloaded data file
    echo "COMMAND: Running DMC.py --file_path ${INPUT_FILE}..."
    python DMC.py --file_path "${INPUT_FILE}"
    echo "COMMAND: DMC.py execution finished."

    echo "INFO: Subtask finished. Review console output for MAE results using ${INPUT_FILE}."
else
    echo "INFO: Subsequent steps skipped due to download failure."
fi
