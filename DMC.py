import streamlit as st
import pandas as pd
from data_processor import process_data # Import the function from data_processor.py
import plotly.express as px

# Page Configuration
st.set_page_config(layout="wide", page_title="Hospital Data Analysis Platform")

st.title("🏥 Hospital Data Analysis Platform")
st.markdown("Upload your Excel data, view transformations, and explore visualizations.")

# --- File Uploader ---
st.sidebar.header("1. Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Choose an Excel file", type=["xlsx", "xls"])

# Initialize session state variables
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'original_df_name' not in st.session_state:
    st.session_state.original_df_name = ""

if uploaded_file is not None:
    try:
        st.sidebar.success(f"File '{uploaded_file.name}' uploaded successfully!")
        original_df = pd.read_excel(uploaded_file)
        st.session_state.original_df_name = uploaded_file.name

        # Process data when file is uploaded or re-uploaded
        with st.spinner("Processing data... This may take a moment."):
            st.session_state.processed_df = process_data(original_df.copy()) # Use .copy()

        if st.session_state.processed_df is None or st.session_state.processed_df.empty:
            st.error("Data processing failed or returned an empty DataFrame. Please check the console for errors from `data_processor.py` or ensure your data format is correct.")
            # Keep the previous data if processing new upload fails, unless it's the first upload.
            if st.session_state.original_df_name != uploaded_file.name : # if it's a new file that failed
                 st.session_state.processed_df = None
        else:
            st.success("Data processed successfully!")

    except Exception as e:
        st.error(f"An error occurred while reading or processing the file: {e}")
        st.session_state.processed_df = None # Clear data on error
else:
    st.sidebar.info("Please upload an Excel file to begin analysis.")

# --- Main Panel Display ---
if st.session_state.processed_df is not None and not st.session_state.processed_df.empty:
    st.header(f"Processed Data from: `{st.session_state.original_df_name}`")

    # --- Display Options ---
    st.subheader("View Data")
    show_raw_data = st.checkbox("Show full processed data table", False)
    if show_raw_data:
        st.dataframe(st.session_state.processed_df)
    else:
        st.dataframe(st.session_state.processed_df.head())

    st.subheader("Summary Statistics")
    st.text("Overall summary statistics for the 'Value' column (includes ED and Trolley counts):")
    st.dataframe(st.session_state.processed_df['Value'].describe())

    st.markdown("---")
    st.header("📊 Visualizations")

    # --- Filters for Visualization ---
    st.sidebar.header("2. Visualization Filters")

    # Get available filter options from the processed data
    hospital_group_options = ['All'] + sorted(st.session_state.processed_df['Hospital Group Name'].unique().tolist())
    selected_hospital_group = st.sidebar.selectbox("Select Hospital Group", hospital_group_options)

    if selected_hospital_group == 'All':
        hospital_options_df = st.session_state.processed_df
    else:
        hospital_options_df = st.session_state.processed_df[st.session_state.processed_df['Hospital Group Name'] == selected_hospital_group]

    hospital_options = ['All'] + sorted(hospital_options_df['Hospital'].unique().tolist())
    selected_hospital = st.sidebar.selectbox("Select Hospital", hospital_options)

    metric_options = ['All'] + sorted(st.session_state.processed_df['Metric'].unique().tolist())
    selected_metric = st.sidebar.selectbox("Select Metric (ED/Trolley)", metric_options)

    time_label_options = ['All'] + sorted(st.session_state.processed_df['TimeLabel'].unique().tolist())
    selected_time_label = st.sidebar.selectbox("Select Time Label (8am, 2pm, 8pm)", time_label_options)


    # --- Apply Filters ---
    filtered_df = st.session_state.processed_df.copy()
    if selected_hospital_group != 'All':
        filtered_df = filtered_df[filtered_df['Hospital Group Name'] == selected_hospital_group]
    if selected_hospital != 'All':
        filtered_df = filtered_df[filtered_df['Hospital'] == selected_hospital]
    if selected_metric != 'All':
        filtered_df = filtered_df[filtered_df['Metric'] == selected_metric]
    if selected_time_label != 'All':
        filtered_df = filtered_df[filtered_df['TimeLabel'] == selected_time_label]

    # --- Line Chart ---
    if not filtered_df.empty:
        st.subheader("Values Over Time")
        fig = px.line(
            filtered_df,
            x='Datetime',
            y='Value',
            color='Metric', # Color lines by Metric (ED/Trolley)
            line_dash='TimeLabel', # Dash lines by TimeLabel (8am, 2pm, 8pm)
            hover_name='Hospital',
            labels={'Value': 'Count', 'Datetime': 'Date and Time'},
            title=f"Counts for {selected_hospital if selected_hospital != 'All' else 'Selected Hospitals'} - {selected_metric if selected_metric != 'All' else 'All Metrics'}"
        )
        fig.update_layout(xaxis_title="Date and Time", yaxis_title="Value (Count)")
        st.plotly_chart(fig, use_container_width=True)

        # --- Chart for Lag and Rolling Features ---
        if selected_metric != 'All' and selected_time_label != 'All' and selected_hospital != 'All': # Most granular view
            st.subheader(f"Lag and Rolling Mean Analysis for {selected_hospital} - {selected_metric} at {selected_time_label}")

            # Ensure no NaNs for plotting these specific features, or handle them
            plot_df_enhanced = filtered_df[['Datetime', 'Value', 'Lag_1Day_Value', 'Rolling_Mean_7D_Value']].copy()
            plot_df_enhanced = plot_df_enhanced.dropna(subset=['Lag_1Day_Value', 'Rolling_Mean_7D_Value'], how='all')

            if not plot_df_enhanced.empty:
                fig_enhanced = px.line(
                    plot_df_enhanced.melt(id_vars=['Datetime'], var_name='Feature', value_name='Count'),
                    x='Datetime',
                    y='Count',
                    color='Feature',
                    title=f"Actual vs. Lag (1D) vs. Rolling Mean (7D) for {selected_hospital} - {selected_metric} - {selected_time_label}"
                )
                st.plotly_chart(fig_enhanced, use_container_width=True)
            else:
                st.warning("Not enough data to display lag/rolling mean chart for the current selection (requires at least 2 days for lag, and some data for rolling mean).")

    elif st.session_state.processed_df is not None: # if processed_df exists but filtered_df is empty
        st.warning("No data matches the current filter selection.")

else:
    if uploaded_file is not None: # If file was uploaded but processing failed
        pass # Error message already shown
    else: # Initial state, no file uploaded yet
        st.info("Upload an Excel file using the sidebar to see the analysis.")

st.sidebar.markdown("---")
st.sidebar.markdown("Developed by AI Agent Jules.")
