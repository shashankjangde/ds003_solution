# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
from preprocess import preprocess_input_data, postprocess_prediction

# Sidebar for navigation
st.sidebar.title("Predictive Maintenance Dashboard")
nav = st.sidebar.radio("Go to", ["Equipment Status", "Failure Predictions",
                       "Maintenance History", "Maintenance Recommendations"])

# Function to create dummy data for testing (to be replaced with actual data integration)


def generate_sample_data():
    return pd.DataFrame({
        'Unit Number': np.random.randint(1, 5, 50),
        'Cycles': np.random.randint(100, 200, 50),
        'Predicted RUL': np.random.normal(100, 20, 50),
        'Expected RUL': np.random.normal(120, 25, 50)
    })


# Main Dashboard Sections
if nav == "Equipment Status":
    st.title("Equipment Status")
    st.write(
        "Real-time status of each equipment unit, with predictive indicators for potential issues.")

    # Generate example data
    data = generate_sample_data()
    st.dataframe(data)

    # Equipment Status Visualization
    fig, ax = plt.subplots()
    sns.lineplot(data=data, x="Cycles", y="Predicted RUL",
                 hue="Unit Number", ax=ax)
    ax.set_title("Predicted RUL over Cycles")
    st.pyplot(fig)

elif nav == "Failure Predictions":
    st.title("Failure Predictions")
    st.write("Predictions for equipment failure based on real-time sensor data.")

    # Prediction input section
    st.subheader("Run a New Prediction")
    model = st.selectbox('Choose the Prediction Model:',
                         ("complete", "FD001", "FD002", "FD003", "FD004"))
    cycles = st.number_input(
        "Cycles", min_value=1, max_value=500, step=1)
    operational_settings = [st.number_input(
        f"Operational Setting {i+1}", 0.0, 1.0, 0.5) for i in range(3)]
    sensor_readings = [st.number_input(
        f"Sensor {i+1}", 0.0, 100.0, 50.0) for i in range(1, 22)]

    base_path = "original_models/"
    model_path = base_path + model + ".pkl"
    with open(model_path, 'rb') as f:
        model_obj = pickle.load(f)
    if st.button("Predict RUL"):

        input_data = list(map(float,[1] + [cycles] + operational_settings + sensor_readings))
        input_data = preprocess_input_data(input_data, model)
        prediction = model_obj.predict(input_data)
        prediction = postprocess_prediction(prediction)
        st.success(f"Predicted Remaining Useful Life (RUL): {
            prediction} cycles")

elif nav == "Maintenance History":
    st.title("Maintenance History")
    st.write("Past maintenance events and actions taken for each equipment unit.")

    # Sample maintenance history data
    history_data = pd.DataFrame({
        'Unit Number': np.random.randint(1, 5, 10),
        'Maintenance Date': pd.date_range(end=pd.Timestamp.today(), periods=10).tolist(),
        'Maintenance Type': np.random.choice(['Preventive', 'Corrective'], 10),
        'Details': ["Routine check-up", "Component replacement", "System reboot"] * 3 + ["Oil change"]
    })
    st.table(history_data)

elif nav == "Maintenance Recommendations":
    st.title("Maintenance Recommendations")
    st.write(
        "Upcoming recommended maintenance schedules based on current RUL predictions.")

    # Sample recommended maintenance data
    recommendations = pd.DataFrame({
        'Unit Number': np.arange(1, 6),
        'Next Maintenance Date': pd.date_range(start=pd.Timestamp.today(), periods=5).tolist(),
        'Recommended Actions': ["Inspect sensors", "Calibrate system", "Replace filters", "Check connections", "Lubricate moving parts"]
    })
    st.table(recommendations)

    st.subheader("Plot RUL Distribution")
    data = generate_sample_data()
    fig, ax = plt.subplots()
    sns.histplot(data["Predicted RUL"], kde=True, ax=ax)
    ax.set_title("Distribution of Predicted RUL")
    st.pyplot(fig)

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    "Predictive Maintenance Dashboard for Manufacturing Equipment.")
