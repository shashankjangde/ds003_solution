import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle

# Define the folder and files for each model
modelDumpFolder = "original_models/"
modelDumpFiles = ['FD001.pkl', 'FD002.pkl',
                  'FD003.pkl', 'FD004.pkl', 'complete.pkl']

scaler_files = [f"pcascaler/scaler{i}" for i in range(1, 5)]
pca_file = "pcascaler/pca.pkl"

# Dictionary to store the loaded models
loadedModels = {}

# Load each model
for file in modelDumpFiles:
    model_path = modelDumpFolder + file
    with open(model_path, 'rb') as f:
        loadedModels[file] = pickle.load(f)

# Load scalers and PCA
scalers = [pickle.load(open(f"{scaler}.pkl", 'rb')) for scaler in scaler_files]
pca = pickle.load(open(pca_file, 'rb'))


def preprocess_input_data(input_data, model, window_size=10):
    """
    Preprocesses the input data to be compatible with the model.

    Args:
    - input_data (list of floats): List of values including unit number, cycle, op settings, and sensor data.
    - window_size (int): Window size used during training for LSTM input.

    Returns:
    - processed_data (ndarray): Preprocessed data compatible with the model.
    """
    # Convert input data to a DataFrame format
    input_df = pd.DataFrame([input_data], columns=["Unit Number", "Cycles(Time)"] +
                            [f"operation_setting{i}" for i in range(1, 4)] +
                            [f"sensor{i}" for i in range(1, 22)])

    # Select a scaler (e.g., first one in list for simplicity, adjust as needed)

    scaler = scalers[0]  # or use logic to choose based on the model type

    # Scale and apply PCA
    scaled_data = scaler.transform(input_df.iloc[:, 2:])
    pca_data = pca.transform(scaled_data)

    # Repeat the PCA-transformed data to match the window size for LSTM model compatibility
    processed_data = np.repeat(pca_data, window_size, axis=0).reshape(
        (1, window_size, pca_data.shape[1]))

    return processed_data


def postprocess_prediction(predicted_rul):
    """
    Converts the model's predicted RUL to an integer format.

    Args:
    - predicted_rul (ndarray or float): The model's raw RUL prediction.

    Returns:
    - int_rul (int): Predicted RUL as an integer.
    """
    # Ensure the predicted RUL is a single value if ndarray
    if isinstance(predicted_rul, np.ndarray):
        predicted_rul = predicted_rul.item()

    # Round the predicted RUL to the nearest integer and ensure it's non-negative
    int_rul = max(int(round(predicted_rul)), 0)

    return int_rul


# Example input
# inputi = "1 1 -0.0007 -0.0004 100.0 518.67 641.82 1589.70 1400.60 14.62 21.61 554.36 2388.06 9046.19 1.30 47.47 521.66 2388.02 8138.62 8.4195 0.03 392 2388 100.00 39.06 23.4190"
# input_data = list(map(float, inputi.split()))

# # Preprocess the input data
# processed_data = preprocess_input_data(input_data, x=0)

# # Make prediction
# model = loadedModels["FD001.pkl"]
# prediction = model.predict(processed_data)

# # Process prediction to integer
# int_prediction = postprocess_prediction(prediction)

# print("Processed Data:", processed_data)
# print("Integer Prediction:", int_prediction)
