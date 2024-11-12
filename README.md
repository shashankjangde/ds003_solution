# ds003_solution 🎛️
This repository contains a Predictive Maintenance solution, developed as a submission for an internship application at Solar Group within the Data Science domain.

---

## 🚀 Problem Statement
### **Predictive Maintenance for Manufacturing Equipment**
Predictive maintenance is critical for improving equipment reliability in manufacturing environments, allowing companies to anticipate equipment failures and minimize unexpected downtime. This project presents a predictive maintenance (PdM) system using data-driven methods, designed to forecast failures before they occur, thus reducing maintenance costs and prolonging equipment life.

---

## 👥 Group: SIIL-TEAM-0002
### Team Members:
1. **Shashank Jangde**
2. **Vidit Singh**
3. **Sushant Pandey**
4. **Yash Mishra**

---

## 🛠️ Project Description
In the manufacturing industry, ensuring equipment reliability directly impacts productivity, safety, and costs. This predictive maintenance project focuses on creating a machine learning model to forecast equipment failures, enabling proactive maintenance. Traditional reactive maintenance (post-failure repair) or scheduled maintenance (regular time-based) is costly and often ineffective. This PdM system leverages **sensor data** and **machine learning models** to provide real-time failure predictions, helping companies reduce unplanned downtime, improve equipment lifespan, and optimize maintenance costs.

### Objectives:
1. **Develop a predictive model** to forecast equipment failures.
2. **Create a user-friendly interface** for real-time equipment monitoring.
3. **Ensure seamless integration** of the PdM system into existing manufacturing environments.

### Key Features:
- **LSTM Model**: Ideal for time-series data to predict Remaining Useful Life (RUL).
- **Data Visualization**: Interactive dashboard with Streamlit to display model results and insights.
- **Adaptability**: Designed to handle diverse manufacturing environments and operating conditions.

---

## 📂 Methodology
### **1. Tools and Technologies**
- **Programming Language**: Python
- **Data Handling and Preprocessing**: `pandas`, `NumPy`
- **Data Normalization and Transformation**: `sklearn.preprocessing.StandardScaler`
- **Dimensionality Reduction**: `sklearn.decomposition.PCA`
- **Machine Learning Model**: LSTM with `Keras` and `TensorFlow`
- **Model Evaluation**: `mean_squared_error`, `r2_score` (sklearn)
- **Visualization**: `matplotlib`, `seaborn`
- **Data Visualization and UI**: `Streamlit`

### **2. Data Source**
The **NASA Turbofan Jet Engine Data Set** (CMAPSS) was selected for its robustness in representing equipment with multiple failure modes and complex operational conditions. This dataset includes vital sensor data necessary for RUL prediction.

### **3. Model Selection Rationale**
The **LSTM model** was chosen for its ability to handle sequence prediction in time-series data. Using **Principal Component Analysis (PCA)** further optimized model performance, helping to focus on the most relevant features and reducing computational costs.

### **4. Development Workflow**
- **Data Preprocessing**: Scaling and dimensionality reduction.
- **Model Training**: LSTM network optimized for time-series prediction.
- **Evaluation**: Performance measured using RUL predictions and evaluated with metrics like `mean_squared_error`.

### **5. Project Structure and Steps**
- **Step 1**: Dataset exploration and preprocessing.
- **Step 2**: Model design and prototyping.
- **Step 3**: Training, tuning, and model evaluation.
- **Step 4**: UI design and integration for real-time insights.

---

## 📊 Results
The PdM system is capable of:
- Predicting **Remaining Useful Life (RUL)** accurately within a reasonable margin.
- Integrating real-time **sensor readings** with environmental settings (altitude, throttle).
- Adapting to **multiple failure modes** and operational conditions for versatile application.

Performance comparisons:
- **LSTM**: Best fit for sequential, time-series data.
- **Random Forest**: Evaluated but found less effective for time-series prediction.

---

## 🔧 Setup Instructions
To set up the project locally, follow these steps:
1. git clone this repository locally.
2. create a virtual environment in python.
3. activate the virtual environment.
4. install all the necessary libraries using "pip install -r requirements.txt"
5. once the project is set up properly and you have successfully installed all the dependencies and activated the environment, type the below command to run the project:
   $stremlit run app.py
6. You'll be directed to chrome browser where you can use the project.
