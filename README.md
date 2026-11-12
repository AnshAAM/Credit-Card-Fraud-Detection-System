# Credit Card Fraud Detection System

A machine learning-based application that identifies potentially fraudulent credit card transactions using a Random Forest classification model. The project includes an interactive Streamlit dashboard for data analysis, transaction prediction, and model information.

## Project Overview

Credit card fraud is a major challenge in financial systems. This project uses machine learning to classify transactions as legitimate or potentially fraudulent based on transaction features.

The application provides an interactive interface where users can upload transaction datasets, explore the data, and test fraud predictions.

## Features

- **Dataset Upload:** Upload a credit card transaction dataset in CSV format.
- **Exploratory Data Analysis:** Visualize transaction distributions and fraud patterns.
- **Transaction Prediction:** Predict whether a transaction is legitimate or fraudulent.
- **Random Transaction Testing:** Select a transaction from the uploaded dataset for prediction.
- **Real Fraud Example:** Test the model using an actual fraudulent transaction from the dataset.
- **Manual Input:** Enter feature values to obtain a prediction.
- **Model Information:** View details about the machine learning model and its input features.

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

## Machine Learning Model

The project uses a **Random Forest Classifier** for binary classification.

The model uses 29 input features:
- V1 to V28
- Amount

A StandardScaler is used for feature scaling.

The trained model, scaler, and feature list are stored separately as pickle files.

## Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── app22.py
├── MLproject.py
├── fraud_detection_model.pkl
├── fraud_scaler.pkl
├── fraud_features.pkl
├── requirements.txt
├── README.md
└── screenshots/
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/Credit-Card-Fraud-Detection.git
```

### 2. Navigate to the project directory

```bash
cd Credit-Card-Fraud-Detection
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the application

```bash
python -m streamlit run app22.py
```

The application will open in your web browser.

## Dataset

The application requires a credit card transaction dataset in CSV format.

The uploaded dataset should contain:
- `Time`
- `Amount`
- `V1` to `V28`
- `Class`

The `Class` column represents the transaction label:
- `0` – Legitimate transaction
- `1` – Fraudulent transaction

**Note:** The dataset is not included in this repository. Obtain the dataset separately and upload it through the application.

## Model Files

The following pre-trained files are required to run the application:

| File | Purpose |
|---|---|
| `fraud_detection_model.pkl` | Trained Random Forest model |
| `fraud_scaler.pkl` | StandardScaler used for feature scaling |
| `fraud_features.pkl` | List of model input features |

## Future Improvements

- Add model evaluation metrics such as precision, recall, F1-score, and confusion matrix.
- Improve prediction visualization and reporting.
- Add support for additional machine learning algorithms.
- Enhance the dashboard with more detailed transaction analytics.

## Author

**Ansh Arora**

Computer Science Engineering Student

