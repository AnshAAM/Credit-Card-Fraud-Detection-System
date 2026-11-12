import pandas as pd
import numpy as np
import joblib

# -------- LOAD MODEL --------
try:
    model = joblib.load("fraud_detection_model.pkl")
    scaler = joblib.load("fraud_scaler.pkl")
    features = joblib.load("fraud_features.pkl")
    print("Model Loaded Successfully\n")
except:
    print("Error: Model files not found")
    exit()

# -------- LOAD DATASET --------
file_path = input("Enter dataset path (creditcard.csv): ")

try:
    df = pd.read_csv(file_path)
    print("Dataset Loaded Successfully\n")
except:
    print("Error loading dataset")
    exit()

# -------- PREPROCESS FUNCTION --------
def preprocess(df):
    df_clean = df.copy()

    if 'Time' in df_clean.columns:
        df_clean = df_clean.drop(['Time'], axis=1)

    X = df_clean.drop('Class', axis=1)
    X_scaled = scaler.transform(X)

    return X_scaled

# -------- MENU --------
while True:
    print("\n===== MENU =====")
    print("1. Random Transaction")
    print("2. Real Fraud Example")
    print("3. Exit")

    choice = input("Enter choice: ")

    # -------- RANDOM TRANSACTION --------
    if choice == '1':
        df_scaled = preprocess(df)

        idx = np.random.randint(0, len(df))

        original_row = df.iloc[idx]
        sample = df_scaled[idx].reshape(1, -1)

        print("\n--- Random Transaction Values ---")
        for col in df.columns:
            if col != "Class":
                print(f"{col}: {original_row[col]}")

        pred = model.predict(sample)[0]
        prob = model.predict_proba(sample)[0][1]

        print("\nFraud Probability:", round(prob*100, 2), "%")

        if pred == 1:
            print("🚨 Fraudulent Transaction")
        else:
            print("✅ Legitimate Transaction")

    # -------- FRAUD EXAMPLE --------
    elif choice == '2':
        fraud_df = df[df['Class'] == 1]

        if len(fraud_df) == 0:
            print("No fraud data found")
        else:
            sample_row = fraud_df.sample(1)
            sample_scaled = preprocess(sample_row)

            print("\n--- Fraud Transaction Values ---")
            for col in sample_row.columns:
                if col != "Class":
                    print(f"{col}: {sample_row.iloc[0][col]}")

            pred = model.predict(sample_scaled)[0]
            prob = model.predict_proba(sample_scaled)[0][1]

            print("\nFraud Probability:", round(prob*100, 2), "%")
            print("🚨 Real Fraud Example Tested")

    elif choice == '3':
        print("Exiting...")
        break

    else:
        print("Invalid choice")