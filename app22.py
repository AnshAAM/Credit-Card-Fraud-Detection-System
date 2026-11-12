
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Fraud Detection System", layout="wide")
st.title("💳 Credit Card Fraud Detection System")

menu = st.sidebar.radio("Navigation", ["Upload", "EDA", "Prediction", "Model Info"])

# ---------------- LOAD MODEL ----------------
try:
    model = joblib.load("fraud_detection_model.pkl")
    scaler = joblib.load("fraud_scaler.pkl")
    features = joblib.load("fraud_features.pkl")
    model_loaded = True
except Exception as error:
    model_loaded = False
    st.error(f"Model loading error: {error}")

# ---------------- SESSION ----------------
if "df" not in st.session_state:
    st.session_state.df = None

# ---------------- PREPROCESS FUNCTION ----------------
def preprocess_full(df):
    df_clean = df.copy()

    required_columns = features + ['Time', 'Class']

    missing_columns = [column for column in required_columns if column not in df_clean.columns]

    if missing_columns:
        raise ValueError(f"Missing columns in dataset: {missing_columns}")

    X = df_clean[features]

    X_scaled = scaler.transform(X)

    X_scaled = pd.DataFrame(X_scaled, columns=features)

    X_scaled['Class'] = df_clean['Class'].values

    return X_scaled

# ---------------- UPLOAD ----------------
if menu == "Upload":
    st.header("📂 Upload Dataset")

    file = st.file_uploader("Upload creditcard.csv", type=["csv"])

    if file:
        try:
            df = pd.read_csv(file)

            st.session_state.df = df

            st.success("Dataset Loaded Successfully")
            st.write("Dataset Shape:", df.shape)
            st.dataframe(df.head())

        except Exception as error:
            st.error(f"Error loading dataset: {error}")

# ---------------- EDA ----------------
elif menu == "EDA":
    st.header("📊 Exploratory Data Analysis")

    df = st.session_state.df

    if df is None:
        st.warning("Upload dataset first")

    elif 'Class' not in df.columns or 'Amount' not in df.columns:
        st.error("Dataset must contain Class and Amount columns")

    else:
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots()

            sns.countplot(x='Class', data=df, ax=ax)

            ax.set_title("Fraud vs Normal Transactions")
            ax.set_xlabel("Transaction Class")
            ax.set_ylabel("Count")

            st.pyplot(fig)
            plt.close(fig)

        with col2:
            fig2, ax2 = plt.subplots()

            sns.histplot(df['Amount'], bins=50, ax=ax2)

            ax2.set_title("Transaction Amount Distribution")
            ax2.set_xlabel("Amount")

            st.pyplot(fig2)
            plt.close(fig2)

# ---------------- MODEL INFO ----------------
elif menu == "Model Info":
    st.header("🤖 Model Details")

    st.markdown("""
    **Model Used:** Random Forest Classifier

    **Features:** V1–V28 (PCA transformed) + Amount

    **Total Features:** 29

    **Key Focus:** Fraud detection using pattern recognition
    """)

# ---------------- PREDICTION ----------------
elif menu == "Prediction":
    st.header("🔍 Fraud Detection")

    if not model_loaded:
        st.error("Model files could not be loaded. Check that all three .pkl files are in the project folder.")

    else:
        df = st.session_state.df

        if df is None:
            st.warning("Upload dataset first")

        else:
            try:
                df_scaled = preprocess_full(df)

                option = st.radio(
                    "Choose Prediction Mode",
                    ["Random Transaction", "Real Fraud Example", "Manual Input"]
                )

                sample = None

                # -------- INPUT SELECTION --------
                if option == "Random Transaction":

                    sample = df_scaled.drop(columns=['Class']).sample(1)

                    st.info("Random transaction selected")

                elif option == "Real Fraud Example":

                    fraud_samples = df_scaled[df_scaled['Class'] == 1]

                    if len(fraud_samples) > 0:

                        sample = fraud_samples.sample(1).drop(columns=['Class'])

                        st.warning("Real fraud sample loaded 🔥")

                    else:
                        st.error("No fraud data found in the uploaded dataset")

                else:
                    st.warning("Enter scaled feature values for manual prediction.")

                    input_data = {}
                    cols = st.columns(4)

                    for i, feature in enumerate(features):

                        with cols[i % 4]:
                            input_data[feature] = st.number_input(
                                feature,
                                value=0.0
                            )

                    sample = pd.DataFrame([input_data], columns=features)

                # -------- PREDICT --------
                if sample is not None:

                    if st.button("🔍 Analyze Transaction"):

                        X_input = sample[features]

                        prediction = model.predict(X_input)[0]

                        prob = model.predict_proba(X_input)[0][1]

                        st.subheader("📊 Prediction Result")

                        st.write(f"**Fraud Probability:** {prob * 100:.2f}%")

                        if prediction == 1:
                            st.error("🚨 Fraudulent Transaction Detected")
                        else:
                            st.success("✅ Legitimate Transaction")

                        # -------- INTERPRETATION --------
                        st.markdown("### 🧠 Interpretation")

                        if prob > 0.7:
                            st.write("High risk → likely fraud")

                        elif prob > 0.4:
                            st.write("Moderate risk")

                        else:
                            st.write("Low risk")

            except Exception as error:
                st.error(f"Prediction error: {error}")
