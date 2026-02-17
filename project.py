import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Churn Predictor", page_icon="📊", layout="wide")

# ---------------- HEADER ----------------
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>Customer Churn Prediction Dashboard</h1>
    <hr>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("Options")
uploaded_file = st.sidebar.file_uploader("Upload churn.csv", type=["csv"])

# ---------------- MAIN ----------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding="latin-1")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -------- Data Cleaning --------
    if "Customer ID" in df.columns:
        df.drop("Customer ID", axis=1, inplace=True)

    if "Churn Label" in df.columns:
        df["Churn Label"] = df["Churn Label"].map({"Yes":1,"No":0})
        df.rename(columns={"Churn Label":"Churn"}, inplace=True)

    df.drop(["Churn Category","Churn Reason"], axis=1, inplace=True, errors="ignore")
    df.drop([
        "Customer Status",
        "Churn Score",
        "CLTV",
        "Total Revenue",
        "Satisfaction Score"
    ], axis=1, inplace=True, errors="ignore")

    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # -------- Split --------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -------- Model --------
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)

    # -------- METRICS --------
    col1, col2 = st.columns(2)
    col1.metric("Model Accuracy", f"{acc*100:.2f}%")
    col2.metric("Total Records", len(df))

    st.markdown("---")

    # -------- GRAPHS --------
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Churn Count")
        fig1, ax1 = plt.subplots()
        sns.countplot(x=y, ax=ax1)
        st.pyplot(fig1)

    with col4:
        if "Age" in df.columns:
            st.subheader("Age vs Churn")
            fig2, ax2 = plt.subplots()
            sns.boxplot(x=y, y=df["Age"], ax=ax2)
            st.pyplot(fig2)

    st.markdown("---")

    # -------- CONFUSION MATRIX --------
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, predictions)
    fig3, ax3 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax3)
    st.pyplot(fig3)

    st.success("Model Trained Successfully!")

else:
    st.info("Please upload churn.csv file from sidebar to start.")

# ---------------- FOOTER ----------------
st.markdown("""
    <hr>
    <p style='text-align: center;'>Made with ❤️ using Streamlit</p>
""", unsafe_allow_html=True)
  
