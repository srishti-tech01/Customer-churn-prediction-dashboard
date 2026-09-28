"""Customer Churn Prediction Dashboard (Streamlit)
Compares Logistic Regression, Decision Tree and Random Forest, then selects the best model by ROC-AUC.
Run:  streamlit run app.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Churn Predictor", page_icon="📊", layout="wide")
st.markdown("<h1 style='text-align:center'>Customer Churn Prediction Dashboard</h1><hr>", unsafe_allow_html=True)

# Columns that leak the answer or are high-cardinality location fields
LEAKY = ["Customer Status", "Churn Score", "CLTV", "Total Revenue", "Satisfaction Score",
         "Churn Category", "Churn Reason"]
GEO = ["Country", "State", "City", "Zip Code", "Latitude", "Longitude", "Quarter"]


@st.cache_data
def prepare(df: pd.DataFrame):
    df = df.copy()
    df["Churn"] = df["Churn Label"].map({"Yes": 1, "No": 0})
    df = df.drop(columns=["Customer ID", "Churn Label"] + LEAKY + GEO, errors="ignore")
    df = df.dropna(subset=["Churn"])
    df = pd.get_dummies(df, drop_first=True).fillna(0)
    return df.drop(columns="Churn"), df["Churn"]


@st.cache_resource
def train_all(X: pd.DataFrame, y: pd.Series):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler().fit(X_tr)
    specs = {
        "Logistic Regression": (LogisticRegression(max_iter=1000), True),
        "Decision Tree": (DecisionTreeClassifier(max_depth=6, random_state=42), False),
        "Random Forest": (RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1), False),
    }
    results, fitted = [], {}
    for name, (model, scaled) in specs.items():
        A, B = (scaler.transform(X_tr), scaler.transform(X_te)) if scaled else (X_tr, X_te)
        model.fit(A, y_tr)
        pred, prob = model.predict(B), model.predict_proba(B)[:, 1]
        results.append({"Model": name,
                        "Accuracy": accuracy_score(y_te, pred),
                        "Precision": precision_score(y_te, pred),
                        "Recall": recall_score(y_te, pred),
                        "F1-Score": f1_score(y_te, pred),
                        "ROC-AUC": roc_auc_score(y_te, prob)})
        fitted[name] = (pred, prob)
    return pd.DataFrame(results), fitted, y_te, specs["Random Forest"][0]


# ---------------- data input ----------------
up = st.sidebar.file_uploader("Upload churn.csv", type=["csv"])
if up is not None:
    raw = pd.read_csv(up, encoding="latin-1")
elif os.path.exists("churn.csv"):
    raw = pd.read_csv("churn.csv", encoding="latin-1")
    st.sidebar.info("Using bundled churn.csv")
else:
    st.info("Please upload churn.csv from the sidebar to start.")
    st.stop()

st.subheader("Dataset Preview")
st.dataframe(raw.head())
X, y = prepare(raw)
metrics, fitted, y_te, rf_model = train_all(X, y)
best = metrics.sort_values("ROC-AUC", ascending=False).iloc[0]["Model"]

c1, c2, c3 = st.columns(3)
c1.metric("Total Records", f"{len(raw):,}")
c2.metric("Overall Churn Rate", f"{y.mean()*100:.1f}%")
c3.metric("Best Model (by ROC-AUC)", best)

st.subheader("Model Comparison")
st.dataframe(metrics.set_index("Model").style.format("{:.3f}").highlight_max(axis=0, color="#c8e6c9"))

left, right = st.columns(2)
with left:
    st.subheader(f"Confusion Matrix – {best}")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_te, fitted[best][0]), annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    st.pyplot(fig)
with right:
    st.subheader("ROC Curves")
    fig, ax = plt.subplots()
    for name, (_, prob) in fitted.items():
        fpr, tpr, _ = roc_curve(y_te, prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_te, prob):.2f})")
    ax.plot([0, 1], [0, 1], "k--"); ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.legend()
    st.pyplot(fig)

st.subheader("Business Insights: what drives churn?")
a, b = st.columns(2)
with a:
    if "Contract" in raw.columns:
        rate = raw.groupby("Contract")["Churn Label"].apply(lambda s: (s == "Yes").mean() * 100)
        fig, ax = plt.subplots(); rate.plot.bar(ax=ax, color="#4c72b0"); ax.set_ylabel("Churn rate (%)"); ax.set_title("Churn rate by contract type")
        plt.xticks(rotation=0); st.pyplot(fig)
with b:
    imp = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values().tail(10)
    fig, ax = plt.subplots(); imp.plot.barh(ax=ax, color="#dd8452"); ax.set_title("Top 10 features (Random Forest)")
    st.pyplot(fig)

st.success("Models trained successfully.")
