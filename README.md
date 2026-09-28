# Customer Churn Prediction Dashboard

An interactive Streamlit dashboard that predicts telecom customer churn and compares three machine learning models.

## Features
- CSV upload (or uses the bundled `churn.csv`)
- Data cleaning: removes target-leakage and high-cardinality location columns, encodes categoricals
- Compares Logistic Regression, Decision Tree and Random Forest
- Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Confusion matrix, ROC curves, churn rate by contract type, top feature importances

## Results (7,043 customers, 80/20 split, seed 42)
| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.852 | 0.710 | 0.911 |
| Random Forest | 0.847 | 0.673 | 0.901 |
| Decision Tree | 0.821 | 0.592 | 0.876 |

Key insight: month-to-month contracts churn at 45.8% vs 2.5% for two-year contracts.

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Streamlit, Matplotlib, Seaborn

## How to Run
pip install -r requirements.txt
streamlit run app.py
