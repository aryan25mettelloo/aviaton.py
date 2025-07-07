# Aviation Prediction System - Multi-Model AI Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
import xgboost as xgb
import streamlit as st

# --- Streamlit UI ---
st.set_page_config(page_title="Aviation Prediction System", layout="wide")
st.title("✈️ Aviation Prediction System")
st.markdown("This AI system predicts flight delays, cancellations, and risk levels.")

# --- Load local Excel file from OneDrive folder ---
try:
    df = pd.read_excel("flight_data.xlsx")
    df = df.dropna()

    st.success("✅ Dataset loaded from OneDrive Excel file!")
    st.subheader("Preview of Dataset")
    st.dataframe(df.head())

    features = ['Airline', 'Origin', 'Destination', 'ScheduledDepTime', 'Weather', 'DayOfWeek']
    target_delay = 'DelayMinutes'
    target_status = 'Delayed'

    df_encoded = pd.get_dummies(df[features])

    X = df_encoded
    y_delay = df[target_delay]
    y_status = df[target_status]

    X_train, X_test, y_delay_train, y_delay_test = train_test_split(X, y_delay, test_size=0.2, random_state=42)
    _, _, y_status_train, y_status_test = train_test_split(X, y_status, test_size=0.2, random_state=42)

    # --- Model 1: Delay Predictor ---
    delay_model = RandomForestRegressor(n_estimators=100, random_state=42)
    delay_model.fit(X_train, y_delay_train)
    delay_preds = delay_model.predict(X_test)
    mae = mean_absolute_error(y_delay_test, delay_preds)
    st.success(f"✅ Delay Prediction MAE: {mae:.2f} minutes")

    # --- Model 2: Delay Classification ---
    delay_clf = LogisticRegression(max_iter=1000)
    delay_clf.fit(X_train, y_status_train)
    status_preds = delay_clf.predict(X_test)
    acc = accuracy_score(y_status_test, status_preds)
    st.success(f"✅ On-time/Delayed Classification Accuracy: {acc:.2f}")

    # Optional: Cancellation Model
    if 'Cancelled' in df.columns:
        y_cancel = df['Cancelled']
        _, _, y_cancel_train, y_cancel_test = train_test_split(X, y_cancel, test_size=0.2, random_state=42)
        cancel_model = xgb.XGBClassifier()
        cancel_model.fit(X_train, y_cancel_train)
        cancel_preds = cancel_model.predict(X_test)
        cancel_acc = accuracy_score(y_cancel_test, cancel_preds)
        st.success(f"✅ Cancellation Prediction Accuracy: {cancel_acc:.2f}")

    # Optional: Crash Risk Model
    if 'CrashRisk' in df.columns:
        crash_features = ['WeatherSeverity', 'TechnicalIssue', 'PilotHours']
        if all(col in df.columns for col in crash_features):
            crash_model = RandomForestClassifier()
            crash_model.fit(df[crash_features], df['CrashRisk'])
            crash_preds = crash_model.predict(df[crash_features])
            st.success("✅ Crash Risk Prediction Completed")

except Exception as e:
    st.error(f"❌ Error loading dataset: {e}")

