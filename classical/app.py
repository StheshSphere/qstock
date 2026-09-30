"""
QStock - Demo App
=========================
A minimal Streamlit UI so a committee member can:
  1. Enter/upload member data
  2. Run the QAOA payout scheduler
  3. Run the quantum risk classifier
  4. View results side-by-side with classical baselines

Run with: streamlit run classical/app.py
"""

import sys
import os
import streamlit as st
import pandas as pd
import numpy as np

# Allow importing from the quantum/ package
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from quantum.qaoa_scheduler import run_qaoa_scheduler, decode_schedule, classical_baseline
from quantum.qml_risk_classifier import (
    compute_quantum_kernel,
    train_qsvm,
    train_classical_baseline,
    make_synthetic_data,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="QStock", page_icon="\U0001FA99")
st.title("QStock")
st.caption("Quantum-optimized stokvel management \u2014 fair payouts, smarter risk tracking.")

tab1, tab2 = st.tabs(["Payout Scheduler", "Risk Classifier"])

# ---------------- Payout Scheduler Tab ----------------
with tab1:
    st.subheader("Fair Payout Rotation (QAOA)")
    st.write("Enter member names and an urgency score (0-1, higher = more urgent need).")

    default_data = pd.DataFrame(
        {
            "name": ["Thabo", "Naledi", "Sipho", "Amahle", "Katlego", "Zanele"],
            "urgency": [0.9, 0.2, 0.7, 0.1, 0.8, 0.3],
        }
    )
    edited = st.data_editor(default_data, num_rows="dynamic", key="scheduler_data")

    if st.button("Run QAOA Scheduler"):
        names = edited["name"].tolist()
        urgencies = edited["urgency"].tolist()

        with st.spinner("Running QAOA optimization..."):
            result = run_qaoa_scheduler(urgencies, reps=3)
            schedule = decode_schedule(result, names)
            baseline = classical_baseline(urgencies, names)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**QAOA Schedule**")
            st.write("Early payout:", schedule["early_payout"])
            st.write("Late payout:", schedule["late_payout"])
        with col2:
            st.markdown("**Classical Baseline (rank-split)**")
            st.write("Early payout:", baseline["early_payout"])
            st.write("Late payout:", baseline["late_payout"])

# ---------------- Risk Classifier Tab ----------------
with tab2:
    st.subheader("Member Risk Classification (Quantum Kernel SVM)")
    st.write("Using synthetic member data for this demo. Swap in real (anonymized) data in production.")

    n_samples = st.slider("Number of synthetic members", 10, 40, 20)

    if st.button("Run Risk Classifier"):
        with st.spinner("Computing quantum kernel and training classifiers..."):
            X, y = make_synthetic_data(n_samples=n_samples)
            # Small range keeps the ZZ kernel in the smooth, unwrapped regime
            # (see qml_risk_classifier.py); (0, pi) collapses all fidelities.
            X_scaled = MinMaxScaler(feature_range=(0, 0.25)).fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.3, random_state=42, stratify=y
            )

            qsvm_result = train_qsvm(X_train, y_train, X_test, y_test)
            baseline_result = train_classical_baseline(X_train, y_train, X_test, y_test)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("QSVM Accuracy", f"{qsvm_result['accuracy']:.2%}")
        with col2:
            st.metric("Classical RBF-SVM Accuracy", f"{baseline_result['accuracy']:.2%}")

        st.markdown("**QSVM Classification Report**")
        st.text(qsvm_result["report"])
        st.markdown("**Classical Baseline Report**")
        st.text(baseline_result["report"])

st.divider()
st.caption(
    "QStock \u2014 built for the ADAPT IT Social Good Hackathon, Quantum Computing Track."
)
