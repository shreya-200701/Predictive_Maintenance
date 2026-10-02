# ✈️ AI-Based Turbofan Predictive Maintenance

An AI-based Predictive Maintenance system that uses **XGBoost Machine Learning** to predict the **Remaining Useful Life (RUL)** of turbofan engines using the NASA C-MAPSS FD001 dataset.

The project includes a trained XGBoost regression model, sensor-based analysis, feature engineering, model explainability using SHAP, and an interactive Streamlit dashboard.

---

## 🚀 Live Demo

🌐 **Streamlit App:**  
https://predictivemaintenance-tclxcw4my2zakesw5mntw.streamlit.app/

---

## 📌 Project Overview

Predictive maintenance uses machine learning to estimate when equipment may require maintenance before failure occurs.

In this project, turbofan engine sensor data is analyzed to estimate the **Remaining Useful Life (RUL)** of an engine.

The system:

1. Processes turbofan engine sensor data
2. Performs feature engineering
3. Creates rolling statistical features
4. Trains an XGBoost regression model
5. Predicts Remaining Useful Life
6. Classifies the current engine condition
7. Visualizes sensor behavior
8. Provides feature importance
9. Uses SHAP to explain individual predictions
10. Provides an interactive Streamlit dashboard

---

## 🧠 Machine Learning Model

### XGBoost Regressor

The project uses **XGBoost (Extreme Gradient Boosting)** as the main machine learning algorithm.

The task is treated as a **regression problem** because the model predicts a numerical RUL value.

### Model Objective

The model learns the relationship between:

**Engine sensor measurements → Remaining Useful Life**

The predicted output is:

```text
Remaining Useful Life (RUL)
