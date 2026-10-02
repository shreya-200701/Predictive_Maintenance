
# ============================================================
# AI-BASED PREDICTIVE MAINTENANCE SYSTEM
# NASA C-MAPSS FD001 + XGBoost
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Turbofan Predictive Maintenance",
    page_icon="✈️",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
    }

    .subtitle {
        font-size: 20px;
        color: #666666;
    }

    .health-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "xgboost_turbofan_rul_model.pkl"
    )

    return model


@st.cache_data
def load_feature_names():

    feature_names = joblib.load(
        "feature_names.pkl"
    )

    return feature_names


@st.cache_data
def load_test_data():

    df = pd.read_csv(
        "test_features.csv"
    )

    return df


model = load_model()

feature_names = load_feature_names()

test_features = load_test_data()


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">'
    '✈️ AI-Based Turbofan Predictive Maintenance'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'NASA C-MAPSS FD001 + XGBoost Remaining Useful Life Prediction'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Control Panel")

st.sidebar.markdown(
    """
    Select a turbofan engine to analyze its
    current health and predicted Remaining
    Useful Life (RUL).
    """
)


# Get available engines

engine_ids = sorted(
    test_features["engine_id"].unique()
)


selected_engine = st.sidebar.selectbox(
    "Select Engine",
    engine_ids
)


# ============================================================
# GET SELECTED ENGINE
# ============================================================

engine_data = test_features[
    test_features["engine_id"]
    == selected_engine
].copy()


# Sort by operating cycle

engine_data = engine_data.sort_values(
    "cycle"
)


# Last available observation

current_data = engine_data.tail(1).copy()


# ============================================================
# PREPARE MODEL INPUT
# ============================================================

X_current = current_data.drop(
    columns=["engine_id"]
)


# Make sure feature order is exactly
# the same as during training.

X_current = X_current[
    feature_names
]


# ============================================================
# PREDICTION
# ============================================================

predicted_rul = model.predict(
    X_current
)[0]


# RUL cannot physically be negative

predicted_rul = max(
    0,
    predicted_rul
)


# ============================================================
# HEALTH CLASSIFICATION
# ============================================================

if predicted_rul > 50:

    status = "🟢 HEALTHY"

    status_message = (
        "The engine has a relatively high "
        "predicted remaining useful life."
    )

elif predicted_rul > 20:

    status = "🟡 WARNING"

    status_message = (
        "The engine is approaching a maintenance "
        "condition. Schedule inspection."
    )

else:

    status = "🔴 CRITICAL"

    status_message = (
        "The predicted remaining useful life "
        "is low. Maintenance should be prioritized."
    )


# ============================================================
# TOP METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Engine ID",
        int(selected_engine)
    )


with col2:

    st.metric(
        "Current Cycle",
        int(current_data["cycle"].iloc[0])
    )


with col3:

    st.metric(
        "Predicted RUL",
        f"{predicted_rul:.1f} cycles"
    )


with col4:

    st.metric(
        "Health Status",
        status
    )


st.divider()


# ============================================================
# MAINTENANCE ALERT
# ============================================================

st.subheader(
    "🔧 Maintenance Assessment"
)

if predicted_rul > 50:

    st.success(
        f"""
        **{status}**

        {status_message}

        Predicted remaining useful life:
        **{predicted_rul:.1f} cycles**
        """
    )

elif predicted_rul > 20:

    st.warning(
        f"""
        **{status}**

        {status_message}

        Predicted remaining useful life:
        **{predicted_rul:.1f} cycles**
        """
    )

else:

    st.error(
        f"""
        **{status}**

        {status_message}

        Predicted remaining useful life:
        **{predicted_rul:.1f} cycles**
        """
    )


# ============================================================
# RUL GAUGE / BAR
# ============================================================

st.subheader(
    "📊 Remaining Useful Life"
)

# Maximum display value

max_display_rul = max(
    100,
    predicted_rul
)

rul_percentage = min(
    predicted_rul / max_display_rul,
    1.0
)

st.progress(
    float(rul_percentage)
)

st.write(
    f"Estimated remaining life: "
    f"**{predicted_rul:.1f} cycles**"
)


# ============================================================
# ENGINE INFORMATION
# ============================================================

st.divider()

st.subheader(
    "📋 Engine Information"
)


info_col1, info_col2 = st.columns(2)


with info_col1:

    st.write(
        f"**Engine ID:** {selected_engine}"
    )

    st.write(
        f"**Current cycle:** "
        f"{int(current_data['cycle'].iloc[0])}"
    )

    st.write(
        f"**Number of observations:** "
        f"{len(engine_data)}"
    )


with info_col2:

    st.write(
        f"**Predicted RUL:** "
        f"{predicted_rul:.2f} cycles"
    )

    st.write(
        f"**Maintenance status:** "
        f"{status}"
    )


# ============================================================
# SENSOR DATA
# ============================================================

st.divider()

st.subheader(
    "📈 Engine Sensor History"
)


sensor_columns = [
    col
    for col in engine_data.columns
    if col.startswith("sensor_")
    and "_mean_" not in col
    and "_std_" not in col
]


if len(sensor_columns) > 0:

    selected_sensor = st.selectbox(
        "Select sensor to visualize",
        sensor_columns
    )

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        engine_data["cycle"],
        engine_data[selected_sensor]
    )

    ax.set_xlabel(
        "Operating Cycle"
    )

    ax.set_ylabel(
        selected_sensor
    )

    ax.set_title(
        f"{selected_sensor} — Engine {selected_engine}"
    )

    ax.grid(True)

    st.pyplot(fig)


# ============================================================
# MULTIPLE SENSOR VIEW
# ============================================================

st.subheader(
    "📊 Sensor Overview"
)


selected_sensors = st.multiselect(
    "Choose sensors",
    sensor_columns,
    default=sensor_columns[:3]
)


if len(selected_sensors) > 0:

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    for sensor in selected_sensors:

        # Normalize sensors because they can
        # have very different scales.

        values = engine_data[sensor]

        normalized = (
            values - values.min()
        ) / (
            values.max() - values.min()
            + 1e-9
        )

        ax.plot(
            engine_data["cycle"],
            normalized,
            label=sensor
        )

    ax.set_xlabel(
        "Operating Cycle"
    )

    ax.set_ylabel(
        "Normalized Sensor Value"
    )

    ax.set_title(
        "Normalized Sensor Trends"
    )

    ax.legend()

    ax.grid(True)

    st.pyplot(fig)


# ============================================================
# CURRENT SENSOR VALUES
# ============================================================

st.divider()

st.subheader(
    "🔍 Current Sensor Readings"
)


current_sensor_values = current_data[
    sensor_columns
].T

current_sensor_values.columns = [
    "Current Value"
]

st.dataframe(
    current_sensor_values,
    use_container_width=True
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.divider()

st.subheader(
    "🎯 XGBoost Feature Importance"
)


importance = pd.Series(
    model.feature_importances_,
    index=feature_names
)

importance = (
    importance
    .sort_values(
        ascending=False
    )
    .head(15)
)


fig, ax = plt.subplots(
    figsize=(10, 6)
)

importance.sort_values().plot(
    kind="barh",
    ax=ax
)

ax.set_xlabel(
    "Importance"
)

ax.set_ylabel(
    "Feature"
)

ax.set_title(
    "Top Features Used by XGBoost"
)

st.pyplot(fig)


# ============================================================
# SHAP EXPLANATION
# ============================================================

st.divider()

st.subheader(
    "🧠 Why did the model make this prediction?"
)

st.write(
    """
    SHAP (SHapley Additive exPlanations) shows
    how individual features influence the XGBoost
    prediction for the selected engine.
    """
)


try:

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer(
        X_current
    )


    # Waterfall plot

    fig = plt.figure(
        figsize=(10, 7)
    )

    shap.plots.waterfall(
        shap_values[0],
        max_display=15,
        show=False
    )

    st.pyplot(
        plt.gcf(),
        clear_figure=True
    )


except Exception as e:

    st.warning(
        "SHAP explanation could not be generated."
    )

    st.write(
        str(e)
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

st.subheader(
    "🤖 Model Information"
)

model_col1, model_col2 = st.columns(2)


with model_col1:

    st.write(
        "**Algorithm:** XGBoost Regressor"
    )

    st.write(
        "**Problem:** Regression"
    )

    st.write(
        "**Target:** Remaining Useful Life (RUL)"
    )


with model_col2:

    st.write(
        "**Dataset:** NASA C-MAPSS FD001"
    )

    st.write(
        "**Task:** Turbofan predictive maintenance"
    )

    st.write(
        "**Output:** Remaining cycles before failure"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI-Based Predictive Maintenance System | "
    "NASA C-MAPSS FD001 | XGBoost"
)

