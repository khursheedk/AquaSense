"""
app.py — AquaSense Streamlit interface.

Run with:
    streamlit run app.py
"""

import pathlib
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AquaSense — Smart Water Consumption Prediction",
    page_icon="💧",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_PATH = pathlib.Path("model.pkl")
DATA_PATH  = pathlib.Path("water_consumption.csv")

FEATURES = [
    "occupants",
    "property_size_m2",
    "num_bathrooms",
    "has_dishwasher",
    "has_washing_machine",
    "avg_temp_celsius",
    "garden_size_m2",
    "days_away_per_year",
]

# ---------------------------------------------------------------------------
# Load model (cached so it is loaded only once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(
            "**Model file not found.**  "
            "Run `python train_model.py` first to train and save the model."
        )
        st.stop()
    artifact = joblib.load(MODEL_PATH)
    # Support both plain model and {"model": ..., "features": ...} dict
    if isinstance(artifact, dict):
        return artifact["model"], artifact.get("features", FEATURES)
    return artifact, FEATURES


@st.cache_data
def load_dataset():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None


model, model_features = load_model()
df_data = load_dataset()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("💧 AquaSense")
st.subheader("Smart Water Consumption Prediction System")
st.markdown(
    """
    Water scarcity is one of the pressing challenges of the 21st century.
    **AquaSense** uses a machine-learning regression model to estimate daily
    household water consumption based on household characteristics.
    Enter your household details below to get an instant estimate.

    > ⚠️ **Note:** This prototype uses a *synthetic* educational dataset modelled
    > on published per-activity water-use research (Aquastat / FAO, Ofwat, AWA).
    > Predictions are estimates and should not be used for engineering or policy decisions.
    """
)

st.divider()

# ---------------------------------------------------------------------------
# Sidebar — Model & Dataset Info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ About the Model")
    st.markdown(
        """
        **Algorithm:** Random Forest Regressor  
        **Target variable:** Water consumption (L/day)  
        **Dataset:** Synthetic (500 households)  
        **Features used:** 8  

        **Model limitations:**
        - Trained on synthetic data; real-world accuracy will differ.
        - Does not account for all local factors (climate zone, tariff structure, etc.).
        - Predictions outside the training range may be less reliable.

        **Dataset basis:**
        - Aquastat / FAO per-capita benchmarks
        - Ofwat (UK) household demand research
        - Australian Water Association (AWA) guidelines
        - US EPA WaterSense programme data
        """
    )
    st.divider()
    st.caption("AquaSense v1.0 | Educational prototype")

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
st.header("🏠 Household Details")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        occupants = st.number_input(
            "Number of occupants",
            min_value=1, max_value=10, value=3, step=1,
            help="Total number of people living in the household."
        )
        property_size_m2 = st.number_input(
            "Property size (m²)",
            min_value=20.0, max_value=400.0, value=95.0, step=5.0,
            help="Total floor area of the property in square metres."
        )
        num_bathrooms = st.number_input(
            "Number of bathrooms",
            min_value=1, max_value=6, value=1, step=1,
            help="Total number of bathrooms (including en-suites)."
        )
        avg_temp_celsius = st.number_input(
            "Average annual temperature (°C)",
            min_value=-5.0, max_value=45.0, value=18.0, step=0.5,
            help="Annual average outdoor temperature at your location."
        )

    with col2:
        has_dishwasher = st.selectbox(
            "Dishwasher", options=[("Yes", 1), ("No", 0)],
            format_func=lambda x: x[0],
            help="Does the household have a dishwasher?"
        )
        has_washing_machine = st.selectbox(
            "Washing machine", options=[("Yes", 1), ("No", 0)],
            format_func=lambda x: x[0],
            help="Does the household have a washing machine?"
        )
        garden_size_m2 = st.number_input(
            "Garden / outdoor area (m²)",
            min_value=0.0, max_value=200.0, value=20.0, step=5.0,
            help="Approximate size of the garden or outdoor irrigated area."
        )
        days_away_per_year = st.number_input(
            "Days away from home per year",
            min_value=0, max_value=365, value=14, step=1,
            help="Approximate number of days per year the household is unoccupied."
        )

    submitted = st.form_submit_button("💧 Predict Water Consumption", use_container_width=True)

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if submitted:
    # Extract scalar values from selectboxes (they return tuples)
    dw_val  = has_dishwasher[1]   if isinstance(has_dishwasher, tuple)      else int(has_dishwasher)
    wm_val  = has_washing_machine[1] if isinstance(has_washing_machine, tuple) else int(has_washing_machine)

    # Validate ranges
    errors = []
    if occupants < 1:
        errors.append("Number of occupants must be at least 1.")
    if property_size_m2 < 20:
        errors.append("Property size must be at least 20 m².")
    if days_away_per_year > 365:
        errors.append("Days away cannot exceed 365.")

    if errors:
        for e in errors:
            st.error(e)
    else:
        input_data = pd.DataFrame([{
            "occupants":            int(occupants),
            "property_size_m2":     float(property_size_m2),
            "num_bathrooms":        int(num_bathrooms),
            "has_dishwasher":       int(dw_val),
            "has_washing_machine":  int(wm_val),
            "avg_temp_celsius":     float(avg_temp_celsius),
            "garden_size_m2":       float(garden_size_m2),
            "days_away_per_year":   int(days_away_per_year),
        }])[model_features]  # enforce feature order

        prediction = float(model.predict(input_data)[0])
        prediction = max(0.0, round(prediction, 1))

        st.divider()
        st.header("📊 Prediction Result")

        m1, m2, m3 = st.columns(3)
        m1.metric("Daily consumption",  f"{prediction:.1f} L/day")
        m2.metric("Monthly (approx.)",  f"{prediction * 30 / 1000:.2f} m³/month")
        m3.metric("Annual (approx.)",   f"{prediction * 365 / 1000:.1f} m³/year")

        # Interpretation
        if prediction < 100:
            level, colour = "very low", "🟢"
        elif prediction < 180:
            level, colour = "low to moderate", "🟢"
        elif prediction < 280:
            level, colour = "moderate", "🟡"
        elif prediction < 400:
            level, colour = "high", "🟠"
        else:
            level, colour = "very high", "🔴"

        st.info(
            f"{colour} **Interpretation:** The estimated daily water consumption for this household is "
            f"**{prediction:.1f} litres per day**, which is classified as **{level}**. "
            f"The global average is approximately **100–200 L/person/day** (Aquastat, FAO). "
            f"For {int(occupants)} occupant(s), this corresponds to roughly "
            f"**{prediction / occupants:.0f} L/person/day**."
        )

        st.caption(
            "⚠️ This is an estimate from a model trained on synthetic data. "
            "Actual consumption depends on local infrastructure, habits, tariff structures, "
            "seasonal variation, and many other factors not captured here."
        )

# ---------------------------------------------------------------------------
# Visualisation — Feature vs consumption scatter (only if dataset available)
# ---------------------------------------------------------------------------
if df_data is not None:
    st.divider()
    st.header("📈 Dataset Overview")
    st.markdown(
        "The chart below shows how the number of **occupants** relates to daily water "
        "consumption across the 500 synthetic households in the training dataset."
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    sc = ax.scatter(
        df_data["occupants"],
        df_data["water_consumption_lpd"],
        alpha=0.4,
        c=df_data["avg_temp_celsius"],
        cmap="YlOrRd",
        edgecolors="none",
        s=25,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Avg. temperature (°C)", fontsize=9)
    ax.set_xlabel("Number of occupants", fontsize=10)
    ax.set_ylabel("Water consumption (L/day)", fontsize=10)
    ax.set_title("Occupants vs. Daily Water Consumption\n(colour = average temperature)", fontsize=11)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "**Dataset note:** This is a synthetic dataset of 500 households generated for "
        "educational purposes. It is modelled on published water-use research "
        "(Aquastat/FAO, Ofwat, AWA) but does not represent real recorded data."
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.markdown(
    "<p style='text-align:center; color:#888; font-size:0.85rem;'>"
    "AquaSense — Smart Water Consumption Prediction System &nbsp;|&nbsp; "
    "Educational prototype &nbsp;|&nbsp; BCA Machine Learning Project"
    "</p>",
    unsafe_allow_html=True,
)
