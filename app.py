import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# -----------------------------
# RAG Function
# -----------------------------

def get_energy_tips(level):
    with open("energy_tips.txt", "r") as file:
        content = file.read()

    sections = content.split("\n\n")

    for section in sections:
        if section.startswith(level):
            return section.split("\n")[1:]

    return []


# -----------------------------
# Load Dataset
# -----------------------------

data = pd.read_csv("railway_energy.csv")

time_mapping = {
    "Morning": 1,
    "Afternoon": 2,
    "Evening": 3,
    "Night": 4
}

data["Time"] = data["Time"].map(time_mapping)

X = data[
    [
        "Passenger_Count",
        "Temperature",
        "Time",
        "Weekend"
    ]
]

y = data["Electricity_Units"]

model = LinearRegression()
model.fit(X, y)

# -----------------------------
# Streamlit Page
# -----------------------------

st.set_page_config(
    page_title="GreenRail AI",
    page_icon="🚆"
)

st.title("🚆 GreenRail AI")

st.caption(
    "Predicts railway station energy usage and provides energy-saving suggestions."
)

st.subheader("Enter Station Details")

passengers = st.number_input(
    "Passenger Count",
    min_value=0,
    value=500
)

temperature = st.number_input(
    "Temperature (°C)",
    value=30
)

time = st.selectbox(
    "Time of Day",
    ["Morning", "Afternoon", "Evening", "Night"]
)

weekend = st.selectbox(
    "Weekend",
    ["No", "Yes"]
)

if st.button("Check Energy Usage"):

    time_num = time_mapping[time]
    weekend_num = 1 if weekend == "Yes" else 0

    new_data = pd.DataFrame({
        "Passenger_Count": [passengers],
        "Temperature": [temperature],
        "Time": [time_num],
        "Weekend": [weekend_num]
    })

    prediction = model.predict(new_data)

    units = prediction[0]
    co2 = units * 0.82

    if units > 1800:
        level = "HIGH"
        score = 40

    elif units > 1200:
        level = "MEDIUM"
        score = 70

    else:
        level = "LOW"
        score = 90

    st.subheader("Results")

    st.metric(
        "⚡ Electricity Consumption",
        f"{round(units)} Units"
    )

    st.metric(
        "🌍 Estimated CO₂ Emission",
        f"{round(co2)} kg"
    )

    st.subheader("🌱 Sustainability Score")

    st.progress(score / 100)
    st.write(f"{score}/100")

    st.subheader("Summary")

    st.write(
        f"The station is expected to consume approximately "
        f"{round(units)} units of electricity."
    )

    if level == "HIGH":
        st.error("High energy consumption expected.")

    elif level == "MEDIUM":
        st.warning("Moderate energy consumption expected.")

    else:
        st.success("Energy consumption is within a healthy range.")

    st.subheader("💡 Energy Saving Suggestions")

    tips = get_energy_tips(level)

    for tip in tips:
        st.write("✅", tip)

    st.markdown("---")

    st.caption(
        "Developed as part of the 1M1B AI for Sustainability Internship."
    ) 