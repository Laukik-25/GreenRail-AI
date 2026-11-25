import pandas as pd
from sklearn.linear_model import LinearRegression

# --------------------------
# RAG Function
# --------------------------

def get_energy_tips(level):

    with open("energy_tips.txt", "r") as file:
        content = file.read()

    sections = content.split("\n\n")

    for section in sections:

        if section.startswith(level):
            return section.split("\n")[1:]

    return []


# --------------------------
# Load Dataset
# --------------------------

data = pd.read_csv("railway_energy.csv")

time_mapping = {
    "Morning": 1,
    "Afternoon": 2,
    "Evening": 3,
    "Night": 4
}

data["Time"] = data["Time"].map(time_mapping)

# Features
X = data[
    ["Passenger_Count",
     "Temperature",
     "Time",
     "Weekend"]
]

# Target
y = data["Electricity_Units"]

# Train model
model = LinearRegression()
model.fit(X, y)

# --------------------------
# User Input
# --------------------------

print("===== GreenRail AI with RAG =====")

passengers = int(input("Enter Passenger Count: "))
temperature = int(input("Enter Temperature (°C): "))
time = int(input(
    "Enter Time (1-Morning,2-Afternoon,3-Evening,4-Night): "
))
weekend = int(input("Weekend? (1-Yes,0-No): "))

new_data = pd.DataFrame({
    "Passenger_Count": [passengers],
    "Temperature": [temperature],
    "Time": [time],
    "Weekend": [weekend]
})

prediction = model.predict(new_data)

units = prediction[0]

print("\nPredicted Electricity Units:",
      round(units))

co2 = units * 0.82

print("Estimated CO2 Emission:",
      round(co2),
      "kg")

# --------------------------
# RAG Retrieval
# --------------------------

if units > 1800:
    level = "HIGH"

elif units > 1200:
    level = "MEDIUM"

else:
    level = "LOW"

tips = get_energy_tips(level)

print("\n===== RAG Recommendations =====")

for tip in tips:
    print("•", tip)

print("\n===== End =====")