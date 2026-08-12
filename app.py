import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("admission_data.csv")

# Features and target
X = data[["GRE_Score", "TOEFL_Score", "University_Rating", "CGPA", "SOP_Rating"]]
y = data["Admission"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print("Admission Prediction Project")
print("Model Accuracy:", round(accuracy * 100, 2), "%")

# Take user input
print("\nEnter student details:")
gre = float(input("GRE Score: "))
toefl = float(input("TOEFL Score: "))
rating = float(input("University Rating (1-5): "))
cgpa = float(input("CGPA: "))
sop = float(input("SOP Rating (1-5): "))

new_student = [[gre, toefl, rating, cgpa, sop]]
prediction = model.predict(new_student)[0]

if prediction == 1:
    print("\nResult: Admission Likely")
else:
    print("\nResult: Admission Not Likely")
