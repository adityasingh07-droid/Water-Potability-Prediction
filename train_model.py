import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("dataset/water_potability.csv")

for column in df.columns:
    if column != "Potability":
        df[column] = df[column].fillna(df[column].median())

X = df.drop("Potability", axis=1)
y = df["Potability"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)

joblib.dump(rf_model, "models/water_potability_model.pkl")

print("Model trained successfully!")
print("Model saved successfully!")
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)