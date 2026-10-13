import joblib

model = joblib.load("models/water_potability_model.pkl")

print(type(model))
print(model)