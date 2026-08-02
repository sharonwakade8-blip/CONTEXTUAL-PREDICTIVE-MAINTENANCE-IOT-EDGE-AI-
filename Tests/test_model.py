import joblib

# Load model
model = joblib.load("Models/lightgbm_model.pkl")

print("Model loaded successfully!")
print(type(model))
