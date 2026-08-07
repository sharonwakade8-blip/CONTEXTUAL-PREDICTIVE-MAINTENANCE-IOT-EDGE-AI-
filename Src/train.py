from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import os

from preprocess import load_and_preprocess_data

# Load data
X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()

# Create model
model = LGBMClassifier(random_state=42)

# Hyperparameter grid
param_grid = {
    "n_estimators": [200],
    "learning_rate": [0.1],
    "max_depth": [10],
    "num_leaves": [31]
}

# Grid Search
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

# Train
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Create Models folder if needed
os.makedirs("Models", exist_ok=True)

# Save model and scaler
joblib.dump(best_model, "Models/lightgbm_model.pkl")
joblib.dump(scaler, "Models/scaler.pkl")

print("Best Parameters:")
print(grid_search.best_params_)

print("\nBest Cross Validation Score:")
print(grid_search.best_score_)

print("\nTraining completed successfully!")

