"""
predict.py
Role: Dashboard & Deployment Engineer (Vishnu)
Project: Contextual Predictive Maintenance (IoT Edge AI)

What this does:
1. Production inference engine.
2. Loads trained LightGBM model & standard scaler (from Member 5).
3. Preprocesses single-row or batch JSON/CSV data.
4. Predicts failure probability and classifies failure type (Binary).
5. Computes SHAP explainability values for the dashboard.
"""
import os
import joblib
import pandas as pd
import numpy as np
import shap

class PredictiveMaintenanceInference:
    def __init__(self, model_path="Models/lightgbm_model.pkl", scaler_path="Models/scaler.pkl"):
        """Initializes the inference pipeline, loading the pre-trained model and scaler."""
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.explainer = None
        self.feature_names = [
            "Type", "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
            "Torque [Nm]", "Tool wear [min]", "TWF", "HDF", "PWF", "OSF", "RNF"
        ]
        
        self.load_artifacts()

    def load_artifacts(self):
        """Load model and scaler from disk."""
        if not os.path.exists(self.model_path) or not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Model or Scaler not found at {self.model_path} / {self.scaler_path}")
        
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        
        self.explainer = shap.TreeExplainer(self.model)
        
    def preprocess_input(self, data_dict: dict) -> pd.DataFrame:
        # Add the missing dummy features from training data leakage
        for col in ["TWF", "HDF", "PWF", "OSF", "RNF"]:
            if col not in data_dict:
                data_dict[col] = 0
                
        df = pd.DataFrame([data_dict])
        df = df[self.feature_names]
        
        scaled_features = self.scaler.transform(df)
        return pd.DataFrame(scaled_features, columns=self.feature_names)
        
    def predict_risk(self, data_dict: dict) -> dict:
        """
        Predict failure probability and compute local SHAP values for explainability.
        Returns a dictionary containing risk score, prediction, and SHAP values.
        """
        # 1. Preprocess
        processed_df = self.preprocess_input(data_dict)
        
        # 2. Predict Probability & Binary Class
        # predict_proba returns [Prob(Normal), Prob(Failure)]
        failure_prob = self.model.predict_proba(processed_df)[0][1]
        binary_pred = self.model.predict(processed_df)[0]
        
        # 3. Categorize Risk (Fleet Manager Persona Requirement)
        if failure_prob >= 0.80:
            status = "CRITICAL"
        elif failure_prob >= 0.50:
            status = "WARNING"
        else:
            status = "NORMAL"
            
        # 4. Generate SHAP values for Root Cause Analysis (Reliability Engineer Persona)
        shap_values = self.explainer.shap_values(processed_df)
        
        # Extract the shap values for the predicted class.
        # For binary classification LightGBM, shap_values might be a list of arrays (one per class) or a single array.
        if isinstance(shap_values, list):
            # Binary classification usually has the positive class at index 1
            local_shap = shap_values[1][0] 
        else:
            local_shap = shap_values[0]
            
        # Map feature names to their SHAP values
        shap_dict = {feat: float(val) for feat, val in zip(self.feature_names, local_shap)}
        
        return {
            "failure_probability": float(failure_prob),
            "predicted_class": int(binary_pred),
            "status": status,
            "shap_values": shap_dict,
            "base_value": float(self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, (list, tuple, np.ndarray)) else self.explainer.expected_value)
        }
        
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run inference on a batch of machines (used for fleet overview).
        """
        # Keep original to return alongside predictions
        results_df = df.copy()
        
        # Ensure we only scale the required columns, drop things like Machine ID or UDI if they exist
        input_df = df[self.feature_names].copy()
        
        scaled_features = self.scaler.transform(input_df)
        
        probs = self.model.predict_proba(scaled_features)[:, 1]
        preds = self.model.predict(scaled_features)
        
        results_df['Failure_Probability'] = probs
        results_df['Predicted_Failure'] = preds
        
        # Assign statuses
        conditions = [
            (results_df['Failure_Probability'] >= 0.80),
            (results_df['Failure_Probability'] >= 0.50)
        ]
        choices = ['CRITICAL', 'WARNING']
        results_df['Status'] = np.select(conditions, choices, default='NORMAL')
        
        return results_df

# For quick local testing
if __name__ == "__main__":
    inference = PredictiveMaintenanceInference()
    sample_data = {
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0,
        "Type": 1 # e.g. L=1, M=2, H=0 based on label encoder mapping
    }
    
    result = inference.predict_risk(sample_data)
    print("--- Single Prediction Test ---")
    print(f"Status: {result['status']}")
    print(f"Probability: {result['failure_probability']:.2%}")
    print(f"SHAP Root Causes: {result['shap_values']}")
