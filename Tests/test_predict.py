import sys
import os
import pytest
import numpy as np
import pandas as pd

# Ensure Src/ is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Src')))

try:
    from predict import PredictiveMaintenanceInference
except ImportError:
    pass

@pytest.fixture
def engine():
    # Attempt to load the real model, if not present we skip tests
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = os.path.join(base_dir, 'Models', 'lightgbm_model.pkl')
    scaler_path = os.path.join(base_dir, 'Models', 'scaler.pkl')
    
    if not os.path.exists(model_path):
        pytest.skip("Trained model not found. Run training first.")
        
    return PredictiveMaintenanceInference(model_path=model_path, scaler_path=scaler_path)

def test_inference_pipeline_single(engine):
    sample_data = {
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0,
        "Type": 1
    }
    
    result = engine.predict_risk(sample_data)
    
    assert "failure_probability" in result
    assert "status" in result
    assert "shap_values" in result
    assert 0.0 <= result["failure_probability"] <= 1.0
    assert result["status"] in ["NORMAL", "WARNING", "CRITICAL"]

def test_inference_batch(engine):
    sample_batch = pd.DataFrame({
        "Air temperature [K]": [298.1, 302.5],
        "Process temperature [K]": [308.6, 315.0],
        "Rotational speed [rpm]": [1551, 1300],
        "Torque [Nm]": [42.8, 65.0],
        "Tool wear [min]": [0, 220],
        "Type": [1, 1]
    })
    
    results = engine.predict_batch(sample_batch)
    
    assert "Failure_Probability" in results.columns
    assert "Status" in results.columns
    assert len(results) == 2
    # Second item is high risk (high temp, low speed, high torque, high wear)
    assert results.iloc[1]["Failure_Probability"] > results.iloc[0]["Failure_Probability"]
