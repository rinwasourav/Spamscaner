import os
import pytest
import pandas as pd
from fastapi.testclient import TestClient

from src.data_generator import generate_synthetic_data, save_synthetic_dataset
from src.train import train_and_evaluate, FEATURE_COLUMNS, TARGET_COLUMN
from src.main import app, load_model

client = TestClient(app)

def test_data_generation():
    df = generate_synthetic_data(num_samples=500, random_state=42)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 500
    for col in FEATURE_COLUMNS:
        assert col in df.columns
    assert TARGET_COLUMN in df.columns
    assert set(df[TARGET_COLUMN].unique()).issubset({0, 1})

def test_dataset_save_file(tmp_path):
    output_path = str(tmp_path / "test_calls.csv")
    res_path = save_synthetic_dataset(output_path=output_path, num_samples=100)
    assert os.path.exists(res_path)
    df = pd.read_csv(res_path)
    assert len(df) == 100

def test_model_training(tmp_path):
    data_path = str(tmp_path / "test_data.csv")
    model_path = str(tmp_path / "test_model.pkl")
    
    save_synthetic_dataset(output_path=data_path, num_samples=1000)
    results = train_and_evaluate(data_path=data_path, model_output_path=model_path)
    
    assert os.path.exists(model_path)
    assert results["roc_auc"] > 0.85
    assert results["precision"] > 0.80
    assert results["f1_score"] > 0.80

def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_api_predict_legit_call():
    # Reload model to ensure latest state
    load_model()
    payload = {
        "calls_per_hour": 2.5,
        "avg_call_duration": 240.0,
        "contact_degree": 8,
        "spam_report_count": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "risk_score" in data
    assert "is_threat" in data
    assert data["is_threat"] is False
    assert data["threat_level"] in ["LOW", "MEDIUM"]

def test_api_predict_spam_call():
    load_model()
    payload = {
        "calls_per_hour": 65.0,
        "avg_call_duration": 4.5,
        "contact_degree": 0,
        "spam_report_count": 18
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "risk_score" in data
    assert "is_threat" in data
    assert data["is_threat"] is True
    assert data["threat_level"] == "HIGH"
