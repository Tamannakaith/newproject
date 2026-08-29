import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_valid_transaction_prediction():
    payload = {
        "transaction_id": "test-txn-valid",
        "user_id": "usr_1001",
        "amount": 150.0,
        "time": 50.0,
        "features": [0.1] * 28
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "fraud_probability" in data
    assert "is_fraud" in data
    assert "action" in data
    assert isinstance(data["is_fraud"], bool)

def test_invalid_feature_length():
    payload = {
        "transaction_id": "test-txn-bad",
        "user_id": "usr_1001",
        "amount": 50.0,
        "time": 10.0,
        "features": [0.1] * 10  # Only 10 features instead of 28
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
