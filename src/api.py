from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Fraud Detection Inference Engine")

model = joblib.load("models/xgb_fraud_model.pkl")
scaler = joblib.load("models/scaler.pkl")

FRAUD_THRESHOLD = 0.40

class Transaction(BaseModel):
    transaction_id: str
    user_id: str
    amount: float
    time: float
    features: list[float]

@app.post("/predict")
def predict_transaction(txn: Transaction):
    if len(txn.features) != 28:
        raise HTTPException(status_code=400, detail="Expected 28 PCA features (V1-V28)")

    scaled_amt_time = scaler.transform([[txn.amount], [txn.time]]).flatten()
    input_vector = np.array(txn.features + [scaled_amt_time[0], scaled_amt_time[1]]).reshape(1, -1)

    fraud_probability = float(model.predict_proba(input_vector)[0][1])
    is_fraud = fraud_probability >= FRAUD_THRESHOLD

    return {
        "transaction_id": txn.transaction_id,
        "fraud_probability": round(fraud_probability, 4),
        "is_fraud": is_fraud,
        "action": "FLAG_AND_ALERT" if is_fraud else "APPROVE"
    }
