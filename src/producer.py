import pandas as pd
import requests
import uuid
import random
import time

API_URL = "http://localhost:8000/predict"
df = pd.read_csv("data/creditcard.csv")

print("🚀 Streaming transactions to Inference Engine...")
for _, row in df.sample(frac=1).iterrows():
    payload = {
        "transaction_id": str(uuid.uuid4()),
        "user_id": f"usr_{random.randint(1000, 9999)}",
        "amount": float(row['Amount']),
        "time": float(row['Time']),
        "features": [float(row[f'V{i}']) for i in range(1, 29)]
    }
    try:
        res = requests.post(API_URL, json=payload, timeout=2)
        if res.status_code == 200:
            data = res.json()
            if data["is_fraud"]:
                print(f"🚨 FRAUD FLAGGED! | Txn ID: {data['transaction_id'][:8]} | Score: {data['fraud_probability']*100:.1f}%")
            else:
                print(f"✅ APPROVED     | Txn ID: {data['transaction_id'][:8]} | Score: {data['fraud_probability']*100:.1f}%")
    except Exception as e:
        print(f"Connection error: Make sure API is running on port 8000")
    time.sleep(0.3)
