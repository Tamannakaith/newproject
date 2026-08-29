import requests
import json
import time

API_URL = "http://localhost:8000/predict"

def process_event(event):
    try:
        res = requests.post(API_URL, json=event, timeout=2)
        if res.status_code == 200:
            data = res.json()
            if data["is_fraud"]:
                print(f"🚨 FRAUD FLAGGED! | Txn ID: {data['transaction_id'][:8]} | Prob: {data['fraud_probability']*100:.1f}%")
            else:
                print(f"✅ APPROVED     | Txn ID: {data['transaction_id'][:8]} | Prob: {data['fraud_probability']*100:.1f}%")
    except Exception as e:
        print(f"Waiting for API server: {e}")

