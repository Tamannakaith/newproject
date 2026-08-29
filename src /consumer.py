import json
import requests
from kafka import KafkaConsumer

consumer = KafkaConsumer(
    'transactions_stream',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='latest',
    group_id='fraud-detection-engine'
)

API_URL = "http://localhost:8000/predict"
WEBHOOK_URL = "https://httpbin.org/post"  # Mock endpoint or replace with Slack/Discord webhook

def send_alert(alert_data):
    payload = {
        "text": f"🚨 FRAUD ALERT: Txn ID {alert_data['transaction_id']} | Risk Score: {alert_data['fraud_probability'] * 100:.2f}%"
    }
    try:
        requests.post(WEBHOOK_URL, json=payload, timeout=2)
    except Exception as e:
        print(f"Webhook delivery failed: {e}")

print("Consumer running. Monitoring transaction stream...")
for msg in consumer:
    tx_data = msg.value
    res = requests.post(API_URL, json=tx_data)
    if res.status_code == 200:
        prediction = res.json()
        if prediction["is_fraud"]:
            print(f"🚨 FLAGGED: {prediction['transaction_id']} (Score: {prediction['fraud_probability']})")
            send_alert(prediction)
        else:
            print(f"✅ APPROVED: {prediction['transaction_id']}")