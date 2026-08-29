import json
import time
import random
import uuid
from kafka import KafkaProducer
import pandas as pd

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

df = pd.read_csv("data/creditcard.csv")

print("Producer started. Streaming transactions...")
for _, row in df.iterrows():
    message = {
        "transaction_id": str(uuid.uuid4()),
        "user_id": f"usr_{random.randint(1000, 9999)}",
        "amount": float(row['Amount']),
        "time": float(row['Time']),
        "features": [float(row[f'V{i}']) for i in range(1, 29)]
    }
    producer.send('transactions_stream', value=message)
    time.sleep(0.05)  # 20 transactions per second