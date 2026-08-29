# Real-Time Financial Fraud Detection Streaming Pipeline

An end-to-end event-driven machine learning system that detects fraudulent transactions in real time with sub-15ms inference latency.

## 🏗️ Architecture
## 🛠️ Tech Stack
* **Language & Analysis:** Python 3.10+, Pandas, NumPy
* **Machine Learning:** Scikit-Learn, Imbalanced-Learn (SMOTE), XGBoost
* **Data Streaming:** Apache Kafka, Zookeeper
* **Inference API:** FastAPI, Uvicorn, Pydantic
* **Containerization:** Docker Compose

## 📊 Model Performance & Class Imbalance Handling
The dataset (Kaggle Credit Card Fraud) contains severe class imbalance (**0.17% fraud rate**).
* **Imbalance Strategy:** Synthetic Minority Over-sampling Technique (**SMOTE**) applied strictly to the training split to prevent data leakage.
* **Feature Scaling:** `RobustScaler` utilized on transaction amounts and timestamps to reduce outlier influence.
* **Evaluation Metric:** Optimized for **PR-AUC (Precision-Recall Curve)** and Recall over raw accuracy to minimize false negatives.

## 🚀 Quickstart Guide

### 1. Clone & Set Up Environment
```bash
git clone [https://github.com/Tamannakaith/newproject.git](https://github.com/Tamannakaith/newproject.git)
cd newproject
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

docker-compose up -d
python src/train.py
uvicorn src.api:app --reload --port 8000
python src/consumer.py
python src/producer.py
