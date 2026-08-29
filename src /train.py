import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

# 1. Load Data
df = pd.read_csv("data/creditcard.csv")

# 2. Scale Amount and Time
scaler = RobustScaler()
df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
df = df.drop(['Time', 'Amount'], axis=1)

X = df.drop('Class', axis=1)
y = df['Class']

# Stratified Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Handle Imbalance on Train Partition Only
smote = SMOTE(sampling_strategy=0.2, random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 4. Train Model
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train_res, y_train_res)

# 5. Evaluate
y_probs = model.predict_proba(X_test)[:, 1]
print(f"ROC-AUC: {roc_auc_score(y_test, y_probs):.4f}")
print(f"PR-AUC: {average_precision_score(y_test, y_probs):.4f}")

# 6. Save Artifacts
joblib.dump(model, "models/xgb_fraud_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("Training complete. Models exported to /models.")