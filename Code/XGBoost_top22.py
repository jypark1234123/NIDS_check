import os
import torch
import numpy as np

from xgboost import XGBClassifier

# ===============================
# 📌 경로
# ===============================
BASE_DIR = r"C:\26-1"

TRAIN_PT = os.path.join(BASE_DIR, "PT_files", "2018_train.pt")
TEST_PT  = os.path.join(BASE_DIR, "PT_files", "2018_test.pt")

FEATURE_DIR = os.path.join(BASE_DIR, "Feature_Text")
PT_DIR = os.path.join(BASE_DIR, "PT_files")

os.makedirs(FEATURE_DIR, exist_ok=True)
os.makedirs(PT_DIR, exist_ok=True)

# ===============================
# 📌 설정
# ===============================
TOP_K = 22

TXT_SAVE = os.path.join(FEATURE_DIR, "2018_XGBoost_Top22.txt")

TRAIN_SAVE = os.path.join(PT_DIR, "2018train_xgb22.pt")
TEST_SAVE  = os.path.join(PT_DIR, "2018test_xgb22.pt")

# ===============================
# 📌 데이터 로드
# ===============================
X_train, y_train, feature_names = torch.load(TRAIN_PT)
X_test, y_test, _ = torch.load(TEST_PT)

X_train_np = X_train.numpy()
y_train_np = y_train.numpy()

print("Train Shape :", X_train_np.shape)
print("Test Shape  :", X_test.shape)

# ===============================
# 📌 XGBoost 학습
# ===============================
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train_np, y_train_np)

# ===============================
# 📌 Feature Importance
# ===============================
importance = model.feature_importances_

idx = np.argsort(importance)[::-1].copy()

top_idx = idx[:TOP_K]

top_features = [feature_names[i] for i in top_idx]

print("\n===== XGBoost Top-22 =====")

for rank, i in enumerate(top_idx, start=1):
    print(f"{rank:2d}. {feature_names[i]:30s} {importance[i]:.6f}")

# ===============================
# 📌 TXT 저장
# ===============================
with open(TXT_SAVE, "w", encoding="utf-8") as f:
    for feat in top_features:
        f.write(feat + "\n")

print(f"\nFeature List Saved : {TXT_SAVE}")

# ===============================
# 📌 PT 생성
# ===============================
X_train_new = X_train[:, top_idx]
X_test_new = X_test[:, top_idx]

torch.save(
    (X_train_new, y_train, top_features),
    TRAIN_SAVE
)

torch.save(
    (X_test_new, y_test, top_features),
    TEST_SAVE
)

print("\n===== Saved =====")
print(TRAIN_SAVE)
print(TEST_SAVE)

print("\nTrain Shape :", X_train_new.shape)
print("Test Shape  :", X_test_new.shape)