import torch
import models
import train_utils
import time

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 2. 데이터 로드
X_train, y_train = torch.load("train_removed.pt")
X_test, y_test = torch.load("test_removed.pt")

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


# DataLoader 생성
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)


# SSAE 모델 생성
input_dim = X_train.shape[1]
hidden_layers = [256, 128, 64]

ssae = models.StackedSparseAutoencoder(input_dim, hidden_layers)

# SSAE 학습
start_ssae = time.time()

ssae, history = train_utils.train_stacked_sae(
    ssae,
    train_loader,
    device,
    lr=1e-3,
    epochs=10
)

end_ssae = time.time()
print(f"\n SSAE Training Time: {end_ssae - start_ssae:.2f} sec")


# Feature 추출 
ssae.eval()

start_feat = time.time()

with torch.no_grad():
    X_train_feat = ssae.forward_features(X_train.to(device)).cpu().numpy()
    X_test_feat = ssae.forward_features(X_test.to(device)).cpu().numpy()

end_feat = time.time()
print(f" Feature Extraction Time: {end_feat - start_feat:.2f} sec")

y_train_np = y_train.numpy()
y_test_np = y_test.numpy()

print("Feature shape:", X_train_feat.shape)


# 7. XGBoost 

clf = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=2,
    n_jobs=-1,
    tree_method='hist',
    eval_metric='logloss'
)

print("\nTraining XGBoost...")

start_xgb = time.time()

clf.fit(X_train_feat, y_train_np)

end_xgb = time.time()
print(f" XGBoost Training Time: {end_xgb - start_xgb:.2f} sec")

# =========================
# 8. 예측 및 평가 (시간 측정)
# =========================
start_pred = time.time()

y_pred = clf.predict(X_test_feat)

end_pred = time.time()
print(f" Prediction Time: {end_pred - start_pred:.2f} sec")

acc = accuracy_score(y_test_np, y_pred)

print("\n✅ Accuracy:", acc)

print("\n📊 Classification Report:")
print(classification_report(y_test_np, y_pred))

print("\n📉 Confusion Matrix:")
print(confusion_matrix(y_test_np, y_pred))