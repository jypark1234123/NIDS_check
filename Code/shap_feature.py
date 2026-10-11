import torch
import numpy as np
import shap
from lightgbm import LGBMClassifier


X_train, y_train, feature_names = torch.load("train.pt")
X_test, y_test, _ = torch.load("test.pt")

X_train_np = X_train.numpy()
y_train_np = y_train.numpy()
X_test_np = X_test.numpy()

print("Train shape:", X_train_np.shape)


model = LGBMClassifier(
    n_estimators=100,
    max_depth=-1,
    learning_rate=0.1,
    n_jobs=-1
)

print("\nTraining LightGBM...")
model.fit(X_train_np, y_train_np)


print("\nComputing SHAP values...")

sample_idx = np.random.choice(len(X_train_np), 10000, replace=False)
X_sample = X_train_np[sample_idx]

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)


if isinstance(shap_values, list):
    shap_values = shap_values[1]

importance = np.abs(shap_values).mean(axis=0)


sorted_idx = np.argsort(importance)[::-1]

k50 = int(len(importance) * 0.5)
k30 = int(len(importance) * 0.3)

top50_idx = sorted_idx[:k50]
top30_idx = sorted_idx[:k30]

print(f"Top 50%: {k50} features")
print(f"Top 30%: {k30} features")


print("\nTop 50% features:")
for i, idx in enumerate(top50_idx):
    print(f"{i+1}. {feature_names[idx]}")

print("\nTop 30% features:")
for i, idx in enumerate(top30_idx):
    print(f"{i+1}. {feature_names[idx]}")


X_train_50 = X_train_np[:, top50_idx]
X_test_50 = X_test_np[:, top50_idx]

X_train_30 = X_train_np[:, top30_idx]
X_test_30 = X_test_np[:, top30_idx]


train50 = (torch.tensor(X_train_50, dtype=torch.float32), y_train)
test50 = (torch.tensor(X_test_50, dtype=torch.float32), y_test)

train30 = (torch.tensor(X_train_30, dtype=torch.float32), y_train)
test30 = (torch.tensor(X_test_30, dtype=torch.float32), y_test)


torch.save(train50, "train50.pt")
torch.save(test50, "test50.pt")

torch.save(train30, "train30.pt")
torch.save(test30, "test30.pt")

print("\nSaved:")
print("train50:", X_train_50.shape, "| test50:", X_test_50.shape)
print("train30:", X_train_30.shape, "| test30:", X_test_30.shape)