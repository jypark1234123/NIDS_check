import torch
import numpy as np

from sklearn.feature_selection import mutual_info_classif

# ==========================================
# 경로 (2017)
# ==========================================
TRAIN_PT = r"C:\26-1\PT_files\2018_train.pt"
TEST_PT  = r"C:\26-1\PT_files\2018_test.pt"

TRAIN_OUT = r"C:\26-1\PT_files\2018train_mi22.pt"
TEST_OUT  = r"C:\26-1\PT_files\2018test_mi22.pt"

FEATURE_TXT = r"C:\26-1\Feature_Text\2018_MI_Top22.txt"

TOP_K = 22

# ==========================================
# Load
# ==========================================
X_train, y_train, feature_names = torch.load(TRAIN_PT)
X_test, y_test, _ = torch.load(TEST_PT)

print(f"Train Shape : {tuple(X_train.shape)}")
print(f"Test Shape  : {tuple(X_test.shape)}")

X_train_np = X_train.numpy()
y_train_np = y_train.numpy()

# ==========================================
# Mutual Information
# ==========================================
print("\nCalculating Mutual Information...")

mi_scores = mutual_info_classif(
    X_train_np,
    y_train_np,
    random_state=42,
    n_jobs=-1
)

# ==========================================
# Top-22
# ==========================================
top_idx = np.argsort(mi_scores)[::-1][:TOP_K].copy()

print("\n===== MI Top-22 =====")

with open(FEATURE_TXT, "w", encoding="utf-8") as f:

    for rank, idx in enumerate(top_idx, start=1):

        name = feature_names[idx]
        score = mi_scores[idx]

        print(f"{rank:2d}. {name:30s} {score:.6f}")

        f.write(f"{rank:2d}. {name}\t{score:.6f}\n")

print(f"\nFeature List Saved : {FEATURE_TXT}")

# ==========================================
# Dataset 생성
# ==========================================
X_train_new = X_train[:, top_idx]
X_test_new = X_test[:, top_idx]

selected_features = [feature_names[i] for i in top_idx]

torch.save(
    (X_train_new, y_train, selected_features),
    TRAIN_OUT
)

torch.save(
    (X_test_new, y_test, selected_features),
    TEST_OUT
)

print("\n===== Saved =====")
print(TRAIN_OUT)
print(TEST_OUT)

print("\nTrain Shape :", X_train_new.shape)
print("Test Shape  :", X_test_new.shape)