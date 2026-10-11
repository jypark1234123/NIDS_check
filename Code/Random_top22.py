import os
import random
import torch

# =====================================
# 경로
# =====================================
TRAIN_PT = r"C:\26-1\PT_files\2018_train.pt"
TEST_PT  = r"C:\26-1\PT_files\2018_test.pt"

OUT_PT = r"C:\26-1\PT_files"
OUT_TXT = r"C:\26-1\Feature_Text"

TOP_K = 22
NUM_REPEAT = 5

# =====================================
# 데이터 로드
# =====================================
X_train, y_train, feature_names = torch.load(TRAIN_PT)
X_test, y_test, _ = torch.load(TEST_PT)

print("Train Shape :", X_train.shape)
print("Test Shape  :", X_test.shape)

num_features = len(feature_names)

# =====================================
# Random 5회 생성
# =====================================
for run in range(1, NUM_REPEAT + 1):

    random.seed(run)      # 재현 가능

    selected_idx = sorted(random.sample(range(num_features), TOP_K))

    selected_features = [feature_names[i] for i in selected_idx]

    print(f"\n========== Random {run} ==========")

    for rank, feat in enumerate(selected_features, start=1):
        print(f"{rank:2d}. {feat}")

    # -----------------------------
    # Feature txt 저장
    # -----------------------------
    txt_path = os.path.join(
        OUT_TXT,
        f"2018_Random22_{run}.txt"
    )

    with open(txt_path, "w", encoding="utf-8") as f:
        for feat in selected_features:
            f.write(feat + "\n")

    # -----------------------------
    # Feature 선택
    # -----------------------------
    X_train_new = X_train[:, selected_idx]
    X_test_new = X_test[:, selected_idx]

    train_save = os.path.join(
        OUT_PT,
        f"2018train_random22_{run}.pt"
    )

    test_save = os.path.join(
        OUT_PT,
        f"2018test_random22_{run}.pt"
    )

    torch.save(
        (X_train_new, y_train, selected_features),
        train_save
    )

    torch.save(
        (X_test_new, y_test, selected_features),
        test_save
    )

    print("Saved :", train_save)
    print("Saved :", test_save)

print("\n========== 완료 ==========")