import torch
import pandas as pd
import os

# ===============================
# 📌 경로
# ===============================
BASE_DIR = r"C:\26-1"

PT_DIR = os.path.join(BASE_DIR, "PT_files")

TRAIN_PT = os.path.join(PT_DIR, "2018train_31.pt")
TEST_PT  = os.path.join(PT_DIR, "2018test_31.pt")

OUT_TRAIN_PT = os.path.join(PT_DIR, "2018train_31_clean.pt")
OUT_TEST_PT  = os.path.join(PT_DIR, "2018test_31_clean.pt")


# ===============================
# 📌 제거할 feature
# ===============================
DROP_FEATURES = [
    "Init Fwd Win Byts",
    "Dst Port",
    "Flow Byts/s"
]


# ===============================
# 📌 pt 로드
# ===============================
def load_pt(path):
    X_t, y_t, feature_names = torch.load(path)

    X = pd.DataFrame(X_t.numpy(), columns=feature_names)
    y = pd.Series(y_t.numpy())

    return X, y, feature_names


# ===============================
# 📌 main
# ===============================
def main():
    # 데이터 로드
    X_train, y_train, features = load_pt(TRAIN_PT)
    X_test, y_test, _ = load_pt(TEST_PT)

    print("원래 feature 수:", len(features))

    # 실제 존재하는 feature만 제거
    drop_cols = [f for f in DROP_FEATURES if f in X_train.columns]

    print("제거되는 feature:", drop_cols)

    # 제거
    X_train = X_train.drop(columns=drop_cols)
    X_test = X_test.drop(columns=drop_cols)

    print("제거 후 feature 수:", X_train.shape[1])

    # tensor 변환
    X_train_t = torch.FloatTensor(X_train.values)
    y_train_t = torch.LongTensor(y_train.values)

    X_test_t = torch.FloatTensor(X_test.values)
    y_test_t = torch.LongTensor(y_test.values)

    new_features = X_train.columns.tolist()

    # 저장
    torch.save((X_train_t, y_train_t, new_features), OUT_TRAIN_PT)
    torch.save((X_test_t, y_test_t, new_features), OUT_TEST_PT)

    print("\n✅ Feature 제거 완료")
    print("저장 위치:")
    print(OUT_TRAIN_PT)
    print(OUT_TEST_PT)


# ===============================
if __name__ == "__main__":
    main()