import torch
import pandas as pd
import os

# ===============================
# 📌 경로
# ===============================
BASE_DIR = r"C:\26-1"

PT_DIR = os.path.join(BASE_DIR, "PT_files")
FEATURE_DIR = os.path.join(BASE_DIR, "Feature_Text")

# 🔥 입력 (Top31 결과)
XGB_TXT = os.path.join(FEATURE_DIR, "2018_xgbPI_top23.txt")
ELASTIC_TXT = os.path.join(FEATURE_DIR, "2018_elasticPI_top23.txt")

# 🔥 기준 데이터 (MI 결과)
TRAIN_PT = os.path.join(PT_DIR, "2018train_mi80.pt")
TEST_PT  = os.path.join(PT_DIR, "2018test_mi80.pt")

# 🔥 출력
OUT_TRAIN_PT = os.path.join(PT_DIR, "2018train_23.pt")
OUT_TEST_PT  = os.path.join(PT_DIR, "2018test_23.pt")

OUT_TXT = os.path.join(FEATURE_DIR, "2018_23_features.txt")


# ===============================
# 📌 txt 읽기
# ===============================
def load_feature_list(path):
    with open(path, 'r') as f:
        return set(line.strip() for line in f.readlines())


# ===============================
# 📌 pt 로드
# ===============================
def load_pt(path):
    X_t, y_t, feature_names = torch.load(path)

    X = pd.DataFrame(X_t.numpy(), columns=feature_names)
    y = pd.Series(y_t.numpy())

    return X, y


# ===============================
# 📌 메인
# ===============================
def main():
    # 1. feature 읽기
    xgb_features = load_feature_list(XGB_TXT)
    elastic_features = load_feature_list(ELASTIC_TXT)

    # 2. 교집합
    final_features = list(xgb_features & elastic_features)

    print("XGB feature 수:", len(xgb_features))
    print("Elastic feature 수:", len(elastic_features))
    print("🔥 최종 교집합 feature 수:", len(final_features))

    # 3. 데이터 로드 (MI 기준)
    X_train, y_train = load_pt(TRAIN_PT)
    X_test, y_test = load_pt(TEST_PT)

    # 4. feature 적용
    X_train_sel = X_train[final_features]
    X_test_sel = X_test[final_features]

    # 5. tensor 변환
    X_train_t = torch.FloatTensor(X_train_sel.values)
    y_train_t = torch.LongTensor(y_train.values)

    X_test_t = torch.FloatTensor(X_test_sel.values)
    y_test_t = torch.LongTensor(y_test.values)

    # 6. 저장
    torch.save((X_train_t, y_train_t, final_features), OUT_TRAIN_PT)
    torch.save((X_test_t, y_test_t, final_features), OUT_TEST_PT)

    # 7. txt 저장
    with open(OUT_TXT, 'w') as f:
        for feat in final_features:
            f.write(feat + '\n')

    print("\n✅ 최종 Robust Feature 생성 완료")
    print("저장 위치:")
    print(OUT_TRAIN_PT)
    print(OUT_TXT)


# ===============================
if __name__ == "__main__":
    main()