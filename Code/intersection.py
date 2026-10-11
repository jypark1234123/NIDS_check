import torch
import pandas as pd
import os

# 📌 기본 경로
BASE_DIR = r"C:\26-1"

PT_DIR = os.path.join(BASE_DIR, "PT_files")
FEATURE_DIR = os.path.join(BASE_DIR, "Feature_Text")

# 📌 txt 파일 경로
XGB_TXT = os.path.join(FEATURE_DIR, "2018_XG_top80.txt")
ELASTIC_TXT = os.path.join(FEATURE_DIR, "2018_elastic_top80.txt")

# 📌 pt 파일 (feature 이름 기준 동일하므로 아무거나 사용 가능)
TRAIN_PT = os.path.join(PT_DIR, "2018train_XG80.pt")
TEST_PT = os.path.join(PT_DIR, "2018test_XG80.pt")

# 📌 출력
OUT_TRAIN_PT = os.path.join(PT_DIR, "2018train_intersection.pt")
OUT_TEST_PT = os.path.join(PT_DIR, "2018test_intersection.pt")

OUT_TXT = os.path.join(FEATURE_DIR, "2018_intersection_features.txt")


# 1. txt 읽기
def load_feature_list(path):
    with open(path, 'r') as f:
        features = [line.strip() for line in f.readlines()]
    return set(features)


# 2. pt 로드
def load_pt(path):
    X_t, y_t, feature_names = torch.load(path)

    X = pd.DataFrame(X_t.numpy(), columns=feature_names)
    y = pd.Series(y_t.numpy())

    return X, y


# 3. main
def main():
    # feature 리스트 읽기
    xgb_features = load_feature_list(XGB_TXT)
    elastic_features = load_feature_list(ELASTIC_TXT)

    # 🔥 교집합
    intersection_features = list(xgb_features & elastic_features)

    print("XGBoost feature 수:", len(xgb_features))
    print("ElasticNet feature 수:", len(elastic_features))
    print("교집합 feature 수:", len(intersection_features))

    # pt 로드
    X_train, y_train = load_pt(TRAIN_PT)
    X_test, y_test = load_pt(TEST_PT)

    # 🔥 feature 선택
    X_train_sel = X_train[intersection_features]
    X_test_sel = X_test[intersection_features]

    # tensor 변환
    X_train_t = torch.FloatTensor(X_train_sel.values)
    y_train_t = torch.LongTensor(y_train.values)

    X_test_t = torch.FloatTensor(X_test_sel.values)
    y_test_t = torch.LongTensor(y_test.values)

    # 저장
    torch.save((X_train_t, y_train_t, intersection_features), OUT_TRAIN_PT)
    torch.save((X_test_t, y_test_t, intersection_features), OUT_TEST_PT)

    # txt 저장
    with open(OUT_TXT, 'w') as f:
        for feat in intersection_features:
            f.write(feat + '\n')

    print("✅ 교집합 feature 저장 완료")
    print("저장 위치:")
    print(OUT_TRAIN_PT)
    print(OUT_TEST_PT)
    print(OUT_TXT)


if __name__ == "__main__":
    main()