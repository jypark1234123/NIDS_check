import torch
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
import os

# 📌 기본 경로
BASE_DIR = r"C:\26-1"

PT_DIR = os.path.join(BASE_DIR, "PT_files")
FEATURE_DIR = os.path.join(BASE_DIR, "Feature_Text")

# 📌 입력 파일 (교집합 결과)
TRAIN_PT = os.path.join(PT_DIR, "2018train_intersection.pt")
TEST_PT = os.path.join(PT_DIR, "2018test_intersection.pt")

# 📌 출력 파일
OUT_TRAIN_PT = os.path.join(PT_DIR, "2018train_mi80.pt")
OUT_TEST_PT = os.path.join(PT_DIR, "2018test_mi80.pt")
OUT_TXT = os.path.join(FEATURE_DIR, "2018_mi_top80.txt")


def load_pt(path):
    X_t, y_t, feature_names = torch.load(path)

    X = pd.DataFrame(X_t.numpy(), columns=feature_names)
    y = pd.Series(y_t.numpy())

    return X, y, feature_names


def mutual_info_selection(X, y, ratio=0.8):
    # 🔥 속도 개선 옵션 (중요)
    mi = mutual_info_classif(
        X,
        y,
        random_state=42,
        n_neighbors=3   # 기본값 유지 (속도 안정)
    )

    feat_df = pd.DataFrame({
        'feature': X.columns,
        'importance': mi
    }).sort_values(by='importance', ascending=False)

    top_n = int(len(feat_df) * ratio)
    selected_features = feat_df['feature'].iloc[:top_n].tolist()

    return selected_features, feat_df


def main():
    # 1. 데이터 로드
    X_train, y_train, feature_names = load_pt(TRAIN_PT)
    X_test, y_test, _ = load_pt(TEST_PT)

    print("Intersection feature 수:", len(feature_names))

    # 🔥 (선택) 속도 개선용 샘플링
    if len(X_train) > 100000:
        print("⚡ 샘플링 적용 (속도 개선)")
        X_sample = X_train.sample(n=50000, random_state=42)
        y_sample = y_train.loc[X_sample.index]
    else:
        X_sample = X_train
        y_sample = y_train

    # 2. MI 기반 선택
    selected_features, feat_df = mutual_info_selection(X_sample, y_sample, ratio=0.8)

    print("MI 선택 feature 수:", len(selected_features))

    # 3. 전체 데이터에 적용
    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]

    # 4. tensor 변환
    X_train_t = torch.FloatTensor(X_train_sel.values)
    y_train_t = torch.LongTensor(y_train.values)

    X_test_t = torch.FloatTensor(X_test_sel.values)
    y_test_t = torch.LongTensor(y_test.values)

    # 5. 저장
    torch.save((X_train_t, y_train_t, selected_features), OUT_TRAIN_PT)
    torch.save((X_test_t, y_test_t, selected_features), OUT_TEST_PT)

    # 6. txt 저장
    with open(OUT_TXT, 'w') as f:
        for feat in selected_features:
            f.write(feat + '\n')

    print("✅ Mutual Information 기반 Top 80% 완료")
    print("저장 위치:")
    print(OUT_TRAIN_PT)
    print(OUT_TEST_PT)
    print(OUT_TXT)


if __name__ == "__main__":
    main()