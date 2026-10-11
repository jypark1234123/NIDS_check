import torch
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
import os

# 📌 기본 경로
BASE_DIR = r"C:\26-1"

PT_DIR = os.path.join(BASE_DIR, "PT_files")
FEATURE_DIR = os.path.join(BASE_DIR, "Feature_Text")

# 📌 입력 파일
TRAIN_PT = os.path.join(PT_DIR, "2018_train.pt")
TEST_PT = os.path.join(PT_DIR, "2018_test.pt")

# 📌 출력 파일
OUT_TRAIN_PT = os.path.join(PT_DIR, "2018train_el80.pt")
OUT_TEST_PT = os.path.join(PT_DIR, "2018test_el80.pt")
FEATURE_TXT = os.path.join(FEATURE_DIR, "2018_elastic_top80.txt")


def load_pt(path):
    X_t, y_t, feature_names = torch.load(path)
    
    X = pd.DataFrame(X_t.numpy(), columns=feature_names)
    y = pd.Series(y_t.numpy())
    
    return X, y, feature_names


def elasticnet_feature_selection(X, y, ratio=0.8):
    # 🔥 scaling 필수
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 🔥 느슨한 LASSO
    model = ElasticNet(
        alpha=0.001,
        l1_ratio=0.8,
        max_iter=5000,
        random_state=42
    )

    model.fit(X_scaled, y)

    coef = np.abs(model.coef_)

    feat_df = pd.DataFrame({
        'feature': X.columns,
        'importance': coef
    }).sort_values(by='importance', ascending=False)

    top_n = int(len(feat_df) * ratio)
    selected_features = feat_df['feature'].iloc[:top_n].tolist()

    return selected_features, feat_df


def save_feature_txt(features, path):
    with open(path, 'w') as f:
        for feat in features:
            f.write(feat + '\n')


def apply_feature_selection(X, selected_features):
    return X[selected_features]


def to_tensor(X, y):
    X_t = torch.FloatTensor(X.values)
    y_t = torch.LongTensor(y.values)
    return X_t, y_t


def main():
    # 1. 데이터 로드
    X_train, y_train, feature_names = load_pt(TRAIN_PT)
    X_test, y_test, _ = load_pt(TEST_PT)

    print("Original feature count:", len(feature_names))

    # 2. ElasticNet feature selection
    selected_features, feat_df = elasticnet_feature_selection(X_train, y_train, ratio=0.8)

    print("Selected feature count:", len(selected_features))

    # 3. 적용
    X_train_sel = apply_feature_selection(X_train, selected_features)
    X_test_sel = apply_feature_selection(X_test, selected_features)

    # 4. tensor 변환
    X_train_t, y_train_t = to_tensor(X_train_sel, y_train)
    X_test_t, y_test_t = to_tensor(X_test_sel, y_test)

    # 5. 저장
    torch.save((X_train_t, y_train_t, selected_features), OUT_TRAIN_PT)
    torch.save((X_test_t, y_test_t, selected_features), OUT_TEST_PT)

    # 6. txt 저장
    save_feature_txt(selected_features, FEATURE_TXT)

    print("✅ ElasticNet Feature Selection 완료")
    print("저장 위치:")
    print(OUT_TRAIN_PT)
    print(OUT_TEST_PT)
    print(FEATURE_TXT)


if __name__ == "__main__":
    main()