import torch
import pandas as pd
import numpy as np
import os

from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

# ===============================
# 📌 경로
# ===============================
BASE_DIR = r"C:\26-1"

PT_DIR = os.path.join(BASE_DIR, "PT_files")
FEATURE_DIR = os.path.join(BASE_DIR, "Feature_Text")

# 입력 (MI 결과)
TRAIN_PT = os.path.join(PT_DIR, "2018train_mi80.pt")
TEST_PT  = os.path.join(PT_DIR, "2018test_mi80.pt")

# 출력 (Top 31)
OUT_TRAIN_PT = os.path.join(PT_DIR, "2018train_elasticPI_23.pt")
OUT_TEST_PT  = os.path.join(PT_DIR, "2018test_elasticPI_23.pt")

OUT_TXT = os.path.join(FEATURE_DIR, "2018_elasticPI_top23.txt")


# ===============================
# 📌 PT 로드
# ===============================
def load_pt(path):
    X_t, y_t, feature_names = torch.load(path)

    X = pd.DataFrame(X_t.numpy(), columns=feature_names)
    y = pd.Series(y_t.numpy())

    return X, y, feature_names


# ===============================
# 📌 ElasticNet PI
# ===============================
def elastic_permutation_importance(X, y):
    # 🔥 scaling 필수
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 🔥 모델
    model = ElasticNet(
        alpha=0.001,
        l1_ratio=0.8,
        max_iter=5000,
        random_state=42
    )

    print("🚀 ElasticNet 학습 중...")
    model.fit(X_scaled, y)

    # 샘플링
    if len(X_scaled) > 50000:
        print("⚡ PI 계산용 샘플링")
        idx = np.random.RandomState(42).choice(len(X_scaled), 50000, replace=False)
        X_sample = X_scaled[idx]
        y_sample = y.iloc[idx]
    else:
        X_sample = X_scaled
        y_sample = y

    print("🚀 Permutation Importance 계산 중...")
    result = permutation_importance(
        model,
        X_sample,
        y_sample,
        scoring='neg_mean_squared_error',  # 🔥 핵심
        n_repeats=5,
        random_state=42,
        n_jobs=-1
    )

    importance = result.importances_mean

    feat_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importance
    }).sort_values(by='importance', ascending=False)

    return feat_df, model, scaler


# ===============================
# 📌 Top-K 선택
# ===============================
def select_top_k(feat_df, k=23):
    return feat_df['feature'].iloc[:k].tolist()


# ===============================
# 📌 Tensor 변환
# ===============================
def to_tensor(X, y):
    X_t = torch.FloatTensor(X.values)
    y_t = torch.LongTensor(y.values)
    return X_t, y_t


# ===============================
# 📌 메인
# ===============================
def main():
    # 1. 데이터 로드
    X_train, y_train, feature_names = load_pt(TRAIN_PT)
    X_test, y_test, _ = load_pt(TEST_PT)

    print("Feature 개수:", len(feature_names))

    # 2. PI
    feat_df, model, scaler = elastic_permutation_importance(X_train, y_train)

    # 3. Top 23 선택
    selected_features = select_top_k(feat_df, k=23)

    print("🔥 선택된 feature 수:", len(selected_features))

    # 4. feature 선택
    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]

    # 5. Tensor 변환
    X_train_t, y_train_t = to_tensor(X_train_sel, y_train)
    X_test_t, y_test_t = to_tensor(X_test_sel, y_test)

    # 6. 저장
    torch.save((X_train_t, y_train_t, selected_features), OUT_TRAIN_PT)
    torch.save((X_test_t, y_test_t, selected_features), OUT_TEST_PT)

    # 7. txt 저장
    with open(OUT_TXT, 'w') as f:
        for feat in selected_features:
            f.write(feat + '\n')

    # 8. 출력
    print("\n🔥 Top 10 Features:")
    print(feat_df.head(10))

    print("\n✅ ElasticNet PI Top31 완료")
    print("저장 위치:")
    print(OUT_TRAIN_PT)
    print(OUT_TXT)


# ===============================
if __name__ == "__main__":
    main()