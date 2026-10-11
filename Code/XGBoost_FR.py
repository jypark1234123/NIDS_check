import torch
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import os

# 📌 기본 경로
BASE_DIR = r"C:\26-1"

PT_DIR = os.path.join(BASE_DIR, "PT_files")
FEATURE_DIR = os.path.join(BASE_DIR, "Feature_Text")

# 📌 입력 파일
TRAIN_PT = os.path.join(PT_DIR, "2018_train.pt")
TEST_PT = os.path.join(PT_DIR, "2018_test.pt")

# 📌 출력 파일
OUT_TRAIN_PT = os.path.join(PT_DIR, "2018train_XG80.pt")
OUT_TEST_PT = os.path.join(PT_DIR, "2018test_XG80.pt")
FEATURE_TXT = os.path.join(FEATURE_DIR, "2018_XG_top80.txt")


def load_pt(path):
    X_t, y_t, feature_names = torch.load(path)
    
    X = pd.DataFrame(X_t.numpy(), columns=feature_names)
    y = pd.Series(y_t.numpy())
    
    return X, y, feature_names


def select_features_xgb(X, y, ratio=0.8):
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        n_jobs=-1,
        random_state=42
    )
    
    model.fit(X, y)
    
    importances = model.feature_importances_
    
    feat_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    }).sort_values(by='importance', ascending=False)
    
    top_n = int(len(feat_df) * ratio)
    selected_features = feat_df['feature'].iloc[:top_n].tolist()
    
    return selected_features, feat_df


def save_feature_txt(features, path):
    with open(path, 'w') as f:
        for feat in features:
            f.write(feat + '\n')


def main():
    # 1. 데이터 로드
    X_train, y_train, feature_names = load_pt(TRAIN_PT)
    X_test, y_test, _ = load_pt(TEST_PT)

    print("Original feature count:", len(feature_names))

    # 2. XGBoost feature selection
    selected_features, _ = select_features_xgb(X_train, y_train, ratio=0.8)

    print("Selected feature count:", len(selected_features))

    # 3. feature 적용
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

    save_feature_txt(selected_features, FEATURE_TXT)

    print("✅ XGBoost Feature Selection 완료")
    print("저장 위치:")
    print(OUT_TRAIN_PT)
    print(OUT_TEST_PT)
    print(FEATURE_TXT)


if __name__ == "__main__":
    main()