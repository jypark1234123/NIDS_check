import os
import glob
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
import torch


BASE_DIR = r"C:\26-1"
DATA_DIR = os.path.join(BASE_DIR, "CICIDS2018_CSV")
PT_DIR = os.path.join(BASE_DIR, "PT_files")

TRAIN_OUT = os.path.join(PT_DIR, "2018_train.pt")
TEST_OUT = os.path.join(PT_DIR, "2018_test.pt")


# =========================
# CSV 로드 + split
# =========================
def load_split_csv(data_dir):
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

    train_list = []
    test_list = []

    for file in csv_files:
        print(f"Loading: {file}")
        df = pd.read_csv(file, encoding='utf-8', low_memory=False)

        if "Friday" in file:
            test_list.append(df)
        else:
            train_list.append(df)

    train_df = pd.concat(train_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    return train_df, test_df


# =========================
# 공통 전처리
# =========================
def basic_clean(df):
    df.columns = df.columns.str.strip()

    # 🔥 헤더 중복 제거
    if 'Dst Port' in df.columns:
        df = df[df['Dst Port'] != 'Dst Port']

    # 컬럼 제거
    drop_cols = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    return df


# =========================
# Feature / Label 분리
# =========================
def split_xy(df):
    X = df.drop(columns=['Label'])
    y = df['Label']

    # Label 처리 (binary)
    y = y.astype(str).str.strip().str.upper()
    y_binary = (y != 'BENIGN').astype(int)

    return X, y_binary


# =========================
# 전체 파이프라인
# =========================
def preprocess():
    train_df, test_df = load_split_csv(DATA_DIR)

    # 1. 기본 정리
    train_df = basic_clean(train_df)
    test_df = basic_clean(test_df)

    # 2. X, y 분리
    X_train, y_train = split_xy(train_df)
    X_test, y_test = split_xy(test_df)

    # 3. 🔥 컬럼 통일 (핵심)
    common_cols = list(set(X_train.columns) & set(X_test.columns))

    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    feature_names = common_cols

    print("Feature 개수:", len(feature_names))

    # 4. 숫자 변환 (문자 제거)
    X_train = X_train.apply(pd.to_numeric, errors='coerce')
    X_test = X_test.apply(pd.to_numeric, errors='coerce')

    # 5. NaN / Inf 처리
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

    # 6. 🔥 Scaling (train 기준)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    print("Train label 분포:")
    print(pd.Series(y_train).value_counts())

    print("Test label 분포:")
    print(pd.Series(y_test).value_counts())

    return X_train, X_test, y_train.values, y_test.values, feature_names


# =========================
# Torch 변환
# =========================
def to_tensor(X, y):
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    return X_tensor, y_tensor


# =========================
# MAIN
# =========================
def main():
    X_train, X_test, y_train, y_test, feature_names = preprocess()

    # 안전 체크
    if len(np.unique(y_train)) < 2:
        raise ValueError("❌ 클래스 하나뿐")

    X_train_t, y_train_t = to_tensor(X_train, y_train)
    X_test_t, y_test_t = to_tensor(X_test, y_test)

    print("Torch Train:", X_train_t.shape)
    print("Torch Test:", X_test_t.shape)

    torch.save((X_train_t, y_train_t, feature_names), TRAIN_OUT)
    torch.save((X_test_t, y_test_t, feature_names), TEST_OUT)

    print("✅ 전처리 완료")
    print(TRAIN_OUT)
    print(TEST_OUT)


if __name__ == "__main__":
    main()