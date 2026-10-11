import os
import glob
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import torch


BASE_DIR = r"C:\26-1"
DATA_DIR = os.path.join(BASE_DIR, "MachineLearningCSV", "MachineLearningCVE")
PT_DIR = os.path.join(BASE_DIR, "PT_files")
TRAIN_OUT = os.path.join(PT_DIR, "2017train.pt")
TEST_OUT = os.path.join(PT_DIR, "2017test.pt")


def load_all_csv(data_dir):
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    df_list = []
    for file in csv_files:
        print(f"Loading: {file}")
        df = pd.read_csv(file, encoding='latin1')
        df_list.append(df)
    
    df_all = pd.concat(df_list, ignore_index=True)
    print(f"Total shape: {df_all.shape}")
    
    return df_all


def preprocess_cicids(df):
    df.columns = df.columns.str.strip()

    drop_cols = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    X = df.drop(columns=['Label'])
    y = df['Label']

    feature_names = X.columns.tolist()

    y = y.astype(str).str.strip().str.upper()
    y_binary = (y != 'BENIGN').astype(int)

    print("Label 분포 (전처리 전):")
    print(y_binary.value_counts())

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_binary.values, feature_names


def make_torch_dataset(X, y):
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    return X_tensor, y_tensor


def main():
    df = load_all_csv(DATA_DIR)

    X, y, feature_names = preprocess_cicids(df)

    print("After preprocessing:", X.shape)

    print("🔥 Label distribution (최종):")
    print(pd.Series(y).value_counts())

    # 🔥 안전 체크
    if len(np.unique(y)) < 2:
        raise ValueError("❌ 클래스가 하나뿐임")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_t, y_train_t = make_torch_dataset(X_train, y_train)
    X_test_t, y_test_t = make_torch_dataset(X_test, y_test)

    print("Train shape:", X_train_t.shape)
    print("Test shape:", X_test_t.shape)

    # 📌 저장 위치 수정됨
    torch.save((X_train_t, y_train_t, feature_names), TRAIN_OUT)
    torch.save((X_test_t, y_test_t, feature_names), TEST_OUT)

    print("✅ 2017 전처리 완료 및 저장됨")
    print(TRAIN_OUT)
    print(TEST_OUT)


if __name__ == "__main__":
    main()