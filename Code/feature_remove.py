import torch
import numpy as np


X_train, y_train, feature_names = torch.load("train.pt")
X_test, y_test, _ = torch.load("test.pt")

X_train_np = X_train.numpy()
X_test_np = X_test.numpy()

print("Original shape:", X_train_np.shape)


feature_names = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min"
]


remove_features = [
    "Subflow Bwd Packets",
    "min_seg_size_forward",
    "Fwd IAT Min",
    "Flow Duration",
    "PSH Flag Count",
    "Total Backward Packets",
    "URG Flag Count",
    "Flow Packets/s",
    "Idle Mean",
    "Active Std",
    "Fwd IAT Mean",
    "Fwd Packets/s",
    "Bwd IAT Max",
    "Idle Min",
    "Fwd IAT Max",
    "Bwd IAT Total"
]


remove_idx = []

for f in remove_features:
    if f in feature_names:
        remove_idx.append(feature_names.index(f))
    else:
        print(f"Warning: {f} not found")

print("Remove index:", remove_idx)


X_train_new = np.delete(X_train_np, remove_idx, axis=1)
X_test_new = np.delete(X_test_np, remove_idx, axis=1)

print("New shape:", X_train_new.shape)


X_train_new = torch.tensor(X_train_new, dtype=torch.float32)
X_test_new = torch.tensor(X_test_new, dtype=torch.float32)


torch.save((X_train_new, y_train), "train_removed.pt")
torch.save((X_test_new, y_test), "test_removed.pt")

print("Complete!")