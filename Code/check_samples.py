import torch

paths = [
    r"C:\26-1\PT_files\2017test_xgbPI_31.pt",
    r"C:\26-1\PT_files\2017test_elasticPI_31.pt",
    r"C:\26-1\PT_files\2018test_xgbPI_31.pt",
    r"C:\26-1\PT_files\2018test_elasticPI_31.pt",
]

for p in paths:
    X_t, y_t, feature_names = torch.load(p)
    print(p, "→", X_t.shape[0], "samples")