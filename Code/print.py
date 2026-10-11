import torch


data = torch.load("train.pt")


X_train, y_train, feature_names = data

print(f"feature : {len(feature_names)}\n")


for i, name in enumerate(feature_names):
    print(f"{i}: {name}")