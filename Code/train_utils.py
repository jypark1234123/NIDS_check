import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import models


# =========================
# 다음 layer 입력 생성
# =========================
def get_next_layer_inputs(encoder, dataloader, device):
    encoder.eval()
    features_list = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            features = torch.sigmoid(encoder(inputs))
            features_list.append(features.cpu())

    return torch.cat(features_list, dim=0)


# =========================
# 단일 SAE 학습
# =========================
def train_sae(sae_layer, dataloader, device,
              lr=1e-3, epochs=10):

    optimizer = optim.Adam(sae_layer.parameters(), lr=lr)

    sae_layer.train()
    loss_history = []

    for epoch in range(epochs):
        total_loss = 0

        for inputs, _ in dataloader:
            inputs = inputs.to(device)

            # forward
            encoded, outputs = sae_layer(inputs)

            # 🔥 핵심: models.py loss 사용
            loss = models.loss_function(outputs, inputs, encoded, sae_layer)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_loss = total_loss / len(dataloader)
        loss_history.append(epoch_loss)

        print(f"[SAE] Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")

    return sae_layer, loss_history


# =========================
# SSAE Layer-wise 학습
# =========================
def train_stacked_sae(ssae, train_loader, device,
                      lr=1e-3, epochs=10, batch_size=256):

    current_loader = train_loader
    history = {}

    for i, sae_layer in enumerate(ssae.saes):
        print(f"\n=== Layer {i+1} Pretraining ===")
        print(f"Input: {sae_layer.encoder.in_features} → Hidden: {sae_layer.encoder.out_features}")

        # 새 SAE 생성
        current_sae = models.SparseAutoencoder(
            sae_layer.encoder.in_features,
            sae_layer.encoder.out_features
        ).to(device)

        # 학습
        current_sae, loss_hist = train_sae(
            current_sae,
            current_loader,
            device,
            lr=lr,
            epochs=epochs
        )

        history[f"layer_{i+1}"] = loss_hist

        # 학습된 encoder 저장
        ssae.saes[i] = current_sae

        # 다음 layer 입력 생성
        if i < len(ssae.saes) - 1:
            print(f"→ Extracting features for next layer...")

            X_next = get_next_layer_inputs(current_sae.encoder, current_loader, device)

            dummy_y = torch.zeros(X_next.shape[0], dtype=torch.long)
            dataset = TensorDataset(X_next, dummy_y)

            current_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True
            )

    print("\n✅ SSAE Pretraining Complete")
    return ssae, history