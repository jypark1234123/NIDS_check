import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ===============================
# 📌 경로
# ===============================
TRAIN_PT = r"C:\26-1\PT_files\2018train_elasticPI_31.pt"
TEST_PT  = r"C:\26-1\PT_files\2018test_elasticPI_31.pt"

MODEL_SAVE = r"C:\26-1\PTH\2018ssae_elasticPI_31.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ===============================
# 📌 데이터 로드
# ===============================
def load_pt(path):
    X_t, y_t, feature_names = torch.load(path)

    print(f"\nLoaded: {path}")
    print(f"Feature Count: {len(feature_names)}")

    return X_t.to(DEVICE), y_t.to(DEVICE)


# ===============================
# 📌 SSAE 모델
# ===============================
class SSAE(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.ReLU(),

            nn.Linear(128, input_dim)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(16, 2)
        )

    def forward(self, x):
        latent = self.encoder(x)
        out = self.classifier(latent)
        return out

    def reconstruct(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon

    def get_latent(self, x):
        return self.encoder(x)

    # Sparse Penalty
    def sparsity_penalty(self, latent, rho=0.05):

        rho_hat = torch.mean(
            torch.sigmoid(latent),
            dim=0
        )

        kl = torch.sum(
            rho * torch.log((rho + 1e-8) / (rho_hat + 1e-8))
            +
            (1 - rho) *
            torch.log(
                (1 - rho + 1e-8)
                /
                (1 - rho_hat + 1e-8)
            )
        )

        return kl


# ===============================
# 📌 평가
# ===============================
def evaluate(model, X, y):

    model.eval()

    with torch.no_grad():

        outputs = model(X)

        probs = torch.softmax(
            outputs,
            dim=1
        )[:, 1].cpu().numpy()

        preds = torch.argmax(
            outputs,
            dim=1
        ).cpu().numpy()

        y_true = y.cpu().numpy()

    acc = accuracy_score(y_true, preds)

    f1 = f1_score(y_true, preds)

    roc = roc_auc_score(y_true, probs)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        preds
    ).ravel()

    far = fp / (fp + tn + 1e-8)

    return acc, f1, roc, far


# ===============================
# 📌 메인
# ===============================
def main():

    # ---------------------------
    # 데이터 로드
    # ---------------------------
    X_train, y_train = load_pt(TRAIN_PT)
    X_test, y_test = load_pt(TEST_PT)

    input_dim = X_train.shape[1]

    print(f"\nInput Dimension: {input_dim}")

    # ---------------------------
    # 모델
    # ---------------------------
    model = SSAE(input_dim).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # ---------------------------
    # 학습
    # ---------------------------
    epochs = 10
    batch_size = 1024

    start_train = time.time()

    for epoch in range(epochs):

        model.train()

        perm = torch.randperm(
            X_train.size(0),
            device=DEVICE
        )

        total_loss = 0

        for i in range(
            0,
            X_train.size(0),
            batch_size
        ):

            idx = perm[i:i+batch_size]

            batch_x = X_train[idx]
            batch_y = y_train[idx]

            optimizer.zero_grad()

            # 분류
            outputs = model(batch_x)

            cls_loss = criterion(
                outputs,
                batch_y
            )

            # 재구성
            recon = model.reconstruct(
                batch_x
            )

            recon_loss = F.mse_loss(
                recon,
                batch_x
            )

            # Sparse Penalty
            latent = model.get_latent(
                batch_x
            )

            sparse_loss = model.sparsity_penalty(
                latent
            )

            # Total Loss
            loss = (
                cls_loss
                + 0.1 * recon_loss
                + 0.001 * sparse_loss
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{epochs} "
            f"Loss: {total_loss:.4f}"
        )

    train_time = (
        time.time() - start_train
    )

    # ---------------------------
    # 평가
    # ---------------------------
    start_inf = time.time()

    acc, f1, roc, far = evaluate(
        model,
        X_test,
        y_test
    )

    inference_time = (
        time.time() - start_inf
    )

    # ---------------------------
    # 메모리 사용량
    # ---------------------------
    if DEVICE == "cuda":
        memory = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 2)
        )
    else:
        import psutil, os
        process = psutil.Process(os.getpid())
        memory = process.memory_info().rss / (1024 ** 2)

    # ---------------------------
    # 파라미터 수
    # ---------------------------
    param_count = sum(
        p.numel()
        for p in model.parameters()
    )

    # ---------------------------
    # 모델 저장
    # ---------------------------
    torch.save(
        model.state_dict(),
        MODEL_SAVE
    )

    print(
        f"\nModel Saved:\n{MODEL_SAVE}"
    )

    # ---------------------------
    # 결과 출력
    # ---------------------------
    print("\n===== 결과 =====")

    print(f"Accuracy : {acc:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {roc:.4f}")
    print(f"FAR      : {far:.6f}")

    print(
        f"\nTraining Time : "
        f"{train_time:.2f} sec"
    )

    print(
        f"Inference Time : "
        f"{inference_time:.4f} sec"
    )

    print(
        f"\nMemory Usage : "
        f"{memory:.2f} MB"
    )

    print(
        f"Parameter Count : "
        f"{param_count:,}"
    )


# ===============================
if __name__ == "__main__":
    main()