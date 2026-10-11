import torch
import torch.nn as nn
import torch.optim as optim
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
TRAIN_PT = r"C:\26-1\PT_files\2017train_elasticPI_31.pt"
TEST_PT  = r"C:\26-1\PT_files\2017test_elasticPI_31.pt"

MODEL_SAVE = r"C:\26-1\PTH\2017_transformer_elasticPI_31.pth"

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
# 📌 Transformer 모델
# ===============================
class TransformerIDS(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()

        self.input_dim = input_dim

        # feature → embedding
        self.embedding = nn.Linear(1, d_model)

        # positional encoding (간단 버전)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, input_dim, d_model)
        )

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        # (batch, feature) → (batch, feature, 1)
        x = x.unsqueeze(-1)

        # embedding
        x = self.embedding(x)

        # positional encoding 추가
        x = x + self.pos_embedding

        # transformer
        x = self.transformer(x)

        # pooling (mean)
        x = torch.mean(x, dim=1)

        # classification
        out = self.classifier(x)

        return out


# ===============================
# 📌 평가 함수 (Batch Evaluation)
# ===============================
def evaluate(model, X, y, batch_size=4096):
    model.eval()

    preds_all = []
    probs_all = []
    y_all = []

    with torch.no_grad():
        for i in range(0, X.size(0), batch_size):
            batch_x = X[i:i + batch_size]
            batch_y = y[i:i + batch_size]

            outputs = model(batch_x)

            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            probs_all.append(probs.cpu())
            preds_all.append(preds.cpu())
            y_all.append(batch_y.cpu())

    probs = torch.cat(probs_all).numpy()
    preds = torch.cat(preds_all).numpy()
    y_true = torch.cat(y_all).numpy()

    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds)
    roc = roc_auc_score(y_true, probs)

    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    far = fp / (fp + tn + 1e-8)

    return acc, f1, roc, far


# ===============================
# 📌 메인
# ===============================
def main():
    # 데이터
    X_train, y_train = load_pt(TRAIN_PT)
    X_test, y_test = load_pt(TEST_PT)

    input_dim = X_train.shape[1]

    print(f"\nInput Dimension: {input_dim}")

    # 모델
    model = TransformerIDS(input_dim).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ===============================
    # 🚀 Training
    # ===============================
    epochs = 10
    batch_size = 1024

    start_train = time.time()

    for epoch in range(epochs):
        model.train()

        perm = torch.randperm(X_train.size(0), device=DEVICE)

        total_loss = 0

        for i in range(0, X_train.size(0), batch_size):
            idx = perm[i:i+batch_size]

            batch_x = X_train[idx]
            batch_y = y_train[idx]

            optimizer.zero_grad()

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss:.4f}")

    train_time = time.time() - start_train

    # ===============================
    # 🚀 평가
    # ===============================
    start_inf = time.time()

    acc, f1, roc, far = evaluate(model, X_test, y_test)

    inference_time = time.time() - start_inf

    # ===============================
    # 📦 정보
    # ===============================
    param_count = sum(p.numel() for p in model.parameters())

    if DEVICE == "cuda":
        memory = torch.cuda.max_memory_allocated() / (1024**2)
    else:
        import psutil, os
        process = psutil.Process(os.getpid())
        memory = process.memory_info().rss / (1024 ** 2)

    # ===============================
    # 💾 저장
    # ===============================
    torch.save(model.state_dict(), MODEL_SAVE)

    print(f"\n📁 모델 저장 완료:\n{MODEL_SAVE}")

    # ===============================
    # 📊 출력
    # ===============================
    print("\n===== 결과 =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC: {roc:.4f}")
    print(f"FAR: {far:.6f}")

    print(f"\nTraining Time: {train_time:.2f} sec")
    print(f"Inference Time: {inference_time:.4f} sec")

    print(f"\nMemory Usage: {memory:.2f} MB")
    print(f"Parameter Count: {param_count:,}")


# ===============================
if __name__ == "__main__":
    main()