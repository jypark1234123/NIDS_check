import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

# ===============================
# 📌 경로
# ===============================
TRAIN_PT = r"C:\26-1\PT_files\2017train_elasticPI_31.pt"
TEST_PT  = r"C:\26-1\PT_files\2017test_elasticPI_31.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ===============================
# 📌 데이터 로드
# ===============================
def load_pt(path):
    X_t, y_t, feature_names = torch.load(path)
    return X_t.to(DEVICE), y_t.to(DEVICE)


# ===============================
# 📌 SAE + Attention 모델 (개선버전)
# ===============================
class SAE_Attention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # 🔥 Encoder (SAE)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        # 🔥 Attention (feature weighting)
        self.attn_layer = nn.Sequential(
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Softmax(dim=1)
        )

        # 🔥 Classifier
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        x = self.encoder(x)

        # attention 적용
        attn_weights = self.attn_layer(x)
        x = x * attn_weights

        out = self.classifier(x)
        return out


# ===============================
# 📌 평가 함수
# ===============================
def evaluate(model, X, y):
    model.eval()

    with torch.no_grad():
        outputs = model(X)
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        y_true = y.cpu().numpy()

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
    # 1. 데이터 로드
    X_train, y_train = load_pt(TRAIN_PT)
    X_test, y_test = load_pt(TEST_PT)

    input_dim = X_train.shape[1]
    print("Feature 수:", input_dim)

    # 🔥 데이터 누수 체크
    print("Train/Test shape:", X_train.shape, X_test.shape)

    train_sample = X_train[:100].cpu().numpy()
    test_sample = X_test[:100].cpu().numpy()

    same_count = np.sum(train_sample == test_sample)

    print("🔥 Train/Test 동일 값 개수:", same_count)

    # 2. 모델
    model = SAE_Attention(input_dim).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ===============================
    # 🔥 Training
    # ===============================
    start_train = time.time()

    epochs = 10
    batch_size = 1024

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(X_train.size(0))

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
    # 🔥 Inference + Evaluation
    # ===============================
    start_inf = time.time()
    acc, f1, roc, far = evaluate(model, X_test, y_test)
    inference_time = time.time() - start_inf

    # ===============================
    # 🔥 Memory
    # ===============================
    if DEVICE == "cuda":                          # ← 들여쓰기 4칸 있어야 함
        memory = torch.cuda.max_memory_allocated() / (1024**2)
    else:
        import psutil, os
        process = psutil.Process(os.getpid())
        memory = process.memory_info().rss / (1024**2)
    # ===============================
    # 🔥 Parameter Count
    # ===============================
    param_count = sum(p.numel() for p in model.parameters())

    # ===============================
    # 🔥 모델 저장
    # ===============================
    torch.save(model.state_dict(), r"C:\26-1\PTH\2017sae_attention_elasticPI_31.pth")

    # ===============================
    # 🔥 결과 출력
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