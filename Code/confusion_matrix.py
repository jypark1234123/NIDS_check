import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ===============================
# 경로
# ===============================
TEST_PT = r"C:\26-1\PT_files\2017test.pt"
MODEL_PATH = r"C:\26-1\PTH\2017ssae_attention_full.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ===============================
# 데이터 로드
# ===============================
def load_pt(path):
    X_t, y_t, feature_names = torch.load(path)

    print(f"\nLoaded : {path}")
    print(f"Feature Count : {len(feature_names)}")

    return X_t.to(DEVICE), y_t.to(DEVICE), feature_names


# ===============================
# SSAE + Attention
# ===============================
class SSAE(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

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

        self.attention = nn.Sequential(
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Softmax(dim=1)
        )

        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.ReLU(),

            nn.Linear(128, input_dim)
        )

        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(16, 2)
        )

    def forward(self, x):
        latent = self.encoder(x)

        attn_weights = self.attention(latent)
        latent = latent * attn_weights

        return self.classifier(latent)

    def reconstruct(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)

    def get_latent(self, x):
        return self.encoder(x)

    def sparsity_penalty(self, latent, rho=0.05):
        rho_hat = torch.mean(torch.sigmoid(latent), dim=0)

        kl = torch.sum(
            rho * torch.log((rho + 1e-8) / (rho_hat + 1e-8))
            + (1 - rho) * torch.log((1 - rho + 1e-8) / (1 - rho_hat + 1e-8))
        )
        return kl


# ===============================
# 평가
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

    return acc, f1, roc, far, tn, fp, fn, tp


# ===============================
# Main
# ===============================
def main():

    X_test, y_test, feature_names = load_pt(TEST_PT)

    input_dim = X_test.shape[1]

    print(f"\nInput Dimension : {input_dim}")

    model = SSAE(input_dim).to(DEVICE)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    print("\nModel Loaded Successfully.")

    acc, f1, roc, far, tn, fp, fn, tp = evaluate(
        model,
        X_test,
        y_test
    )

    print("\n========== Evaluation ==========")

    print(f"Accuracy : {acc:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {roc:.4f}")
    print(f"FAR      : {far:.6f}")

    print("\n========== Confusion Matrix ==========")
    print(f"TN : {tn}")
    print(f"FP : {fp}")
    print(f"FN : {fn}")
    print(f"TP : {tp}")


if __name__ == "__main__":
    main()