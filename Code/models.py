import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# Sparse Autoencoder
# =========================
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SparseAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Linear(input_dim, hidden_dim)
        
        # Decoder
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # sigmoid activation (논문 기준)
        encoded = torch.sigmoid(self.encoder(x))
        decoded = torch.sigmoid(self.decoder(encoded))
        return encoded, decoded


# =========================
# KL Divergence (Sparsity)
# =========================
def kl_divergence(rho, rho_hat):
    rho_hat = torch.clamp(rho_hat, 1e-7, 1 - 1e-7)
    
    return (
        rho * torch.log(rho / rho_hat) +
        (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
    )


# =========================
# Loss Function
# =========================
def loss_function(recon_x, x, hidden_activations, model,
                  sparsity_param=0.05,
                  sparsity_weight=1e-3,
                  weight_decay=1e-5):

    # 1. Reconstruction Loss (MSE)
    mse = F.mse_loss(recon_x, x, reduction='mean') * 0.5

    # 2. Sparsity (KL Divergence)
    rho_hat = torch.mean(hidden_activations, dim=0)
    kl = torch.sum(kl_divergence(sparsity_param, rho_hat))
    sparsity_term = sparsity_weight * kl

    # 3. Weight decay (L2)
    l2 = 0.0
    for param in model.parameters():
        l2 += torch.norm(param, 2) ** 2

    l2_term = (weight_decay / 2.0) * l2

    return mse + sparsity_term + l2_term


# =========================
# Stacked Sparse Autoencoder
# =========================
class StackedSparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_layers):
        super(StackedSparseAutoencoder, self).__init__()

        self.saes = nn.ModuleList()

        current_dim = input_dim
        for hidden_dim in hidden_layers:
            self.saes.append(SparseAutoencoder(current_dim, hidden_dim))
            current_dim = hidden_dim

    def forward_features(self, x):
        """
        SSAE를 통과한 feature 추출
        (decoder 안씀)
        """
        out = x
        for sae in self.saes:
            out = torch.sigmoid(sae.encoder(out))
        return out