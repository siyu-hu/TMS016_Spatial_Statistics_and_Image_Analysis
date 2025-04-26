# siamese.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseNetwork(nn.Module):
    """
    Lightweight Siamese CNN for 300 * 300 grayscale fingerprints.

    Input  : two tensors of shape [B, 1, 300, 300]
    Output : two L2-normalised embedding tensors of shape [B, embedding_dim]

    Total parameters ≈ 0.11 M (vs. ≈ 90 M in the original design).
    """

    def __init__(self, embedding_dim: int = 128):
        super().__init__()

        # -------- Convolutional feature extractor --------
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),   # [B, 16, 300, 300]
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),                  # [B, 16, 150, 150]

            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # [B, 32, 150, 150]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),                  # [B, 32, 75, 75]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # [B, 64, 75, 75]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),                  # [B, 64, 25, 25]

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # [B, 128, 25, 25]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))                  # [B, 128, 1, 1]
        )

        # -------- Projection to low-dimensional embedding --------
        self.projection = nn.Sequential(
            nn.Flatten(),                                 # [B, 128]
            nn.Linear(128, embedding_dim)                 # [B, embedding_dim]
        )

    # Forward pass for one branch
    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.projection(x)
        x = F.normalize(x, p=2, dim=1)                   # L2 normalisation
        return x

    # Siamese forward: return embeddings for both inputs
    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        return self.forward_once(x1), self.forward_once(x2)


# ------------- quick self-test -------------
if __name__ == "__main__":
    net = SiameseNetwork()
    dummy_a = torch.randn(4, 1, 300, 300)   # batch = 4
    dummy_b = torch.randn(4, 1, 300, 300)
    emb_a, emb_b = net(dummy_a, dummy_b)
    print(emb_a.shape, emb_b.shape)         # torch.Size([4, 128]) torch.Size([4, 128])
