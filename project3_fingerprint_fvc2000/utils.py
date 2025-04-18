import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
import matplotlib.pyplot as plt
import os

class SiameseDataset(Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file, allow_pickle=True)
        self.pairs = data["pairs"]
        self.labels = data["labels"]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path = self.pairs[idx]
        label = self.labels[idx]

        img1 = np.load(img1_path)
        img2 = np.load(img2_path)

        img1_tensor = torch.tensor(img1).unsqueeze(0).float()  # [1, 300, 300]
        img2_tensor = torch.tensor(img2).unsqueeze(0).float()
        label_tensor = torch.tensor(label).float()

        return img1_tensor, img2_tensor, label_tensor


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        distances = torch.nn.functional.pairwise_distance(output1, output2)
        loss = torch.mean(
            (1 - label) * torch.pow(distances, 2) +
            label * torch.pow(torch.clamp(self.margin - distances, min=0.0), 2)
        )
        return loss


def plot_loss(train_losses, val_losses, save_path=None):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
        print(f"Loss plot saved to {save_path}")
    else:
        plt.show()


def save_checkpoint(model, path="checkpoints/best_model.pt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")
