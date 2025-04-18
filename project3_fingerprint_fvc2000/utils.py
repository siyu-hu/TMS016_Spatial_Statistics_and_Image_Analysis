import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import torch.nn.functional as F
from tqdm import tqdm

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




def plot_distance_distribution(model, dataloader, device="cpu", save_path=None):
    import torch.nn.functional as F
    model.eval()

    pos_distances = []
    neg_distances = []

    with torch.no_grad():
        for img1, img2, label in dataloader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            out1, out2 = model(img1, img2)
            dist = F.pairwise_distance(out1, out2)

            for d, l in zip(dist, label):
                if l == 1:
                    pos_distances.append(d.item())
                else:
                    neg_distances.append(d.item())


    plt.figure(figsize=(8,5))
    plt.hist(pos_distances, bins=50, alpha=0.6, label="Positive (same finger)")
    plt.hist(neg_distances, bins=50, alpha=0.6, label="Negative (different finger)")
    plt.xlabel("Distance")
    plt.ylabel("Count")
    plt.title("Distance Distribution")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
        print(f" Distance histogram saved to {save_path}")
    else:
        plt.show()




def plot_roc_curve(model, dataloader, device="cpu", save_path=None):
    import torch.nn.functional as F
    model.eval()

    all_distances = []
    all_labels = []

    with torch.no_grad():
        for img1, img2, label in dataloader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            out1, out2 = model(img1, img2)
            dist = F.pairwise_distance(out1, out2)

            all_distances.extend(dist.cpu().numpy())
            all_labels.extend(label.cpu().numpy())


    # score = -distance
    fpr, tpr, thresholds = roc_curve(all_labels, -1 * np.array(all_distances)) 
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")

    if save_path:
        plt.savefig(save_path)
        print(f" ROC curve saved to {save_path}")
    else:
        plt.show()

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

def plot_accuracy_vs_threshold(model, dataloader, device="cpu", save_path=None):
    model.eval()
    thresholds = np.linspace(0.01, 0.15, 10)  
    accuracies = []

    all_distances = []
    all_labels = []

    print(" Extracting embeddings...")
    with torch.no_grad():
        for img1, img2, label in tqdm(dataloader, desc="Forward pass"):
            img1, img2 = img1.to(device), img2.to(device)
            out1, out2 = model(img1, img2)
            dist = F.pairwise_distance(out1, out2)
            all_distances.extend(dist.cpu().numpy())
            all_labels.extend(label.numpy())

    all_distances = np.array(all_distances)
    all_labels = np.array(all_labels)

    print("Calculating accuracy for thresholds...")
    for t in tqdm(thresholds, desc="Threshold"):
        predictions = (all_distances < t).astype(np.float32)
        correct = (predictions == all_labels).sum()
        acc = correct / len(all_labels)
        accuracies.append(acc)

    best_idx = np.argmax(accuracies)
    best_threshold = thresholds[best_idx]
    best_acc = accuracies[best_idx]
    print(f"\n Best threshold = {best_threshold:.3f} → Accuracy = {best_acc*100:.2f}%")

    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, accuracies, marker='o')
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Threshold")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        print(f"Accuracy vs Threshold plot saved to {save_path}")
    else:
        plt.show()