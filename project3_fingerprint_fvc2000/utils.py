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
from sklearn.metrics import precision_score, recall_score, f1_score


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

        if isinstance(img1_path, str):
            img1 = np.load(img1_path)
        else:
            img1 = img1_path  #  array

        if isinstance(img2_path, str):
            img2 = np.load(img2_path)
        else:
            img2 = img2_path

        img1 = torch.tensor(img1).unsqueeze(0)  # [1, H, W]
        img2 = torch.tensor(img2).unsqueeze(0)
        label = torch.tensor(label).float()

        return img1, img2, label


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


def print_classification_report(tp, tn, fp, fn):
    total = tp + tn + fp + fn
    acc = (tp + tn) / total * 100
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n Classification Report:")
    print(f"  Accuracy  : {acc:.2f}%")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"   F1 Score  : {f1:.4f}")
    print(f"  TP={tp} | TN={tn} | FP={fp} | FN={fn}")

# def plot_accuracy_vs_threshold(model, dataloader, device="cpu", save_path=None):
#     model.eval()
#     thresholds = np.linspace(0.01, 0.15, 10)  
#     accuracies = []

#     all_distances = []
#     all_labels = []

#     print(" Extracting embeddings...")
#     with torch.no_grad():
#         for img1, img2, label in tqdm(dataloader, desc="Forward pass"):
#             img1, img2 = img1.to(device), img2.to(device)
#             out1, out2 = model(img1, img2)
#             dist = F.pairwise_distance(out1, out2)
#             all_distances.extend(dist.cpu().numpy())
#             all_labels.extend(label.cpu().numpy())

#     all_distances = np.array(all_distances)
#     all_labels = np.array(all_labels)

#     print("Calculating accuracy for thresholds...")
#     for t in tqdm(thresholds, desc="Threshold"):
#         predictions = (all_distances < t).astype(np.float32)
#         correct = (predictions == all_labels).sum()
#         acc = correct / len(all_labels)
#         accuracies.append(acc)

#     best_idx = np.argmax(accuracies)
#     best_threshold = thresholds[best_idx]
#     best_acc = accuracies[best_idx]
#     print(f"\n Best threshold = {best_threshold:.3f} → Accuracy = {best_acc*100:.2f}%")

#     plt.figure(figsize=(7, 5))
#     plt.plot(thresholds, accuracies, marker='o')
#     plt.xlabel("Threshold")
#     plt.ylabel("Accuracy")
#     plt.title("Accuracy vs Threshold")
#     plt.grid(True)

#     if save_path:
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         plt.savefig(save_path)
#         print(f"Accuracy vs Threshold plot saved to {save_path}")
#     else:
#         plt.show()

#     # Save the best threshold to a text file
#     threshold_txt_path = os.path.join(os.path.dirname(save_path), "best_threshold.txt")
#     with open(threshold_txt_path, "w") as f:
#         f.write(f"{best_threshold:.6f}")
#     print(f"Best threshold saved to {threshold_txt_path}")


def plot_metrics_vs_threshold(model, dataloader, device="cpu", save_path=None):
    model.eval()
    thresholds = np.linspace(0.01, 0.3, 40)
    all_distances = []
    all_labels = []

    print("Extracting embeddings...")
    with torch.no_grad():
        for img1, img2, label in tqdm(dataloader, desc="Forward pass"):
            img1, img2 = img1.to(device), img2.to(device)
            out1, out2 = model(img1, img2)
            dist = F.pairwise_distance(out1, out2)
            all_distances.extend(dist.cpu().numpy())
            all_labels.extend(label.numpy())

    all_distances = np.array(all_distances)
    all_labels = np.array(all_labels)

    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    print("Calculating metrics for thresholds...")
    for t in tqdm(thresholds, desc="Threshold"):
        preds = (all_distances < t).astype(int)
        accuracies.append((preds == all_labels).mean())
        precisions.append(precision_score(all_labels, preds, zero_division=0))
        recalls.append(recall_score(all_labels, preds, zero_division=0))
        f1s.append(f1_score(all_labels, preds, zero_division=0))

    best_idx = np.argmax(f1s)
    best_threshold = thresholds[best_idx]
    best_f1 = f1s[best_idx]
    print(f"\n Best threshold = {best_threshold:.6f} → F1 Score = {best_f1:.6f}")

    # Plot all metrics
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, accuracies, label="Accuracy")
    plt.plot(thresholds, precisions, label="Precision")
    plt.plot(thresholds, recalls, label="Recall")
    plt.plot(thresholds, f1s, label="F1 Score")
    plt.xlabel("Threshold")
    plt.ylabel("Metric Value")
    plt.title("Metrics vs Threshold")
    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Metrics plot saved to {save_path}")
    else:
        plt.show()

    with open("./project3_fingerprint_fvc2000/outputs/best_threshold.txt", "w") as f:
        f.write(f"{best_threshold:.6f}")
    print("Best threshold saved to outputs/best_threshold.txt")

    return best_threshold
