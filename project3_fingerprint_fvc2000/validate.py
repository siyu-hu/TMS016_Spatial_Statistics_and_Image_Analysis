import torch
import numpy as np
from siamese_model import SiameseNetwork
from utils import SiameseDataset, print_classification_report
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import plot_distance_distribution, plot_roc_curve, plot_metrics_vs_threshold
import os



def evaluate_accuracy(model, dataloader, threshold=0.5, device="cpu"):
    print(f"\n📏 Threshold used: {threshold}")
    model.eval()
    correct = 0
    total = 0
    tp = tn = fp = fn = 0

    with torch.no_grad():
        for img1, img2, label in tqdm(dataloader, desc="Validating"):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)

            out1, out2 = model(img1, img2)
            distance = F.pairwise_distance(out1, out2)

            # distance < threshold means similar (1), distance >= threshold means dissimilar (0)
            prediction = (distance < threshold).float()

            correct += (prediction == label).sum().item()
            total += label.size(0)

            tp += ((prediction == 1) & (label == 1)).sum().item()
            tn += ((prediction == 0) & (label == 0)).sum().item()
            fp += ((prediction == 1) & (label == 0)).sum().item()
            fn += ((prediction == 0) & (label == 1)).sum().item()

    accuracy = correct / total * 100
    print_classification_report(tp, tn, fp, fn)
    return accuracy

def main():
    os.makedirs("./project3_fingerprint_fvc2000/outputs", exist_ok=True)
    val_data_path = "./project3_fingerprint_fvc2000/data/val_pairs.npz"

    # IMPORTANT: Change the model path to your trained model
    ckpt_path = "./project3_fingerprint_fvc2000/checkpoints/model_augFalse_bs4_ep5_lr0.0005_mg1.0.pt"

    batch_size = 4

    threshold = 0.06

    # Load the best threshold from file if it exists
    # threshold_file = "./project3_fingerprint_fvc2000/outputs/best_threshold.txt"
    # if os.path.exists(threshold_file):
    #     with open(threshold_file, "r") as f:
    #         threshold = float(f.read().strip())
    #     print(f"Loaded best threshold: {threshold}")
    # else:
    #     threshold = 0.05  # fallback 
    #     print(f"No threshold file found, using default threshold = {threshold}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Loaded model from {ckpt_path}")

    val_dataset = SiameseDataset(val_data_path)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    evaluate_accuracy(model, val_loader, threshold, device)
    plot_distance_distribution(model, val_loader, device, save_path="./project3_fingerprint_fvc2000/outputs/distance_hist.png")
    plot_roc_curve(model, val_loader, device, save_path="./project3_fingerprint_fvc2000/outputs/roc_curve.png")
    #plot_accuracy_vs_threshold(model, val_loader, device,save_path="./project3_fingerprint_fvc2000/outputs/accuracy_vs_threshold.png")
    plot_metrics_vs_threshold(model, val_loader, device, save_path="./project3_fingerprint_fvc2000/outputs/metrics_vs_threshold.png")

if __name__ == "__main__":
    main()
