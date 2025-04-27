import torch
import numpy as np
from siamese_model import SiameseNetwork
from utils import SiameseDataset, print_classification_report
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import plot_distance_distribution, plot_roc_curve, plot_metrics_vs_threshold
import os
import argparse



def evaluate_accuracy(model, dataloader, threshold=0.5, device="cpu"):
    print(f"\n Threshold used: {threshold}")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_data", required=True, help="Path to validation pairs npz file")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint file", default="./checkpoints/model_augTrue_blTrue_bs8_ep20_lr0.001_mg2.0.pt")
    parser.add_argument("--threshold", type=float, default=0.814286, help="Threshold for distance")
    args = parser.parse_args()

    os.makedirs("./outputs", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = 8

    # Load the best threshold from file if it exists
    # threshold_file = "./project3_fingerprint_fvc2000/outputs/best_threshold.txt"
    # if os.path.exists(threshold_file):
    #     with open(threshold_file, "r") as f:
    #         threshold = float(f.read().strip())
    #     print(f"Loaded best threshold: {threshold}")
    # else:
    #     threshold = 0.05  # fallback 
    #     print(f"No threshold file found, using default threshold = {threshold}")

    # IMPORTANT: Change the model path to your trained model
    ckpt_path = args.ckpt

    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Loaded model from {ckpt_path}")

    val_dataset = SiameseDataset(args.val_data, root_dir="..")
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    evaluate_accuracy(model, val_loader, threshold=args.threshold, device=device)

    plot_distance_distribution(model, val_loader, device, save_path="./outputs/distance_hist.png")
    plot_roc_curve(model, val_loader, device, save_path="./outputs/roc_curve.png")
    plot_metrics_vs_threshold(model, val_loader, device, save_path="./outputs/metrics_vs_threshold.png")

if __name__ == "__main__":
    main()

