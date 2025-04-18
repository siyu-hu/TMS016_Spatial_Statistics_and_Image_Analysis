import os
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from preprocess import normalize
from create_train_pairs import load_images_by_finger_tif, create_pairs
from siamese_model import SiameseNetwork
from utils import print_classification_report


def inference_batch(model, pairs, labels, threshold=0.041, device="cpu"):
    model.eval()
    correct = 0
    total = 0
    tp = tn = fp = fn = 0

    for (img1_path, img2_path), label in tqdm(zip(pairs, labels), total=len(pairs), desc="Inferencing"):
        img1 = normalize(img1_path)
        img2 = normalize(img2_path)

        img1 = torch.from_numpy(img1).unsqueeze(0).unsqueeze(0).to(device)  # shape: [1, 1, H, W]
        img2 = torch.from_numpy(img2).unsqueeze(0).unsqueeze(0).to(device)
        label = torch.tensor(label).to(device)

        with torch.no_grad():
            out1, out2 = model(img1, img2)
            dist = F.pairwise_distance(out1, out2).item()
            prediction = 1.0 if dist < threshold else 0.0

            correct += (prediction == label.item())
            total += 1

            # 更新统计
            if prediction == 1.0 and label.item() == 1:
                tp += 1
            elif prediction == 0.0 and label.item() == 0:
                tn += 1
            elif prediction == 1.0 and label.item() == 0:
                fp += 1
            elif prediction == 0.0 and label.item() == 1:
                fn += 1

    acc = correct / total * 100
    print_classification_report(tp, tn, fp, fn)


def main():
    db3_path = "./project3_fingerprint_fvc2000/data/original/DB3_B"
    model_path = "./project3_fingerprint_fvc2000/checkpoints/best_model.pt"
    threshold = 0.041
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f" Loaded model from {model_path}")

    # load DB3_B images and create pairs
    finger_dict = load_images_by_finger_tif(db3_path)
    selected_fingers = sorted(finger_dict.keys())
    pairs, labels = create_pairs(finger_dict, selected_fingers)

    inference_batch(model, pairs, labels, threshold=threshold, device=device)
    

if __name__ == "__main__":
    main()
