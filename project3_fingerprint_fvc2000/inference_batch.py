import os
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from preprocess import normalize
from create_train_pairs import load_images_by_finger_tif, create_pairs
from siamese_model import SiameseNetwork
from utils import print_classification_report
from random import sample


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

            # Update confusion matrix
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


def auto_calibrate_threshold(model, calib_pairs, calib_labels, device="cpu",
                             search_min=0.0, search_max=1.2, steps=120):
    """
    Given a batch of calibrated pairs + labels, scan the threshold to find the highest F1 point.
    Returns: best_threshold, best_f1
    """
    model.eval()
    dists = []

    with torch.no_grad():
        for (p1, p2), _ in zip(calib_pairs, calib_labels):
            a = torch.from_numpy(np.load(p1)).unsqueeze(0).unsqueeze(0).to(device)
            b = torch.from_numpy(np.load(p2)).unsqueeze(0).unsqueeze(0).to(device)
            f1, f2 = model(a, b)
            dists.append(F.pairwise_distance(f1, f2).item())

    dists = np.array(dists)
    labs  = np.array(calib_labels)

    thresholds = np.linspace(search_min, search_max, steps)
    best_f1, best_t = 0.0, thresholds[0]

    for t in thresholds:
        pred = (dists < t).astype(int)
        tp = np.sum((pred == 1) & (labs == 1))
        fp = np.sum((pred == 1) & (labs == 0))
        fn = np.sum((pred == 0) & (labs == 1))
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)

        if f1 > best_f1:
            best_f1, best_t = f1, t

    return best_t, best_f1


def main():
    # ------------ paths ------------
    #inference_data_path = "./project3_fingerprint_fvc2000/data/original/DB3_B"
    inference_data_path = "./project3_fingerprint_fvc2000/data/original/DB2_B"
    # IMPORTANT: Change the model path to your trained model
    model_path = "./project3_fingerprint_fvc2000/checkpoints/model_augTrue_blTrue_bs8_ep20_lr0.001_mg2.0.pt"
    

    # ------------ device / model ------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded model from {model_path}")

    # ------------ build pairs for this DB ------------
    finger_dict = load_images_by_finger_tif(inference_data_path)
    pairs, labels = create_pairs(finger_dict, sorted(finger_dict.keys()),
                                 augment_positive=False, num_augments=0,
                                 balance_negatives=False)
    
    # ------------ 20 % calibrate, 80 % infer ------------
    idx = np.arange(len(pairs))
    np.random.seed(42)
    np.random.shuffle(idx)

    split = int(0.2 * len(idx))
    calib_idx, infer_idx = idx[:split], idx[split:]

    calib_pairs  = [pairs[i]  for i in calib_idx]
    calib_labels = [labels[i] for i in calib_idx]
    infer_pairs  = [pairs[i]  for i in infer_idx]
    infer_labels = [labels[i] for i in infer_idx]

    print(f"Calibration pairs : {len(calib_pairs)}")
    threshold, f1_calib = auto_calibrate_threshold(model, calib_pairs, calib_labels, device)
    print(f"Auto-calibrated threshold = {threshold:.4f}  (F1 on calib = {f1_calib:.4f})")

    # ------------ final inference ------------
    inference_batch(model, infer_pairs, infer_labels,
                    threshold=threshold, device=device, desc="Inference-80%")
    

if __name__ == "__main__":
    main()
