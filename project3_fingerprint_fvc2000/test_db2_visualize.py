# db2_visualize.py  ─────────────────────────────────────────
"""
Draw distance histogram for DB2_B using the already-trained model.
out put file: ./outputs/db2_distance_hist.png
"""

import os, torch, numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from preprocess import normalize                      
from create_train_pairs import load_images_by_finger_tif, create_pairs
from siamese_model import SiameseNetwork
import matplotlib.pyplot as plt

# ---------- 0. paths ----------
db2_raw   = "./project3_fingerprint_fvc2000/data/original/DB2_B"
ckpt_path = "./project3_fingerprint_fvc2000/checkpoints/model_augTrue_blTrue_bs8_ep20_lr0.001_mg2.0.pt"
out_png   = "./project3_fingerprint_fvc2000/outputs/db2_distance_hist.png"
os.makedirs(os.path.dirname(out_png), exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 1. build all pairs from raw tif ----------
finger_dict = load_images_by_finger_tif(db2_raw)
pairs, labels = create_pairs(finger_dict, sorted(finger_dict.keys()),
                             augment_positive=False, num_augments=0,
                             balance_negatives=False)
print(f"Total pairs = {len(pairs)} (pos={sum(labels)}, neg={len(labels)-sum(labels)})")

# ---------- 2. tiny Dataset that reads tif on-the-fly ----------
class RawPairDataset(Dataset):
    def __init__(self, pairs, labels):
        self.pairs, self.labels = pairs, labels
    def __len__(self):  return len(self.pairs)
    def __getitem__(self, idx):
        p1, p2 = self.pairs[idx]
        lab    = self.labels[idx]
        return normalize(p1), normalize(p2), lab

loader = DataLoader(RawPairDataset(pairs, labels), batch_size=32, shuffle=False)

# ---------- 3. load model ----------
model = SiameseNetwork().to(device)
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()
print("Model loaded.")

# ---------- 4. collect distances ----------
pos_d, neg_d = [], []
with torch.no_grad():
    for x1, x2, lab in tqdm(loader, desc="Forward"):
        a = x1.unsqueeze(1).to(device)   # [B,1,300,300]
        b = x2.unsqueeze(1).to(device)
        f1, f2 = model(a, b)
        d = F.pairwise_distance(f1, f2).cpu().numpy()
        for dist, l in zip(d, lab):
            (pos_d if l==1 else neg_d).append(dist)

# ---------- 5. plot ----------
plt.figure(figsize=(8,5))
plt.hist(pos_d, bins=50, alpha=0.6, label="Positive (same)")
plt.hist(neg_d, bins=50, alpha=0.6, label="Negative (different)")
plt.xlabel("Distance"); plt.ylabel("Count"); plt.title("DB2_B Distance Distribution")
plt.legend()
plt.savefig(out_png, dpi=120)
print(f"Histogram saved → {out_png}")
# ────────────────────────────────────────────────────────────
