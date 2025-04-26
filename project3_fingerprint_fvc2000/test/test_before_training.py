import numpy as np
import torch
import torch.nn.functional as F
from siamese_model import SiameseNetwork

def test_one_pair(pair_path="./project3_fingerprint_fvc2000/data/train_pairs.npz"):
    # load 
    model = SiameseNetwork()
    model.eval()

    # load one pair
    data = np.load(pair_path, allow_pickle=True)
    pairs = data["pairs"]
    labels = data["labels"]

    (img1_path, img2_path), label = pairs[0], labels[0]
    img1 = np.load(img1_path)
    img2 = np.load(img2_path)

    # tranfer to  tensor
    img1_tensor = torch.tensor(img1).unsqueeze(0).unsqueeze(0).float()
    img2_tensor = torch.tensor(img2).unsqueeze(0).unsqueeze(0).float()

    with torch.no_grad():
        out1, out2 = model(img1_tensor, img2_tensor)
        distance = F.pairwise_distance(out1, out2)

    print(f"Label: {label}, Distance: {distance.item():.4f}")

if __name__ == "__main__":
    test_one_pair()

import numpy as np, glob, os
sample = np.load(glob.glob("project3_fingerprint_fvc2000/data/processed/DB1_B/*.npy")[0])
print(sample.min(), sample.max(), sample.dtype)
