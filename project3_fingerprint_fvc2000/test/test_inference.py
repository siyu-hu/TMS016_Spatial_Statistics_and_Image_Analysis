import torch
import torch.nn.functional as F
import numpy as np
import os
from preprocess import normalize  
from siamese_model import SiameseNetwork

def preprocess_image(image_path):

    img = normalize(image_path, size=(300, 300))  
    img = np.expand_dims(img, axis=0)  # shape: [1, 300, 300]
    img = np.expand_dims(img, axis=0)  # shape: [1, 1, 300, 300]
    return torch.from_numpy(img)

def inference(model, img1_tensor, img2_tensor, threshold=0.041):
    model.eval()
    with torch.no_grad():
        out1, out2 = model(img1_tensor, img2_tensor)
        distance = F.pairwise_distance(out1, out2).item()
        is_same = distance < threshold
        return distance, is_same

def main():
    model_path = "./project3_fingerprint_fvc2000/checkpoints/best_model.pt"
    image1_path = "./project3_fingerprint_fvc2000/data/original/DB3_B/101_1.tif"
    image2_path = "./project3_fingerprint_fvc2000/data/original/DB3_B/101_2.tif"
    threshold = 0.04 # validation threshold
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f" Loaded model from {model_path}")

    # load test images and preprocess
    img1 = preprocess_image(image1_path).to(device)
    img2 = preprocess_image(image2_path).to(device)

    # inference get distance and prediction
    distance, is_same = inference(model, img1, img2, threshold)
    print(f"\n Inference Results")
    print(f" Distance: {distance:.4f}")
    print(" Prediction:", "Same Finger " if is_same else "Different Fingers ")

if __name__ == "__main__":
    main()
