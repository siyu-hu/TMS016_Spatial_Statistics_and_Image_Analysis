from preprocess import batch_preprocess_images
from create_train_pairs import load_images_by_finger, create_pairs, save_pairs
import numpy as np
from tqdm import tqdm

def main():

    input_folder = "./project3_fingerprint_fvc2000/data/original/DB1_B"
    output_folder = "./project3_fingerprint_fvc2000/data/processed/DB1_B"

    batch_preprocess_images(input_folder, output_folder)

    all_fingers = sorted(load_images_by_finger(output_folder).keys())
        
    train_fingers = all_fingers[:6]   # training set = 101 - 106 
    val_fingers = all_fingers[6:]     # validating set = 107 - 110
    finger_dict = load_images_by_finger(output_folder)

    train_pairs, train_labels = create_pairs(finger_dict, train_fingers)
    save_pairs(train_pairs, train_labels, "./project3_fingerprint_fvc2000/data/train_pairs")

    val_pairs, val_labels = create_pairs(finger_dict, val_fingers)
    save_pairs(val_pairs, val_labels, "./project3_fingerprint_fvc2000/data/val_pairs")

    data = np.load("./project3_fingerprint_fvc2000/data/train_pairs.npz", allow_pickle=True)
    pairs = data["pairs"]
    labels = data["labels"]

    for (img1_path, img2_path), label in tqdm(zip(pairs, labels), total=len(pairs)):
        img1 = np.load(img1_path)
        img2 = np.load(img2_path)

if __name__ == "__main__":
    main()