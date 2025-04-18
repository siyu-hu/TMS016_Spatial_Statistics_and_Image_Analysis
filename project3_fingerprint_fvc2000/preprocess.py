from create_train_pairs import load_images_by_finger, create_pairs, save_pairs
import numpy as np
from tqdm import tqdm
import os
import cv2

def check_image_sizes(folder_path, expected_size=(300, 300)):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.tif'):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Failed to read {filename}")
                continue
            if img.shape != expected_size:
                print(f"{filename} size is not expected, actual size is {img.shape}")

def normalize(image_path, size=(300, 300)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    if img.shape != size:
        img = cv2.resize(img, size)
    img_normalized = img.astype('float32') / 255.0
    return img_normalized

def batch_preprocess_images(input_folder, output_folder, size=(300, 300)):
    check_image_sizes(input_folder, expected_size=size)
    os.makedirs(output_folder, exist_ok=True)

    count = 0
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.tif'):
            input_path = os.path.join(input_folder, filename)
            output_filename = filename.replace('.tif', '.npy')
            output_path = os.path.join(output_folder, output_filename)

            try:
                processed_img = normalize(input_path, size=size)
                np.save(output_path, processed_img)
                count += 1
            except Exception as e:
                print(f"Error: {filename} : {e}")

    print(f"\n{count} images have been processed, output path is {output_folder}")


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