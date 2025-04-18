import os
import cv2
import numpy as np

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


