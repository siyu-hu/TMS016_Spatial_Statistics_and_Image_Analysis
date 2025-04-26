import numpy as np
import matplotlib.pyplot as plt
from augment_images import random_affine_transform

def main():
    # Load an example fingerprint image (npy format)
    img_path = "./project3_fingerprint_fvc2000/data/processed/DB1_B/101_1.npy" 
    img = np.load(img_path)

    # Generate multiple augmented versions
    augmented_imgs = [random_affine_transform(img) for _ in range(5)]  # Generate 5 variants

    # Plot original and augmented images
    plt.figure(figsize=(15, 3))
    plt.subplot(1, 6, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    for i, aug_img in enumerate(augmented_imgs, start=2):
        plt.subplot(1, 6, i)
        plt.imshow(aug_img, cmap='gray')
        plt.title(f"Augmented {i-1}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
