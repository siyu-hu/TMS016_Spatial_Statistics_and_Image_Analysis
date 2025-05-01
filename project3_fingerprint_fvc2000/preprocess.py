"""
Unified fingerprint-image preprocessing module
Key features
1. Advanced enhancement pipeline
   Gamma correction → Zero-mean / unit-variance gray normalization →
   Orientation field estimation → Overlapping-block Gabor filtering → CLAHE

2. Backward-compatible public entry point*
   `normalize(image_path) → float32 array [300 * 300], range 0-1`

3. Batch processing 
   Saves two copies simultaneously  
     • Enhanced **tif** → ./data/original/<*_new>  
     • Training **npy**  → ./data/processed/<*_new>

What changed compared with the old version?
-------------------------------------------
✓ **CHANGED** - The former “simple” normalization was replaced; the external
                calling method stays the same.  
✓ **NEW**      - `normalize_gray`, core preprocessing functions,
                and a re-worked `batch_preprocess`.  
✓ **KEPT**     - `check_image_sizes` 
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse


def check_image_sizes(folder_path, expected_size=(300, 300)):
    """Iterate through tif files and report those whose size is unexpected."""
    for fname in os.listdir(folder_path):
        if fname.lower().endswith('.tif'):
            path = os.path.join(folder_path, fname)
            img  = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f" ERROR:  Failed to read {fname}")
                continue
            if img.shape != expected_size:
                print(f"  {fname} has size {img.shape} ≠ {expected_size}")

# ------------------------------------------------------------------------- #
#                      advanced pipeline by Qi Wang                     #
# ------------------------------------------------------------------------- #
def normalize_gray(img):
    """Zero-mean / unit-variance, then stretch to 0–255 (uint8)."""
    mean, std = img.mean(), img.std()
    z = (img - mean) / (std + 1e-5)
    z = (z - z.min()) / (z.max() - z.min()) * 255
    return z.astype(np.uint8)

def gamma_correction(img, gamma=0.3):
    inv = 1.0 / gamma
    table = ((np.arange(256) / 255.0) ** inv * 255).astype("uint8")
    return cv2.LUT(img, table)

def compute_orientation_field(img, block_size=16):
    rows, cols = img.shape
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    orient = np.zeros((rows // block_size, cols // block_size))

    for i in range(0, rows - block_size, block_size):
        for j in range(0, cols - block_size, block_size):
            gx = sobelx[i:i+block_size, j:j+block_size]
            gy = sobely[i:i+block_size, j:j+block_size]
            Vx = 2 * np.sum(gx * gy)
            Vy = np.sum(gx**2 - gy**2)
            orient[i//block_size, j//block_size] = 0.5 * np.arctan2(Vx, Vy)
    return orient

def gabor_filter_overlap(img, orientation_field,
                         block_size=16, freq=0.1, sigma=4.0):
    rows, cols = img.shape
    enhanced = np.zeros((rows, cols), np.float32)
    weights  = np.zeros((rows, cols), np.float32)
    stride   = block_size // 2

    for i in range(0, rows - block_size, stride):
        for j in range(0, cols - block_size, stride):
            theta  = orientation_field[i//block_size, j//block_size]
            kernel = cv2.getGaborKernel((block_size, block_size),
                                        sigma, theta, 1.0/freq, 0.5, 0,
                                        ktype=cv2.CV_32F)
            block    = img[i:i+block_size, j:j+block_size]
            filtered = cv2.filter2D(block, cv2.CV_32F, kernel)

            enhanced[i:i+block_size, j:j+block_size] += filtered
            weights[i:i+block_size, j:j+block_size]  += 1.0

    enhanced /= np.where(weights == 0, 1.0, weights)
    return cv2.normalize(enhanced, None, 0, 255,
                         cv2.NORM_MINMAX).astype(np.uint8)

def apply_clahe(img, clipLimit=3.0, tileGridSize=(8, 8)):
    """Local contrast enhancement (CLAHE)."""
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(img)

def preprocess_fingerprint(img_path):
    """
    Full preprocessing of a single image (returns uint8, original size).
    Gamma → normalize_gray → orientation field → Gabor → CLAHE
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"ERROR: Cannot open {img_path}")

    stage1 = gamma_correction(img)
    stage2 = normalize_gray(stage1)
    field  = compute_orientation_field(stage2)
    gabor  = gabor_filter_overlap(stage2, field)
    return apply_clahe(gabor)


def normalize(image_path, size=(300, 300)):
    """
    Backward-compatible wrapper.

    Parameters
    ----------
    image_path : str
        Path to a grayscale fingerprint image.
    size : tuple, default (300, 300)
        Output spatial size.

    Returns
    -------
    ndarray, float32, shape = size
        Pixel range 0 - 1, ready for the network.
    """
    img = preprocess_fingerprint(image_path)          # uint8
    img = cv2.resize(img, size)
    return img.astype('float32') / 255.0

# ------------------------------------------------------------------------- #
#       Batch processing (save tif + npy to the specified locations)        #
# ------------------------------------------------------------------------- #
def batch_preprocess(input_dir,
                     output_dir_npy="./data/processed/DB1_B_new_1",
                     output_dir_tif="./data/original/DB1_B_new_1",
                     size=(300, 300)):
    """
    Convert all tif/bmp/jpg/png in `input_dir`.
    • Enhanced tif (uint8, 0 - 255) → `output_dir_tif`
    • Training npy (float32, 0 - 1) → `output_dir_npy`
    """
    os.makedirs(output_dir_npy, exist_ok=True)
    os.makedirs(output_dir_tif, exist_ok=True)

    count = 0
    for fname in tqdm(os.listdir(input_dir), desc="preprocessing"):
        if fname.lower().endswith(('.tif', '.bmp', '.jpg', '.png')):
            in_path = os.path.join(input_dir, fname)
            try:
                img = preprocess_fingerprint(in_path)
                img = cv2.resize(img, size)

                stem = os.path.splitext(fname)[0]
                # Save tif
                cv2.imwrite(os.path.join(output_dir_tif, f"{stem}.tif"), img)
                # Save npy (float32, 0–1)
                np.save(os.path.join(output_dir_npy, f"{stem}.npy"),
                        img.astype('float32') / 255.0)
                count += 1
            except Exception as e:
                print(f" {fname}: {e}")

    print(f"\n {count} images saved to")
    print(f"   tif : {output_dir_tif}")
    print(f"   npy : {output_dir_npy}")

# ------------------------------------------------------------------------- #
#                                  CLI                                    #
# ------------------------------------------------------------------------- #
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Fingerprint batch preprocessing (advanced pipeline)")
    parser.add_argument("--input", "-i", required=True,
                        help="Directory containing raw fingerprint images")
    parser.add_argument("--out-npy", default="./data/processed/DB1_B_new_1",
                        help="Destination folder for processed .npy files")
    parser.add_argument("--out-tif", default="./data/original/DB1_B_new_1",
                        help="Destination folder for enhanced .tif files")
    parser.add_argument("--size", type=int, default=300,
                        help="Output width/height after resize (default: 300)")

    args = parser.parse_args()
    
    batch_preprocess(input_dir=args.input,
                     output_dir_npy=args.out_npy,
                     output_dir_tif=args.out_tif,
                     size=(args.size, args.size))
