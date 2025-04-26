import os
import numpy as np
import random
from itertools import combinations
from augment_images import random_affine_transform
from sklearn.utils import shuffle


def load_images_by_finger(data_path):
    """key = finger_id; value = image path list"""
    finger_dict = {}
    for file in os.listdir(data_path):
        if file.endswith('.npy'):
            finger_id = file.split('_')[0]
            finger_dict.setdefault(finger_id, []).append(os.path.join(data_path, file))
    for fid in finger_dict:
        finger_dict[fid] = sorted(finger_dict[fid])
    return finger_dict

def load_images_by_finger_tif(data_path):
    """load tif (for DB3_B test images)"""
    finger_dict = {}
    for file in os.listdir(data_path):
        if file.lower().endswith('.tif'):
            finger_id = file.split('_')[0]
            finger_dict.setdefault(finger_id, []).append(os.path.join(data_path, file))
    for fid in finger_dict:
        finger_dict[fid] = sorted(finger_dict[fid])
    return finger_dict


def create_pairs(finger_dict, selected_fingers, augment_positive=False, num_augments=2,balance_negatives=False):
    """
    Create positive and negative pairs.
    If augment_positive=True, perform data augmentation on positive pairs.
    If balance_negatives=True, sample negative pairs to match positive pairs count.
    
    Args:
        finger_dict: dict of {finger_id: list of npy file paths}
        selected_fingers: list of selected finger ids
        augment_positive: whether to augment positive pairs
        num_augments: how many augmentations per positive pair
        balance_negatives: whether to balance negative samples
    """
    pairs = []
    labels = []

    for fid in selected_fingers:
        images = finger_dict[fid]

        # postive pairs (same finger)
        for img1, img2 in combinations(images, 2):
            pairs.append([img1, img2])
            labels.append(1) # label = 1 ->> positive pair (same finger)

            if augment_positive:
                # add data augmentation for positive pairs
                img1_arr = np.load(img1)
                img2_arr = np.load(img2)

                for _ in range(num_augments):
                    aug1 = random_affine_transform(img1_arr)
                    aug2 = random_affine_transform(img2_arr)

                    # save temporary augmented images in memory
                    pairs.append([aug1, aug2])
                    labels.append(1)
        
    num_positive = sum(1 for l in labels if l == 1)

    # negative pairs (different fingers, randomly selected)
    negative_pairs = []
    all_fingers = list(selected_fingers)
    for i in range(len(all_fingers)):
        for j in range(i+1, len(all_fingers)):
            imgs1 = finger_dict[all_fingers[i]]
            imgs2 = finger_dict[all_fingers[j]]
            for img1 in imgs1:
                for img2 in imgs2:
                    negative_pairs.append(([img1, img2], 0))

    if balance_negatives:
        # Randomly sample same number of negative pairs as positive pairs
        negative_pairs = random.sample(negative_pairs, min(num_positive, len(negative_pairs)))

    # Add negative pairs
    for pair, label in negative_pairs:
        pairs.append(pair)
        labels.append(label)

    pairs, labels = shuffle(pairs, labels, random_state=42) 
    return pairs, labels



def save_pairs(pairs, labels, output_file):
    pairs = np.array(pairs, dtype=object)  # [img1_path, img2_path]
    labels = np.array(labels)
    np.savez(output_file, pairs=pairs, labels=labels)
    print(f"Saved {len(pairs)} pairs to {output_file}.npz")
