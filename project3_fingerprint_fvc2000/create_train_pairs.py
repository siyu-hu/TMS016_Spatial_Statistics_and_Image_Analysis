import os
import numpy as np
import random
from itertools import combinations
from augment_images import random_affine_transform

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


def create_pairs(finger_dict, selected_fingers, augment_positive=False, num_augments=1):
    """
    Create positive and negative pairs.
    If augment_positive=True, perform data augmentation on positive pairs.
    Args:
        finger_dict: dict of {finger_id: list of npy file paths}
        selected_fingers: list of selected finger ids
        augment_positive: whether to augment positive pairs
        num_augments: how many augmentations per positive pair
    """
    pairs = []
    labels = []

    for fid in selected_fingers:
        images = finger_dict[fid]
        # positive pairs (same finger)
        for img1_path, img2_path in combinations(images, 2):
            pairs.append([img1_path, img2_path])
            labels.append(1)

            if augment_positive:
                # Load image arrays
                img1 = np.load(img1_path)
                img2 = np.load(img2_path)

                for _ in range(num_augments):
                    aug1 = random_affine_transform(img1)
                    aug2 = random_affine_transform(img2)

                    # Save temporary augmented images in memory
                    pairs.append([aug1, aug2])
                    labels.append(1)

    # negative pairs (different fingers)
    all_fingers = list(selected_fingers)
    for i in range(len(all_fingers)):
        for j in range(i+1, len(all_fingers)):
            imgs1 = finger_dict[all_fingers[i]]
            imgs2 = finger_dict[all_fingers[j]]
            for img1_path in imgs1:
                for img2_path in imgs2:
                    pairs.append([img1_path, img2_path])
                    labels.append(0)

    return pairs, labels

# def create_pairs(finger_dict, selected_fingers):
#    "basic version - without augmentation" 
#     pairs = []
#     labels = []

#     for fid in selected_fingers:
#         images = finger_dict[fid]
#         # postive pairs(same finger)
#         for img1, img2 in combinations(images, 2):
#             pairs.append([img1, img2])
#             labels.append(1) # label = 1 ->> positive piar( same finger)

#     # negative  pairs(different fingers, randomly selected)
#     all_fingers = list(selected_fingers)
#     for i in range(len(all_fingers)):
#         for j in range(i+1, len(all_fingers)):
#             imgs1 = finger_dict[all_fingers[i]]
#             imgs2 = finger_dict[all_fingers[j]]
#             for img1 in imgs1:
#                 for img2 in imgs2:
#                     pairs.append([img1, img2])
#                     labels.append(0)

#     return pairs, labels

def save_pairs(pairs, labels, output_file):
    pairs = np.array(pairs, dtype=object)  # [img1_path, img2_path]
    labels = np.array(labels)
    np.savez(output_file, pairs=pairs, labels=labels)
    print(f"Saved {len(pairs)} pairs to {output_file}.npz")
