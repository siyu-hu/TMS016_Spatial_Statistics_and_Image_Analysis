import numpy as np
import cv2
import random

def random_affine_transform(image, max_rotation=10, max_scale=0.1):
    """
    Apply random affine transformation (small rotation + scaling) to a single image.

    Args:
        image: numpy array, single-channel (H, W)
        max_rotation: maximum rotation angle (± degrees)
        max_scale: maximum scale variation (± percentage)

    Returns:
        Transformed image with the same size.
    """

    h, w = image.shape

    # 1. Random rotation angle
    angle = random.uniform(-max_rotation, max_rotation)

    # 2. Random scaling factor
    scale = 1.0 + random.uniform(-max_scale, max_scale)

    # 3. Build the affine transformation matrix
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)

    # 4. Apply the affine transformation
    transformed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    return transformed
