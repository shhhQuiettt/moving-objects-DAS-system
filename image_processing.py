import cv2
import matplotlib.pyplot as plt
import numpy as np


def proper_opening(image, kernel=(3, 3)):
    close1 = cv2.morphologyEx(
        image,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, kernel),
        iterations=1,
    )
    open1 = cv2.morphologyEx(
        close1,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, kernel),
        iterations=1,
    )
    close2 = cv2.morphologyEx(
        open1,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, kernel),
        iterations=1,
    )
    return np.minimum(image, close2)


def generate_colors(num_colors: int) -> np.ndarray:
    # Generate random RGB colors
    hsv_colors = [(i / num_colors, 1.0, 1.0) for i in range(num_colors)]
    rgb_colors = [plt.cm.hsv(h)[:3] for h, _, _ in hsv_colors]
    rgb_colors = (np.array(rgb_colors) * 255).astype(np.uint8)
    np.random.shuffle(rgb_colors)
    return rgb_colors
