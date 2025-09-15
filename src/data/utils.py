import cv2
import numpy as np


def compute_canny(
        image: np.ndarray,
        low: int = 100,
        high: int = 200) -> np.ndarray:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    edges = cv2.Canny(gray, low, high)
    edges = edges.astype(np.float32) / 255.0
    return edges
