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

def pipeline_processing(img):

    img_g = cv2.GaussianBlur(img, (5, 5), 0)
    tr, img_i = cv2.threshold(img_g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_i = cv2.morphologyEx(
        img_i,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
        iterations=1,
    )
    return img_i

def clustering(img):
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import MinMaxScaler

    X = np.nonzero(img)
    X = np.vstack(X).T

    X.shape
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    clustering = DBSCAN(eps=0.08, min_samples=300).fit(X_scaled)

    no_of_clusters = np.max(clustering.labels_)+1
    # print(no_of_clusters)
    return no_of_clusters

def generate_colors(num_colors):
    hsv_colors = [(i / num_colors, 1.0, 1.0) for i in range(num_colors)]
    rgb_colors = [plt.cm.hsv(h)[:3] for h, _, _ in hsv_colors]
    rgb_colors = (np.array(rgb_colors) * 255).astype(np.uint8)
    np.random.shuffle(rgb_colors)
    return rgb_colors