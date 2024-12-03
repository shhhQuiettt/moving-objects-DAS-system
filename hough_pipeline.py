import cv2
import numpy as np
import numpy.typing as npt
from image_processing import generate_colors
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from visualisation2 import plot_numpy_with_lines
from data import velocity_from_slope


def get_slope_and_intercept(x1, y1, x2, y2):
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope, intercept


def detect_velocities(img: npt.NDArray):
    """
    img: npt.NDArray
        Preprocessed image, ready for line detection
    """

    aspect_ratio = 12 / 16

    h, w = img.shape

    new_w = int(aspect_ratio * h)

    img = cv2.resize(img, (new_w, h), interpolation=cv2.INTER_NEAREST)

    lines = cv2.HoughLinesP(
        img,  # Input edge image
        1,  # Distance resolution in pixels
        np.pi / 180,  # Angle resolution in radians
        threshold=500,  # Min number of votes for valid line
        minLineLength=1500,  # Min allowed length of line
        maxLineGap=400,  # Max allowed gap between line for joining them
    )

    if lines is None:
        print("No lines detected")
        return

    # ensuring that the lines are not vertical
    valid_lines = lines[~(lines[:, :, 0] == lines[:, :, 2]), :]
    if valid_lines.size == 0:
        print("No non-vertical lines detected")
        return

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(valid_lines)

    clustering = DBSCAN(eps=1, min_samples=5).fit(X_scaled)

    no_of_clusters = np.max(clustering.labels_) + 1

    average_lines = []
    velocities = []
    for cluster_id in range(no_of_clusters):
        average_line = np.mean(valid_lines[clustering.labels_ == cluster_id], axis=0)
        average_lines.append(average_line)

        slope, _ = get_slope_and_intercept(*average_line)
        velocities.append(velocity_from_slope(slope))

    plot_numpy_with_lines(
        img, [(*get_slope_and_intercept(*l), 0, img.shape[1]) for l in average_lines]
    )
