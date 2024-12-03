import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from image_processing import generate_colors
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from visualisation2 import plot_numpy
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
    img_before = img.copy()

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

    img_lines = cv2.cvtColor(img_before, cv2.COLOR_GRAY2RGB)
    color_palette = generate_colors(len(valid_lines))
    for i in range(len(valid_lines)):
        cv2.line(
            img_lines,
            (valid_lines[i, 0], valid_lines[i, 1]),
            (valid_lines[i, 2], valid_lines[i, 3]),
            [int(c) for c in color_palette[i]],
            1,
            cv2.LINE_AA,
        )

    valid_lines_directional = np.apply_along_axis(
        lambda x: np.array([*get_slope_and_intercept(*x)]), axis=1, arr=valid_lines
    )

    valid_lines_centers = np.apply_along_axis(
        lambda x: np.array([(x[0] + x[2]) / 2, (x[1] + x[3]) / 2]),
        axis=1,
        arr=valid_lines,
    )

    print("Validafsd", valid_lines_directional.shape)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(valid_lines_centers)
    # X_scaled = valid_lines_directional

    clustering = DBSCAN(eps=0.15, min_samples=16).fit(X_scaled)

    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clustering.labels_)

    no_of_clusters = np.max(clustering.labels_) + 1

    img_clusters = cv2.cvtColor(img_before, cv2.COLOR_GRAY2RGB)
    color_palette = generate_colors(no_of_clusters)
    line_colors = np.apply_along_axis(lambda x: color_palette[x], 0, clustering.labels_)
    for i in range(len(valid_lines)):
        cv2.line(
            img_clusters,
            (valid_lines[i, 0], valid_lines[i, 1]),
            (valid_lines[i, 2], valid_lines[i, 3]),
            [int(c) for c in line_colors[i]],
            1,
            cv2.LINE_AA,
        )

    average_lines = []
    velocities = []
    for cluster_id in range(no_of_clusters):
        average_line = np.mean(valid_lines[clustering.labels_ == cluster_id], axis=0)
        average_lines.append(average_line)

        slope, _ = get_slope_and_intercept(*average_line)
        velocities.append(velocity_from_slope(slope))

    img_average_lines = cv2.cvtColor(img_before, cv2.COLOR_GRAY2RGB)
    for i in average_lines:
        cv2.line(
            img_average_lines,
            (int(i[0]), int(i[1])),
            (int(i[2]), int(i[3])),
            (0, 255, 0),
            3,
            cv2.LINE_AA,
        )

    plot_numpy(img_before, img_lines, img_clusters, img_average_lines)
