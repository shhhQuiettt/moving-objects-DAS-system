import numpy as np
import numpy.typing as npt
from visualisation import plot_numpy, plot_numpy_with_lines
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from data import prepocess, velocity_from_slope, mps_to_kmph
from image_processing import proper_opening, generate_colors, frequency_lowpass
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

from utils import put_velocity_on_image


def detect_velocities(img: npt.NDArray, original_img: npt.NDArray, index: int = 0, save: bool = False) -> list[float]:
    X = np.nonzero(img)
    X = np.vstack(X).T

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    clustering = DBSCAN(eps=0.08, min_samples=300).fit(X_scaled)

    no_of_clusters = int(np.max(clustering.labels_) + 1)

    colors = generate_colors(no_of_clusters)

    img_clusters = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for cluster_id in range(no_of_clusters):
        idx = X[clustering.labels_ == cluster_id, :]

        # Coloring the points
        img_clusters[idx.T[0], idx.T[1], :] = colors[cluster_id]

    # Detecting the lines
    lines = []
    velocities = []
    for cluster_id in range(no_of_clusters):
        idx = X[clustering.labels_ == cluster_id, :]
        x_coords = idx[:, 1].reshape(-1, 1)
        y_coords = idx[:, 0]

        cluster_center = np.mean(idx, axis=0).astype(np.int32)

        model = LinearRegression()
        model.fit(x_coords, y_coords)

        slope = model.coef_[0]
        intercept = model.intercept_

        x_start = np.min(x_coords)
        x_end = np.max(x_coords)

        r2 = model.score(x_coords, y_coords)

        # Discarding objects with R2 < 0.3
        if r2 < 0.3:
            continue

        velocity = velocity_from_slope(slope)
        velocities.append(velocity)

        lines.append((slope, intercept, x_start, x_end))

    del clustering, scaler, X_scaled, X

    plot_numpy(img_clusters, title="Detected clusters_" + f"{index:02}", save=save)
    plot_numpy_with_lines(original_img, lines, title="Detected lines Regression_" + f"{index:02}", save=save)

    for velocity in velocities:
        print(f"Detected velocity: {velocity} m/s)")
