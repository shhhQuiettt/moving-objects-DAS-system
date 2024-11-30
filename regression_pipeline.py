import numpy as np
from visualisation2 import plot_numpy, plot_numpy_with_lines
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from data import prepocess, velocity_from_slope, mps_to_kmph
from image_processing import proper_opening, generate_colors
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression


def detect_velocities(data: pd.DataFrame, verbose=True) -> list[float]:
    img = data.to_numpy()

    # Preparing the data to easier work with it
    img = prepocess(img)

    if verbose:
        plot_numpy(img, title="Original image")

    # Thresholding the image
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if verbose:
        plot_numpy(img, title="Thresholded image")

    img = proper_opening(img)

    if verbose:
        plot_numpy(img, title="Opened image")

    X = np.nonzero(img)
    X = np.vstack(X).T

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    clustering = DBSCAN(eps=0.08, min_samples=300).fit(X_scaled)

    no_of_clusters = int(np.max(clustering.labels_) + 1)

    colors = generate_colors(no_of_clusters)

    if verbose:
        colored_clusters = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        for cluster_id in range(no_of_clusters):
            # Indices of the points in the cluster
            idx = X[clustering.labels_ == cluster_id, :]

            # Coloring the points
            colored_clusters[idx.T[0], idx.T[1], :] = colors[cluster_id]

        plot_numpy(colored_clusters, title="Clusters")

    # Detecting the lines
    lines = []
    velocities = []
    for cluster_id in range(no_of_clusters):
        idx = X[clustering.labels_ == cluster_id, :]
        x_coords = idx[:, 1].reshape(-1, 1)
        y_coords = idx[:, 0]

        model = LinearRegression()
        model.fit(x_coords, y_coords)

        slope = model.coef_[0]
        intercept = model.intercept_

        x_start = np.min(x_coords)
        x_end = np.max(x_coords)

        r2 = model.score(x_coords, y_coords)
        print(f"R2 of cluster {cluster_id}: {r2}")

        # Discarding objects with R2 < 0.8
        if r2 < 0.8:
            if verbose:
                print(f"Discarded cluster {cluster_id} with R2={r2} (<0.8)")
            continue

        velocity = velocity_from_slope(slope)
        if verbose:
            print(f"Detected velocity: {velocity} m/s ({ mps_to_kmph(velocity) } km/h)")
            print()
        velocities.append(velocity)

        lines.append((slope, intercept, x_start, x_end))

    if verbose:
        plot_numpy_with_lines(img, lines)

    return velocities
