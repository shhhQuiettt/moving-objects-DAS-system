import numpy as np
from visualisation2 import plot_numpy, plot_numpy_with_lines
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from data import prepocess, velocity_from_slope, mps_to_kmph
from image_processing import proper_opening, generate_colors, frequency_lowpass
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression


def detect_velocities(data: pd.DataFrame, verbose=True) -> list[float]:
    img = data.to_numpy()

    # Preparing the data to easier work with it
    img = prepocess(img)

    if verbose:
        plot_numpy(img, title="Original image")

    # img = frequency_lowpass(img, 0.25)
    # if verbose:
    #     plot_numpy(img, title="Low-pass filtered image")

    img = cv2.fastNlMeansDenoising(img, templateWindowSize=7, searchWindowSize=21, h=14)
    # if verbose:
    #     plot_numpy(img, title="Denoised image")

    img = cv2.blur(img, (3, 41))
    # img = cv2.GaussianBlur(img, (3, 3), 0)
    if verbose:
        plot_numpy(img, title="Blurred image")

    img = cv2.morphologyEx(
        img,
        cv2.MORPH_ERODE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9)),
        iterations=1,
    )

    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if verbose:
        plot_numpy(img, title="Thresholded image")

    # img = cv2.morphologyEx(
    #     img,
    #     cv2.MORPH_OPEN,
    #     cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
    #     iterations=1,
    # )

    if verbose:
        plot_numpy(img, title="Opened image")

    # img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)

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

        # Discarding objects with R2 < 0.6
        if r2 < 0.5:
            if verbose:
                print(f"Discarded cluster {cluster_id} with R2={r2} (<0.5)")
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
