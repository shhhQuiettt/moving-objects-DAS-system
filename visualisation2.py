import numpy as np
import cv2

from data import DX, DT
import matplotlib.pyplot as plt
import numpy.typing as npt
from matplotlib.colors import Normalize


def plot_numpy(data: npt.NDArray):
    # display data but strech it horizontaly
    fig = plt.figure(figsize=(12, 16))

    ax = plt.axes()

    if len(data.shape) == 2:
        ax.imshow(
            data, vmin=0, vmax=255, aspect="auto", interpolation="none", cmap="gray"
        )

    else:
        ax.imshow(data, aspect="auto", interpolation="none")

    # remove axis
    ax.axis("off")
    plt.show()


def plot_numpy_with_lines(
    data: npt.NDArray, lines: list[tuple[float, float, int, int]]
):
    """
    lines: list of (slope, intercept) tuples
    """
    if len(data.shape) == 2:
        data = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
    fig = plt.figure(figsize=(12, 16))

    ax = plt.axes()

    plt.ylabel("Time [s]")
    plt.xlabel("Position [m]")

    nx = data.shape[1]
    step_x = int(nx / 6)
    x_positions = np.arange(0, nx, step_x)
    x_labels = np.arange(0, nx, step_x) * DX
    ax.set_xticks(x_positions, np.round(x_labels))

    ny = data.shape[0]
    step_y = int(ny / 6)
    y_positions = np.arange(0, ny, step_y)
    y_labels = np.arange(0, ny, step_y) * DT
    ax.set_yticks(y_positions, np.round(y_labels))

    ax.imshow(data, aspect="auto", interpolation="none")

    for line in lines:
        slope, intercept, x_min, x_max = line
        y_min = 0
        y_max = data.shape[0]

        x = np.arange(x_min, x_max)
        y = slope * x + intercept

        y_valid_mask = (y >= y_min) & (y < y_max)
        x_valid_mask = (x > x_min) & (x < x_max)

        valid_mask = y_valid_mask & x_valid_mask

        x = x[valid_mask]
        y = y[valid_mask]

        ax.plot(x, y, color="red")

    plt.show()
