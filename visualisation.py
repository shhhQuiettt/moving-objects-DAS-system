import matplotlib.pyplot as plt
import numpy.typing as npt
import numpy as np
from matplotlib.colors import Normalize


def set_axis(x, no_labels=7) -> tuple[np.array, np.array]:
    """Sets the x-axis positions and labels for a plot.

    Args:
        x (np.array): The x-axis data.
        no_labels (int, optional): The number of labels to display. Defaults to 7.

    Returns:
        tuple[np.array, np.array]: A tuple containing:
            - The positions of the labels on the x-axis.
            - The labels themselves.
    """
    nx = x.shape[0]
    step_x = int(nx / (no_labels - 1))
    x_positions = np.arange(0, nx, step_x)
    x_labels = x[::step_x]
    return x_positions, x_labels


def plot_timeframe(df):
    fig = plt.figure(figsize=(12, 16))
    ax = plt.axes()

    # This is an example transformation and should be converted to the proper algorithm
    df -= df.mean()
    df = np.abs(df)
    low, high = np.percentile(df, [3, 99])
    norm = Normalize(vmin=low, vmax=high, clip=True)

    im = ax.imshow(df, interpolation="none", aspect="auto", norm=norm)
    plt.ylabel("time")
    plt.xlabel("space [m]")

    cax = fig.add_axes(
        [
            ax.get_position().x1 + 0.06,
            ax.get_position().y0,
            0.02,
            ax.get_position().height,
        ]
    )
    plt.colorbar(im, cax=cax)
    x_positions, x_labels = set_axis(df.columns)
    ax.set_xticks(x_positions, np.round(x_labels))
    y_positions, y_labels = set_axis(df.index.time)
    ax.set_yticks(y_positions, y_labels)
    plt.show()
    # save plot to file


def plot_numpy(arr: npt.NDArray):

    fig = plt.figure(figsize=(12, 16))
    ax = plt.axes()

    arr -= arr.mean()
    arr = np.abs(arr)
    low, high = np.percentile(arr, [3, 99])
    norm = Normalize(vmin=low, vmax=high, clip=True)

    im = ax.imshow(arr, interpolation="none", aspect="auto", norm=norm)
    plt.ylabel("time")
    plt.xlabel("space [m]")

    cax = fig.add_axes(
        [
            ax.get_position().x1 + 0.06,
            ax.get_position().y0,
            0.02,
            ax.get_position().height,
        ]
    )
    plt.colorbar(im, cax=cax)
    plt.show()
