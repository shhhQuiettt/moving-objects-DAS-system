import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt
from matplotlib.colors import Normalize


def plot_numpy(data: npt.NDArray):
    # display data but strech it horizontaly
    fig = plt.figure(figsize=(12, 16))

    ax = plt.axes()
    ax.imshow(data, aspect="auto", interpolation="none", cmap="gray")
    # remove axis
    ax.axis("off")
    plt.show()
