import numpy as np
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
        print("gere")
        ax.imshow(data, aspect="auto", interpolation="none")

    # remove axis
    ax.axis("off")
    plt.show()
