import matplotlib.pyplot as plt
import cv2
import numpy as np
import numpy.typing as npt


def put_velocity_on_image(
    img: npt.NDArray, velocity, line, color=(255, 255, 0), org=None
):
    org = (int((line[0] + line[2]) / 2), int((line[1] + line[3]) / 2))

    cv2.putText(
        img,
        f"{velocity:.2f} m/s",
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        # font_scale,
        5,
        color,
        4,
        cv2.LINE_AA,
    )
    return img


def get_slope_and_intercept(x1, y1, x2, y2):
    assert x2 - x1 != 0
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope, intercept


def generate_colors(num_colors: int) -> npt.NDArray:
    # Generate random RGB colors
    hsv_colors = [(i / num_colors, 1.0, 1.0) for i in range(num_colors)]
    rgb_colors = [plt.cm.hsv(h)[:3] for h, _, _ in hsv_colors]
    rgb_colors = (np.array(rgb_colors) * 255).astype(np.uint8)
    np.random.shuffle(rgb_colors)
    return rgb_colors


def mps_to_kmph(velocity: float) -> float:
    return round(velocity * (3.6), 2)
