import cv2
import numpy.typing as npt
import matplotlib.pyplot as plt
import numpy as np


def proper_opening(image, kernel=(3, 3)):
    close1 = cv2.morphologyEx(
        image,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, kernel),
        iterations=1,
    )
    open1 = cv2.morphologyEx(
        close1,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, kernel),
        iterations=1,
    )
    close2 = cv2.morphologyEx(
        open1,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, kernel),
        iterations=1,
    )
    return np.minimum(image, close2)


def generate_colors(num_colors: int) -> np.ndarray:
    # Generate random RGB colors
    hsv_colors = [(i / num_colors, 1.0, 1.0) for i in range(num_colors)]
    rgb_colors = [plt.cm.hsv(h)[:3] for h, _, _ in hsv_colors]
    rgb_colors = (np.array(rgb_colors) * 255).astype(np.uint8)
    np.random.shuffle(rgb_colors)
    return rgb_colors


def to_frequency_domain(image: npt.NDArray) -> npt.NDArray:
    return np.fft.fftshift(np.fft.fft2(image))


def to_spatial_domain(image: npt.NDArray) -> npt.NDArray:
    return np.fft.ifft2(np.fft.ifftshift(image)).real.astype(np.uint8)


def get_mask(shape: tuple[int, int], freq_fraction: float):
    mask = np.zeros(shape, np.float32)
    return cv2.circle(
        mask, (shape[0] // 2, shape[1] // 2), int(min(shape) * freq_fraction), 1, -1
    )


def frequency_lowpass(image: npt.NDArray, freq_fraction: float):
    image_freq = to_frequency_domain(image)
    mask = get_mask(image_freq.shape, freq_fraction)

    return to_spatial_domain(image_freq * mask)


def frequency_highpass(image: npt.NDArray, freq_fraction: float):
    image_freq = to_frequency_domain(image)
    mask = get_mask(image_freq.shape, freq_fraction)

    return to_spatial_domain(image_freq * (1 - mask))
