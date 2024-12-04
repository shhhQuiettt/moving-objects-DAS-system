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


def generate_colors(num_colors: int) -> npt.NDArray:
    # Generate random RGB colors
    raise ValueError("Change import to utils")
    hsv_colors = [(i / num_colors, 1.0, 1.0) for i in range(num_colors)]
    rgb_colors = [plt.cm.hsv(h)[:3] for h, _, _ in hsv_colors]
    rgb_colors = (np.array(rgb_colors) * 255).astype(np.uint8)
    np.random.shuffle(rgb_colors)
    return rgb_colors


def initial_preprocess(img):
    img = np.abs(img)
    high = np.percentile(img, 99)
    img = np.minimum(img, high)

    img = np.around(255 * (img - np.min(img)) / (np.max(img) - np.min(img))).astype(
        np.uint8
    )
    return img
    # print(high, np.max(data))


def preprocess(img):
    img = np.abs(img)
    high = np.percentile(img, 99)
    img = np.minimum(img, high)

    img = np.around(255 * (img - np.min(img)) / (np.max(img) - np.min(img))).astype(
        np.uint8
    )
    # print(high, np.max(data))

    img = cv2.fastNlMeansDenoising(img, templateWindowSize=7, searchWindowSize=21, h=14)

    img = cv2.blur(img, (3, 41))

    img = cv2.morphologyEx(
        img,
        cv2.MORPH_ERODE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9)),
        iterations=1,
    )

    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return img


def pipeline_processing(img):
    img_g = cv2.GaussianBlur(img, (5, 5), 0)
    tr, img_i = cv2.threshold(img_g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_i = cv2.morphologyEx(
        img_i,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1,
    )
    return img_i


def clustering(img):
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import MinMaxScaler

    X = np.nonzero(img)
    X = np.vstack(X).T

    X.shape
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    clustering = DBSCAN(eps=0.08, min_samples=300, metric="manhattan").fit(X_scaled)

    no_of_clusters = np.max(clustering.labels_) + 1
    # print(no_of_clusters)
    return no_of_clusters, X, clustering


def generate_colors(num_colors):
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
