import numpy as np

# from visualisation import plot_timeframe, plot_numpy
from visualisation2 import plot_numpy
import numpy as np
from data import load_from_file, prepocess

import cv2


filename = "090332.npy"

data = load_from_file(filename)
# plot_timeframe(data)

img = data.to_numpy()

img = prepocess(img)

plot_numpy(img)


img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

img = cv2.morphologyEx(
    img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
)


# print(np.min(img), np.max(img))


# closed_img = cv2.morphologyEx(
#     img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# )

# plot_timeframe(closed_img)


# from skimage.morphology import skeletonize

# skeletonized = skeletonize(closed_img)
# plot_numpy(skeletonized)


# from skimage.filters import threshold_otsu


# # normalize the image
# from sklearn.preprocessing import MinMaxScaler,


# high = np.percentile(skeletonized, 99)
# low = np.percentile(skeletonized, 3)
# scaler = MinMaxScaler(feature_range=(low, high), clip=True)
# normalized = scaler.fit_transform(skeletonized)
