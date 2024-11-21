from visualisation import plot_timeframe
import numpy as np
from data import load_from_file

import cv2


filename = "090332.npy"

data = load_from_file(filename)
# plot_timeframe(data)

img = dat.to_numpy()


closed_img = cv2.morphologyEx(
    img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
)

plot_timeframe(closed_img)


from skimage.morphology import skeletonize

skeletonized = skeletonize(closed_img)
plot_numpy(skeletonized)


from skimage.filters import threshold_otsu


# normalize the image
from sklearn.preprocessing import MinMaxScaler, 


high = np.percentile(skeletonized, 99)
low = np.percentile(skeletonized, 3)
scaler = MinMaxScaler(feature_range=(low, high), clip=True)
normalized = scaler.fit_transform(skeletonized)

