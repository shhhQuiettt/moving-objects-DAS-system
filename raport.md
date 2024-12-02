# Detecting Moving Objects in DAS Recordings

## Introduction
In this project, we analyzed Distributed Acoustic Sensing (DAS) data captured as a 2D numpy matrix, representing strain rates along a fiber optic cable installed on Jana Pawła II street. The data reflects the vibrations caused by passing trams, trucks, and cars. Using image processing and signal analysis techniques, we identified vehicle, tracked their movement, and estimated their velocities, providing valuable insights into traffic patterns on this busy street.

## Algorithm
<!-- import image -->
![flow chart](flow_chart.png)

The algorithm consists of the following steps:
1. **Preprocessing**: The data is preprocessed to remove noise and normalize the values.
2. **Background Subtraction**: The background is estimated using a rolling median filter.
3. **Thresholding**: A threshold is applied to the data to detect moving objects.
4. **Connected Components**: Connected components are identified in the thresholded image.
5. **Bounding Boxes**: Bounding boxes are drawn around the connected components.
6. **Tracking**: The objects are tracked using the Hungarian algorithm.
7. **Velocity Estimation**: The velocity of the objects is estimated based on their movement.