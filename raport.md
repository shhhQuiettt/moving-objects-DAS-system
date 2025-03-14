# Detecting Moving Objects in DAS Recordings

## Authors:
[Krzysztof Skrobała](https://github.com/shhhQuiettt) 156039<br>
[Wojciech Bogacz](https://github.com/wojbog) 156034

## Introduction
In this project, we analyzed Distributed Acoustic Sensing (DAS) data captured as a 2D numpy matrix, representing strain rates along a fiber optic cable installed on Jana Pawła II street. The data reflects the vibrations caused by passing trams, trucks, and cars. Using image processing and signal analysis techniques, we identified vehicle, tracked their movement, and estimated their velocities, providing valuable insights into traffic patterns on this busy street.

The task is very difficult, therefore we tested 2 approches. None of these is the best, however, some scenarios are better in one case and some in other.

## FIRST APPROACH

### Algorithm
<!-- import image -->
![flow chart](flow_chart.png)


To approach this challenge, we analyzed the data and applied the following phase of the preprocessing:
1. **Absolute value** - To ensure all data points are non-negative.
2. **reject 1% of the data from both sides of the distribution** - To remove outliers and improve the robustness of the analysis.
3. **Standardize the data in range 0-255** - To normalize the data for consistent comparison and processing.
-Wojtek
4. **Gaussian blur** - To reduce noise and detail in the data.
5. **Binary conversion** - To simplify the data by converting it to a binary format.
6. **Opening - morphological operation** - To remove small objects from the foreground and smooth the data.
-Krzyś
7. **Non-local Means Denoising algorithm** - To reduce noise while preserving important details and textures in the data.
8. **blur** - To smooth the data and reduce high-frequency noise.
9. **Erosion - morphological operation** - To remove small noise and detach objects that are close to each other.
10. **Binary conversion** - To simplify the data by converting it to a binary format.

Analazying the data, we can conlude that there are not just one line but more, therefore we had to introduce the algorithm, which would separate the data to different vehicles. To obtain that we applied **DBSCAN** clustering algorithm.

- DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a clustering algorithm that groups together points that are closely packed together, marking as outliers points that lie alone in low-density regions. It requires two parameters: **eps** (the maximum distance between two points to be considered neighbors) and **min_samples** (the minimum number of points required to form a dense region). DBSCAN is particularly effective for <u>discovering clusters of arbitrary shape</u> and handling noise in the data. It does not require specifying the number of clusters in advance, making it a versatile choice for exploratory data analysis.

Calculating the velocity of the vehicle, it is required to have approximation of the distance and time. The data gives us the time, However, to discover the distance we had to perform other techniques. Observing the images, we noticed linearity of the paths therefore we tested Linear regression algorithm to detect the papth from the data.

**Linear regression** is a statistical method used to model the relationship between a dependent variable and one or more independent variables. The goal is to find the best-fitting straight line (called the regression line) through the data points that minimizes the sum of the squared differences between the observed values and the values predicted by the line. This method is widely used for predictive analysis and to understand the strength and nature of relationships between variables.

Here are the results of the first approach:

## SECOND APPROACH
preprocessing


In the second method, we approached the issue from the other side. First, we generated lines using Houghline and then, we clusterd them.

**The Hough Line Transform** is a feature extraction technique used in image analysis and computer vision to detect straight lines in an image. It works by transforming points in the image space to the parameter space, where each point in the image space corresponds to a sinusoidal curve in the parameter space. Lines in the image space are identified by finding intersections of these curves in the parameter space. This method is robust to noise and can detect lines even if they are broken or partially obscured. It is commonly used in applications like lane detection in autonomous driving and shape analysis.

To cluster lines we used the same algorithm like in the first method - **DBSCAN**.

Here are the results of the second method: