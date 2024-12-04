import matplotlib
matplotlib.use('Agg')
from visualisation import plot_numpy
import hough_pipeline
import regression_pipeline
from image_processing import preprocess, initial_preprocess
from data import load_all_files, get_names

files_generator = load_all_files()
files_names = get_names()

i = 0
for file in files_generator:
    if i == 5:
        break
    i+=1
    print(f"Processing file {files_names[i]}")
    original = initial_preprocess(file.to_numpy())
    img = preprocess(file.to_numpy())
    plot_numpy(original, title="original_"+ f"{i:02}", save=True)
    print("Hough line detection")
    hough_pipeline.detect_velocities(img, original, index=i, save=True)
    print("Regression line detection")
    regression_pipeline.detect_velocities(img,  original, index=i, save=True)
