from operator import index
import os 
import glob 

import cv2 
import pandas as pd 

from uvs.sce import compute_focal_length
from utils import Tracker 

DATA_DIR = "../data/"
IMG_EXTENSION = ".png"
CENTIMETER = 5 
CSV_NAME = "focal_length.csv"

if __name__ ==  "__main__": 
    # get all the image paths from target directory and initialize dataframe to store all the data 
    image_paths = glob.glob(DATA_DIR + '*' + IMG_EXTENSION)
    df = pd.DataFrame(columns=["fname", "depth", "baseline", "pix_focal"])

    # get two x coordinate of two points from each image 
    for i, path in enumerate(image_paths): 
        # read image and put the tracker/points of reference 
        img = cv2.imread(path)
        tracker = Tracker()
        tracker.register_points(img, f"Measure focal length with the width of {CENTIMETER} centimeters.")

        assert tracker.points.shape == (2, 2), "Exactly two points are required."

        #NOTE: following operation is a specific opeartion for the stored image, so please replace it with appropriate operation if you are using other data source
        depth = float(path.split('_')[0].split('\\')[-1])

        # get focal length, store the results 
        f_pix = compute_focal_length(tracker.points[0, 0], tracker.points[1, 0], depth, CENTIMETER)
        df.loc[i] = [path.split('/')[-1], depth, CENTIMETER, f_pix] 

    df.to_csv(DATA_DIR + CSV_NAME, index=False)