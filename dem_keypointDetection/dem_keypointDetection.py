#%%

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import dem_numericalMethods as nuMet
from dem_concentricCircleDetection import findKeypoints


#------------------ input data ---------------------#
#xNum, yNum             = 10, 6
#xLength, yLength       = 105, 58
#coordOuter, coordInner = 7.0, 2.0
#fieldOuter, fieldInner = 6.0, 2.0
#coordRatio, innerRatio = coordOuter / coordInner, fieldOuter / fieldInner
#camNum                 = 2
#imgNamePrefix          = ["left", "right"]
#imgDirectory           = ("C:/Users/66879/Desktop/SSCs-gui-Flask-cali4/Pics1", "C:/Users/66879/Desktop/SSCs-gui-Flask-cali4/Pics2")
#datDirectory           = ("C:/Users/66879/Desktop/SSCs-gui-Flask-cali4/Pics1/data1", "C:/Users/66879/Desktop/SSCs-gui-Flask-cali4/Pics2/data2")
#imgNumber              = [1, 5]
#sd, biFac              = 2.0, 1.5  # standard deviation and binarization threshold 
#---------------------------------------------------#
def dectectKeypoints():
    xNum, yNum             = 10, 6
    xLength, yLength       = 105, 58
    coordOuter, coordInner = 7.0, 2.0
    fieldOuter, fieldInner = 6.0, 2.0
    coordRatio, innerRatio = coordOuter / coordInner, fieldOuter / fieldInner
    camNum                 = 2
    imgNamePrefix          = ["left", "right"]
    imgDirectory           = ("C:/Sxmm/4th years/Project/exe_srtaincal/guy/SSCs-gui-Flask-gt (2)/SSCs-gui-Flask-cali4/SSCs-gui-Flask-cali4/Pics1", "C:/Sxmm/4th years/Project/exe_srtaincal/guy/SSCs-gui-Flask-gt (2)/SSCs-gui-Flask-cali4/SSCs-gui-Flask-cali4/Pics2")
    datDirectory           = ("C:/Sxmm/4th years/Project/exe_srtaincal/guy/SSCs-gui-Flask-gt (2)/SSCs-gui-Flask-cali4/SSCs-gui-Flask-cali4/Pics1/data1", "C:/Sxmm/4th years/Project/exe_srtaincal/guy/SSCs-gui-Flask-gt (2)/SSCs-gui-Flask-cali4/SSCs-gui-Flask-cali4/Pics2/data2")
    imgNumber              = [1, 5]
    sd, biFac              = 2.0, 1.5  # standard deviation and binarization threshold 
    (first, last)  =  imgNumber
    viewNum        =  (last - first) + 1
    loc            =  np.zeros((viewNum, xNum, yNum, 2), dtype=float)
    shape          =  loc[0].shape

    for i in range(viewNum):
        for j in range(camNum):
            # 1. directory existence check => creating one if not existing
            outdir = Path(f"{datDirectory[j]}/{imgNamePrefix[j]}")
            if not os.path.exists(outdir):
                os.makedirs(outdir, exist_ok=True)
            print(f"\n################ Camera {j + 1} ################")
        
            # 2. determination of all 2D subpixel keypoint locations on the calibration grid from all viewNum pictures (views)
        #for i in range(viewNum):
            print(f"\n######## View {j + 1}.{i + 1} ########")

            savedFile = Path(f"{outdir}/{imgNamePrefix[j]}{i + first}.xml")
            if savedFile.exists():
                loc[i] = pd.read_csv(savedFile, header=None).to_numpy().reshape(shape[0], shape[1], shape[2])
                print(f"All keypoints from View {j + 1}.{i + 1} successfully loaded")
                continue
            
            # Corrected image path without extra directory level
            img_path = f"{imgDirectory[j]}/{imgNamePrefix[j]}{i + first}.jpg"
            print(f"Attempting to load image: {img_path}")

            # Check if the image file exists
            if not Path(img_path).exists():
                print(f"Error: The file {img_path} does not exist.")
                continue
            
            img = cv2.imread(img_path)

            # Check if the image loaded correctly
            if img is None:
                print(f"Error: Unable to load image {img_path}. Please check the file path and ensure the file exists.")
                continue
            
            sd, biFac = findKeypoints(img, coordRatio, innerRatio, xNum, yNum, loc[i], sd, biFac)
            # Convert loc[i] to a 2D array, convert the 2D array into a dataframe and save the dataframe as a csv file
            pd.DataFrame(loc[i].reshape(shape[0], -1)).to_csv(savedFile, header=False, index=False)

    # Add file existence check for all images
    for directory in imgDirectory:
        print(f"Checking files in directory: {directory}")
        files = os.listdir(directory)
        print("Files found:", files)



# %%
