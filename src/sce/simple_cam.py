#TODO: try out the well known opencv methodology to get the calibration matrix 
#NOTE: we can compute the projection matrix from DLT

import glob 

import cv2 
import numpy as np 

""" 
Dev note: 
- Get intrinsic parameters 
- Use visual odometer to figure out the R and t 
- From full projection matrix, estimate ray and the depth. 
""" 

def get_intrinsics_from_checker(img_dir: str, board_dims: tuple=(7, 6)): 
    """Compute intrinsic parameters of camera based on the series of checkerboard images. 

    Parameters 
    ----------
    img_dir: str 
        Path to the directory which contains set of images. 
    board_dims: tuple 
        Dimension of board region to extract the corners. 

    Note
    ----------
    Reference for implementation: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    """
    # termination criteria 
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) 
    
    # define 3D world coordinates 
    obj_points = np.zeros((np.prod(board_dims, dtype=int), 3), dtype=np.float32) 
    obj_points[:, :2] = np.mgrid[0:board_dims[0], 0:board_dims[1]].T.reshape(-1, 2) 
    
    # lists to store the results 
    objp = []   # 3D point in real world coordinate system 
    imgp = []   # 3D points on image plane. 

    # extract image paths 
    img_paths = glob.glob(img_dir + "*")
    
    for path in img_paths: 
        img = cv2.imread(path) 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        # find chessboard corners 
        ret, corners = cv2.findChessboardCorners(gray, board_dims, None) 

        # check if desired number of corners are detected, refine pixel coordinates and store them 
        if ret is True: 
            objp.append(obj_points) 

            # refine pixel coordinates 
            refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria) 
            imgp.append(refined_corners) 
            
            ##### Plot for debug purpose #####
            cv2.drawChessboardCorners(img, board_dims, refined_corners, ret) 
            cv2.imshow("test", img) 
            cv2.watiKey(0) 
        
    cv2.destroyAllWindows() 

    # compute camera calibration with opencv default calibration function 
    ret, mat, dist, rvecs, tvecs = cv2.calibrateCamera(objp, imgp, gray.shape[::-1], None, None) 

    return mat, dist, rvecs, tvecs, objp, imgp  

if __name__ == "__main__": 
    # compute camera calibration
    directory = "../../data/checker/" 
    mat, dist, rvecs, tvecs, objp, imgp = get_intrinsics_from_checker(directory) 
    print(mat) 
    print(dist) 
