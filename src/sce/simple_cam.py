#TODO: try out the well known opencv methodology to get the calibration matrix 
#NOTE: we can compute the projection matrix from DLT
"""
Basic Idea: 
- Estimate intrinsic parameters from set of checkerboard images 
- Estimate initial R, t from visual odometry, and recover 3D point based on the pseudo inverse 
- From pseudo inverse, recover 3D coordinates 
- Or just compute 3D coordinates from SFM (<-- probably this is it)
- Judge the closeness from the given SFM coordinates and translational vector 
"""

import glob 

import cv2 
import numpy as np
from sklearn.preprocessing import scale 

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
            cv2.waitKey(0) 
        
    cv2.destroyAllWindows() 

    # compute camera calibration with opencv default calibration function 
    ret, mat, dist, rvecs, tvecs = cv2.calibrateCamera(objp, imgp, gray.shape[::-1], None, None) 

    return mat, dist, rvecs, tvecs, objp, imgp

def estimate_ray_with_proj(P: np.ndarray, point: np.ndarray, focal_len: float, mu: float = 100): 
    """Given a projection matrix and 2D point on image plane, compute the 3D vector."""
    M, p4 = P[:, :3], np.expand_dims(P[:, 3], axis=-1) 
    track_pts = np.ones((3, 1))
    track_pts[:2, :] = point.T 

    # compute 3D ray estimation 
    Minv = np.linalg.inv(M)
    ray = Minv @ (mu * track_pts - p4)
    ray = ray.T[0]
    
    # scale x and y vectors according to the ratio of focal length and computed z vector 
    scaled_ray = np.zeros((3, 1))
    scaled_ray[:2] = np.expand_dims(ray[:2] * focal_len / ray[-1], axis=1) 
    scaled_ray[2] = focal_len 
    scaled_ray = scaled_ray.T[0] 
    
    return ray, scaled_ray 

class Odometry(object): 
    """
    Note
    ---------
    Implementation reference: https://github.com/alishobeiri/Monocular-Video-Odometery/blob/master/monovideoodometery.py
    """
    def __init__(
        self, 
        init_img: np.ndarray, 
        focal_len: float, 
        cam_center: np.ndarray, 
        error_thresh: float = 1.8, 
        min_features: int = 2000, 
        lk_params: dict = dict(winSize=(32, 32), maxLevel=8, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 9, 0.02555)), 
        detector: cv2.FastFeatureDetector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
    ): 
        # if given image not in grayscale, convert it 
        if len(init_img.shape) > 2: 
            self.prev_frame = cv2.cvtColor(init_img, cv2.COLOR_BGR2GRAY)
        else: 
            self.prev_frame = init_img
        self.focal_len = focal_len
        self.cam_center = cam_center 
        self.error_thresh = error_thresh 
        self.min_features = min_features
        self.lk_params = lk_params
        self.detector = detector 
        
        # internal parameters 
        self.R = np.zeros((3, 3))
        self.t = np.zeros((3, 1))
        self.img_idx = 0 
        self.n_features = 0 
        
        # initialize parameters from given initial image 
        self.p0 = self._detect(self.prev_frame)
        self.p1 = None 

        # create intrinsic matrix 
        self.K = np.diag([self.focal_len, self.focal_len, 1])
        self.K[:2, 2] = self.cam_center.T if self.cam_center.shape == (1, 2) else self.cam_center
        self.Rt = np.zeros((3, 4))
        self.P = self.K @ self.Rt 
        
    def _detect(self, img: np.ndarray): 
        # detect the feature points 
        fp = self.detector.detect(img)
        return np.array([x.pt for x in fp], dtype=np.float32).reshape(-1, 1, 2)
    
    def compute_odometry(self, img: np.ndarray): 
        """Compuote the rotation matrix and translation vector.

        intrinsic_mat: np.ndarray 
            Intrinsic matrix of shape (3, 3)
        """
        # convert length of images 
        if len(img.shape) > 2: 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # extract features 
        if self.n_features < self.min_features: 
            self.p0 = self._detect(self.prev_frame) 
        
        # compute optical flow 
        p1, status, error = cv2.calcOpticalFlowPyrLK(self.prev_frame, img, self.p0, None, **self.lk_params)
        sqrt_error = np.math.sqrt(error[status==1].mean())

        # extract the effective feature points from p0 and computed p1 
        p0_extracted = np.expand_dims(self.p0[status == 1], axis=1) 
        p1_extracted = np.expand_dims(p1[status == 1], axis=1) 

        # compute essential matrix and extrinsic parameters 
        E, _ = cv2.findEssentialMat(p1_extracted, 
                                    p0_extracted, 
                                    self.focal_len, 
                                    self.cam_center, 
                                    cv2.RANSAC, 
                                    0.999, 
                                    1.0, 
                                    None) 

        # compute rotation matrix and translation vector 
        _, R, t, _ = cv2.recoverPose(E, 
                                            p0_extracted, 
                                            p1_extracted, 
                                            R=self.R, t=self.t, 
                                            focal=self.focal_len, 
                                            pp=self.cam_center, 
                                            mask=None) 
         
        # Update number of extracted features 
        if sqrt_error > self.error_thresh or self.img_idx < 1: 
            # update camera matrix and related matrices 
            self.Rt[:, :3] = R @ self.R 
            self.Rt[:, 3] = self.t[:, 0] + self.R @ t[:, 0]
            self.R = self.Rt[:, :3]
            self.t = np.expand_dims(self.Rt[:, 3], axis=1)   
            self.P = self.K @ self.Rt
        
        self.n_features = p1_extracted.shape[0]
        self.prev_frame = img 
        self.img_idx += 1 