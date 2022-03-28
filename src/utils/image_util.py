#TODO: Develop image utility functions 
from collections.abc import Callable 

import cv2 
import numpy as np 

def _register_points(img: np.ndarray, window_name: str=None): 
    """Register the points to track.

    Parameters 
    ---------- 
    img: np.ndarray 
        Image to use for registering point(s) to track.

    Returns
    ----------
    points: np.ndarray 
        Spatial 2D coordinates in image to track. 
    """
    global res 
    res = []
    
    def click_event(event, x, y, flags, params): 
        global res 
        # store and show the clicked coordinates when left click on mouse happens 
        if event == cv2.EVENT_LBUTTONDOWN: 
           res.append([x, y]) 
           cv2.circle(img, (x, y), 3, (0, 255, 0), 2)

    # initialize opencv window
    if window_name is None: 
        window_name = "Register" 

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, click_event)
    
    # run until 'q' key is pressed 
    while 1: 
        cv2.imshow(window_name, img)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break 

    # return coordinates in numpy array format 
    points = res.copy() 
    points = np.asarray(points, dtype=int) 
    cv2.destroyAllWindows()
    return points  

def _init_tracker(img: np.ndarray, corners: np.ndarray): 
    """Initialize tracker with first frame"""
    # globalize variables 
    global prev_frame, p0 
    prev_frame = img.copy()
    p0 = corners.astype(np.float32)
    
def _update_tracker(img: np.ndarray): 
    """Update the trackers."""
    global prev_frame, p0 
    lk_params = dict(winSize=(32, 32), 
                    maxLevel=8, 
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 9, 0.02555))
    p1, _, _ = cv2.calcOpticalFlowPyrLK(prev_frame, img, p0, None, **lk_params)
    prev_frame = img.copy()
    p0 = p1.copy()
    
    return p1

def run_ray_estimation(func: Callable): 
    # NOTE: We might change this function to class 
    """Estimate the direction vector (i.e. ray) from camera to target object.
    
    Parameters 
    ---------- 
    func: Callable 
        Function that computes ray from camera to object. 
    """ 
    
    # Initialize image capturing 
    cap = cv2.VideoCapture(0) 
    if not (cap.isOpened()): 
        raise Exception("Could not open video capturing device.") 

    # register tracker points and initialize tracker 
    win_name = "Ray Estimation" 
    _, init_img = cap.read()
    tracker = _register_points(init_img, window_name=win_name) 
    _init_tracker(init_img, tracker) 

    # TODO: track points, and compute ray 
