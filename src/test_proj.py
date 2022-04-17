import cv2 
import numpy as np 

from sce import Odometry 
from utils import Tracker, draw_points, draw_arrows

if __name__ == "__main__": 
    ##### For test purpose, we use the known intrinsic params ##### 
    FOCAL = 710 
    IN_DEGREE = True 
    VID_PATH = "../data/test_vid/test.mp4"
    OUT_PATH = "../data/test_vid/odometer_test_out.mp4" 
    WINDOW_NAME = "Test"

    # read video and configure the image  
    vid = cv2.VideoCapture(VID_PATH) 

    success, img = vid.read()
    if not success: 
        raise Exception("Video not loaded successfully.") 
    size = (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5)) 
    out = cv2.VideoWriter(OUT_PATH, -1, 20.0, size) 
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA) 
    center = ((img.shape[1]-1)//2+1, (img.shape[0]-1)//2+1)
    center = np.array(center, dtype=float) 

    # initialize tracker 
    tracker = Tracker() 
    tracker.register_points(img, WINDOW_NAME) 

    # initialize odometer 
    odom = Odometry(img, FOCAL, center) 

    cnt = 0
    orig_coord = None 
    while success: 
        # read next frame 
        success, img = vid.read() 
        if not success: 
            break 
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA) 

        # update tracker and odometry 
        tracker.update_tracker(img) 
        odom.compute_odometry(img) 
        
        if cnt == 0: 
            # compute the original estimated 3D point of the tracker 
            Pplus = np.linalg.pinv(odom.P) 
            homogenous_pix = np.ones((3, 1)) 
            homogenous_pix[:2, :] = tracker.points.T 
            orig_coord = (Pplus @ homogenous_pix).T[0]
            orig_coord = orig_coord[:3]/orig_coord[3] 
            print(orig_coord) 

        # compute ray 
        ray = orig_coord - odom.t.T 
        draw_points(img, tracker.points) 
        cv2.putText(img, f"{ray}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, f"{np.linalg.norm(ray)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, f"{odom.t.T}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, f"{orig_coord}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow("WINDOW_NAME", img) 
        cv2.waitKey(1) 
        cnt += 1 


    vid.release() 
    out.release() 
    cv2.destroyAllWindows() 
